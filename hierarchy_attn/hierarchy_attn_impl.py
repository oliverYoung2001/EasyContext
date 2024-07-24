import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from typing import Optional, Tuple
from collections.abc import Iterable
from functools import partial
from .utils import IntraComm, InterComm, OverlappedInterComm, update_out_and_lse

DTYPE32 = torch.float32
# DTYPE32 = torch.bfloat32

# [NOTE]: we don't scale to float32 both in forward and backward !!!

def intra_attn_forward(
    process_groups,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    out: torch.Tensor = None,
    lse: torch.Tensor = None,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
):
    comm = IntraComm(process_groups, PROC_INFO)
    mbs = q.shape[0]
    assert q.shape[0] == k.shape[0] == v.shape[0]
    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]
    out = None
    lse = None
    kv = torch.cat([k, v], dim=0)
    kvs = [kv, torch.empty_like(kv)]


    def forward(q, k, v, causal):
        block_out, _, _, _, _, block_lse, _, _ = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        block_lse = block_lse.transpose(-2, -1).contiguous().unsqueeze(dim=-1)
        return block_out, block_lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]
    
    if causal:
        for step in range(comm.local_size):
            kv = kvs[step & 1]
            k, v = kv[:mbs], kv[mbs:]
            next_kv = kvs[(step & 1) ^ 1]
            if step + 1 != comm.local_size:
                comm.send_recv(kv, next_kv)
                comm.commit()

            # Comp
            if step == 0:
                block_out, block_lse = forward(q, k, v, causal=True)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            elif step <= comm.rank:
                k0 = k[:, :block_seq_len]
                v0 = v[:, :block_seq_len]
                block_out, block_lse = forward(q, k0, v0, causal=False)
                out, lse = update_out_and_lse(out, lse, block_out, block_lse)
            else:
                block_out, block_lse = forward(q1, k, v, causal=False)
                out, lse = update_out_and_lse(
                    out,
                    lse,
                    block_out,
                    block_lse,
                    slice_=(slice(None), slice(block_seq_len, None)),
                )

            if step + 1 != comm.local_size:
                comm.wait()
    else:
        for step in range(comm.local_size):
            kv = kvs[step & 1]
            k, v = kv[:mbs], kv[mbs:]
            next_kv = kvs[(step & 1) ^ 1]
            if step + 1 != comm.local_size:
                comm.send_recv(kv, next_kv)
                comm.commit()

            block_out, block_lse = forward(q, k, v, causal=False)   # [mbs, S, Nh, D], [mbs, S, Nh, 1]
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)   # [mbs, S, Nh, D], [mbs, S, Nh, 1]

            if step + 1 != comm.local_size:
                comm.wait()
    # if out is not None:
    #     out = out.to(q.dtype)
    # if lse is not None:
    #     lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]
    
    
def hierarchy_attn_forward(
    process_groups,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
):
    rank = PROC_INFO['rank']
    world_size = PROC_INFO['world_size']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    assert world_size % local_size == 0
    node_num = world_size // local_size
    comm = InterComm(process_groups, PROC_INFO)
    out, lse = None, None
    _o, _lse = None, None
    qs = [q, torch.empty_like(q)]
    
    o_, lse_ = intra_attn_forward(  # [mbs, S / sp, Nh, D], [mbs, S / sp, Nh, 1]
        process_groups,
        q, k, v,
        softmax_scale,
        None, None,
        dropout_p,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        PROC_INFO
    )
    out, lse = update_out_and_lse(out, lse, o_, lse_)   # [mbs, S / sp, Nh, D], [mbs, S / sp, Nh, 1]
    if node_id == 0:
        os = [None, None]
    else:
        os = [
            torch.empty_like(o_),
            torch.empty_like(lse_),
        ]
    for step in range(node_num - 1):
        _q = comm.send_recv_q(step, q, qs[1])
        comm.commit()
        comm.wait()

        if _o is not None:
            out, lse = update_out_and_lse(out, lse, _o, _lse,) # [mbs, S, Nh, D], [mbs, S, Nh, 1]
        if _q is not None:
            o_, lse_ = intra_attn_forward(  # [mbs, S, Nh, D], [mbs, S, Nh, 1]
                process_groups,
                _q, k, v,
                softmax_scale,
                None, None,
                dropout_p,
                False,
                window_size,
                alibi_slopes,
                deterministic,
                PROC_INFO
            )
        else:
            o_, lse_ = None, None
        _o, _lse = comm.send_recv_o(step, o_, lse_, os[0], os[1]) # [mbs, S, Nh, D], [mbs, S, Nh, 1]
    comm.commit()
    comm.wait()
    if _o is not None:
        out, lse = update_out_and_lse(out, lse, _o, _lse,)

    # out = out.to(q.dtype)
    # lse = lse.squeeze(dim=-1).transpose(1, 2)   # [mbs, S, Nh, D], [mbs, Nh, S]
    return out, lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]


def overlapped_hierarchy_attn_forward(
    process_groups,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
):
    rank = PROC_INFO['rank']
    world_size = PROC_INFO['world_size']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    # print(f'rank: {rank}, world_size: {world_size}, local_rank: {local_rank}, local_size: {local_size}, node_id: {node_id}')
    assert world_size % local_size == 0
    node_num = world_size // local_size
    comm = OverlappedInterComm(process_groups, PROC_INFO)
    out, lse = None, None
    if node_id == node_num - 1:
        qs = [None, None]
    else:
        qs = [torch.empty_like(q), torch.empty_like(q)]
    q_buf = [None, None]
    o_prod_buf = [None, None]
    lse_prod_buf = [None, None]
    o_cons_buf = [None, None]
    lse_cons_buf = [None, None]
    
    # start
    q_buf[0] = comm.send_recv_q(0, q, qs[0])
    comm.commit()
    o_local, lse_local = intra_attn_forward(  # [mbs, S / sp, Nh, D], [mbs, S / sp, Nh, 1]
        process_groups,
        q, k, v,
        softmax_scale,
        None, None,
        dropout_p,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        PROC_INFO
    )
    out, lse = update_out_and_lse(out, lse, o_local, lse_local)   # [mbs, S / sp, Nh, D], [mbs, S / sp, Nh, 1]
    comm.wait()
    
    if node_id == 0:
        os = [None, None]
        lses = [None, None]
    else:
        os = [o_local, torch.empty_like(o_local)]
        lses = [lse_local, torch.empty_like(lse_local)]
    
    for step in range(1, node_num + 1):
        # Comm
        q_buf[1] = comm.send_recv_q(step, q, qs[step & 1])
        o_cons_buf[1], lse_cons_buf[1] = comm.send_recv_o(step, 
                                                        o_prod_buf[0], lse_prod_buf[0], 
                                                        os[step & 1], lses[step & 1]) # [mbs, S, Nh, D], [mbs, S, Nh, 1]
        comm.commit()
        
        # Comp
        if o_cons_buf[0] is not None:
            out, lse = update_out_and_lse(out, lse, o_cons_buf[0], lse_cons_buf[0]) # [mbs, S, Nh, D], [mbs, S, Nh, 1]
        if q_buf[0] is not None:
            o_prod_buf[1], lse_prod_buf[1] = intra_attn_forward(  # [mbs, S, Nh, D], [mbs, S, Nh, 1]
                process_groups,
                q_buf[0], k, v,
                softmax_scale,
                None, None,
                dropout_p,
                False,
                window_size,
                alibi_slopes,
                deterministic,
                PROC_INFO
            )
        else:
            o_prod_buf[1], lse_prod_buf[1] = None, None
        comm.wait()
        # left shift
        for buf in [q_buf, o_prod_buf, lse_prod_buf, o_cons_buf, lse_cons_buf]:
            for _ in range(len(buf) - 1):
                buf[_] = buf[_ + 1]
    
    if o_cons_buf[0] is not None:
        out, lse = update_out_and_lse(out, lse, o_cons_buf[0], lse_cons_buf[0],)

    return out, lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]

def hierarchy_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    pass

def overlapped_hierarchy_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
):
    pass
    
class HierarchyAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        groups,
        PROC_INFO,
        overlapped,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        if overlapped:
            out, softmax_lse = overlapped_hierarchy_attn_forward(
                groups,
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=False,
                PROC_INFO=PROC_INFO,
            )
        else:
            out, softmax_lse = hierarchy_attn_forward(
                groups,
                q,
                k,
                v,
                softmax_scale=softmax_scale,
                dropout_p=dropout_p,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=False,
                PROC_INFO=PROC_INFO,
            )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.groups = groups
        ctx.overlapped = overlapped
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        if ctx.overlapped:
            dq, dk, dv = overlapped_hierarchy_attn_backward(
                ctx.group,
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                softmax_scale=ctx.softmax_scale,
                dropout_p=ctx.dropout_p,
                causal=ctx.causal,
                window_size=ctx.window_size,
                alibi_slopes=ctx.alibi_slopes,
                deterministic=ctx.deterministic,
            )
        else:
            dq, dk, dv = hierarchy_attn_backward(
                ctx.group,
                dout,
                q,
                k,
                v,
                out,
                softmax_lse,
                softmax_scale=ctx.softmax_scale,
                dropout_p=ctx.dropout_p,
                causal=ctx.causal,
                window_size=ctx.window_size,
                alibi_slopes=ctx.alibi_slopes,
                deterministic=ctx.deterministic,
            )
        return dq, dk, dv, None, None, None, None, None, None, None, None


def hierarchy_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    groups=None,
    PROC_INFO=None,
    overlapped=False,
):
    return HierarchyAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        groups,
        PROC_INFO,
        overlapped,
    )
