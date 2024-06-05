import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from typing import Optional, Tuple
from collections.abc import Iterable

DTYPE32 = torch.float32
# DTYPE32 = torch.bfloat32

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    print_rank_0(f'out: {out.shape}, lse: {lse.shape}, block_out: {block_out.shape}, block_lse: {block_lse.shape}')

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


def wait_recursively(reqs):
    if reqs is None:
        return
    for req in reqs:
        if hasattr(req, 'wait'):
            req.wait()
        elif isinstance(req, Iterable):
            wait_recursively(req)
        else:
            raise Exception("Unknown type of reqs in wait_recursively !!!")

            
class InterComm:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        assert self.world_size % self.local_size == 0
        self.node_num = self.world_size // self.local_size
        if 'world' not in process_groups.keys():
            process_groups['world'] = None
        if 'inter' not in process_groups.keys():
            for i in range(self.local_size):
                ranks = range(i, self.world_size, self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['inter'] = group
        
        self.process_groups = process_groups       
        self._ops = []
        self._reqs = None            
    
    def send_recv_q(self, step, q: torch.Tensor, _q: Optional[torch.Tensor] = None):
        offset = step // 2 + 1 if step % 2 == 0 else self.node_num - (step // 2 + 1)
        send_node_id = self.node_id - offset
        recv_node_id = self.node_id + offset
        
        if 0 <= send_node_id < self.node_num:
            send_op = dist.P2POp(dist.isend, q, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_op)
        if 0 <= recv_node_id < self.node_num:
            if _q is None:
                _q = torch.empty_like(q)
            recv_op = dist.P2POp(dist.irecv, _q, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_op)
        return _q

    def send_recv_o(self, step, o: torch.Tensor, lse: torch.Tensor, _o: Optional[torch.Tensor] = None, _lse: Optional[torch.Tensor] = None):
        # [TODO]: combine o and lse
        offset = step // 2 + 1 if step % 2 == 0 else self.node_num - (step // 2 + 1)
        send_node_id = self.node_id + offset
        recv_node_id = self.node_id - offset
        
        if 0 <= send_node_id < self.node_num:
            send_o_op = dist.P2POp(dist.isend, o, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_o_op)
            send_lse_op = dist.P2POp(dist.isend, lse, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_lse_op)
            
        if 0 <= recv_node_id < self.node_num:
            assert(_o is not None and _lse is not None)
            recv_o_op = dist.P2POp(dist.irecv, _o, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_o_op)
            recv_lse_op = dist.P2POp(dist.irecv, _lse, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_lse_op)
        return _o, _lse
    
    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        if len(self._ops) > 0:
            self._reqs = dist.batch_isend_irecv(self._ops)
            self._ops = []
    
    def wait(self):
        wait_recursively(self._reqs)
        self._reqs = None
        
        
class IntraComm:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        if 'intra' not in process_groups.keys():
            for i in range(self.world_size // self.local_size):
                ranks = range(i * self.local_size, (i + 1) * self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['intra'] = group
        self.process_groups = process_groups
        self.send_rank = self.node_id * self.local_size + (self.local_rank + 1) % self.local_size
        self.recv_rank = self.node_id * self.local_size + (self.local_rank - 1) % self.local_size
        self._ops = []
        self._reqs = None
        
    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self.process_groups['intra'])
        recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self.process_groups['intra'])
        self._ops.append(recv_op)
        self._ops.append(send_op)
        # print(f'id(res): {id(res)}')
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        if len(self._ops) > 0:
            self._reqs = dist.batch_isend_irecv(self._ops)
            self._ops = []

    def wait(self):
        wait_recursively(self._reqs)
        self._reqs = None
    
def inner_attn_forward(
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

    block_seq_len = q.shape[1] // 2
    q1 = q[:, block_seq_len:]
    out = None
    lse = None
    ks = [k, torch.empty_like(k)]
    vs = [v, torch.empty_like(v)]


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
        # print_rank_0(f'block_out: {block_out.dtype}, block_lse: {block_lse.dtype}') # bf16, bf32
        return block_out, block_lse
    
    if causal:
        for step in range(comm.world_size):
            k, v = ks[step & 1], vs[step & 1]
            next_k, next_v = ks[(step & 1) ^ 1], vs[(step & 1) ^ 1]
            if step + 1 != comm.world_size:
                comm.send_recv(k, next_k)
                comm.send_recv(v, next_v)
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

            if step + 1 != comm.world_size:
                comm.wait()
    else:
        for step in range(comm.world_size):
            k, v = ks[step & 1], vs[step & 1]
            next_k, next_v = ks[(step & 1) ^ 1], vs[(step & 1) ^ 1]
            if step + 1 != comm.world_size:
                comm.send_recv(k, next_k)
                # print(f'id(next_k): {id(next_k)}')
                comm.send_recv(v, next_v)
                # print(f'id(next_v): {id(next_v)}')
                comm.commit()

            block_out, block_lse = forward(q, k, v, causal=False)
            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            if step + 1 != comm.world_size:
                comm.wait()
    # print_rank_0(f'out: {out.shape}, {out.dtype}, lse: {lse.shape}, {lse.dtype}')   # [mbs, S / sp, Nh, D], fp32, [mbs, S / sp, Nh, 1], fp32
    if out is not None:
        out = out.to(q.dtype)
    # if lse is not None:
    #     lse = lse.squeeze(dim=-1).transpose(1, 2)
    return out, lse
    
    
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
    _o, _lse = None, None
    # print_rank_0(f'qkv: {q.shape}, {k.shape}, {v.shape}')   # [mbs, S / sp, Nh, D]
    
    out, lse = inner_attn_forward(
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
    # print_rank_0(f'out: {out.shape}, {out.dtype}, lse: {lse.shape}, {lse.dtype}') # [mbs, S / sp, Nh, D], [mbs, Nh, S / sp]
    if node_id == 0:
        os = [None, None]
    else:
        os = [
            torch.empty_like(out),
            torch.empty_like(lse),
        ]
    for step in range(node_num - 1):
        _q = comm.send_recv_q(step, q)
        comm.commit()
        comm.wait()
        if _o is not None:
            out, lse = update_out_and_lse(out, lse, _o, _lse,)
        if _q is not None:
            o_, lse_ = inner_attn_forward(
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
        _o, _lse = comm.send_recv_o(step, o_, lse_, os[0], os[1])
    comm.wait()
    if _o is not None:
        out, lse = update_out_and_lse(out, lse, _o, _lse,)

    out = out.to(q.dtype)   # ?
    lse = lse.squeeze(dim=-1).transpose(1, 2)   # ?
    return out, lse
    
    
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
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
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
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
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
    )