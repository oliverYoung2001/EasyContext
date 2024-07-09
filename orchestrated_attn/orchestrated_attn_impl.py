import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from search_algo.execute_plan import Execution_Plan
from .utils import *
from .global_vars import *
from search_algo.dependent_graph import Cuda_Kernel, Comp_Kernel, Comm_Kernel

def execute_kernel(kernel: Cuda_Kernel, data_dict: dict, PROC_INFO, comp_func, comm: IntraComm, idata_buf: dict):
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    # get cuda stream on which kernel is executed
    if isinstance(kernel, Comp_Kernel):
        kernel.stream = get_global_var('streams')[0]
    else:
        if kernel.key[3] == local_rank:
            kernel.stream = get_global_var('streams')[1]    # Send
        else:
            kernel.stream = get_global_var('streams')[2]    # Recv
    with torch.cuda.stream(kernel.stream):
        # step1: wait for precursors
        for precursor in kernel.precursors:
            if hasattr(precursor, 'in_ranks') and local_rank in precursor.in_ranks:
                kernel.stream.wait_event(precursor.event)
        
        # step2: execute kernel
        # comp: (b_id, h_id, r_id, c_id, gpuid) -> Cuda_Kernel
        # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
        if not hasattr(kernel, 'in_ranks'): # [NOTE]: maybe need a lock here !!!
            kernel.in_ranks = set()
        kernel.in_ranks.add(local_rank)
        if isinstance(kernel, Comp_Kernel):
            bid, hid, rid, cid = kernel.key[0: 4]
            causal = rid == cid
            out = comp_func(data_dict[(bid, hid, rid, 'i', 'r')], data_dict[(bid, hid, cid, 'i', 'c')], causal=causal)
            o_keys = (bid, hid, rid, 'o', 'r'), (bid, hid, cid, 'o', 'c')   # (or, oc)
            for t in range(2):  # 0 -> r, 1 -> c
                if o_keys[t] in data_dict.keys():
                    data_dict[o_keys[t]].reduce(out[t])
                else:
                    data_dict[o_keys[t]] = out[t]
        else:
            d_key = kernel.key[: 3] + kernel.key[5:]
            if kernel.key[3] == local_rank: # Send
                assert d_key in data_dict.keys()
                comm.send(kernel.key[4], data_dict[d_key], kernel.stream)
            else:                           # Recv
                idata_tmp = idata_buf[kernel.key[-2:]]
                comm.recv(kernel.key[3], idata_tmp, kernel.stream)
                if d_key in data_dict.keys():
                    data_dict[d_key].reduce(idata_tmp)
                else:
                    data_dict[d_key] = idata_tmp
        
        # step3: record event after kernel execution for successors
        kernel.event = torch.cuda.Event()
        kernel.event.record()

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
    execution_plan: Execution_Plan = None,
) -> Integrated_Data:
    rank = PROC_INFO['rank']
    world_size = PROC_INFO['world_size']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    assert world_size % local_size == 0
    node_num = world_size // local_size
    assert local_size == execution_plan.tot_sp
    if not is_exist_global_var('streams'):
        streams = []
        streams.append(torch.cuda.current_stream())
        for _ in range(1, execution_plan.stream_num):
            streams.append(torch.cuda.Stream(torch.cuda.current_device()))
        set_global_var('streams', streams)
    streams = get_global_var('streams')
    data_dict = {}  # (b_id, h_id, r/c_id, i/o, r/c) -> Integrated_Data
    
    # initialize Comm
    comm = IntraComm(process_groups, PROC_INFO)
    
    # Comp Func
    def fwd_comp_func(inp_row: Input_Row_Fwd, inp_col: Input_Col_Fwd, causal) -> tuple:
        O, _, _, _, _, lse, _, _ = _flash_attn_forward(
            inp_row.Q,
            inp_col.K,
            inp_col.V,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
        )
        return (Output_Row_Fwd(O), Output_Col_Fwd())
        lse = lse.transpose(-2, -1).contiguous().unsqueeze(dim=-1) # block_out, block_lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]
        return (Output_Row_Fwd(O, lse), Output_Col_Fwd())
    
    # initial data:
    ir_idata, ic_idata = Input_Row_Fwd(q), Input_Col_Fwd(k, v)
    data_dict[(0, 0, local_rank, 'i', 'r')] = ir_idata
    data_dict[(0, 0, local_rank, 'i', 'c')] = ic_idata
    idata_buf = {
        ('i', 'r'): Input_Row_Fwd.from_idata(ir_idata),
        ('i', 'c'): Input_Col_Fwd.from_idata(ic_idata),
        ('o', 'r'): Output_Row_Fwd.from_execution_plan(execution_plan, q),
        ('o', 'c'): Output_Col_Fwd(),
    }
    for kernel in execution_plan.gpu_kernel_lists[local_rank]:
        execute_kernel(kernel, data_dict, PROC_INFO, fwd_comp_func, comm, idata_buf)
    print(f'rank{rank}, Out !!!', flush=True)
    return data_dict[(0, 0, local_rank, 'o', 'r')]
    
    
def orchestrated_attn_forward(
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
    execution_plan: Execution_Plan = None,
) -> Output_Row_Fwd:
    # [NOTE]: Now not support batch mechanism and only support tot_sp = world_size
    # [NOTE]: Now only support bs_split == 1, Nh_split == 1
    da_config = execution_plan.da_config
    assert execution_plan.tot_sp == PROC_INFO['world_size']
    assert execution_plan.split_degrees[2] == 1 and execution_plan.split_degrees[3] == 1
    if execution_plan.da_config.SP[0] == 1:
        return intra_attn_forward(
            process_groups,
            q, k, v,
            softmax_scale,
            None, None,
            dropout_p,
            execution_plan.da_config.causal,
            window_size,
            alibi_slopes,
            deterministic,
            PROC_INFO,
            execution_plan,
        )
    raise NotImplementedError

def orchestrated_attn_backward(
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
    PROC_INFO=None,
    execution_plan: Execution_Plan = None,
):
    pass

class OrchestratedAttnFunc(torch.autograd.Function):
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
        execution_plan: Execution_Plan,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out_row = orchestrated_attn_forward(
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
            execution_plan=execution_plan,
        )
        out, softmax_lse = out_row.O, out_row.lse
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.groups = groups
        ctx.PROC_INFO = PROC_INFO
        ctx.execution_plan = execution_plan
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = orchestrated_attn_backward(
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
            PROC_INFO=ctx.PROC_INFO,
            execution_plan=ctx.execution_plan,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None

def orchestrated_attn_func(
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
    execution_plan: Execution_Plan = None,
):
    return OrchestratedAttnFunc.apply(
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
        execution_plan,
    )
