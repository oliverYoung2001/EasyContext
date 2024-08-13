import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
# from flash_attn_burst.flash_attn_burst_interface import _flash_attn_burst_forward, _flash_attn_burst_backward
from search_algo.execute_plan import Execution_Plan, Fused_Execution_Plan
from .utils import *
from .global_vars import *
from search_algo.dependent_graph import Cuda_Kernel, Comp_Kernel, Comm_Kernel
import copy

def execute_kernel_old(kernel: Cuda_Kernel, data_dict: dict, PROC_INFO, comp_func, comm: IntraComm, idata_buf: dict, causal):
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    # print(f'rank{local_rank}, execute_kernel: {kernel.key}', flush=True)
    # get cuda stream on which kernel is executed
    with torch.cuda.stream(kernel.stream):
        with torch.cuda.nvtx.range(f'{kernel.key}'):
        # if True:
            # step1: wait for precursors
            for precursor in kernel.precursors:
                if hasattr(precursor, 'in_ranks') and rank in precursor.in_ranks:
                    kernel.stream.wait_event(precursor.event)
                    pass
            
            # step2: execute kernel
            # comp: (b_id, h_id, r_id, c_id, gpuid) -> Cuda_Kernel
            # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
            if isinstance(kernel, Comp_Kernel):
                # bid, hid, rid, cid = 0, 0, local_rank, local_rank
                bid, hid, rid, cid = kernel.key[0: 4]
                kernel_causal = causal and isinstance(rid, int) and rid == cid
                # print(f'rank{local_rank}, causal: {kernel_causal}, rid: {rid}, cid: {cid}', flush=True)
                out = comp_func(data_dict[(bid, hid, rid, 'i', 'r')], data_dict[(bid, hid, cid, 'i', 'c')], causal=kernel_causal)
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
                    comm.send(kernel.key[4], data_dict[d_key], kernel.stream, kernel.ncclcomm)
                else:                           # Recv
                    idata_tmp = idata_buf[kernel.key[-2:]]  # ('i'/'o', 'r'/'c')
                    comm.recv(kernel.key[3], idata_tmp, kernel.stream, kernel.ncclcomm)
                    if d_key in data_dict.keys():
                        data_dict[d_key].reduce(idata_tmp)
                    else:
                        data_dict[d_key] = idata_tmp
            
            # step3: record event after kernel execution for successors
            kernel.event = torch.cuda.Event()
            kernel.event.record(kernel.stream)

def execute_kernel(kernel: Cuda_Kernel, data_dict: dict, PROC_INFO, comp_func, comm: IntraComm, buf_dict: dict, causal):
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    # print(f'rank{local_rank}, execute_kernel: {kernel.key}', flush=True)
    # get cuda stream on which kernel is executed
    with torch.cuda.stream(kernel.stream):
        with torch.cuda.nvtx.range(f'{kernel.key}'):
        # if True:
            # step1: wait for precursors
            for precursor in kernel.precursors:
                if hasattr(precursor, 'in_ranks') and rank in precursor.in_ranks:
                    kernel.stream.wait_event(precursor.event)
                    pass
            
            # step2: execute kernel
            # comp: (b_id, h_id, r_id, c_id, gpuid) -> Cuda_Kernel
            # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
            if isinstance(kernel, Comp_Kernel):
                # bid, hid, rid, cid = 0, 0, local_rank, local_rank
                bid, hid, rid, cid = kernel.key[0: 4]
                kernel_causal = causal and isinstance(rid, int) and rid == cid
                rcs = kernel.key[2: 4]
                low_bds = (
                    len(rcs[0]) if isinstance(rcs[0], tuple) else 1,
                    len(rcs[1]) if isinstance(rcs[1], tuple) else 1, 
                )
                # print(f'rank{local_rank}, causal: {kernel_causal}, rid: {rid}, cid: {cid}', flush=True)
                # out = comp_func(data_dict[(bid, hid, rid, 'i', 'r')], data_dict[(bid, hid, cid, 'i', 'c')], causal=kernel_causal)
                out = comp_func(
                    buf_dict[(('i', 'r'), low_bds[0])], buf_dict[(('i', 'c'), low_bds[1])],
                    buf_dict[(('o', 'r'), low_bds[0])], buf_dict[(('o', 'c'), low_bds[1])],
                    kernel_causal
                )
                # o_keys = (bid, hid, rid, 'o', 'r'), (bid, hid, cid, 'o', 'c')   # (or, oc)
                # for t in range(2):  # 0 -> r, 1 -> c
                #     if o_keys[t] in data_dict.keys():
                #         data_dict[o_keys[t]].reduce(out[t])
                #     else:
                #         data_dict[o_keys[t]] = out[t]
            else:
                # d_key = kernel.key[: 3] + kernel.key[5:]
                buf_key = (kernel.key[-2:], 1)  # (('i'/'o', 'r'/'c'), 1)
                if kernel.key[3] == local_rank: # Send
                    # assert d_key in data_dict.keys()
                    # comm.send(kernel.key[4], data_dict[d_key], kernel.stream, kernel.ncclcomm)
                    comm.send(kernel.key[4], buf_dict[buf_key], kernel.stream, kernel.ncclcomm)
                else:                           # Recv
                    # idata_tmp = idata_buf[kernel.key[-2:]]  # ('i'/'o', 'r'/'c')
                    # comm.recv(kernel.key[3], idata_tmp, kernel.stream, kernel.ncclcomm)
                    # if d_key in data_dict.keys():
                    #     data_dict[d_key].reduce(idata_tmp)
                    # else:
                    #     data_dict[d_key] = idata_tmp
                    comm.recv(kernel.key[3], buf_dict[buf_key], kernel.stream, kernel.ncclcomm)
            
            # step3: record event after kernel execution for successors
            kernel.event = torch.cuda.Event()
            kernel.event.record(kernel.stream)

def execute_inter_kernel(kernel: Cuda_Kernel, data_dict: dict, PROC_INFO, comp_func, comm: InterComm, buf_dict: dict, causal):
    rank = PROC_INFO['rank']
    # local_rank = PROC_INFO['local_rank']
    node_id = PROC_INFO['nodeid']
    # print(f'rank{local_rank}, execute_kernel: {kernel.key}', flush=True)
    # get cuda stream on which kernel is executed
    # with torch.cuda.stream(kernel.stream):
    if True:
        with torch.cuda.nvtx.range(f'{kernel.key}'):
        # if True:
            # step1: wait for precursors
            for precursor in kernel.precursors:
                if hasattr(precursor, 'in_ranks') and rank in precursor.in_ranks:
                    if hasattr(precursor, 'sub_streams'):       # precursor is a comp kernel !!!
                        assert isinstance(kernel, Comm_Kernel)  # Current is a comm kernel !!!
                        assert hasattr(precursor, 'events')
                        assert not hasattr(precursor, 'event')
                        kernel.stream.wait_event(precursor.events[0])   # wait Comp Stream
                        kernel.stream.wait_event(precursor.events[2])   # wait Recv Stream
                    else:                                       # precursor is a comm kernel !!!
                        if hasattr(kernel, 'sub_streams'):      # Current is a comp kernel !!!
                            assert isinstance(kernel, Comp_Kernel)
                            kernel.sub_streams[0].wait_event(precursor.event)   # Comp Stream waits ...
                            kernel.sub_streams[1].wait_event(precursor.event)   # Send Stream waits ...
                        else:                                   # Current is a comm kernel !!!
                            kernel.stream.wait_event(precursor.event)   # [NOTE]: hardcode !!!
            
            # step2: execute kernel
            # comp: (b_id, h_id, r_id, c_id, gpuid) or (b_id, h_id, (r_ids, ), (c_ids, ), gpuid) -> Cuda_Kernel
            # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
            if isinstance(kernel, Comp_Kernel):
                assert hasattr(kernel, 'execution_plan')
                intra_execution_plan = kernel.execution_plan
                # bid, hid, rid, cid = 0, 0, local_rank, local_rank
                bid, hid, rid, cid = kernel.key[0: 4]
                kernel_causal = causal and isinstance(rid, int) and rid == cid  # inter causal, both are int, equal
                # print(f'rank{local_rank}, causal: {kernel_causal}, rid: {rid}, cid: {cid}', flush=True)
                out = comp_func(
                    # data_dict[(bid, hid, rid, 'i', 'r')], data_dict[(bid, hid, cid, 'i', 'c')], 
                    inp_row=intra_execution_plan.buf_dict['inp_row'],
                    inp_col=intra_execution_plan.buf_dict['inp_col'],
                    causal=kernel_causal,
                    execution_plan=intra_execution_plan,
                )
                # o_keys = (bid, hid, rid, 'o', 'r'), (bid, hid, cid, 'o', 'c')   # (or, oc)
                # for t in range(2):  # 0 -> r, 1 -> c
                #     if o_keys[t] in data_dict.keys():
                #         data_dict[o_keys[t]].reduce(out[t])
                #     else:
                #         data_dict[o_keys[t]] = out[t]
            else:
                # d_key = kernel.key[: 3] + kernel.key[5:]
                buf_key = (kernel.key[-2:], 1)  # (('i'/'o', 'r'/'c'), 1)
                if kernel.key[3] == node_id: # Send
                    comm.send(kernel.key[4], buf_dict[buf_key], kernel.stream, kernel.ncclcomm)
                else:                           # Recv
                    comm.recv(kernel.key[3], buf_dict[buf_key], kernel.stream, kernel.ncclcomm)
                    # if d_key in data_dict.keys():
                    #     data_dict[d_key].reduce(idata_tmp)
                    # else:
                    #     data_dict[d_key] = idata_tmp
            
            # step3: record event after kernel execution for successors
            if hasattr(kernel, 'sub_streams'):  # Current is a comp kernel !!!
                assert isinstance(kernel, Comp_Kernel)
                kernel.events = [
                    torch.cuda.Event(), # Comp Stream
                    None,               # Send Stream
                    torch.cuda.Event(), # Recv Stream
                ]
                kernel.events[0].record(kernel.sub_streams[0])
                kernel.events[2].record(kernel.sub_streams[2])
            else:                               # Current is a comm kernel !!!
                kernel.event = torch.cuda.Event()
                kernel.event.record(kernel.stream)


def intra_attn_forward(
    inp_row: Input_Row_Fwd,
    inp_col: Input_Col_Fwd,
    softmax_scale,
    dropout_p,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    PROC_INFO,
    execution_plan: Union[Execution_Plan, Fused_Execution_Plan],
    # buf_dict: Union[dict, None],
) -> Integrated_Data:
    local_rank = PROC_INFO['local_rank']
    assert PROC_INFO['tasks_per_node'] == execution_plan.Y * execution_plan.X if hasattr(execution_plan, 'X') else execution_plan.hierarchy_sp
    intra_streams = get_global_var('streams')['intra']
    # Parse buf_dict
    buf_dict = execution_plan.buf_dict
    
    # Comp Func
    def fwd_comp_func(inp_row: Input_Row_Fwd, inp_col: Input_Col_Fwd, 
                      out_row: Output_Row_Fwd, out_col: Output_Col_Fwd, causal) -> tuple:
        # print_rank_0(f'configuous Q: {inp_row.Q.is_contiguous()}, K: {inp_col.K.is_contiguous()}, V: {inp_col.V.is_contiguous()}, O: {out_row.O.is_contiguous()}')
        # print_rank_0(f'shape Q: {inp_row.Q.shape}, K: {inp_col.K.shape}, V: {inp_col.V.shape}, O: {out_row.O.shape}')
        # print(f'rank{local_rank}, dropout_p: {dropout_p}, softmax_scale: {softmax_scale}, , causal: {causal}, window_size: {window_size}, alibi_slopes: {alibi_slopes}, return_softmax: {True and dropout_p > 0}', flush=True)
        O, _, _, _, _, lse, _, _ = _flash_attn_forward(     # O: [mbs, S, Nh, D], lse: [mbs, Nh, S]
            inp_row.Q,
            inp_col.K,
            inp_col.V,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            return_softmax=True and dropout_p > 0,
            out=out_row.O,
        )
        return (out_row, out_col)
        lse = lse.transpose(-2, -1).contiguous().unsqueeze(dim=-1) # block_out, block_lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]
        return (Output_Row_Fwd(O, lse), Output_Col_Fwd())
    
    if buf_dict['graph_type'] == 'general':    # general cases        
        # initialize Comm
        comm = IntraComm(PROC_INFO)
        p_fwd_comp_func = fwd_comp_func
        for kernel in execution_plan.gpu_kernel_lists[local_rank]:
            execute_kernel(kernel, None, PROC_INFO, p_fwd_comp_func, comm, buf_dict, causal)
        return buf_dict['out_row'], buf_dict['out_col']
    elif buf_dict['graph_type'] == 'fused':   # (X, Y) cases
        assert causal == False, 'Intra attn XY not support causal == True'
        X, Y = execution_plan.X, execution_plan.Y
        cur_x_id = local_rank % X
        cur_y_id = local_rank // X  
        comm = IntraComm_fused(PROC_INFO, X, Y)
        or_nelems = buf_dict['or_nelems']
        out_row = buf_dict['or'][or_nelems * cur_x_id: or_nelems * (cur_x_id + 1)]

        event_ir = torch.cuda.Event()
        event_ic = torch.cuda.Event()
        event_comp = torch.cuda.Event()
        # Allgather rc
        comm.all_gather('c', inp_col, buf_dict['ic'], intra_streams[1])
        event_ic.record(intra_streams[1])
        comm.all_gather('r', inp_row, buf_dict['ir'], intra_streams[1])
        event_ir.record(intra_streams[1])
        
        # Comp
        with torch.cuda.stream(intra_streams[0]):
            # step1: input data layout transform
            # if execution_plan.da_config.bs != 1:    # always bs == 1 !!!
            if inp_row.Q.shape[0] != 1:     # bs == 1
                raise NotImplementedError
            Q_shape = list(inp_row.Q.shape)
            assert len(Q_shape) == 4    # (bs, Sq, Nh, D)
            Q_shape_row = copy.deepcopy(Q_shape)
            Q_shape_row[1] *= X         # (bs, X * Sq, Nh, D)
            inp_row_tot = Input_Row_Fwd(buf_dict['ir'].view(Q_shape_row))
            
            K_shape = list(inp_col.K.shape) # (bs, Skv, Nh, D)
            K_shape_col = copy.deepcopy(K_shape)
            K_shape_col[1] *= Y             # (bs, Y * Skv, Nh, D)
            K_shape_col_tot = copy.deepcopy(K_shape)
            K_shape_col_tot = [Y, 2] + K_shape_col_tot  # (Y, 2, bs, Skv, Nh, D)
            intra_streams[0].wait_event(event_ic)
            inp_col_tot_data = buf_dict['ic'].view(K_shape_col_tot).transpose(0, 1).transpose(1, 2).flatten(2, 3).contiguous()    # [2, bs, Y * Skv, Nh, D]
            K_tot = inp_col_tot_data[0]  # [bs, Y * Skv, Nh, D]
            V_tot = inp_col_tot_data[1]  # [bs, Y * Skv, Nh, D]
            inp_col_tot = Input_Col_Fwd(K_tot, V_tot, inp_col_tot_data.flatten())
            
            out_row_tot = Output_Row_Fwd(buf_dict['or'].view(Q_shape_row))
            out_col_tot = None
            
            # step2: execute flashattn kernel
            intra_streams[0].wait_event(event_ir)
            fwd_comp_func(inp_row_tot, inp_col_tot, out_row_tot, out_col_tot, causal)
            
            # step3: output data layout transform
            pass
        
            event_comp.record(intra_streams[0])
        
        # ReduceScatter rc
        intra_streams[2].wait_event(event_comp)
        comm.reduce_scatter('r', buf_dict['or'], out_row, intra_streams[2])
        # comm.all_gather('c', buf_dict['oc'], out_col, streams[2])
        return out_row
    else:
        raise Exception(f"graph_type {buf_dict['graph_type']} not supported !!!")

    
def orchestrated_attn_forward(
    inp_row: Input_Row_Fwd,
    inp_col: Input_Col_Fwd,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
    execution_plan_dict: dict = None,
    # buf_dict: Union[dict, None] = None,     # 'tensor_buf'
    extra_dict: Union[dict, None] = None,   # 'da_config', 'exp_config
) -> Output_Row_Fwd:
    # [NOTE]: Now not support batch mechanism and only support tot_sp = world_size
    # [NOTE]: Now only support bs_split == 1, Nh_split == 1
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    inter_execution_plan: Execution_Plan = execution_plan_dict['inter']
    
    # Parse buf_dict
    buf_dict = inter_execution_plan.buf_dict
        
    da_config = inter_execution_plan.da_config
    assert inter_execution_plan.hierarchy_sp == PROC_INFO['node_num']
    assert inter_execution_plan.split_degrees[2] == 1 and inter_execution_plan.split_degrees[3] == 1
    
    # # Inter-level
    # data_dict = {}  # (b_id, h_id, r/c_id, i/o, r/c) -> Integrated_Data
    # initialize Comm
    comm = InterComm(PROC_INFO)
    
    p_fwd_inter_comp_func = partial(
        intra_attn_forward, 
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        # execution_plan.da_config.causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        PROC_INFO=PROC_INFO,
        
    )    # [NOTE]: fuse of not fuse !!!
    for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
        execute_inter_kernel(kernel, None, PROC_INFO, p_fwd_inter_comp_func, comm, buf_dict, causal)
    return inter_execution_plan.buf_dict['out_row'], inter_execution_plan.buf_dict['out_col']

def intra_attn_backward(
    inp_row: Input_Row_Bwd,
    inp_col: Input_Col_Bwd,
    softmax_scale,
    dropout_p,
    causal,
    window_size,
    alibi_slopes,
    deterministic,
    PROC_INFO,
    execution_plan: Union[Execution_Plan, Fused_Execution_Plan],
    # buf_dict: Union[dict, None],
) -> Integrated_Data:
    local_rank = PROC_INFO['local_rank']
    assert PROC_INFO['tasks_per_node'] == execution_plan.Y * execution_plan.X if hasattr(execution_plan, 'X') else execution_plan.hierarchy_sp
    intra_streams = get_global_var('streams')['intra']
    # Parse buf_dict
    buf_dict = execution_plan.buf_dict
    
    # Comp Func
    def bwd_comp_func(inp_row: Input_Row_Bwd, inp_col: Input_Col_Bwd, 
                      out_row: Output_Row_Bwd, out_col: Output_Col_Bwd, causal) -> tuple:
        # print(f'rank{local_rank}, dropout_p: {dropout_p}, softmax_scale: {softmax_scale}, , causal: {causal}, window_size: {window_size}, alibi_slopes: {alibi_slopes}, return_softmax: {True and dropout_p > 0}', flush=True)
        _flash_attn_backward(
            inp_row.dO,
            inp_row.Q,
            inp_col.K,
            inp_col.V,
            inp_row.Q, # dummy O # inp_row.O,
            inp_row.lse,
            out_row.dQ,
            out_col.dK,
            out_col.dV,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            alibi_slopes,
            deterministic,
            rng_state=None,
            softmax_d=inp_row.D,
        )
        return (out_row, out_col)
    
    if buf_dict['graph_type'] == 'general':    # general cases
        # initialize Comm
        comm = IntraComm(PROC_INFO)
        p_bwd_comp_func = bwd_comp_func
        for kernel in execution_plan.gpu_kernel_lists[local_rank]:
            execute_kernel(kernel, None, PROC_INFO, p_bwd_comp_func, comm, buf_dict, causal)
        return buf_dict['out_row'], buf_dict['out_col']
    elif buf_dict['graph_type'] == 'fused':   # (X, Y) cases
        assert causal == False, 'Intra attn XY not support causal == True'
        X, Y = execution_plan.X, execution_plan.Y
        cur_x_id = local_rank % X   # [0, X)
        cur_y_id = local_rank // X  # [0, Y)
        comm = IntraComm_fused(PROC_INFO, X, Y)
        ir_nelems= buf_dict['ir_nelems']
        or_nelems = buf_dict['or_nelems']
        oc_nelems = buf_dict['oc_nelems']
        out_row = buf_dict['or'][or_nelems * cur_x_id: or_nelems * (cur_x_id + 1)]
        out_col = buf_dict['oc'][oc_nelems * cur_y_id: oc_nelems * (cur_y_id + 1)]

        event_ir = torch.cuda.Event()
        event_ic = torch.cuda.Event()
        event_oc = torch.cuda.Event()
        event_comp = torch.cuda.Event()
        # # Allgather rc
        comm.all_gather('r', inp_row, buf_dict['ir'], intra_streams[1])
        event_ir.record(intra_streams[1])
        comm.all_gather('c', inp_col, buf_dict['ic'], intra_streams[1])
        event_ic.record(intra_streams[1])
        
        # Comp
        with torch.cuda.stream(intra_streams[0]):
            # step1: input data layout transform
            # if execution_plan.da_config.bs == 1:    # always true !!!
            if inp_row.Q.shape[0] == 1:     # bs == 1
                Q_shape = list(inp_row.Q.shape)     # (bs, Sq, Nh, D)
                Q_nelem = math.prod(Q_shape)
                K_shape = list(inp_col.K.shape)     # (bs, Skv, Nh, D)
                K_nelem = math.prod(K_shape)
                lse_shape = list(inp_row.lse.shape) # (bs, Nh, Sq)
                lse_nelem = math.prod(lse_shape)
                assert len(Q_shape) == 4        # (bs, Sq, Nh, D)
                assert len(K_shape) == 4        # (bs, Skv, Nh, D)
                assert len(lse_shape) == 3      # (bs, Nh, Sq)

                # ir_tot
                intra_streams[0].wait_event(event_ir)
                Q_shape_row = copy.deepcopy(Q_shape)
                Q_shape_row[1] *= X         # (bs, X * Sq, Nh, D)
                Q_nelem_row = math.prod(Q_shape_row)
                lse_shape_row = copy.deepcopy(lse_shape)
                assert lse_shape_row[1] == 1, 'Now Nh > 1 not supported in fused backward !!!'    # Nh == 1
                lse_shape_row[2] *= X       # (bs, Nh, X * Sq)
                lse_nelem_row = math.prod(lse_shape_row)
                t_list = []
                offsets = [0, Q_nelem, 2 * Q_nelem, 2 * Q_nelem + 2 * lse_nelem, 2 * Q_nelem + 3 * lse_nelem]   # Q, dO, D, lse
                assert ir_nelems == offsets[-1]
                for idx in range(len(offsets) - 1):
                    for x in range(X):
                        t_list.append(buf_dict['ir'][ir_nelems * x + offsets[idx]: ir_nelems * x + offsets[idx + 1]].flatten())
                inp_row_tot_data = torch.cat(t_list, out=buf_dict['ir_'])    # [NOTE]: passed !!!
                # inp_row_tot_data = torch.cat(t_list)                       # failed !!!
                assert inp_row_tot_data.numel() == 2 * Q_nelem_row + 3 * lse_nelem_row
                
                Q_tot = inp_row_tot_data[: Q_nelem_row].view(Q_shape_row)
                dO_tot = inp_row_tot_data[Q_nelem_row: 2 * Q_nelem_row].view(Q_shape_row)
                D_tot = inp_row_tot_data[2 * Q_nelem_row: - lse_nelem_row].view(FULL_DTYPE).view(lse_shape_row)
                lse_tot = inp_row_tot_data[- lse_nelem_row: ].view(lse_shape_row)
                inp_row_tot = Input_Row_Bwd(Q_tot, dO_tot, D_tot, lse_tot, inp_row_tot_data)
                
                # ic_tot
                K_shape_col = copy.deepcopy(K_shape)
                K_shape_col[1] *= Y     # (bs, Y * Skv, Nh, D)
                K_nelem_col = math.prod(K_shape_col)
                K_shape_col_tot = copy.deepcopy(K_shape)
                K_shape_col_tot = [Y, 2] + K_shape_col_tot  # (Y, 2, bs, Skv, Nh, D)
                intra_streams[0].wait_event(event_ic)
                inp_col_tot_data = buf_dict['ic'].view(K_shape_col_tot).transpose(0, 1).transpose(1, 2).flatten(2, 3).contiguous() # [2, bs, Y * Skv, Nh, D]
                K_tot = inp_col_tot_data[0]  # [bs, Y * Skv, Nh, D]
                V_tot = inp_col_tot_data[1]  # [bs, Y * Skv, Nh, D]
                inp_col_tot = Input_Col_Bwd(K_tot, V_tot, inp_col_tot_data.flatten())
                
                # or_tot
                out_row_tot = Output_Row_Bwd(buf_dict['or'].view(Q_shape_row))
                
                # oc_tot
                dK_tot = buf_dict['oc'][: K_nelem_col].view(K_shape_col)
                dV_tot = buf_dict['oc'][K_nelem_col: ].view(K_shape_col)
                out_col_tot = Output_Col_Bwd(dK_tot, dV_tot, buf_dict['oc'].flatten())
            else:
                raise NotImplementedError()
            # step2: execute flashattn kernel
            # print_rank_0(f"buf['ir']: {buf_dict['ir'].data_ptr()}, buf['ic']: {buf_dict['ic'].data_ptr()}, "
            #              f"buf['or']: {buf_dict['or'].data_ptr()}, buf['oc']: {buf_dict['oc'].data_ptr()}")
            # print_rank_0(f'inp_row_tot: {inp_row_tot.data.data_ptr()}, inp_col_tot: {inp_col_tot.data.data_ptr()}, '
            #              f'out_row_tot: {out_row_tot.data.data_ptr()}, out_col_tot: {out_col_tot.data.data_ptr()}')
            bwd_comp_func(inp_row_tot, inp_col_tot, out_row_tot, out_col_tot, causal)
            event_comp.record(intra_streams[0])
        
        # ReduceScatter rc
        intra_streams[2].wait_event(event_comp)
        comm.reduce_scatter('r', buf_dict['or'], out_row, intra_streams[2])
        with torch.cuda.stream(intra_streams[0]):
            # step3: output  layout transform
            out_col_tot_data = buf_dict['oc'].view((2, Y, K_nelem)).transpose(0, 1).contiguous()    # [Y, 2, bs, Skv, Nh, D]
            event_oc.record(intra_streams[0])
        intra_streams[2].wait_event(event_oc)
        comm.reduce_scatter('c', out_col_tot_data, out_col, intra_streams[2])
        return (out_row, out_col)
    else:
        raise Exception(f"graph_type {buf_dict['graph_type']} not supported !!!")
    
    
def orchestrated_attn_backward(
    inp_row: Input_Row_Bwd,
    inp_col: Input_Col_Bwd,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    PROC_INFO=None,
    execution_plan_dict: dict = None,    
    # buf_dict: Union[dict, None] = None,     # 'tensor_buf'
    extra_dict: Union[dict, None] = None,   # 'da_config', 'exp_config
) -> tuple: # (Output_Row_Bwd, Output_Col_Bwd)
        # [NOTE]: Now not support batch mechanism and only support tot_sp = world_size
    # [NOTE]: Now only support bs_split == 1, Nh_split == 1
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    inter_execution_plan: Execution_Plan = execution_plan_dict['inter']
    
    # Parse buf_dict
    buf_dict = inter_execution_plan.buf_dict
        
    da_config = inter_execution_plan.da_config
    assert inter_execution_plan.hierarchy_sp == PROC_INFO['node_num']
    assert inter_execution_plan.split_degrees[2] == 1 and inter_execution_plan.split_degrees[3] == 1
    
    # # Inter-level
    # data_dict = {}  # (b_id, h_id, r/c_id, i/o, r/c) -> Integrated_Data
    # initialize Comm
    comm = InterComm(PROC_INFO)
    
    p_bwd_inter_comp_func = partial(
        intra_attn_backward,
        softmax_scale=softmax_scale,
        dropout_p=dropout_p,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        PROC_INFO=PROC_INFO,
    )
    for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
        execute_inter_kernel(kernel, None, PROC_INFO, p_bwd_inter_comp_func, comm, buf_dict, causal)
    return inter_execution_plan.buf_dict['out_row'], inter_execution_plan.buf_dict['out_col']

class OrchestratedAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp_row: Input_Row_Fwd,
        inp_col: Input_Col_Fwd,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        PROC_INFO,
        execution_plan_dict: dict,
        # buf_dict: Union[dict, None],
        extra_dict: Union[dict, None],
    ):
        if softmax_scale is None:
            softmax_scale = inp_row.Q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        # k = k.contiguous()
        # v = v.contiguous()
        # return
        out_row = orchestrated_attn_forward(
            inp_row,
            inp_col,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            PROC_INFO=PROC_INFO,
            execution_plan_dict=execution_plan_dict,
            # buf_dict=buf_dict,
            extra_dict=extra_dict,
        )
        return
        out, softmax_lse = out_row.O, out_row.lse
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.PROC_INFO = PROC_INFO
        ctx.execution_plan = execution_plan
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = orchestrated_attn_backward(
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
    inp_row: Input_Row_Fwd, 
    inp_col: Input_Col_Fwd,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    PROC_INFO=None,
    execution_plan_dict: dict = None,
    # buf_dict: Union[dict, None] = None,
    extra_dict: Union[dict, None] = None,
):
    return OrchestratedAttnFunc.apply(
        inp_row,
        inp_col,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        PROC_INFO,
        execution_plan_dict,
        # buf_dict,
        extra_dict,
    )
