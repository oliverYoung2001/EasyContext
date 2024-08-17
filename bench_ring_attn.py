from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import torch.distributed as dist
import torch.distributed
import torch.distributed
from ring_flash_attn import (
    # ring_flash_attn_qkvpacked_func,
    # zigzag_ring_flash_attn_qkvpacked_func,
    # stripe_flash_attn_qkvpacked_func,
    ring_flash_attn_func,
    zigzag_ring_flash_attn_func,
    stripe_flash_attn_func,
    ring_flash_attn_backward,
    zigzag_ring_flash_attn_backward,
    stripe_flash_attn_backward,
)
import torch.distributed
from hierarchy_attn.hierarchy_attn_impl import hierarchy_attn_func
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func, orchestrated_attn_backward
import torch.cuda
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from easy_context.dist_flash_attn.lightseq_async_attn import attention as lightseq_attn
from easy_context.dist_flash_attn.async_communication import initialize_distributed
from search_algo.search_engine import Dist_Attn_Config, Evaluation_Configs, Inter_Comp_Profile_Map
from search_algo.dependent_graph import Cuda_Kernel, Comp_Kernel, Comm_Kernel
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from orchestrated_attn.global_vars import *
from orchestrated_attn.utils import *
from search_algo.utils import select_best_schedule_in_node_profile_data
from search_algo.execute_plan import Execution_Plan, Fused_Execution_Plan
import inspect
import warnings
import time
import socket
import pickle
from functools import partial
import regex as re
import copy
warnings.filterwarnings("ignore")   # disable warning caused by lightseq

PROC_INFO: dict
DTYPE = torch.bfloat16
placeholder_op = None
def dummy_placeholder_op(*args, **kwargs):
    pass
    
# def zigzag_ring_flash_attn_func_opt(*args, **kwargs):
#     print(f'args: {args}, kwargs: {kwargs}')
#     return zigzag_ring_flash_attn_func(*args, **kwargs, opt=True)
zigzag_ring_flash_attn_func_opt = partial(zigzag_ring_flash_attn_func, opt=True)
zigzag_ring_flash_attn_func_opt.__name__ = 'zigzag_ring_flash_attn_func_opt'
overlapped_hierarchy_attn_func = partial(hierarchy_attn_func, overlapped=True)
overlapped_hierarchy_attn_func.__name__ = 'overlapped_hierarchy_attn_func'

def lightseq_attn_func(q, k, v, causal, sm_scale):
    return lightseq_attn(q, k, v, causal, sm_scale)
   
def parse_slurm_tasks_per_node(tasks_per_node):
    # 4(x2), 8, ...
    return int(tasks_per_node.split('(')[0])
     
def get_proc_info():
    if os.getenv('SLURM_PROCID', None) is not None:    # launch with Slurm
        rank = int(os.environ['SLURM_PROCID'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        ip = os.environ['SLURM_STEP_NODELIST']
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.environ['SLURM_CLUSTER_NAME']
        nodeid = int(os.environ['SLURM_NODEID'])
        nodename = os.environ['SLURMD_NODENAME']
        tasks_per_node = parse_slurm_tasks_per_node(os.environ['SLURM_TASKS_PER_NODE'])
        
    elif os.getenv('OMPI_COMM_WORLD_RANK', None) is not None: # launch with OpenMPI
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # ip = os.environ['SLURM_STEP_NODELIST']
        ip = None
        hostname = socket.gethostname()
        hostip = socket.gethostbyname(hostname)
        clustername = os.getenv('CLUSTER_NAME', 'Unknown Cluster')
        # nodeid = int(os.environ['SLURM_NODEID'])
        # nodename = os.environ['SLURMD_NODENAME']
        nodename = None
        # tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
        tasks_per_node = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        nodeid = rank // tasks_per_node
        
    else:
        raise Exception("Unknown Launcher !!!")
    proc_info = {
        'clustername': clustername,
        'hostname': hostname,
        'nodename': nodename,
        'nodeid': nodeid,
        'world_size': world_size,
        'tasks_per_node': tasks_per_node,
        'rank': rank,
        'local_rank': local_rank,
        'hostip': hostip,
        'ip': ip,
        'deviceid': local_rank,
    }
    proc_info['node_num'] = world_size // tasks_per_node
    # print(f'proc_info: {proc_info}')
    return proc_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attn-mode", type=str, choices=['zigzag_ring', 'lightseq', 'local_flash'], default="flash")
    parser.add_argument('--profiler-with-tensorboard', action='store_true', default=False, help='whether to profile with tensorboard')
    parser.add_argument('--tb-dir', default=None, type=str, help='where to storage tensorboard files')

    args = parser.parse_args()
    return args

def filter_kwargs(func, kwargs):
    sig = inspect.signature(func)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}

def calc_flops(mbs, S: tuple, Nh, D, causal=True, fob=0):
    flops = 2 * 2 * mbs * S[0] * S[1] * Nh * D
    if causal:
        flops = flops // 2
    if fob == 0:
        m_flops = h_flops = flops
    elif fob == 1:
        m_flops = 2 * flops
        h_flops = 2.5 * flops
    elif fob == 2:
        m_flops = (1 + 2) * flops
        h_flops = (1 + 2.5) * flops
    return m_flops, h_flops # model flops & hardware flops
   
def get_XY_from_plan_path(plan_path: str):
    pat = re.compile(r'^.*Y=(\d+)_X=(\d+)_.*$',)
    res = pat.match(plan_path)
    assert res is not None, f'Invalid plan_path: {plan_path}'
    return (int(res.group(2)), int(res.group(1)))   # (X, Y)
     
def run_func(f, inputs, num_iter, main_stream, forward_only, qkv=None, dout=None):
    torch.cuda.synchronize()
    torch.distributed.barrier()
        
    event_start = torch.cuda.Event(enable_timing=True)
    event_end = torch.cuda.Event(enable_timing=True)
    placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
    event_start.record(stream=main_stream)
    if forward_only:
        with torch.no_grad():
            for _ in range(num_iter):
                _ = f(**inputs)
    else:
        for _ in range(num_iter):
            qkv.grad = None
            out = f(**inputs)
            out.backward(dout)
    event_end.record(stream=main_stream)
    torch.cuda.synchronize()
    td = event_start.elapsed_time(event_end) / 1000 # s
    
    td = torch.tensor(td, device=torch.cuda.current_device())
    torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MAX, async_op=False)
    torch.cuda.synchronize()
    td = td.cpu().item()
    return td
            
def benchmark(args, f, da_config: Dist_Attn_Config, tensor_buf, global_group, warmup=11, num_iter=20, fob=0, log=True):
    warmup = 2
    num_iter = 4
    main_stream = torch.cuda.current_stream()
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.time()
    global PROC_INFO
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    if rank == 0:
        print(f'# {f.__name__}, {"fwd" if fob == 0 else "bwd"}', flush=True)
    if fob == 1:
        fwd_f = f
        if f == ring_flash_attn_func:
            f = ring_flash_attn_backward
        elif f == zigzag_ring_flash_attn_func:
            f = zigzag_ring_flash_attn_backward
        elif f == stripe_flash_attn_func:
            f = stripe_flash_attn_backward
        else:
            raise Exception(f'Invalid f: {f}')
        
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    # torch.cuda.set_device(device)

    batch_size = da_config.bs
    Sq, Skv = da_config.S_per_gpu   # Sq, Skv per gpu !!!
    nheads = da_config.Nh[0]    # Nh, Ng
    d = da_config.D
    if f == flash_attn_func:
        Sq *= world_size
        Skv *= world_size
        world_size = 1
    dropout_p = 0
    causal = da_config.causal
    deterministic = False

    # assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    Q_shape = (batch_size, Sq, nheads, d)
    K_shape = (batch_size, Skv, nheads, d)
    lse_shape = (batch_size, nheads, Sq)
    Q_numel = math.prod(Q_shape)
    K_numel = math.prod(K_shape)
    lse_numel = math.prod(lse_shape)
    q = tensor_buf[: Q_numel].view(Q_shape)
    k = tensor_buf[Q_numel: Q_numel + K_numel].view(K_shape)
    v = tensor_buf[Q_numel + K_numel: Q_numel + K_numel * 2].view(K_shape)
    dout = tensor_buf[Q_numel + K_numel * 2: Q_numel * 2 + K_numel * 2].view(Q_shape)
    out = tensor_buf[Q_numel * 2 + K_numel * 2: Q_numel * 3 + K_numel * 2].view(Q_shape)
    softmax_lse = tensor_buf[Q_numel * 3 + K_numel * 2: Q_numel * 3 + K_numel * 2 + lse_numel * 2].view(FULL_DTYPE).view(lse_shape)
    qkv = tensor_buf[: Q_numel + K_numel * 2]
    # qkv.requires_grad = True
    # bsz, nh, seq_len, hdim
    if f == lightseq_attn_func:
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        dout = dout.permute(0, 2, 1, 3)
    
    inputs = {
        "q": q,
        "k": k,
        "v": v,
        "dropout_p": dropout_p,
        "causal": causal,
        "deterministic": deterministic,
        "sm_scale": q.shape[-1] ** (-0.5),
        "PROC_INFO": PROC_INFO,
        "groups": {},
    }
    if fob == 1:
        # # get groups from executing fwd !!!
        # inputs = filter_kwargs(fwd_f, inputs)
        # fwd_f(**inputs)

        inputs = {
            "process_group": {},
            "dout": dout,
            "q": q,
            "k": k,
            "v": v,
            "out": out,
            "softmax_lse": softmax_lse,
            "dropout_p": dropout_p,
            "causal": causal,
            "deterministic": deterministic,
            "sm_scale": q.shape[-1] ** (-0.5),
            "softmax_scale": q.shape[-1] ** (-0.5),
            "PROC_INFO": PROC_INFO,
            "groups": {},
        }
    inputs = filter_kwargs(f, inputs)
    
    # warmup
    # global placeholder_op
    placeholder_op = dummy_placeholder_op
    placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
    with torch.no_grad():
        for _ in range(warmup):
            _ = f(**inputs)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)   
    # print_rank_0(f'Warmup done !!!')
    
    use_cudagraph = False
    
    is_runned = False
    t1 = time.time()

    # warmup cudagraph
    if use_cudagraph:
        for _ in range(warmup):
            g.replay()
            torch.cuda.synchronize()
            torch.distributed.barrier(group=global_group)  
        # print_rank_0(f'Warmup cudagraph done !!!')

                
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group) 
    t2 = time.time()
    
    td = - 1
    # if args.profiler_with_tensorboard and not hasattr(args, "tb_profiled"):
    if args.profiler_with_tensorboard:
        args.tb_profiled = True
        is_runned = True
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'{os.environ["TRACE_NAME"]}_w{world_size}_r{rank}_S({Sq},{Skv})_bs{batch_size}_Nh{nheads}_D{nheads}_{f.__name__}_{"f" if forward_only else "f+b"}'
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=f'{args.tb_dir}', 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter in range(TOTAL_TURNS):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                if use_cudagraph:
                    g.replay()
                else:
                    if iter % BARRIER_FREQ == 0:
                        placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
                    if forward_only:
                        with torch.no_grad():
                            _ = f(**inputs)
                    else:
                        qkv.grad = None
                        out = f(**inputs)
                        out.backward(dout)
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier(group=global_group)
                prof.step()
        
        num_iter = TOTAL_TURNS
        
    if not is_runned: 
        # run
        if use_cudagraph:
            for _ in range(num_iter):
                g.replay()
        else:
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
            event_start.record(stream=main_stream)
            # preprocess
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = f(**inputs)
            # postprocess
            event_end.record(stream=main_stream)
            torch.cuda.synchronize()
            td = event_start.elapsed_time(event_end) / 1000 # s
            # if fob == 0:
            #     td = run_func(f, inputs, num_iter, main_stream, foward_only=True)
            # else:
            #     td0 = run_func(f, inputs, num_iter, main_stream, foward_only=True)
            #     td1 = run_func(f, inputs, num_iter, main_stream, foward_only=False, qkv=qkv, dout=dout)
            #     td = td1 - td0
    
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    t3 = time.time()
    assert td >= 0, f'Invalid td: {td}'
    if td < 0:
        td = t3 - t2
    else:
        td = torch.tensor(td, device=device)
        torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MAX, async_op=False)
        torch.cuda.synchronize()
        td = td.cpu().item()

    if rank == 0 and log:
        m_flops, h_flops = calc_flops(batch_size, (Sq * world_size, Skv * world_size), nheads, d, causal, fob = fob)
        mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
        print(f"mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)

def all_wait_main_stream(stream_list: list, main_stream: torch.cuda.Stream):
    for stream in stream_list:
        if stream.cuda_stream != main_stream.cuda_stream:
            stream.wait_stream(main_stream)

def main_stream_wait_all(stream_list: list, main_stream: torch.cuda.Stream):
    for stream in stream_list:
        if stream.cuda_stream != main_stream.cuda_stream:
            main_stream.wait_stream(stream)
        
def benchmark_ops(streams, global_group, device, f, inputs, \
                warmup, warmup_cudagraph, num_iter, use_cudagraph, TRACE_NAME, args):
    # print_rank_0(f'streams: {streams}')
    torch.cuda.empty_cache()
    main_stream = streams['intra'][0] # intra comp stream
    stream_list = streams['intra'] + streams['inter']
    # warmup
    global placeholder_op
    placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
    # preprocess
    all_wait_main_stream(stream_list, main_stream)
    with torch.no_grad():
        for _ in range(warmup):
            _ = f(**inputs)
    # postprocess
    main_stream_wait_all(stream_list, main_stream)
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)   
    # # [NOTE]: we don't barrier here to prevent WARN of 
    # # "[W CUDAGraph.cpp:145] Warning: Waiting for pending NCCL work to finish before starting graph capture. (function operator())"
    # print_rank_0(f'Warmup done !!!')

    assert use_cudagraph == False, "Not support cudagraph in this version !!!"
    if use_cudagraph:
        if args.profiler_with_tensorboard:
            with torch.profiler.profile():  # workaround of issue 75504 of PyTorch
                pass
        # Capture cuda graph
        # torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        # return
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=streams[0]):
            pass
            # preprocess
            for stream in streams[1:]:
                stream.wait_stream(torch.cuda.current_stream())
            with torch.no_grad():
                _ = f(**inputs)
            # postprocess
            for stream in streams[1:]:
                torch.cuda.current_stream().wait_stream(stream)
        
    is_runned = False
    
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    t1 = time.time()
    
    # warmup cudagraph
    if use_cudagraph:
        for _ in range(warmup_cudagraph):
            g.replay()
            # torch.cuda.synchronize()
            # torch.distributed.barrier(group=global_group)
        # print_rank_0(f'Warmup cudagraph done !!!')

                
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    
    t2 = time.time()
    td = - 1
    # assert args.profiler_with_tensorboard == False, "Not support profiler_with_tensorboard in this version !!!"
    # if args.profiler_with_tensorboard and not hasattr(args, "tb_profiled"):
    if args.profiler_with_tensorboard:
        args.tb_profiled = True
        is_runned = True
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=f'{args.tb_dir}', 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter in range(TOTAL_TURNS):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                if use_cudagraph:
                    g.replay()
                else:
                    if iter % BARRIER_FREQ == 0:
                        placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
                    # preprocess
                    all_wait_main_stream(stream_list, main_stream)
                    with torch.no_grad():
                        _ = f(**inputs)
                    # postprocess
                    main_stream_wait_all(stream_list, main_stream)
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier(group=global_group)
                prof.step()
        
        num_iter = TOTAL_TURNS
        
    if not is_runned: 
        # run
        if use_cudagraph:
            for _ in range(num_iter):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                g.replay()
                # torch.cuda.synchronize()    # almost no effect on performance
                # torch.distributed.barrier(group=global_group)   # 64TFlops -> 43TFlops
        else:
            for i in range(3):
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                placeholder_op(stream=main_stream) # [NOTE]: aim to eliminate cpu overhead
                event_start.record(stream=main_stream)
                # preprocess
                all_wait_main_stream(stream_list, main_stream)
                with torch.no_grad():
                    for _ in range(num_iter):
                        _ = f(**inputs)
                # postprocess
                main_stream_wait_all(stream_list, main_stream)
                event_end.record(stream=main_stream)
                torch.cuda.synchronize()
                td = event_start.elapsed_time(event_end) / 1000 # s

    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    t3 = time.time()
    if td < 0:
        td = t3 - t2
    else:
        td = torch.tensor(td, device=device)
        torch.distributed.all_reduce(td, op=torch.distributed.ReduceOp.MAX, async_op=False)
        torch.cuda.synchronize()
        td = td.cpu().item()
    torch.cuda.empty_cache()
    return t1, t2, t3, td

def create_buf_dict(da_config: Dist_Attn_Config, exp_config: Evaluation_Configs, execution_plan: Union[Execution_Plan, Fused_Execution_Plan], fused: bool, \
                    batch_degrees: tuple, tensor_buf: torch.Tensor, level_rank: int) -> dict:
    assert len(batch_degrees) == 2, f'Invalid batch_degrees: {batch_degrees}'   # [Q_batch_degree, KV_batch_degree]
    bs = da_config.bs
    Sq, Skv = da_config.S_per_gpu
    Sq *= batch_degrees[0]
    Skv *= batch_degrees[1]
    Nhq, Nhg = da_config.Nh
    d = da_config.D
    
    fob = exp_config.fob
    # buf_dict : (('i'/'o', 'r'/'c'), batch_degree) -> Integrated_Data
    if not fused:
        integrated_data_types = {
            ('i', 'r'): Input_Row_Fwd if fob == 0 else Input_Row_Bwd,
            ('i', 'c'): Input_Col_Fwd if fob == 0 else Input_Col_Bwd,
            ('o', 'r'): Output_Row_Fwd if fob == 0 else Output_Row_Bwd,
            ('o', 'c'): Output_Col_Fwd if fob == 0 else Output_Col_Bwd,
        }
        # Q_buf = tensor_buf[2 * bs * Skv * Nhg * d * 2: ]    # last 2 stands for the largest low_dg of all execution plans !!!
        Q_buf = tensor_buf
        KV_buf = tensor_buf
        # batch_degrees = (1, 1):   # for input/output and comm kernels
        buf_dict = {
            (('i', 'r'), 1): integrated_data_types[('i', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * 1),
            (('i', 'c'), 1): integrated_data_types[('i', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * 1),
            (('o', 'r'), 1): integrated_data_types[('o', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * 1),
            (('o', 'c'), 1): integrated_data_types[('o', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * 1),
        }
        for kernel in execution_plan.gpu_kernel_lists[level_rank]:
            # [TODO]: Ignore Comm Kenrel here, because kernel fusion of comm kernel is not supported now
            if isinstance(kernel, Comp_Kernel):
                rcs = kernel.key[2: 4]
                low_bds = (
                    len(rcs[0]) if isinstance(rcs[0], tuple) else 1,
                    len(rcs[1]) if isinstance(rcs[1], tuple) else 1, 
                )
                if (('i', 'r'), low_bds[0]) not in buf_dict.keys():
                    buf_dict[(('i', 'r'), low_bds[0])] = integrated_data_types[('i', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * low_bds[0])
                    buf_dict[(('o', 'r'), low_bds[0])] = integrated_data_types[('o', 'r')].from_da_config_with_buf(da_config, Q_buf, batch_degrees[0] * low_bds[0])
                if (('i', 'c'), low_bds[1]) not in buf_dict.keys():
                    buf_dict[(('i', 'c'), low_bds[1])] = integrated_data_types[('i', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * low_bds[1])
                    buf_dict[(('o', 'c'), low_bds[1])] = integrated_data_types[('o', 'c')].from_da_config_with_buf(da_config, KV_buf, batch_degrees[1] * low_bds[1])
        
        buf_dict['graph_type'] = 'general'
        buf_dict['tensor_buf'] = tensor_buf
        buf_dict['inp_row'] = buf_dict[(('i', 'r'), 1)]
        buf_dict['inp_col'] = buf_dict[(('i', 'c'), 1)]
        buf_dict['out_row'] = buf_dict[(('o', 'r'), 1)]
        buf_dict['out_col'] = buf_dict[(('o', 'c'), 1)]
    else:
        assert isinstance(execution_plan, Fused_Execution_Plan), f'Invalid execution_plan: {execution_plan}'
        Y, X = execution_plan.Y, execution_plan.X
        if fob == 0:    # forward
            ir_nelems = bs * Sq * Nhq * d   # q
            ic_nelems = bs * Skv * Nhg * (d * 2)   # k, v
            or_nelems = bs * Sq * Nhq * d   # o, (lse)
            oc_nelems = 0
        else:   # backward
            ir_nelems = bs * Sq * Nhq * (d * 2 + 1 * (2 + 1))   # q, do, D, lse
            ic_nelems = bs * Skv * Nhg * (d * 2)   # k, v
            or_nelems = bs * Sq * Nhq * d        # dq
            oc_nelems = bs * Skv * Nhg * (d * 2)  # dk, dv

        ir_tot = ir_nelems * X
        ic_tot = ic_nelems * Y
        or_tot = or_nelems * X
        oc_tot = oc_nelems * Y
        # print_rank_0(f'ir_tot: {ir_tot}, ic_tot: {ic_tot}, or_tot: {or_tot}, oc_tot: {oc_tot}')
        buf = tensor_buf[: ir_tot + ic_tot + or_tot + oc_tot]
        cur_offset = ir_tot + ic_tot + or_tot + oc_tot
        # buf = torch.empty(ir_tot + ic_tot + or_tot + oc_tot, dtype=DTYPE, device=torch.cuda.current_device())
        # print_rank_0(f'buf: {buf.numel() * 2} B')
        buf_dict = {
            'ir': buf[: ir_tot],
            'ic': buf[ir_tot: ir_tot + ic_tot],
            'or': buf[ir_tot + ic_tot: ir_tot + ic_tot + or_tot],
            'oc': buf[ir_tot + ic_tot + or_tot: ],
            'ir_nelems': ir_nelems,
            'ic_nelems': ic_nelems,
            'or_nelems': or_nelems,
            'oc_nelems': oc_nelems,
            'ir_tot': ir_tot,
            'ic_tot': ic_tot,
            'or_tot': or_tot,
            'oc_tot': oc_tot,
            'graph_type': 'fused',
        }
        if fob == 1:
            # inp_row_extra_buf = torch.empty(ir_tot, dtype=DTYPE, device=torch.cuda.current_device())
            inp_row_extra_buf = tensor_buf[cur_offset: cur_offset + ir_tot]
            cur_offset += ir_tot
            buf_dict['ir_'] = buf_dict['ir']
            buf_dict['ir'] = inp_row_extra_buf
        ir_class = Input_Row_Fwd if fob == 0 else Input_Row_Bwd
        ic_class = Input_Col_Fwd if fob == 0 else Input_Col_Bwd
        cur_x_id = level_rank % X   # [0, X)
        cur_y_id = level_rank // X  # [0, Y)
        buf_dict['inp_row'] = ir_class.from_da_config_with_buf(da_config=da_config, buf=buf_dict['ir'][ir_nelems * cur_x_id: ir_nelems * (cur_x_id + 1)], batch_degree=batch_degrees[0])
        buf_dict['inp_col'] = ic_class.from_da_config_with_buf(da_config=da_config, buf=buf_dict['ic'][ic_nelems * cur_y_id: ic_nelems * (cur_y_id + 1)], batch_degree=batch_degrees[1])
    return buf_dict

def benchmark_orchestrate(args, raw_f, da_config: Dist_Attn_Config, tensor_buf: torch.Tensor, warmup=11, num_iter=20, log=True, 
                          exp_configs: list = [], global_group=None, ncclcomm_global: PyNcclCommunicator = None, 
                          use_cudagraph=False, mode='test', inter_comp_plans_dicts: Optional[dict] = None):
    if mode == 'profile':
        assert inter_comp_plans_dicts is not None, "inter_comp_plans_dicts should be determined for profiling"
    # print_rank_0(f'[INFO]: use_cudagraph: {use_cudagraph}')
    warmup_cudagraph = 100
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    global PROC_INFO
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    node_id = PROC_INFO['nodeid']
    node_num = PROC_INFO['node_num']
    # print(f'rank{rank}, node_id: {node_id}, node_num: {node_num}', flush=True)
    if rank == 0:
        print(f'# {raw_f.__name__}', flush=True)
    world_size = dist.get_world_size()
    # device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    # torch.cuda.set_device(device)

    # Configs:
    batch_size = da_config.bs
    Sq, Skv = da_config.S_per_gpu   # Sq, Skv per gpu !!!
    nheads = da_config.Nh[0]    # Nh, Ng
    d = da_config.D
    dropout_p = 0
    deterministic = False

    assert d % 8 == 0
    
    def create_inputs_and_buf_dict(exp_config: Evaluation_Configs, inter_execution_plan: Execution_Plan, inter_comp_plans: dict):
        # qkvdo = tensor_buf[: 4 * batch_size * seqlen * nheads * d].view(4 * batch_size, seqlen, nheads, d)
        # # qkv.requires_grad = True
        # q, k, v, do = qkvdo.chunk(4, dim=0)
        # D_buf = tensor_buf[4 * batch_size * seqlen * nheads * d: ]
        # D = D_buf[: 2 * batch_size * seqlen * nheads * 1].view(FULL_DTYPE).view(batch_size, nheads, seqlen)   # [mbs, Nh, S], torch.float32, 2 stands for 2 torch.bfloat16
        # lse_buf = D_buf[2 * batch_size * seqlen * nheads * 1: ]
        # # q, k, v, do, o = qkvdoo.chunk(5, dim=0)
        # # lse_buf = tensor_buf[5 * batch_size * seqlen * nheads * d: ]
        # lse = lse_buf[: batch_size * seqlen * nheads * 1].view(batch_size, nheads, seqlen)   # [mbs, Nh, S]
        
        # Create buf_dict for inter_comp_plans
        
        extra_dict = {
            'da_config': da_config,
            'exp_configs': exp_config,
        }
        inter_buf_dict = create_buf_dict(
            da_config, exp_config, inter_execution_plan, 
            isinstance(inter_execution_plan, Fused_Execution_Plan), 
            (1, 1), tensor_buf, node_id)
        inter_execution_plan.buf_dict =  inter_buf_dict
        # Create buf_dict for inter_comp_plans
        for inter_comp_key, intra_execution_plan in inter_comp_plans.items():
            intra_execution_plan.buf_dict = create_buf_dict(    # [TODO]
                da_config, exp_config, intra_execution_plan, 
                isinstance(intra_execution_plan, Fused_Execution_Plan), 
                inter_comp_key[0], tensor_buf, local_rank)
        
        inputs = {
            "inp_row": inter_buf_dict['inp_row'], 
            "inp_col": inter_buf_dict['inp_col'],
            "dropout_p": dropout_p,
            "causal": None,
            "deterministic": deterministic,
            "sm_scale": d ** (-0.5),
            "softmax_scale": d ** (-0.5),
            "PROC_INFO": PROC_INFO,
            # 'buf_dict': buf_dict,
            'extra_dict': extra_dict,
        }
        return inputs


    for exp_config in exp_configs:
        # print_rank_0(f'exp_config: {exp_config}')
        # print(f'rank{rank}, exp_config: {exp_config}', flush=True)
        fob = exp_config.fob
        plan_path = exp_config.plan_path
        plan_type = exp_config.plan_type
        inter_comp_profile_map = exp_config.inter_comp_profile_map
        
        torch.cuda.empty_cache()
        t0 = time.time()
        SP = da_config.SP
        Ss = (Sq * world_size, Skv * world_size)
        Nhs = (nheads, nheads)
        bs = batch_size
        if exp_config.execution_plan is not None:
            inter_execution_plan = exp_config.execution_plan
        else:
            # load inter plan
            with open(plan_path, 'rb') as fin:
                inter_execution_plan = pickle.load(fin)
        # print_rank_0(f'inter_execution_plan:')
        # if rank == 0:
        #     if isinstance(inter_execution_plan, Fused_Execution_Plan):
        #         print_rank_0(f'fused_inter_execution_plan: {inter_execution_plan}')
        #     else:
        #         inter_execution_plan.print_lp_result()
        
        # path for intra execution plans:
        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{local_size}_fob={fob}'
        if mode == 'test':
            # load intra plans and form a dict for test mode
            split_degrees = (da_config.SP[0], da_config.SP[0], 1, 1)
            inter_comp_plans = {}   # batch_degrees -> plan, ranks with the same node_id have the same inter_comp_plans
            inter_comp_configs = {}   # batch_degrees -> configs 
            if isinstance(inter_execution_plan, Fused_Execution_Plan):
                assert da_config.causal == False, "Causal is not supported for fused execution plan"
                batch_degrees = (inter_execution_plan.X, inter_execution_plan.Y)
                inter_comp_key = (batch_degrees, False)
                if inter_comp_profile_map is not None:
                    map_key = inter_comp_profile_map.get_comp_map_key(da_config, inter_comp_key[0], split_degrees, inter_comp_key[1])
                    intra_full_attn_config = inter_comp_profile_map.get_value_from_map_key(map_key)[fob] # (Y, X, fused, Time, ~~causal~~)
                else:
                    # raise NotImplementedError
                    intra_full_attn_config = (da_config.SP[1], 1, da_config.Nh[0] == 1, - 0.0, False)  # kv
                    # intra_full_attn_config = (1, da_config.SP[1], da_config.Nh[0] == 1, - 0.0, False)  # qo
                inter_comp_configs[inter_comp_key] = intra_full_attn_config
                if intra_full_attn_config[2] == 0:  # not fused, to load intra execution plan
                    plan_path = f'{par_dir}/SP={local_size}_fob={fob}_Y={intra_full_attn_config[0]}_X={intra_full_attn_config[1]}_dim=0.pkl'
                    with open(plan_path, 'rb') as fin:
                        inter_comp_plans[inter_comp_key] = pickle.load(fin)
                else:
                    inter_comp_plans[inter_comp_key] = Fused_Execution_Plan(intra_full_attn_config[0], intra_full_attn_config[1], intra_full_attn_config[3], False, fob=fob)
                # necessary for fused execution plan !!!
                inter_execution_plan.kernel_execution_plan = inter_comp_plans[inter_comp_key]
            else:
                for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
                    if not isinstance(kernel, Comp_Kernel):
                        continue
                    kernel_causal = da_config.causal and isinstance(kernel.key[2], int) and kernel.key[2] == kernel.key[3]  # global causal, both are int, equal
                    rcs = kernel.key[2: 4]
                    batch_degrees = (
                        len(rcs[0]) if isinstance(rcs[0], tuple) else 1,
                        len(rcs[1]) if isinstance(rcs[1], tuple) else 1, 
                    )
                    inter_comp_key = (batch_degrees, kernel_causal) # (batch_degrees, causal)
                    if inter_comp_key not in inter_comp_plans.keys():
                        if inter_comp_key[1] == True:   # Causal
                            # HACK
                            old_sp = da_config.SP
                            old_S = da_config.S
                            da_config.SP = (1, old_sp[1])
                            da_config.S = (old_S[0] // old_sp[0], old_S[1] // old_sp[0])
                            plan_name_prefix = da_config.get_plan_name(fob=fob)
                            da_config.SP = old_sp
                            da_config.S = old_S
                            # HACK Done
                            
                            suffix = f'_ablation1'
                            plan_path = f'{par_dir}_causal=True/{plan_name_prefix}{suffix}.pkl'
                            with open(plan_path, 'rb') as fin:
                                inter_comp_plans[inter_comp_key] = pickle.load(fin)
                            inter_comp_configs[inter_comp_key] = (- 1, - 1, 1, - 0.0, True)
                        else:                           # non causal
                            if inter_comp_profile_map is not None:
                                map_key = inter_comp_profile_map.get_comp_map_key(da_config, inter_comp_key[0], split_degrees, inter_comp_key[1])
                                intra_full_attn_config = inter_comp_profile_map.get_value_from_map_key(map_key)[fob] # (Y, X, fused, Time, ~~causal~~)
                            else:
                                # raise NotImplementedError
                                intra_full_attn_config = (da_config.SP[1], 1, da_config.Nh[0] == 1, - 0.0, kernel_causal)  # kv
                                # intra_full_attn_config = (1, da_config.SP[1], da_config.Nh[0] == 1, - 0.0, kernel_causal)  # qo

                            inter_comp_configs[inter_comp_key] = intra_full_attn_config
                            if intra_full_attn_config[2] == 0:  # not fused, to load intra execution plan
                                plan_path = f'{par_dir}/SP={local_size}_fob={fob}_Y={intra_full_attn_config[0]}_X={intra_full_attn_config[1]}_dim=0.pkl'
                                with open(plan_path, 'rb') as fin:
                                    inter_comp_plans[inter_comp_key] = pickle.load(fin)
                            else:
                                inter_comp_plans[inter_comp_key] = Fused_Execution_Plan(intra_full_attn_config[0], intra_full_attn_config[1], intra_full_attn_config[3], False, fob=fob)
                    kernel.execution_plan = inter_comp_plans[inter_comp_key]
                            
            # print(f'rank{rank}, node_id: {node_id}, inter_comp_configs: {inter_comp_configs}', flush=True)
            inter_comp_plans_dicts = [inter_comp_plans]
            
        # continue  # above OK !!!
        causal = da_config.causal
        if fob == 0:
            f = raw_f
        else:
            f = orchestrated_attn_backward
        
        no_fused = True
        for inter_comp_plans in inter_comp_plans_dicts:
            if mode == 'profile' and isinstance(list(inter_comp_plans.values())[0], Fused_Execution_Plan) and no_fused:
                if rank == 0:
                    print(f'# {raw_f.__name__} fused', flush=True)
                no_fused = False
            if mode == 'profile':   # Set execution_plan for the only comp kernel in inter_execution_plan
                assert node_num == 1, "Not support multi-node for profiling"
                assert len(inter_execution_plan.gpu_kernel_lists[node_id]) == 1, "Not support multi-kernel for profiling"
                only_kernel = inter_execution_plan.gpu_kernel_lists[node_id][0]
                assert isinstance(only_kernel, Comp_Kernel), "The only kernel in profiling must be Comp_Kernel"
                assert len(inter_comp_plans.values()) == 1, "Only 1 intra execution plan is needed for profiling"
                only_kernel.execution_plan = list(inter_comp_plans.values())[0]
            
            inputs = create_inputs_and_buf_dict(exp_config, inter_execution_plan, inter_comp_plans)
            inputs['causal'] = causal
            # Mark in_ranks on execution_plans to judge whether kernel is on current rank easily
            if isinstance(inter_execution_plan, Execution_Plan):
                for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
                    kernel.in_ranks = set([rank])
            for _, intra_execution_plan in inter_comp_plans.items():
                if not isinstance(intra_execution_plan, Execution_Plan):
                    continue
                for kernel in intra_execution_plan.gpu_kernel_lists[local_rank]:
                    kernel.in_ranks = set([rank])
            # Modify da_config
            if isinstance(inter_execution_plan, Execution_Plan):
                if inter_execution_plan.da_config.S != Ss:
                    inter_execution_plan.da_config.S = Ss
                if inter_execution_plan.da_config.Nh != Nhs:
                    inter_execution_plan.da_config.Nh = Nhs
            # Create streams for both inter and intra execution plans
            if exp_config.plan_type == 'ablation0':
                raise NotImplementedError
                stream_num = 1 # comp stream
                for kernel in execution_plan.gpu_kernel_lists[local_rank]:
                    if not isinstance(kernel, Comp_Kernel): # Comp
                        stream_num += 1
                execution_plan.stream_num = stream_num
            if not is_exist_global_var('streams'):
                streams = {
                    'inter': [],    # Comms
                    'intra': [],    # Comp, Comms
                }
            else:
                streams = get_global_var('streams')
            # Create Streams for Inter Execution Plan
            if len(streams['inter']) < inter_execution_plan.stream_num:
                priorities = [0, - 1, - 2]
                priorities = [0] * inter_execution_plan.stream_num
                for _ in range(len(streams['inter']), inter_execution_plan.stream_num):
                    streams['inter'].append(torch.cuda.Stream(torch.cuda.current_device(), priority=priorities[_]))
            #  Create Streams for Intra Execution Plans
            for _, intra_execution_plan in inter_comp_plans.items():
                if not isinstance(intra_execution_plan, Execution_Plan):
                    intra_stream_num = 3
                else:
                    intra_stream_num = intra_execution_plan.stream_num
                if len(streams['intra']) < intra_stream_num:
                    priorities = [0, - 1, - 2]
                    priorities = [0] * intra_stream_num
                    for _ in range(len(streams['intra']), intra_stream_num):
                        streams['intra'].append(torch.cuda.Stream(torch.cuda.current_device(), priority=priorities[_]))
            set_global_var('streams', streams)
                
            # Set streams for each kernel of both inter and intra execution plans
            if exp_config.plan_type == 'ablation0':
                raise NotImplementedError
                comm_stream_id = 1
                for kernel in execution_plan.gpu_kernel_lists[local_rank]:
                    if isinstance(kernel, Comp_Kernel):
                        kernel.stream = streams[0]
                    else:
                        kernel.stream = streams[comm_stream_id]
                        comm_stream_id += 1
                assert comm_stream_id == execution_plan.stream_num
            else:
                # Set Streams for Intra Execution Plans
                for _, intra_execution_plan in inter_comp_plans.items():
                    if not isinstance(intra_execution_plan, Execution_Plan):
                        continue
                    for kernel in intra_execution_plan.gpu_kernel_lists[local_rank]:
                        if isinstance(kernel, Comp_Kernel):
                            kernel.stream = streams['intra'][0]
                        else:
                            if kernel.key[3] == local_rank:
                                kernel.stream = streams['intra'][1]    # Send
                            else:
                                kernel.stream = streams['intra'][2]    # Recv
                # Set Streams for Inter Execution Plan
                if isinstance(inter_execution_plan, Execution_Plan):
                    for kernel in inter_execution_plan.gpu_kernel_lists[node_id]:
                        # print_rank_0(f'inter kernel: {kernel.key}, {isinstance(kernel, Comp_Kernel)}, {isinstance(kernel, Comm_Kernel)}')
                        if isinstance(kernel, Comp_Kernel):
                            # kernel.stream = streams['inter'][0]
                            assert len(streams['intra']) >= 3
                            kernel.sub_streams = streams['intra'][: 3] # [NOTE]: hardcode here !!! not support ablation0 !!!
                            assert not hasattr(kernel, 'stream')
                        else:
                            if kernel.key[3] == node_id:
                                kernel.stream = streams['inter'][1]    # Send
                            else:
                                kernel.stream = streams['inter'][2]    # Recv
                            assert not hasattr(kernel, 'sub_streams')
            torch.cuda.synchronize()
            torch.distributed.barrier(group=global_group)
            # print(f'rank{rank}, Create Streams Done !!!', flush=True)
            
            # Build nccl communicator for each pair of ranks
            ncclcomm_dict = get_global_var('ncclcomm_dict')
                # For Inter
            if isinstance(inter_execution_plan, Fused_Execution_Plan):
                # [TODO]
                # Create Row&Col PyNcclCommunicator
                Y = inter_execution_plan.Y
                X = inter_execution_plan.X
                cur_x_id = node_id % X
                cur_y_id = node_id // X
                # r_key = tuple(range(node_id * local_size + cur_y_id * X, node_id * local_size + (cur_y_id + 1) * X))
                # c_key = tuple(range(node_id * local_size + cur_x_id, (node_id + 1) * local_size, X))
                r_key = tuple(range(cur_y_id * X * local_size + local_rank, (cur_y_id + 1) * X * local_size + local_rank, local_size))
                c_key = tuple(range(cur_x_id * local_size + local_rank, world_size, X * local_size))
                # print(f'rank{rank}, r_key: {r_key}, c_key: {c_key}', flush=True)
                assert rank in r_key and rank in c_key
                if r_key not in ncclcomm_dict.keys():
                    ncclcomm_dict[r_key] = PyNcclCommunicator(global_group, ranks=r_key, device=torch.cuda.current_device())
                if c_key not in ncclcomm_dict.keys():
                    ncclcomm_dict[c_key] = PyNcclCommunicator(global_group, ranks=c_key, device=torch.cuda.current_device())
            else:
                for kernel in inter_execution_plan.valid_kernels:
                    if isinstance(kernel, Comm_Kernel):
                        key = (kernel.key[3] * local_size + local_rank, kernel.key[4] * local_size + local_rank)    # (send, recv)
                        if rank in key:
                            if key not in ncclcomm_dict.keys():
                                ncclcomm_dict[key] = PyNcclCommunicator(global_group, ranks=key, device=torch.cuda.current_device())
                            kernel.ncclcomm = ncclcomm_dict[key]
                # For Intra
            for _, intra_execution_plan in inter_comp_plans.items():
                    # print(f'rank{rank}, batch_degrees: {batch_degrees}, intra_execution_plan: {intra_execution_plan}', flush=True)
                if not isinstance(intra_execution_plan, Execution_Plan):    # fused intra execution plan
                    assert isinstance(intra_execution_plan, Fused_Execution_Plan)
                    # Create Row&Col PyNcclCommunicator
                    Y = intra_execution_plan.Y
                    X = intra_execution_plan.X
                    # print(f'rank{rank}, batch_degrees: {batch_degrees}, intra_execution_plan: {intra_execution_plan}', flush=True)
                    cur_x_id = local_rank % X
                    cur_y_id = local_rank // X
                    r_key = tuple(range(node_id * local_size + cur_y_id * X, node_id * local_size + (cur_y_id + 1) * X))
                    c_key = tuple(range(node_id * local_size + cur_x_id, (node_id + 1) * local_size, X))
                    assert rank in r_key and rank in c_key
                    if r_key not in ncclcomm_dict.keys():
                        ncclcomm_dict[r_key] = PyNcclCommunicator(global_group, ranks=r_key, device=torch.cuda.current_device())
                    if c_key not in ncclcomm_dict.keys():
                        ncclcomm_dict[c_key] = PyNcclCommunicator(global_group, ranks=c_key, device=torch.cuda.current_device())
                else:                                                      # non-fused intra execution plan
                    for kernel in intra_execution_plan.valid_kernels:
                        if isinstance(kernel, Comm_Kernel):
                            key = (node_id * local_size + kernel.key[3], node_id * local_size + kernel.key[4])    # (send, recv)
                            if rank in key:
                                if key not in ncclcomm_dict.keys():
                                    ncclcomm_dict[key] = PyNcclCommunicator(global_group, ranks=key, device=torch.cuda.current_device())
                                kernel.ncclcomm = ncclcomm_dict[key]
            set_global_var('ncclcomm_dict', ncclcomm_dict)
            # set_global_var('cpu_group_dict', cpu_group_dict)
            # print kernel orders
            # for r in range(local_size):
            #     print_rank_0(f'rank{r}:')
            #     for kernel in execution_plan.gpu_kernel_lists[r]:
            #         # if isinstance(kernel, Comp_Kernel) or kernel.key[- 2] == 'o': # comm + output comm
            #         if kernel.key[- 2] == 'i':  # only input comm
            #             print_rank_0(f'{kernel.key}')
            
            inputs['execution_plan_dict'] = {
                'inter': inter_execution_plan,
                'intra': inter_comp_plans,
            }
            inputs = filter_kwargs(f, inputs)
        
            torch.cuda.synchronize()
            torch.distributed.barrier(group=global_group)
            # continue
            TRACE_NAME = f'{os.environ["TRACE_NAME"]}_SP({node_num},{local_size})_w{world_size}_r{rank}_S({Sq},{Skv})_bs{batch_size}_Nh{nheads}_D{nheads}_' \
                        f'{"causal" if causal else "noncausal"}_{f.__name__}'
            t1, t2, t3, td = benchmark_ops(streams, global_group, torch.cuda.current_device(), f, inputs, warmup, warmup_cudagraph, num_iter, use_cudagraph, TRACE_NAME, args)

            if rank == 0 and log:
            # if True:
                m_flops, h_flops = calc_flops(batch_size, (Sq * world_size, Skv * world_size), nheads, d, causal, fob=fob)
                mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
                # print(f"suffix: {plan_path.split('/')[-1]}, mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)
                print(f"mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, "
                    f"{td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)

def run_all_inter_attn(args, ncclcomm_global, gloo_global_group):
    global PROC_INFO
    baseline_funcs = [
        ring_flash_attn_func,
        zigzag_ring_flash_attn_func,      # baseline
        # zigzag_ring_flash_attn_func_opt,  # sol1
        stripe_flash_attn_func,
        # # lightseq_attn_func,
        # # flash_attn_func,
        # # hierarchy_attn_func,                # one case
        # # overlapped_hierarchy_attn_func,     # another case
    ]
    
    # # modified PROC_INFO for debug, version1:  1 * 8 -> 8 * 8
    # assert PROC_INFO['node_num'] == 1 and PROC_INFO['tasks_per_node'] == 8
    # PROC_INFO['node_num'] = 8
    # PROC_INFO['nodeid'] = PROC_INFO['local_rank']
    # PROC_INFO['world_size'] = PROC_INFO['node_num'] * PROC_INFO['tasks_per_node']
    # # modify finished
    
    # # modified PROC_INFO for debug, version2:  1 * 8 -> 4 * 2
    # assert PROC_INFO['node_num'] == 1 and PROC_INFO['tasks_per_node'] == 8
    # PROC_INFO['node_num'] = 4
    # PROC_INFO['tasks_per_node'] = 2
    # PROC_INFO['nodeid'] = PROC_INFO['rank'] // PROC_INFO['tasks_per_node']
    # PROC_INFO['local_rank'] = PROC_INFO['rank'] % PROC_INFO['tasks_per_node']
    # # modify finished
    
    # # modified PROC_INFO for debug, version3:  1 * 8 -> 8 * 1
    # assert PROC_INFO['node_num'] == 1 and PROC_INFO['tasks_per_node'] == 8
    # PROC_INFO['node_num'] = 8
    # PROC_INFO['tasks_per_node'] = 1
    # PROC_INFO['nodeid'] = PROC_INFO['rank'] // PROC_INFO['tasks_per_node']
    # PROC_INFO['local_rank'] = PROC_INFO['rank'] % PROC_INFO['tasks_per_node']
    # # modify finished
    
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_num = PROC_INFO['node_num']
    rank = PROC_INFO['rank']
    assert node_num >= 1 # and local_size == 8
    
    # fix configs:
    bs = 1
    D = 128
    causals = [
        False,
        # True,
    ]
    SPs = (node_num, local_size)
    
    # variable configs:
    fobs = [
        0, 
        1,
    ]
    Nhs = [
        1,
        32, 
    ]
    
    # experiment variables
    WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    
    # S_BOUND = [256, 64 * 1024]
    # S_BOUND = [256, 32 * 1024]  # lower-bound and upper-bound of S per GPU, for (4, 8)
    S_BOUND = [256, 16 * 1024]  # lower-bound and upper-bound of S per GPU, for (8, 8) causal and (4, 8) noncausal
    # S_BOUND = [1 * 1024, 16 * 1024] # for debug
    # S_BOUND = [32 * 1024, 32 * 1024]
    
    S_base = [1 << logS for logS in range(int(math.log2(S_BOUND[0])), int(math.log2(S_BOUND[1])) + 1)]
    multiplying_powers = [1, 2, 3, 4, 5, 6, 7]
    multiplying_powers = [1]
    Sqkvs = [S * power for S in S_base for power in multiplying_powers if S * power <= S_BOUND[1]]
    Sqkvs = sorted(list(set(Sqkvs)))    # Sq per GPU
    # Sqs = [327680 // world_size]
    # Sqkvs = [256]  # for debug
    # Sqkvs = [1 * 1024]  # for debug
    # Sqkvs = [64 * 1024]
    print_rank_0(f'Sqkvs: {Sqkvs}')
    
    # pre-allocated buffer:
    MAX_SEQ = max(Sqkvs)
    # Nh=32, max_batch_degrees = (2, 3) (q, do, D, lse), (k, v, dk, dv)
    print_rank_0(f'total_size: {(bs * MAX_SEQ * max(Nhs) * (D + 1)) * (8 * 8) * (3 + 4)}')
    tensor_buf = torch.empty(
        # (6 * bs * MAX_SEQ * max(Nhs) * D), 
    #     (bs * MAX_SEQ * max(Nhs) * D * 4) * 3                       # k, v, dk, dv
    #   + (bs * MAX_SEQ * max(Nhs) * (D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    #   + (bs * MAX_SEQ * max(Nhs) * (D * 2) + (1 * (2 + 1))) * 2,    # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
        # (bs * MAX_SEQ * max(Nhs) * (D + 1)) * (4 * 8) * (3 + 4)),
        ((bs * MAX_SEQ * max(Nhs) * (D + 1)) * (4 * 8) * (4)),
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 1 * (32 * 1024) * (32 * 128) * (8 * 8)
    # print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')

    INTER_COMP_FILE_NAME = f'./prof_data/wrapper_intra_SP={local_size}_all.log'
    
    # get_comp_map_key
    if os.path.exists(INTER_COMP_FILE_NAME):
        inter_comp_profile_map = Inter_Comp_Profile_Map(select_best_schedule_in_node_profile_data(INTER_COMP_FILE_NAME, local_size))
    else:
        inter_comp_profile_map = None
    # print_rank_0(f'inter_comp_profile_map: {inter_comp_profile_map.profile_map}')
    
    for causal in causals:
        for fob in fobs:
            print_rank_0(f'causal={causal}, fob={fob}:')
            for Nh in Nhs:
                for Sqkv in Sqkvs: # S per GPU
                    da_config = Dist_Attn_Config(SP=SPs, S=(Sqkv * world_size, Sqkv * world_size), Nh=(Nh, Nh), D=D, bs=bs, causal=causal)
                    print_rank_0(f'da_config: {da_config}:')
                    
                    
                    plan_types = ['automatic', 'ablation0']  # [NOTE]: useful !!!
                    plan_types = ['automatic']
                    exp_configs = []
                    
                    if causal:  # For causal !!!
                        # exp_configs: 4 = 2 * 2 (w/wo fuse, ILP vs Flexflow)
                        ablation_suffixes = ['', '_fused', '_ablation1', '_fused_ablation1']
                        # ablation_suffixes = ['_fused']      # for torch.profiler
                        # ablation_suffixes = ['']            # for SP0 = 1
                        # ablation_suffixes = []
                        
                        
                        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/inter_SP{node_num}_fob={fob}'
                        plan_paths = []
                        
                        # HACK
                        # if da_config.SP[1] != 8:
                        old_sp = da_config.SP
                        old_S = da_config.S
                        da_config.SP = (old_sp[0], 8)
                        da_config.S = (old_S[0] * da_config.SP[1] // old_sp[1], old_S[1] * da_config.SP[1] // old_sp[1])
                        plan_name_prefix = da_config.get_plan_name(fob=fob)
                        # Restore
                        da_config.SP = old_sp
                        da_config.S = old_S
                        
                        for suffix in ablation_suffixes:
                            plan_paths.append(f'{par_dir}/{plan_name_prefix}{suffix}.pkl')
                        print(f'plan_paths: {plan_paths}')
                        for plan_type in plan_types:
                            for plan_path in plan_paths:
                                # load inter execution plans
                                with open(plan_path, 'rb') as fin:
                                    inter_execution_plan = pickle.load(fin)
                                exp_config = Evaluation_Configs(
                                        plan_type=plan_type,
                                        MAX_QUEUE_SIZE=0,
                                        fob=fob,
                                        plan_path=plan_path,
                                        inter_comp_profile_map=inter_comp_profile_map,
                                        execution_plan=inter_execution_plan,
                                    )
                                exp_configs.append(exp_config)
                    else:   # For non-causal !!!
                        # (log2(SP[0]) + 1) * 2 types of inter_execution_plans, 2 stands for w/wo fuse
                        # inter execution plans !!!
                        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{node_num}_fob={fob}'
                        YXs = [(y, node_num // y) for y in range(1, node_num + 1) if node_num % y == 0]
                        print_rank_0(f'YXs: {YXs}')
                        for plan_type in plan_types:
                            # non fused
                            for yx in YXs:
                                plan_path = f'{par_dir}/SP={node_num}_fob={fob}_Y={yx[0]}_X={yx[1]}_dim=0.pkl'
                                with open(plan_path, 'rb') as fin:
                                    inter_execution_plan = pickle.load(fin)
                                exp_config = Evaluation_Configs(
                                        plan_type=plan_type,
                                        MAX_QUEUE_SIZE=0,
                                        fob=fob,
                                        plan_path=plan_path,
                                        inter_comp_profile_map=inter_comp_profile_map,
                                        execution_plan=inter_execution_plan,
                                    )
                                exp_configs.append(exp_config)
                            
                            # fused !!!
                            if fob == 0 or Nh == 1:
                                for yx in YXs:
                                # for yx in YXs[: 1]: # qo
                                    fused_execution_plan = Fused_Execution_Plan(yx[0], yx[1], - 0.0, False, fob=fob)
                                    fused_execution_plan.stream_num = 3
                                    exp_config = Evaluation_Configs(
                                            plan_type=plan_type,
                                            MAX_QUEUE_SIZE=0,
                                            fob=fob,
                                            plan_path=None,
                                            inter_comp_profile_map=inter_comp_profile_map,
                                            execution_plan=fused_execution_plan,
                                        )
                                    exp_configs.append(exp_config)
                    
                    # Execution:
                    # 1 baselines
                    for f in baseline_funcs:
                        if causal == False and f in [zigzag_ring_flash_attn_func, stripe_flash_attn_func]:
                            continue
                        benchmark(args, f, da_config, tensor_buf, gloo_global_group, fob=fob, log=True)
                    
                    # 2 orchestrated_attn_func:
                    benchmark_op = partial(benchmark_orchestrate,
                        args, orchestrated_attn_func, da_config, tensor_buf, log=True, exp_configs=exp_configs, 
                        global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
                        warmup=WARMUP, num_iter=NUM_ITER,
                    )
                    benchmark_op(use_cudagraph=False)          

def profile_all_intra_attn(args, ncclcomm_global, gloo_global_group):
    global PROC_INFO
    baseline_funcs = [
        ring_flash_attn_func,
        zigzag_ring_flash_attn_func,      # baseline
        
        # zigzag_ring_flash_attn_func_opt,  # sol1
        stripe_flash_attn_func,
        
        # lightseq_attn_func,
        # flash_attn_func,
        # hierarchy_attn_func,                # one case
        # overlapped_hierarchy_attn_func,     # another case
    ]
    
    world_size = PROC_INFO['world_size']
    local_size = PROC_INFO['tasks_per_node']
    node_num = PROC_INFO['node_num']
    rank = PROC_INFO['rank']
    assert node_num == 1 # and local_size == 8
    
    # fix configs:
    bs = 1
    D = 128
    causals = [
        # False,
        True,
    ]
    SPs = (node_num, local_size)
    
    # variable configs:
    fobs = [
        # 0, 
        1,
    ]
    Nhs = [
        1,
        32, 
    ]
    
    # experiment variables
    WARMUP, NUM_ITER = 11, 20 # most, best performance for most cases
    # WARMUP, NUM_ITER = 4, 4 # most, best performance for most cases
    # WARMUP, NUM_ITER = 2, 4 # intermediate, best performance for some cases !!!
    # WARMUP, NUM_ITER = 1, 2 # later, bad performance
    
    S_BOUND = [256, 64 * 1024]  # lower-bound and upper-bound of S per GPU, for (1, 8)
    # S_BOUND = [256, 16 * 1024] # for debug
    # S_BOUND = [16 * 1024, 16 * 1024] # for debug
    
    # pre-allocated buffer:
    # Nh=32, max_batch_degrees = (2, 3) (q, do, D, lse), (k, v, dk, dv)
    tensor_buf = torch.empty(
        (bs * S_BOUND[1] * max(Nhs) * D * 4) * 3                       # k, v, dk, dv
    + (bs * S_BOUND[1] * max(Nhs) * (D * 3) + (1 * (2 + 1))) * 2     # q, do, D, lse, dq
    + (bs * S_BOUND[1] * max(Nhs) * (D * 2) + (1 * (2 + 1))) * 2,    # q, do, D, lse (for inp_row_extra_buf because of bugs of bwd of FA)
        device=torch.cuda.current_device(), dtype=DTYPE, requires_grad=False
    )   # 6 * 512MB = 3GB
    # print_rank_0(f'tensor_buf: {tensor_buf.numel() * 2} B')
    inter_comp_profile_map = None

    for fob in fobs:
        inter_ablation_suffixes = ['']            # for SP0 = 1, i.e. profile
        plan_types = ['automatic']  # for profile is OK
        exp_configs = []
        par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/inter_SP{node_num}_fob={fob}'
        plan_paths = []
    
        # Create exp_configs
        dummy_da_config = Dist_Attn_Config(SP=SPs, S=(256 * world_size, 256 * world_size), Nh=(1, 1), D=D, bs=bs, causal=True)
        plan_name_prefix = dummy_da_config.get_plan_name(fob=fob)
        for suffix in inter_ablation_suffixes:
            plan_paths.append(f'{par_dir}/{plan_name_prefix}{suffix}.pkl')
        for plan_type in plan_types:
            for plan_path in plan_paths:
                exp_config = Evaluation_Configs(
                        plan_type=plan_type,
                        MAX_QUEUE_SIZE=0,
                        fob=fob,
                        plan_path=plan_path,
                        inter_comp_profile_map=inter_comp_profile_map,
                    )
                exp_configs.append(exp_config)
                                
        for causal in causals:
            S_base = [1 << logS for logS in range(int(math.log2(S_BOUND[0])), int(math.log2(S_BOUND[1])) + 1)]
            if not causal:  # for non-causal
                multiplying_powers = [1, 2, 3, 4, 5, 6, 7]
            else:
                multiplying_powers = [1]
            Sqkvs = [S * power for S in S_base for power in multiplying_powers if S * power <= S_BOUND[1]]
            Sqkvs = sorted(list(set(Sqkvs)))    # Sq per GPU
            # Sqs = [327680 // world_size]
            # Sqkvs = [256, 512]  # for debug
            # Sqkvs = [1 * 1024, 2 * 1024, 4 * 1024, 8 * 1024]  # for debug
            # Sqkvs = [1 * 1024]  # for debug
            # Sqkvs = [64 * 1024]
            # print_rank_0(f'Sqkvs: {Sqkvs}')
            print_rank_0(f'fob={fob}')
            if not causal:
                par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{local_size}_fob={fob}'
                plan_paths = []
                for plan_name in os.listdir(par_dir):
                    plan_paths.append(f'{par_dir}/{plan_name}')
                plan_paths.sort()   # Y=1(qo), Y=2, ..., Y=N(kv)
                # plan_paths = plan_paths[0:1]    # only one plan
                # print_rank_0(f'fob={fob}, plan_paths: {plan_paths}')
                
                # Create all inter_comp_plans for profiling for non-causal
                # 4 * 2 = 8 types (4: Y=1, Y=2, Y=4, Y=8; 2: fused, non-fused)
                # 4 non-fused
                nonfused_inter_comp_plans_dicts = []
                for plan_path in plan_paths:
                    with open(plan_path, 'rb') as fin:
                        inter_comp_plans_dict = {
                            ((1, 1), causal): pickle.load(fin)
                        }
                        nonfused_inter_comp_plans_dicts.append(inter_comp_plans_dict)
                # 4 fused
                fused_inter_comp_plans_dicts = []
                for _ in range(len(nonfused_inter_comp_plans_dicts)):
                    intra_execution_plan = list(nonfused_inter_comp_plans_dicts[_].values())[0]
                    fused_inter_comp_plans_dict = {
                        ((1, 1), causal): Fused_Execution_Plan(intra_execution_plan.Y, intra_execution_plan.X, - 0.0, causal, fob=fob)
                    }
                    fused_inter_comp_plans_dicts.append(fused_inter_comp_plans_dict)
                    
            for Nh in Nhs:
                if not causal:
                    inter_comp_plans_dicts = copy.copy(nonfused_inter_comp_plans_dicts)
                    if Nh == 1: # only Nh == 1, fused is useful !!!
                        inter_comp_plans_dicts += fused_inter_comp_plans_dicts
                    
                for Sq in Sqkvs: # S per GPU
                    for Skv in Sqkvs: # S per GPU
                        if causal and Sq != Skv:
                            continue
                        da_config = Dist_Attn_Config(SP=SPs, S=(Sq * world_size, Skv * world_size), Nh=(Nh, Nh), D=D, bs=bs, causal=causal)
                        print_rank_0(f'da_config: {da_config}:')
                        
                        # Create all inter_comp_plans for profiling for causal
                        if causal:
                            inter_comp_plans_dicts = []
                            # 2 * 2 = 4 types (2: ILP, Flexflow; 2: non-fused, fused)
                            intra_ablation_suffixes = ['', '_fused', '_ablation1', '_fused_ablation1']
                            # intra_ablation_suffixes = ['', '_ablation1']
                            # intra_ablation_suffixes = ['']  # for torch.profiler
                            # intra_ablation_suffixes = []
                            par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{local_size}_fob={fob}_causal=True'
                            plan_name_prefix = da_config.get_plan_name(fob=fob)
                            # load intra plans
                            for suffix in intra_ablation_suffixes:
                                plan_path = f'{par_dir}/{plan_name_prefix}{suffix}.pkl'
                                with open(plan_path, 'rb') as fin:
                                    cur_execution_plan: Execution_Plan = pickle.load(fin)
                                    inter_comp_plans_dict = {
                                        ((1, 1), causal): cur_execution_plan,
                                    }
                                    # print_rank_0(f'plan_path: {plan_path}')
                                    # if rank == 0:
                                    #     cur_execution_plan.print_lp_result()
                                    inter_comp_plans_dicts.append(inter_comp_plans_dict)
                        
                        # assert len(inter_comp_plans_dicts) == 4 if (causal or (Nh > 1)) else 8
                        
                        # Execution:
                        # 1 baselines
                        if fob == 0 and causal:
                            for f in baseline_funcs:
                                benchmark(args, f, da_config, tensor_buf, gloo_global_group, forward_only=True, log=True)
                        
                        # 2 orchestrated_attn_func:
                        benchmark_op = partial(benchmark_orchestrate,
                            args, orchestrated_attn_func, da_config, tensor_buf, log=True, exp_configs=exp_configs, 
                            global_group=gloo_global_group, ncclcomm_global=ncclcomm_global,
                            warmup=WARMUP, num_iter=NUM_ITER, mode='profile', inter_comp_plans_dicts=inter_comp_plans_dicts,
                        )
                        benchmark_op(use_cudagraph=False)          


def main(args):
    global PROC_INFO
    PROC_INFO = get_proc_info()
    
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    # MASTER_ADDR = 'localhost'
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    # print(f'init_method: {init_method}')
    dist.init_process_group(backend="nccl", init_method=init_method, rank=PROC_INFO['rank'], world_size=PROC_INFO['world_size'])
    gloo_global_group = dist.new_group(ranks=list(range(PROC_INFO['world_size'])), backend='gloo')
    ncclcomm_global = PyNcclCommunicator(gloo_global_group, ranks=list(range(PROC_INFO['world_size'])), device=PROC_INFO['local_rank'])
    # [NOTE]: we create a gloo global group because we use it to barrier in benchmark_orchestrate to prevent cudagraph overlapped with nccl ops !!!
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f'rank{rank}, world_size{world_size}, hostname: {socket.gethostname()}')
    initialize_distributed()    # used by lightseq

    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    torch.cuda.set_device(device)
    
    # preprocess placeholder_op
    global placeholder_op
    SYNC_SIZE = 8 * pow(1024, 3) # 8GB
    sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=device)
    placeholder_op = partial(ncclcomm_global.all_reduce, sync_tensor)
    


    # PROC_INFO['tasks_per_node'] = 2 # for test 4 x 2
    # PROC_INFO['tasks_per_node'] = 4 # for test 2 x 4
    # PROC_INFO['local_rank'] %= PROC_INFO['tasks_per_node']
    # PROC_INFO['nodeid'] = PROC_INFO['rank'] // PROC_INFO['tasks_per_node']
    
    # run_all_intra_attn(args, ncclcomm_global, gloo_global_group)
    # return
    # profile_all_intra_attn(args, ncclcomm_global, gloo_global_group)
    run_all_inter_attn(args, ncclcomm_global, gloo_global_group)    # E2E !!!
    return

    forward_only = False
    
    funcs = [
        # ring_flash_attn_func,
        # zigzag_ring_flash_attn_func,      # baseline
        # # zigzag_ring_flash_attn_func_opt,  # sol1
        # stripe_flash_attn_func,
        # lightseq_attn_func,
        # flash_attn_func,
        # hierarchy_attn_func,                # one case
        # overlapped_hierarchy_attn_func,     # another case
        orchestrated_attn_func,
    ]
    # Configs:
    bs = 1
    D = 128
    # Ss = [
    #     # 2 * 1024,  # 2K
    #     4 * 1024,   # 4K
    #     # 8 * 1024,   # 8K
    #     # 16 * 1024,  # 16K
    #     # 32 * 1024,   # 32K
    #     # 64 * 1024,   # 64K
    #     # 128 * 1024,   # 128K
    #     # 256 * 1024,   # 256K
    #     # 512 * 1024,   # 512K
    #     # 1024 * 1024,   # 1M
    # ]
    Ss_per_gpu = [
        # 256,
        # 512,
        1 * 1024,   # 1K
        # 2 * 1024,
        # 4 * 1024,
        # 8 * 1024,
        # 16 * 1024,
        # 32 * 1024,
        # 64 * 1024,
        # 128 * 1024, # 128K, # [NOTE]: ERROR in backward
    ]
    Ss = [S * world_size for S in Ss_per_gpu]
    Nhs = [ # configs of llama2
        1,  # for test
    #    32,  # 7b
    #    40,  # 13b
    #    80,  # 70b 
    ]
    fob = 0
    # fob = 1
    causal = True
    # causal = False
    
    tensor_buf = torch.empty(
        (6 * bs * max(Ss) * max(Nhs) * D), device=device, dtype=DTYPE, requires_grad=False
    )
    qkv_buf = tensor_buf[: (3 * bs * max(Ss) * max(Nhs) * D)]
    dout_buf = tensor_buf[(3 * bs * max(Ss) * max(Nhs) * D): ]

    for Nh in Nhs:
        for S in Ss:
            S = (S + world_size - 1)  // world_size * world_size
            if torch.distributed.get_rank() == 0:
                print(f'Nh={Nh}, S={S}')
            shapes = {
                'bs': 1,
                'S': S // world_size,  # partitioned
                'Nh': Nh,
                'D': 128,
            }

            for f in funcs:
                torch.cuda.empty_cache()
                if f == orchestrated_attn_func:
                # if False:
                    fob = 0
                    SPs = (PROC_INFO['node_num'], PROC_INFO['tasks_per_node'])
                    da_config = Dist_Attn_Config(SP=SPs, S=(S, S), Nh=(Nh, Nh), D=D, bs=bs, causal=causal)
                    # Config1: Algorithms for intra causal attention
                    exp_configs = []
                    plan_types = ['ablation0', 'automatic', 'ablation1']
                    # plan_types = ['ablation1', 'automatic']
                    plan_types = ['ablation0', 'automatic']
                    
                    plan_name = da_config.get_plan_name(fob=fob)
                    
                    plan_suffixes = []
                    # NUM_ALG = 100
                    # for _ in range(NUM_ALG):
                    #     plan_suffixes.append(f'_alg{_}')
                    plan_suffixes = ['_example', '_kv', '_qo']
                        
                    plan_files = []
                    for plan_suffix in plan_suffixes:
                        plan_files.append(f'{plan_name}{plan_suffix}.pkl')
                        
                    for plan_file in plan_files:
                        for plan_type in plan_types:
                            par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/SP{da_config.SP}_S{da_config.S}'
                            if plan_type == 'ablation1':
                                par_dir = f'{par_dir}_{plan_type}'
                            assert os.path.exists(par_dir), f'{par_dir} not exists'
                            
                            exp_config = Evaluation_Configs(
                                plan_type=plan_type,
                                MAX_QUEUE_SIZE=0,
                                fob=fob,
                            )
                            plan_path = f'{par_dir}/{plan_file}'
                            assert os.path.exists(plan_path), f'{plan_path} not exists'
                            exp_config.plan_path = plan_path
                            exp_configs.append(exp_config)
                        
                    # # Config2: Blocking algorithms for intra full attention
                    # par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{SPs[1]}_fob={fob}'
                    # plan_paths = []
                    # for plan_name in os.listdir(par_dir):
                    #     plan_paths.append(f'{par_dir}/{plan_name}')
                    # # # kv filter
                    # # kv_filter = lambda x: 'X=1' in x
                    # # for i in range(len(plan_paths) - 1, - 1, - 1):
                    # #     if not kv_filter(plan_paths[i]):
                    # #         plan_paths.pop(i)
                    # # X=2 filter
                    # x_filter = lambda x: 'X=2' in x
                    # for i in range(len(plan_paths) - 1, - 1, - 1):
                    #     if not x_filter(plan_paths[i]):
                    #         plan_paths.pop(i)
                    # print_rank_0(f'plan_paths: {plan_paths}')
                    
                    # normal comp&comm
                    benchmark_op = partial(benchmark_orchestrate,
                        args, f, shapes, tensor_buf, log=True, exp_configs=exp_configs, 
                        global_group=gloo_global_group, ncclcomm_global=ncclcomm_global
                    )
                    benchmark_op(use_cudagraph=False)
                    # benchmark_op(use_cudagraph=True)
                    
                    # # fused comp&comm
                    # benchmark_op = partial(benchmark_fused,
                    #     args, f, shapes, tensor_buf, log=True, plan_paths=plan_paths, 
                    #     global_group=gloo_global_group, ncclcomm_global=ncclcomm_global
                    # )
                    # benchmark_op(use_cudagraph=False)
                else:
                    benchmark(args, f, shapes, qkv_buf, dout_buf, forward_only=True, log=True)
                # benchmark(args, f, shapes, forward_only=False, log=True)
    

if __name__ == "__main__":
    main(parse_args())