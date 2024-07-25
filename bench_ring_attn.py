from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
import torch
import torch.distributed as dist
from ring_flash_attn import (
    # ring_flash_attn_qkvpacked_func,
    # zigzag_ring_flash_attn_qkvpacked_func,
    # stripe_flash_attn_qkvpacked_func,
    ring_flash_attn_func,
    zigzag_ring_flash_attn_func,
    stripe_flash_attn_func,
)
import torch.distributed
from hierarchy_attn.hierarchy_attn_impl import hierarchy_attn_func
from orchestrated_attn.orchestrated_attn_impl import orchestrated_attn_func
import torch.cuda
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from easy_context.dist_flash_attn.lightseq_async_attn import attention as lightseq_attn
from easy_context.dist_flash_attn.async_communication import initialize_distributed
from search_algo.search_engine import Dist_Attn_Config
from search_algo.dependent_graph import Cuda_Kernel, Comp_Kernel, Comm_Kernel
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from orchestrated_attn.global_vars import *
from orchestrated_attn.utils import *
import inspect
import warnings
import time
import socket
import pickle
from functools import partial
warnings.filterwarnings("ignore")   # disable warning caused by lightseq

PROC_INFO: dict
DTYPE = torch.bfloat16


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

def calc_flops(mbs, S, Nh, D, causal=True, forward_only=False):
    flops = 2 * 2 * mbs * S * S * Nh * D
    if causal:
        flops = flops // 2
    if forward_only:
        m_flops = h_flops = flops
    else:
        m_flops = (1 + 2) * flops
        h_flops = (1 + 2.5) * flops
    return m_flops, h_flops # model flops & hardware flops
    
def benchmark(args, f, shapes:dict, qkv_buf, dout_buf, warmup=11, num_iter=20, forward_only=True, log=True):
    # warmup = 0
    # num_iter = 20
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t0 = time.time()
    global PROC_INFO
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    if rank == 0:
        print(f'# {f.__name__}, {"fwd" if forward_only else "fwd + bwd"}', flush=True)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    torch.cuda.set_device(device)

    batch_size = shapes['bs']
    seqlen = shapes['S']
    nheads = shapes['Nh']
    d = shapes['D']
    if f == flash_attn_func:
        seqlen *= world_size
        world_size = 1
    dropout_p = 0
    causal = True
    deterministic = False

    # assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    # qkv = torch.empty(
    #     3 * batch_size, seqlen, nheads, d, device=device, dtype=DTYPE, requires_grad=True
    # )
    # dout = torch.empty(batch_size, seqlen, nheads, d, device=device, dtype=DTYPE)
    qkv = qkv_buf[: 3 * batch_size * seqlen * nheads * d].view(3 * batch_size, seqlen, nheads, d)
    qkv.requires_grad = True
    dout = dout_buf[: batch_size * seqlen * nheads * d].view(batch_size, seqlen, nheads, d)
    # bsz, nh, seq_len, hdim
    if f == lightseq_attn_func:
        qkv = qkv.permute(0, 2, 1, 3)
        dout = dout.permute(0, 2, 1, 3)
    q, k, v = qkv.chunk(3, dim=0)
    
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
    inputs = filter_kwargs(f, inputs)
    
    # warmup
    if forward_only:
        with torch.no_grad():
            for _ in range(warmup):
                _ = f(**inputs)
                torch.cuda.synchronize()
                torch.distributed.barrier()
    else:
        for _ in range(warmup):
            qkv.grad = None
            out = f(**inputs)
            out.backward(dout)
    # print_rank_0(f'Warmup done !!!')
    
    use_cudagraph = False
    
    is_runned = False
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t1 = time.time()
    
    # warmup cudagraph
    if use_cudagraph:
        for _ in range(warmup):
            g.replay()
            torch.cuda.synchronize()
            torch.distributed.barrier()
        # print_rank_0(f'Warmup cudagraph done !!!')

                
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t2 = time.time()
    
    # if args.profiler_with_tensorboard and not hasattr(args, "tb_profiled"):
    if args.profiler_with_tensorboard:
        sync_tensor = torch.empty((1 * 1024 * 1024 * 1024), dtype=DTYPE, device=device)
        args.tb_profiled = True
        is_runned = True
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'{os.environ["TRACE_NAME"]}_w{world_size}_r{rank}_S{seqlen}_bs{batch_size}_Nh{nheads}_D{nheads}_{f.__name__}_{"f" if forward_only else "f+b"}'
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
                    if forward_only:
                        with torch.no_grad():
                            _ = f(**inputs)
                    else:
                        qkv.grad = None
                        out = f(**inputs)
                        out.backward(dout)
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                prof.step()
        
        num_iter = TOTAL_TURNS
        
    if not is_runned: 
        # run
        if use_cudagraph:
            for _ in range(num_iter):
                g.replay()
        else:
            if forward_only:
                with torch.no_grad():
                    for _ in range(num_iter):
                        _ = f(**inputs)
                        # print(f'rank{rank}, cpu out !!!', flush=True)
                        # torch.cuda.synchronize()
                        # print(f'rank{rank}, sync out !!!', flush=True)
                        # torch.distributed.barrier()
                        # print(f'rank{rank}, real out !!!', flush=True)
                        # for kernel in execution_plan.gpu_kernel_lists[rank]:
                        #     print(f'rank{rank}, {kernel.key}, {kernel.in_ranks} !!!', flush=True)
            else:
                for _ in range(num_iter):
                    qkv.grad = None
                    out = f(**inputs)
                    out.backward(dout)
     
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    t3 = time.time()
    td = t3 - t2

    if rank == 0 and log:
        m_flops, h_flops = calc_flops(batch_size, seqlen * world_size, nheads, d, causal, forward_only)
        mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
        print(f"mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)

def benchmark_orchestrate(args, f, shapes:dict, qkv_buf, dout_buf, warmup=11, num_iter=20, forward_only=True, log=True, 
                          plan_paths: list = [''], global_group=None, ncclcomm_global: PyNcclCommunicator = None, 
                          use_cudagraph=False):
    print_rank_0(f'[INFO]: use_cudagraph: {use_cudagraph}')
    warmup = 11
    warmup_cudagraph = 100
    num_iter = 20
    torch.cuda.synchronize()
    torch.distributed.barrier(group=global_group)
    global PROC_INFO
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    local_size = PROC_INFO['tasks_per_node']
    if rank == 0:
        print(f'# {f.__name__}, {"fwd" if forward_only else "fwd + bwd"}', flush=True)
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    torch.cuda.set_device(device)

    batch_size = shapes['bs']
    seqlen = shapes['S']
    nheads = shapes['Nh']
    d = shapes['D']
    dropout_p = 0
    deterministic = False

    # assert seqlen % (2 * world_size) == 0
    assert d % 8 == 0

    # qkv = torch.empty(
    #     3 * batch_size, seqlen, nheads, d, device=device, dtype=DTYPE, requires_grad=True
    # )
    # dout = torch.empty(batch_size, seqlen, nheads, d, device=device, dtype=DTYPE)
    qkv = qkv_buf[: 3 * batch_size * seqlen * nheads * d].view(3 * batch_size, seqlen, nheads, d)
    qkv.requires_grad = True
    dout = dout_buf[: batch_size * seqlen * nheads * d].view(batch_size, seqlen, nheads, d)
    q, k, v = qkv.chunk(3, dim=0)
    
    inputs = {
        # "q": q,
        # "k": k,
        # "v": v,
        "inp_row": Input_Row_Fwd(q), 
        "inp_col": Input_Col_Fwd(k, v),
        "dropout_p": dropout_p,
        "causal": None,
        "deterministic": deterministic,
        "sm_scale": q.shape[-1] ** (-0.5),
        "PROC_INFO": PROC_INFO,
        "groups": {},
    }

    SYNC_SIZE = 8 * pow(1024, 3) # 8GB
    sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=device)
    placeholder_op = partial(ncclcomm_global.all_reduce, sync_tensor)
    for plan_path in plan_paths:
        t0 = time.time()
        SP = (1, world_size)
        Ss = (seqlen * world_size, seqlen * world_size)
        Nhs = (nheads, nheads)
        bs = batch_size
        # load plan
        with open(plan_path, 'rb') as fin:
            execution_plan = pickle.load(fin)   # [NOTE]: this obj shared by all processors in memory !!!
        causal = execution_plan.da_config.causal
        inputs['causal'] = causal
        # if rank == 0:
        #     execution_plan.print_lp_result()
        # preprocess
        for kernel in execution_plan.gpu_kernel_lists[local_rank]:
            kernel.in_ranks = set([local_rank])
        # modify da_config
        if execution_plan.da_config.S != Ss:
            # print(f'modify {execution_plan.da_config.S} to {Ss}')
            execution_plan.da_config.S = Ss
        # create streams
        if not is_exist_global_var('streams'):
            streams = []
            # streams.append(torch.cuda.current_stream())
            # print(f'rank{rank}, current_device: {torch.cuda.current_device()}')
            priorities = [0, - 1, - 2]
            priorities = [0, 0, 0]
            for _ in range(0, execution_plan.stream_num):
                streams.append(torch.cuda.Stream(torch.cuda.current_device(), priority=priorities[_]))
            set_global_var('streams', streams)
        streams = get_global_var('streams')
        # set stream for eash kernel
        for kernel in execution_plan.gpu_kernel_lists[local_rank]:
            if isinstance(kernel, Comp_Kernel):
                kernel.stream = streams[0]
            else:
                if kernel.key[3] == local_rank:
                    kernel.stream = streams[1]    # Send
                else:
                    kernel.stream = streams[2]    # Recv
        # build nccl communicator for each pair of ranks
        ncclcomm_dict = get_global_var('ncclcomm_dict')
        # create gloo group for each pair of ranks
        group_dict = {}
        for kernel in execution_plan.valid_kernels:
            if isinstance(kernel, Comm_Kernel):
                key = tuple(sorted([kernel.key[3], kernel.key[4]]))
                if key not in group_dict.keys():
                    group_dict[key] = torch.distributed.new_group(key, backend='gloo')
        for kernel in execution_plan.valid_kernels:
            if isinstance(kernel, Comm_Kernel):
                key = (kernel.key[3], kernel.key[4])    # (send, recv)
                if key not in ncclcomm_dict.keys():
                    # print_rank_0(f'key: {key}')
                    # new_group = torch.distributed.new_group(key, backend='gloo')    # group must be create on every process ???
                    new_group = group_dict[tuple(sorted(key))]
                    if rank in key:
                        # print(f'rank{rank}, key: {key}', flush=True)
                        ncclcomm_dict[key] = PyNcclCommunicator(new_group, device=local_rank)
                if rank in key:
                    kernel.ncclcomm = ncclcomm_dict[key]
        set_global_var('ncclcomm_dict', ncclcomm_dict)
        # print kernel orders
        # for r in range(local_size):
        #     print_rank_0(f'rank{r}:')
        #     for kernel in execution_plan.gpu_kernel_lists[r]:
        #         # if isinstance(kernel, Comp_Kernel) or kernel.key[- 2] == 'o': # comm + output comm
        #         if kernel.key[- 2] == 'i':  # only input comm
        #             print_rank_0(f'{kernel.key}')
                
        inputs['execution_plan'] = execution_plan
        inputs = filter_kwargs(f, inputs)
    
        torch.cuda.synchronize()
        torch.distributed.barrier(group=global_group)
        
        # warmup
        if forward_only:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = f(**inputs)
                    
        else:
            for _ in range(warmup):
                qkv.grad = None
                out = f(**inputs)
                out.backward(dout)
        torch.cuda.synchronize()
        torch.distributed.barrier(group=global_group)   
        # # [NOTE]: we don't barrier here to prevent WARN of 
        # # "[W CUDAGraph.cpp:145] Warning: Waiting for pending NCCL work to finish before starting graph capture. (function operator())"
        # print_rank_0(f'Warmup done !!!')
    
        
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
                
                if forward_only:
                    with torch.no_grad():
                        _ = f(**inputs)
                else:
                    qkv.grad = None
                    out = f(**inputs)
                    out.backward(dout)
                    
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
        # SYNC_SIZE = 64 * 1024 * 1024 # 64MB
        # sync_tensor = torch.empty((SYNC_SIZE), dtype=torch.int8, device=device)
        
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
            TRACE_NAME = f'{os.environ["TRACE_NAME"]}_w{world_size}_r{rank}_S{seqlen}_bs{batch_size}_Nh{nheads}_D{nheads}_{f.__name__}_{"f" if forward_only else "f+b"}'
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
                            placeholder_op(stream=streams[0]) # [NOTE]: aim to eliminate cpu overhead
                        # preprocess
                        for stream in streams[1:]:
                            stream.wait_stream(streams[0])
                        if forward_only:
                            with torch.no_grad():
                                _ = f(**inputs)
                        else:
                            qkv.grad = None
                            out = f(**inputs)
                            out.backward(dout)
                        # postprocess
                        for stream in streams[1:]:
                            streams[0].wait_stream(stream)
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
                event_start = torch.cuda.Event(enable_timing=True)
                event_end = torch.cuda.Event(enable_timing=True)
                placeholder_op(stream=streams[0]) # [NOTE]: aim to eliminate cpu overhead
                event_start.record(stream=streams[0])
                # preprocess
                for stream in streams[1:]:
                    stream.wait_stream(streams[0])
                if forward_only:
                    with torch.no_grad():
                        for _ in range(num_iter):
                            _ = f(**inputs)
                else:
                    for _ in range(num_iter):
                        qkv.grad = None
                        out = f(**inputs)
                        out.backward(dout)
                # postprocess
                for stream in streams[1:]:
                    streams[0].wait_stream(stream)
                event_end.record(stream=streams[0])
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
            

        if rank == 0 and log:
        # if True:
            m_flops, h_flops = calc_flops(batch_size, seqlen * world_size, nheads, d, causal, forward_only)
            mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
            print(f"suffix: {plan_path.split('/')[-1]}, mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td / num_iter:.3e} s/iter, ({(t1 - t0):.3f}, {(t2 - t1):.3f}, {td:.3f}) sec", flush=True)


def main(args):
    global PROC_INFO
    PROC_INFO = get_proc_info()
    
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    MASTER_ADDR = 'localhost'
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    # print(f'init_method: {init_method}')
    dist.init_process_group(backend="nccl", init_method=init_method, rank=PROC_INFO['rank'], world_size=PROC_INFO['world_size'])
    gloo_global_group = dist.new_group(ranks=list(range(PROC_INFO['world_size'])), backend='gloo')
    ncclcomm_global = PyNcclCommunicator(gloo_global_group, device=PROC_INFO['local_rank'])
    # [NOTE]: we create a gloo global group because we use it to barrier in benchmark_orchestrate to prevent cudagraph overlapped with nccl ops !!!
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f'rank{rank}, world_size{world_size}, hostname: {socket.gethostname()}')
    initialize_distributed()    # used by lightseq

    device = torch.device(f"cuda:{PROC_INFO['deviceid']}")
    torch.cuda.set_device(device)


    # PROC_INFO['tasks_per_node'] = 2 # for test 4 x 2
    # PROC_INFO['tasks_per_node'] = 4 # for test 2 x 4
    # PROC_INFO['local_rank'] %= PROC_INFO['tasks_per_node']
    # PROC_INFO['nodeid'] = PROC_INFO['rank'] // PROC_INFO['tasks_per_node']
    

    forward_only = False
    
    funcs = [
        # ring_flash_attn_func,
        # zigzag_ring_flash_attn_func,      # baseline
        # zigzag_ring_flash_attn_func_opt,  # sol1
        # stripe_flash_attn_func,
        # lightseq_attn_func,
        # flash_attn_func,
        # hierarchy_attn_func,                # one case
        # overlapped_hierarchy_attn_func,     # another case
        orchestrated_attn_func,
    ]
    bs = 1
    D = 128
    Ss = [
        # 4 * 1024,   # 4K
        8 * 1024,   # 8K
        # 16 * 1024,  # 16K
        # 32 * 1024,   # 32K
        # 64 * 1024,   # 64K
        # 128 * 1024,   # 128K
        # 256 * 1024,   # 256K
        # 512 * 1024,   # 512K
        # 1024 * 1024,   # 1M
    ]
    Nhs = [ # configs of llama2
       32,  # 7b
    #    40,  # 13b
    #    80,  # 70b 
    ]
    qkv_buf = torch.randn(
        (3 * bs * max(Ss) * max(Nhs) * D), device=device, dtype=DTYPE, requires_grad=False
    )
    dout_buf = torch.randn((bs * max(Ss) * max(Nhs) * D), device=device, dtype=DTYPE)

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
                    SPs = (PROC_INFO['node_num'], PROC_INFO['tasks_per_node'])
                    # da_config = Dist_Attn_Config(SP=SPs, S=(S, S), Nh=(Nh, Nh), D=D, bs=bs, causal=False)
                    # par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/SP{da_config.SP}_S{da_config.S}'
                    # # print(f'{os.path.exists(par_dir)}')
                    # # plan_name = da_config.get_plan_name(fob=0 if forward_only else 1)
                    # plan_file = f'{par_dir}/{plan_name}{plan_suffix}.pkl'
                    
                    plan_suffixes = ['_example', '_qo', '_kv', '_kv', '_qo', '_example']
                    plan_suffixes = ['_example_old', '_qo_old', '_kv_old', '_example', '_qo', '_kv']
                    plan_suffixes = ['_example', '_qo', '_kv', '_example_old', '_qo_old', '_kv_old', '_example', '_qo', '_kv']
                    plan_suffixes = ['_example', '_example', '_qo', '_qo', '_kv', '_kv', '_example', '_example', '_qo', '_qo', '_kv', '_kv']
                    
                    plan_suffixes = ['_example', '_qo', '_kv']
                    # plan_suffixes = ['_kv']
                    # plan_suffixes = ['_example']
                    # NUM_ALG = 100
                    # for _ in range(NUM_ALG):
                    # # for _ in range(NUM_ALG - 1, - 1, - 1):
                    #     plan_suffixes.append(f'_alg{_}')
                    # plan_suffixes = ['_example_old', '_alg12']
                    # plan_suffixes = ['_alg12']
                    
                    # NUM_ALG = 4
                    # plan_suffixes = []
                    # for _ in range(1, NUM_ALG):
                    #     plan_suffixes.append(f'_alg{_}')
                    
                    par_dir = f'{os.path.dirname(__file__)}/search_algo/execution_plans/intra_SP{SPs[1]}'
                    plan_paths = []
                    for plan_name in os.listdir(par_dir):
                        plan_paths.append(f'{par_dir}/{plan_name}')
                    # print(f'plan_paths: {plan_paths}')
                    benchmark_op = partial(benchmark_orchestrate,
                        args, f, shapes, qkv_buf, dout_buf, forward_only=True, log=True, plan_paths=plan_paths, 
                        global_group=gloo_global_group, ncclcomm_global=ncclcomm_global
                    )
                    benchmark_op(use_cudagraph=False)
                    # benchmark_op(use_cudagraph=True)
                else:
                    benchmark(args, f, shapes, qkv_buf, dout_buf, forward_only=True, log=True)
                # benchmark(args, f, shapes, forward_only=False, log=True)
    

if __name__ == "__main__":
    main(parse_args())