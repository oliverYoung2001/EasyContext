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
from hierarchy_attn.hierarchy_attn_impl import hierarchy_attn_func
import torch.cuda
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from easy_context.dist_flash_attn.lightseq_async_attn import attention as lightseq_attn
from easy_context.dist_flash_attn.async_communication import initialize_distributed
import inspect
import warnings
import time
import socket
warnings.filterwarnings("ignore")   # disable warning caused by lightseq

PROC_INFO: dict

def lightseq_attn_func(q, k, v, causal, sm_scale):
    return lightseq_attn(q, k, v, causal, sm_scale)
    
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
        tasks_per_node = os.environ['SLURM_TASKS_PER_NODE']
        
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
    }
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
    
def benchmark(args, f, shapes:dict, warmup=5, num_iter=100, forward_only=True, log=True):
    global PROC_INFO
    rank = PROC_INFO['rank']
    local_rank = PROC_INFO['local_rank']
    if rank == 0:
        print(f'# {f.__name__}, {"fwd" if forward_only else "fwd + bwd"}', flush=True)
    dtype = torch.bfloat16
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
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

    qkv = torch.randn(
        3 * batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    dout = torch.randn(batch_size, seqlen, nheads, d, device=device, dtype=dtype)
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

    is_runned = False
    
    
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
    torch.distributed.barrier()
    # begin = torch.cuda.Event(enable_timing=True)
    # begin.record()
    ts = time.time()
    
    if args.profiler_with_tensorboard and not hasattr(args, "tb_profiled"):
        args.tb_profiled = True
        is_runned = True
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
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
        if forward_only:
            with torch.no_grad():
                for _ in range(num_iter):
                    _ = f(**inputs)

        else:
            for _ in range(num_iter):
                qkv.grad = None
                out = f(**inputs)
                out.backward(dout)
     
    
    torch.cuda.synchronize()
    torch.distributed.barrier()
    te = time.time()
    # end = torch.cuda.Event(enable_timing=True)
    # end.record()
    # td = begin.elapsed_time(end) / 1000.0
    td = te - ts

    if rank == 0 and log:
        m_flops, h_flops = calc_flops(batch_size, seqlen * world_size, nheads, d, causal, forward_only)
        mfu, hfu = (round(flops / pow(1000, 4) / (td / num_iter * world_size), 3) for flops in (m_flops, h_flops))
        print(f"mfu: {mfu} Tflops/s, hfu: {hfu} Tflops/s, {num_iter / td:.3f} iter/s, {td:.3f} sec", flush=True)

def main(args):
    global PROC_INFO
    PROC_INFO = get_proc_info()
    MASTER_ADDR = os.getenv('MASTER_ADDR', None)
    MASTER_PORT = os.getenv('MASTER_PORT', None)
    init_method = f'tcp://[{MASTER_ADDR}]:{MASTER_PORT}'
    # print(f'rank{RANK}, init_method: {init_method}')
    dist.init_process_group(backend="nccl", init_method=init_method, rank=PROC_INFO['rank'], world_size=PROC_INFO['world_size'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    # print(f'rank{rank}, world_size{world_size}, hostname: {socket.gethostname()}')
    initialize_distributed()    # used by lightseq


    forward_only = False
    
    funcs = [
        # ring_flash_attn_func,
        # zigzag_ring_flash_attn_func,
        # stripe_flash_attn_func,
        # lightseq_attn_func,
        # flash_attn_func,
        hierarchy_attn_func,
    ]
    S = 8 * 1024   # 8K
    S = 16 * 1024   # 8K
    S = 32 * 1024   # 32K
    S = 128 * 1024   # 128K
    Ss = [
        # 8 * 1024,   # 8K
        # 16 * 1024,  # 16K
        # 32 * 1024,   # 32K
        128 * 1024,   # 128K
    ]
    for S in Ss:
        if torch.distributed.get_rank() == 0:
            print(f'S={S}')
        shapes = {
            'bs': 1,
            'S': S // world_size,  # partitioned
            'Nh': 32,
            'D': 128,
        }

        for f in funcs:
            torch.cuda.empty_cache()
            benchmark(args, f, shapes, forward_only=True, log=True)
            # benchmark(args, f, shapes, forward_only=False, log=True)
    

if __name__ == "__main__":
    main(parse_args())