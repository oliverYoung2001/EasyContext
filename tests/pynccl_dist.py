import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import multiprocessing
from typing import Dict, List

# import pytest
import torch
import torch.distributed

from tests.distributed.communication_op import (  # noqa
    tensor_model_parallel_all_reduce)
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from tests.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
from tests.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             get_world_group, graph_capture,
                                             init_distributed_environment)
from tests.utils import update_environment_variables
from functools import partial


#[NOTE]: nccl2.21.5 is OK, but nccl2.20.5 is not OK !!!

def execute_func_2(a, b, d, send_comm: PyNcclCommunicator, recv_comm: PyNcclCommunicator, streams: list, group = None):  # send + recv
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    peer_send = (rank + 1) % world_size
    peer_recv = (rank - 1 + world_size) % world_size
    send_group_rank = 0 if rank + 1 < world_size else 1
    recv_group_rank = 1 if rank > 0 else 0
    # send_buf = a[: a.nelement() // world_size * (rank + 1)]
    # recv_buf = b[: b.nelement() // world_size * (peer_recv + 1)]
    # Comm
    if rank == 0:   # avoid deadlock  # overlap failed on every N
        # send_comm.group_start()
        send_comm.send(a, send_group_rank ^ 1, stream=streams[1])
        # send_comm.group_end()
        
        # recv_comm.group_start()
        recv_comm.recv(b, recv_group_rank ^ 1, stream=streams[2])
        # recv_comm.group_end()
    else:
        # recv_comm.group_start()
        recv_comm.recv(b, recv_group_rank ^ 1, stream=streams[2])
        # recv_comm.group_end()
        
        # send_comm.group_start()
        send_comm.send(a, send_group_rank ^ 1, stream=streams[1])
        # send_comm.group_end()
    # if pynccl_comm.rank == 0:   # avoid deadlock  # overlap failed only on N = 2
    #     reqs = torch.distributed.isend(send_buf, peer_send)
    #     reqr = torch.distributed.irecv(recv_buf, peer_recv)
    # else:
    #     reqr = torch.distributed.irecv(recv_buf, peer_recv)
    #     reqs = torch.distributed.isend(send_buf, peer_send)
    # reqs.wait()
    # Comp
    # d = a + b
    
    # reqr.wait()
    # if pynccl_comm.rank == 0:   # avoid deadlock # overlap failed on every N
    #     torch.distributed.send(a, peer_send)
    #     torch.distributed.recv(b, peer_recv)
    # else:
    #     torch.distributed.recv(b, peer_recv)
    #     torch.distributed.send(a, peer_send)
    return a, b

def execute_func_1(a, b, d, pynccl_comm: PyNcclCommunicator, streams: list):    # send/recv + comp
    peer = (pynccl_comm.rank + 1) % pynccl_comm.world_size
    if pynccl_comm.rank == 0:
        pynccl_comm.send(a, peer, stream=streams[1])
    else:
        pynccl_comm.recv(b, peer, stream=streams[2])
    torch.cuda.current_stream().wait_stream(streams[2])
    c = a + b
    c *= (pynccl_comm.rank + 2)
    streams[1].wait_stream(torch.cuda.current_stream())
    if pynccl_comm.rank == 0:   # avoid deadlock
        pynccl_comm.recv(d, peer, stream=streams[2])
    else:
        pynccl_comm.send(c, peer, stream=streams[1])
    torch.cuda.current_stream().wait_stream(streams[2])
    e = d + (pynccl_comm.rank + 3)
    return c, e
  
def execute_func(a, b, d, send_comm: PyNcclCommunicator, recv_comm: PyNcclCommunicator, streams: list, group = None):  # send + recv + comp
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    send_group_rank = 0 if rank + 1 < world_size else 1
    recv_group_rank = 1 if rank > 0 else 0
    # Comm
    if rank == 0:   # avoid deadlock  # overlap failed on every N
        send_comm.send(a, send_group_rank ^ 1, stream=streams[1])
        recv_comm.recv(b, recv_group_rank ^ 1, stream=streams[2])
    else:
        recv_comm.recv(b, recv_group_rank ^ 1, stream=streams[2])
        send_comm.send(a, send_group_rank ^ 1, stream=streams[1])
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     pynccl_comm.send(a, peer, stream=streams[1])
    #     pynccl_comm.recv(b, peer, stream=streams[2])
    # else:
    #     pynccl_comm.recv(b, peer, stream=streams[2])
    #     pynccl_comm.send(a, peer, stream=streams[1])
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     torch.distributed.isend(a, peer)
    #     reqr = torch.distributed.irecv(b, peer)
    # else:
    #     reqr = torch.distributed.irecv(b, peer)
    #     torch.distributed.isend(a, peer)
    torch.cuda.current_stream().wait_stream(streams[2])
    # reqr.wait()
    c = a + b
    c *= (rank + 2)
    streams[1].wait_stream(torch.cuda.current_stream())
    if rank == 0:   # avoid deadlock  # overlap failed on every N
        send_comm.send(c, send_group_rank ^ 1, stream=streams[1])
        recv_comm.recv(d, recv_group_rank ^ 1, stream=streams[2])
    else:
        recv_comm.recv(d, recv_group_rank ^ 1, stream=streams[2])
        send_comm.send(c, send_group_rank ^ 1, stream=streams[1])
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     pynccl_comm.send(c, peer, stream=streams[1])
    #     pynccl_comm.recv(d, peer, stream=streams[2])
    # else:
    #     pynccl_comm.recv(d, peer, stream=streams[2])
    #     pynccl_comm.send(c, peer, stream=streams[1])
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     torch.distributed.isend(c, peer)
    #     reqr = torch.distributed.irecv(d, peer)
    # else:
    #     reqr = torch.distributed.irecv(d, peer)
    #     torch.distributed.isend(c, peer)
    torch.cuda.current_stream().wait_stream(streams[2])
    # reqr.wait()
    d += rank + 3
    return c, d
                
def worker_fn_with_cudagraph_multistreams():
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    new_group = torch.distributed.new_group(range(torch.distributed.get_world_size()))
    nelem = 1024 * 1024 * 1024  # 1G
    # nelem = 128 * 1024 * 1024  # 128M
    # nelem = 1024  # 1K
    # nelem = 16
    
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group, ranks=list(range(world_size)),
                                         device=get_world_group().device)
        # ring pattern groups:
        for r in range(world_size):
            peer_r = (r + 1) % world_size
            ranks = sorted([r, peer_r])
            # new_group = torch.distributed.new_group(ranks, backend='gloo')
            if rank in ranks:
                pycomm = PyNcclCommunicator(get_world_group().cpu_group, ranks=ranks, device=rank)
                if r == rank:
                    send_comm = pycomm
                else:
                    recv_comm = pycomm
        
        # run something in the default stream to initialize torch engine
        a = torch.ones(nelem, device=f'cuda:{pynccl_comm.rank}', dtype=torch.bfloat16)
        b = torch.zeros(nelem, device=f'cuda:{pynccl_comm.rank}', dtype=torch.bfloat16)
        # c = torch.empty(nelem, device=f'cuda:{pynccl_comm.rank}', dtype=torch.bfloat16)
        d = torch.zeros(nelem, device=f'cuda:{pynccl_comm.rank}', dtype=torch.bfloat16)
        a *= pynccl_comm.rank + 1
        # print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
        streams = [
            torch.cuda.current_stream(),    # Comp
            torch.cuda.Stream(torch.cuda.current_device()), # Send
            torch.cuda.Stream(torch.cuda.current_device()), # Recv
        ]
        # print(f'streams: {streams[0].cuda_stream} {streams[1].cuda_stream}, {streams[2].cuda_stream}')
        rank = pynccl_comm.rank
        world_size = pynccl_comm.world_size
        torch.cuda.synchronize()
        
        use_cudagraph = False
        use_cudagraph = True
        if use_cudagraph:
            with torch.profiler.profile():  # workaround of issue 75504 of PyTorch
                pass
            
            with torch.cuda.graph(graph):
                # preprocess
                for stream in streams[1:]:
                    stream.wait_stream(torch.cuda.current_stream())
                
                # execute kernels
                c, e = execute_func(a, b, d, send_comm, recv_comm, streams)
                
                # postprocess
                for stream in streams[1:]:
                    torch.cuda.current_stream().wait_stream(stream)

        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'test_cudagraph_w{world_size}_r{rank}'
        TB_DIR = f'./prof_results/tb_cg'
        os.makedirs(TB_DIR, exist_ok=True)
        
        # for iter in range(TOTAL_TURNS):
        #     # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
        #     graph.replay()
        #     # print(f'rank{rank}, iter{iter}', flush=True)
        #     if (iter + 1) % BARRIER_FREQ == 0:
        #         torch.cuda.synchronize()
        #         torch.distributed.barrier()
                
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=TB_DIR, 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
        # if True:
            for iter in range(TOTAL_TURNS):
                if use_cudagraph:
                    graph.replay()
                else:
                    # preprocess
                    for stream in streams[1:]:
                        stream.wait_stream(torch.cuda.current_stream())
                    
                    # execute kernels
                    c, e = execute_func(a, b, d, send_comm, recv_comm, streams)
                    
                    # postprocess
                    for stream in streams[1:]:
                        torch.cuda.current_stream().wait_stream(stream)
                
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                    print(f'rank{rank}, a: {a.mean()}, b: {b.mean()}, c: {c.mean()}, d: {d.mean()}, e: {e.mean()}', flush=True)
                if 'prof' in locals().keys():
                    locals()['prof'].step()
        # print(f'rank{rank}, Out !!!')
        # print(f'rank{rank}, a: {a.mean()}, b: {b.mean()}, c: {c.mean()}, d: {d.mean()}', flush=True)
        
# @pytest.mark.skipif(torch.cuda.device_count() < 2,
#                     reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph_multistreams():
    worker_fn_with_cudagraph_multistreams()
    

def main():
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs to run the test."
    # init envs
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    env = {}
    # env['RANK'] = str(i)
    env['LOCAL_RANK'] = str(rank)
    # env['WORLD_SIZE'] = str(number_of_processes)
    env['LOCAL_WORLD_SIZE'] = str(world_size)
    # env['MASTER_ADDR'] = 'localhost'
    update_environment_variables(env)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    init_distributed_environment()
    
    # run test
    test_pynccl_with_cudagraph_multistreams()
    
if __name__ == '__main__':
    main()
