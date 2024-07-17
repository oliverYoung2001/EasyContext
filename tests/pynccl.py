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


#[NOTE]: nccl2.21.5 is OK, but nccl2.20.5 is not OK !!!

def distributed_run(fn, world_size):
    number_of_processes = world_size
    processes: List[multiprocessing.Process] = []
    for i in range(number_of_processes):
        env: Dict[str, str] = {}
        env['RANK'] = str(i)
        env['LOCAL_RANK'] = str(i)
        env['WORLD_SIZE'] = str(number_of_processes)
        env['LOCAL_WORLD_SIZE'] = str(number_of_processes)
        env['MASTER_ADDR'] = 'localhost'
        # env['MASTER_PORT'] = '14721'
        p = multiprocessing.Process(target=fn, args=(env, ))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    for p in processes:
        assert p.exitcode == 0


def worker_fn_wrapper(fn):
    # `multiprocessing.Process` cannot accept environment variables directly
    # so we need to pass the environment variables as arguments
    # and update the environment variables in the function
    def wrapped_fn(env):
        update_environment_variables(env)
        local_rank = os.environ['LOCAL_RANK']
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        init_distributed_environment()
        fn()

    return wrapped_fn

def execute_func(a, b, d, pynccl_comm: PyNcclCommunicator, streams: list):  # send + recv
    peer = (pynccl_comm.rank + 1) % pynccl_comm.world_size
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     pynccl_comm.send(a, peer, stream=streams[1])
    #     pynccl_comm.recv(b, peer, stream=streams[2])
    # else:
    #     pynccl_comm.recv(b, peer, stream=streams[2])
    #     pynccl_comm.send(a, peer, stream=streams[1])
    if pynccl_comm.rank == 0:   # avoid deadlock
        torch.distributed.isend(a, peer)
        reqr = torch.distributed.irecv(b, peer)
    else:
        reqr = torch.distributed.irecv(b, peer)
        torch.distributed.isend(a, peer)
    reqr.wait()
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
  
def execute_func_0(a, b, d, pynccl_comm: PyNcclCommunicator, streams: list):  # send + recv + comp
    peer = (pynccl_comm.rank + 1) % pynccl_comm.world_size
    if pynccl_comm.rank == 0:   # avoid deadlock
        pynccl_comm.send(a, peer, stream=streams[1])
        pynccl_comm.recv(b, peer, stream=streams[2])
    else:
        pynccl_comm.recv(b, peer, stream=streams[2])
        pynccl_comm.send(a, peer, stream=streams[1])
    if pynccl_comm.rank == 0:   # avoid deadlock
        torch.distributed.isend(a, peer)
        reqr = torch.distributed.irecv(b, peer)
    else:
        reqr = torch.distributed.irecv(b, peer)
        torch.distributed.isend(a, peer)
    torch.cuda.current_stream().wait_stream(streams[2])
    reqr.wait()
    c = a + b
    c *= (pynccl_comm.rank + 2)
    streams[1].wait_stream(torch.cuda.current_stream())
    # if pynccl_comm.rank == 0:   # avoid deadlock
    #     pynccl_comm.send(c, peer, stream=streams[1])
    #     pynccl_comm.recv(d, peer, stream=streams[2])
    # else:
    #     pynccl_comm.recv(d, peer, stream=streams[2])
    #     pynccl_comm.send(c, peer, stream=streams[1])
    if pynccl_comm.rank == 0:   # avoid deadlock
        torch.distributed.isend(c, peer)
        reqr = torch.distributed.irecv(d, peer)
    else:
        reqr = torch.distributed.irecv(d, peer)
        torch.distributed.isend(c, peer)
    torch.cuda.current_stream().wait_stream(streams[2])
    reqr.wait()
    d += pynccl_comm.rank + 3
    return c
                
@worker_fn_wrapper
def worker_fn_with_cudagraph_multistreams():
    nelem = 1024 * 1024 * 1024  # 1G
    # nelem = 128 * 1024 * 1024  # 128M
    # nelem = 1024  # 1K
    # nelem = 16
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                         device=get_world_group().device)
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
        
        # with torch.profiler.profile():  # workaround of issue 75504 of PyTorch
        #     pass
        
        # with torch.cuda.graph(graph), pynccl_comm.change_state(enable=True):
        #     # preprocess
        #     for stream in streams[1:]:
        #         stream.wait_stream(torch.cuda.current_stream())
            
        #     # execute kernels
        #     c, e = execute_func(a, b, d, pynccl_comm, streams)
            
        #     # postprocess
        #     for stream in streams[1:]:
        #         torch.cuda.current_stream().wait_stream(stream)

        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'test_cudagraph_w{world_size}_r{rank}'
        
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
                dir_name=f'tb_cg', 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
        # if True:
            for iter in range(TOTAL_TURNS):
                # graph.replay()
                
                # # preprocess
                # for stream in streams[1:]:
                #     stream.wait_stream(torch.cuda.current_stream())
                
                # execute kernels
                with pynccl_comm.change_state(enable=True):
                    c, e = execute_func(a, b, d, pynccl_comm, streams)
                
                # # postprocess
                # for stream in streams[1:]:
                #     torch.cuda.current_stream().wait_stream(stream)
                    
                # torch.cuda.synchronize()
                
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
    distributed_run(worker_fn_with_cudagraph_multistreams, 2)
    

def main():
    assert torch.cuda.device_count() >= 2, "Need at least 2 GPUs to run the test."
    test_pynccl_with_cudagraph_multistreams()
    
if __name__ == '__main__':
    main()
