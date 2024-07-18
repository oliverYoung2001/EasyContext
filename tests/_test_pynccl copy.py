import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import multiprocessing
from typing import Dict, List

import pytest
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


@worker_fn_wrapper
def worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    tensor = torch.ones(16, 1024, 1024,
                        dtype=torch.float32).cuda(pynccl_comm.rank)
    with pynccl_comm.change_state(enable=True):
        pynccl_comm.all_reduce(tensor)
    result = tensor.mean().cpu().item()
    assert result == pynccl_comm.world_size


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl():
    distributed_run(worker_fn, 2)


@worker_fn_wrapper
def multiple_allreduce_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 1], backend="gloo"),
        torch.distributed.new_group(ranks=[2, 3], backend="gloo")
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 1] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    with pynccl_comm.change_state(enable=True):
        # two groups can communicate independently
        if torch.distributed.get_rank() in [0, 1]:
            pynccl_comm.all_reduce(tensor)
            pynccl_comm.all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 4
        else:
            pynccl_comm.all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_allreduce():
    # this tests pynccl for multiple tp groups, in a standalone way
    # i.e. call `pynccl_comm.all_reduce` directly
    distributed_run(multiple_allreduce_worker_fn, 4)


@worker_fn_wrapper
def multiple_allreduce_with_vllm_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    ensure_model_parallel_initialized(2, 2)
    tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    with graph_capture():
        # two tp groups can communicate independently
        if torch.distributed.get_rank() in [0, 1]:
            tensor = tensor_model_parallel_all_reduce(tensor)
            tensor = tensor_model_parallel_all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 4
        else:
            tensor = tensor_model_parallel_all_reduce(tensor)
            result = tensor.mean().cpu().item()
            assert result == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_allreduce_with_vllm():
    # this tests pynccl for multiple tp groups, together with vllm
    # i.e. call `tensor_model_parallel_all_reduce`
    distributed_run(multiple_allreduce_with_vllm_worker_fn, 4)


@worker_fn_wrapper
def worker_fn_with_cudagraph():
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                         device=get_world_group().device)
        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=f'cuda:{pynccl_comm.rank}')
        torch.cuda.synchronize()
        with torch.cuda.graph(
                graph, stream=pynccl_comm.stream), pynccl_comm.change_state(
                    enable=True):
            # operation during the graph capture is recorded but not executed
            # see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#creating-a-graph-using-stream-capture # noqa
            pynccl_comm.all_reduce(a)
        pynccl_comm.stream.synchronize()
        assert a.mean().cpu().item() == pynccl_comm.world_size**0
        # print(f'a: {a}')
        graph.replay()
        pynccl_comm.stream.synchronize()
        assert a.mean().cpu().item() == pynccl_comm.world_size**1
        # print(f'a: {a}')


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph():
    distributed_run(worker_fn_with_cudagraph, 2)
    
@worker_fn_wrapper
def worker_fn_with_cudagraph_multistreams():
    with torch.no_grad():
        graph = torch.cuda.CUDAGraph()
        pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                         device=get_world_group().device)
        # run something in the default stream to initialize torch engine
        a = torch.ones((4, 4), device=f'cuda:{pynccl_comm.rank}')
        b = torch.empty((4, 4), device=f'cuda:{pynccl_comm.rank}')
        d = torch.empty((4, 4), device=f'cuda:{pynccl_comm.rank}')
        a *= pynccl_comm.rank + 1
        # print(f'torch.cuda.current_device(): {torch.cuda.current_device()}')
        streams = [
            torch.cuda.current_stream(),    # Comp
            torch.cuda.Stream(torch.cuda.current_device()), # Send
            torch.cuda.Stream(torch.cuda.current_device()), # Recv
        ]
        rank = pynccl_comm.rank
        world_size = pynccl_comm.world_size
        torch.cuda.synchronize()
        with torch.cuda.graph(graph), pynccl_comm.change_state(enable=True):
            # preprocess
            for stream in streams[1:]:
                stream.wait_stream(torch.cuda.current_stream())
            
            # execute kernels
            peer = (pynccl_comm.rank + 1) % pynccl_comm.world_size
            if pynccl_comm.rank == 0:   # avoid deadlock
                pynccl_comm.send(a, peer, stream=streams[1])
                pynccl_comm.recv(b, peer, stream=streams[2])
            else:
                pynccl_comm.recv(b, peer, stream=streams[2])
                pynccl_comm.send(a, peer, stream=streams[1])
            torch.cuda.current_stream().wait_stream(streams[2])
            c = (a + b) * (pynccl_comm.rank + 2)
            streams[1].wait_stream(torch.cuda.current_stream())
            if pynccl_comm.rank == 0:   # avoid deadlock
                pynccl_comm.send(c, peer, stream=streams[1])
                pynccl_comm.recv(d, peer, stream=streams[2])
            else:
                pynccl_comm.recv(d, peer, stream=streams[2])
                pynccl_comm.send(c, peer, stream=streams[1])
            torch.cuda.current_stream().wait_stream(streams[2])
            d += pynccl_comm.rank + 3
            
            # postprocess
            for stream in streams[1:]:
                torch.cuda.current_stream().wait_stream(stream)
        
        # torch.cuda.synchronize()
        # torch.distributed.barrier()
        
        BARRIER_FREQ = 4
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        # WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'test_cudagraph_w{world_size}_r{rank}'
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
            for iter in range(TOTAL_TURNS):
                # torch.distributed.all_reduce(sync_tensor, async_op=False)    # for sync and alignment
                graph.replay()
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    torch.distributed.barrier()
                prof.step()
        # print(f'rank{rank}, a: {a.mean()}, b: {b.mean()}, c: {c.mean()}, d: {d.mean()}', flush=True)
        
@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_with_cudagraph_multistreams():
    distributed_run(worker_fn_with_cudagraph_multistreams, 2)


@worker_fn_wrapper
def send_recv_worker_fn():
    pynccl_comm = PyNcclCommunicator(get_world_group().cpu_group,
                                     device=get_world_group().device)
    if pynccl_comm.rank == 0:
        tensor = torch.ones(16, 1024, 1024,
                            dtype=torch.float32).cuda(pynccl_comm.rank)
    else:
        tensor = torch.empty(16, 1024, 1024,
                             dtype=torch.float32).cuda(pynccl_comm.rank)
    with pynccl_comm.change_state(enable=True):
        if pynccl_comm.rank == 0:
            pynccl_comm.send(tensor,
                             dst=(pynccl_comm.rank + 1) %
                             pynccl_comm.world_size)
        else:
            pynccl_comm.recv(tensor,
                             src=(pynccl_comm.rank - 1) %
                             pynccl_comm.world_size)
    result = tensor.mean().cpu().item()
    assert result == 1


@pytest.mark.skipif(torch.cuda.device_count() < 2,
                    reason="Need at least 2 GPUs to run the test.")
def test_pynccl_send_recv():
    distributed_run(send_recv_worker_fn, 2)


@worker_fn_wrapper
def multiple_send_recv_worker_fn():
    device = torch.device(f"cuda:{torch.distributed.get_rank()}")
    groups = [
        torch.distributed.new_group(ranks=[0, 2], backend="gloo"),
        torch.distributed.new_group(ranks=[1, 3], backend="gloo")
    ]
    group = groups[0] if torch.distributed.get_rank() in [0, 2] else groups[1]
    pynccl_comm = PyNcclCommunicator(group=group, device=device)
    if torch.distributed.get_rank() == 0:
        tensor = torch.ones(16, 1024, 1024, dtype=torch.float32, device=device)
    elif torch.distributed.get_rank() == 1:
        tensor = 2 * torch.ones(
            16, 1024, 1024, dtype=torch.float32, device=device)
    else:
        tensor = torch.empty(16,
                             1024,
                             1024,
                             dtype=torch.float32,
                             device=device)
    with pynccl_comm.change_state(enable=True):
        if torch.distributed.get_rank() in [0, 1]:
            pynccl_comm.send(tensor,
                             dst=(pynccl_comm.rank + 1) %
                             pynccl_comm.world_size)
        else:
            pynccl_comm.recv(tensor,
                             src=(pynccl_comm.rank - 1) %
                             pynccl_comm.world_size)
    result = tensor.mean().cpu().item()
    if torch.distributed.get_rank() in [0, 2]:
        assert result == 1
    else:
        assert result == 2


@pytest.mark.skipif(torch.cuda.device_count() < 4,
                    reason="Need at least 4 GPUs to run the test.")
def test_pynccl_multiple_send_recv():
    distributed_run(multiple_send_recv_worker_fn, 4)


def test_ncclGetUniqueId():
    lib = NCCLLibrary()
    unique_id = lib.ncclGetUniqueId()
    # `list(unique_id.internal)` is something like this:
    # [34, -16, 23, 83, 109, -19, 59, 95, 2, 0, -86, 55, 10, -128, 0, 29, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    # 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # as long as the function doesn't raise an exception, we're good
    assert unique_id is not None
