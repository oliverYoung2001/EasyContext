import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
import pytest
try:
    from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
except ImportError:
    pass
    # print(f'[WARN]: flash_attn not found, disable flash_attn', flush=True)
# from orchestrated_attn.utils import Input_Row_Fwd, Input_Col_Fwd, Output_Row_Fwd, Output_Col_Fwd

MBS = 1
S = 4 * 1024    # 4k
Nh = 32
D = 128
DTYPES = torch.bfloat16
GRAPH_STREAM = torch.cuda.Stream(device='cuda')
USE_TORCH_PROFILER = True
# USE_TORCH_PROFILER = False

# Comp Func
# def fwd_comp_func(inp_row: Input_Row_Fwd, inp_col: Input_Col_Fwd,
def fwd_comp_func(Q, K, V,
                  causal, dropout_p, softmax_scale, window_size, alibi_slopes) -> tuple:
    # inp_row, inp_col = Input_Row_Fwd(Q), Input_Col_Fwd(K, V)
    # print(f'rank{0}, dropout_p: {dropout_p}, softmax_scale: {softmax_scale}, , causal: {causal}, window_size: {window_size}, alibi_slopes: {alibi_slopes}, return_softmax: {True and dropout_p > 0}', flush=True)
    O, _, _, _, _, lse, _, _ = _flash_attn_forward(
        # inp_row.Q,
        # inp_col.K,
        # inp_col.V,
        Q,
        K,
        V,
        dropout_p,
        softmax_scale,
        causal=causal,
        window_size=window_size,
        alibi_slopes=alibi_slopes,
        return_softmax=True and dropout_p > 0,
    )
    return O, lse
    return (Output_Row_Fwd(O), Output_Col_Fwd())
    lse = lse.transpose(-2, -1).contiguous().unsqueeze(dim=-1) # block_out, block_lse # [mbs, S, Nh, D], [mbs, S, Nh, 1]
    return (Output_Row_Fwd(O, lse), Output_Col_Fwd())

@pytest.mark.parametrize("mbs", [MBS])
@pytest.mark.parametrize("S", [S])
@pytest.mark.parametrize("Nh", [Nh])
@pytest.mark.parametrize("D", [D])
@pytest.mark.parametrize("dtype", [DTYPES])
@pytest.mark.parametrize("dropout_p", [0])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("window_size", [(-1, -1)])
@pytest.mark.parametrize("alibi_slopes", [None])
def test_cuda_graph_with_flashattn(
    mbs, S, Nh, D, dtype,
    dropout_p: float, causal: bool, 
    window_size: tuple, alibi_slopes: torch.Tensor
):
    try:
        import flash_attn
    except ImportError:
        print(f'[WARN]: flash_attn not found, disable flash_attn !!!', flush=True)
        return
    WARMUP = 11
    WARMUP = 0
    TIMES = 20
    Q = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    K = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    V = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    sm_scale = Q.shape[-1] ** (-0.5)
    for _ in range(WARMUP):
        for __ in range(4):
            outs = fwd_comp_func(
                # inp_row, inp_col, 
                Q, K, V,
                causal, dropout_p, sm_scale, window_size, alibi_slopes
            )
        torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    # graph_stream = torch.cuda.Stream(device='cuda')
    global GRAPH_STREAM
    print(f'current stream: {torch.cuda.current_stream().cuda_stream}', flush=True)
    print(f'GRAPH_STREAM: {GRAPH_STREAM.cuda_stream}', flush=True)
    # with torch.cuda.graph(g, stream=torch.cuda.current_stream()):   # stream must not be the same as default stream, i.e. stream 0 !!!
    with torch.cuda.graph(g, stream=GRAPH_STREAM):
        print(f'current stream: {torch.cuda.current_stream().cuda_stream}', flush=True) # new current stream
        # for __ in range(4):
        with torch.no_grad():
            with torch.cuda.stream(GRAPH_STREAM):
                with torch.cuda.nvtx.range(f'fwd_comp_func'):
                    outs = fwd_comp_func(
                        # inp_row, inp_col, 
                        Q, K, V,
                        causal, dropout_p, sm_scale, window_size, alibi_slopes
                    )
    torch.cuda.synchronize()
    # print(f'outs: {outs[0].data.shape}, {outs[1].data.shape}')
    for _ in range(TIMES):
        g.replay()
    torch.cuda.synchronize()
    # print(f'outs: {outs[0].data.shape}, {outs[1].data.shape}')


def test_cuda_graph_with_add():
    global USE_TORCH_PROFILER
    a = torch.tensor([1, 2], dtype=torch.int32, device='cuda')
    b = torch.tensor([3, 4], dtype=torch.int32, device='cuda')
    g = torch.cuda.CUDAGraph()
    global GRAPH_STREAM
    with torch.cuda.graph(g, stream=GRAPH_STREAM):
        with torch.cuda.stream(GRAPH_STREAM):
            with torch.cuda.nvtx.range(f'add'):
                c = a + b
    if USE_TORCH_PROFILER:
        BARRIER_FREQ = 4
        BARRIER_FREQ = 1
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 1, BARRIER_FREQ * 1, BARRIER_FREQ * 3, 1
        WAIT, WARMUP, ACTIVE, REPEAT = BARRIER_FREQ * 0, BARRIER_FREQ * 0, BARRIER_FREQ * 1, 1
        TOTAL_TURNS = (WAIT + WARMUP + ACTIVE) * (REPEAT)
        TRACE_NAME = f'test_cuda_graph_with_add'
        DIR_NAME = f'./prof_results/tb_cg'
        os.makedirs(DIR_NAME, exist_ok=True)
        
        with torch.profiler.profile():  # workaround of issue 75504 of PyTorch
            pass
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=WAIT, warmup=WARMUP, active=ACTIVE, repeat=REPEAT),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                dir_name=DIR_NAME, 
                worker_name=TRACE_NAME,
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for iter in range(TOTAL_TURNS):
                g.replay()
                # a.copy_(torch.tensor([- 1, - 2], dtype=torch.int32, device='cuda'))
                # g.replay()
                if (iter + 1) % BARRIER_FREQ == 0:
                    torch.cuda.synchronize()
                    # torch.distributed.barrier()
                prof.step()
    else:
        # print(f'c: {c}')
        g.replay()
        assert c.allclose(a + b)
        # print(f'c: {c}')
        
        a.copy_(torch.tensor([- 1, - 2], dtype=torch.int32, device='cuda'))
        g.replay()
        assert c.allclose(a + b)
        # print(f'c: {c}')
    torch.cuda.synchronize()

if __name__ == '__main__':
    test_cuda_graph_with_flashattn(MBS, S, Nh, D, DTYPES, 0, True, (-1, -1), None)
    test_cuda_graph_with_add()

