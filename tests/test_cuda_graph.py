import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
import pytest
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from orchestrated_attn.utils import Input_Row_Fwd, Input_Col_Fwd, Output_Row_Fwd, Output_Col_Fwd

MBS = 1
S = 4 * 1024    # 4k
Nh = 32
D = 128
DTYPES = torch.bfloat16

# Comp Func
def fwd_comp_func(inp_row: Input_Row_Fwd, inp_col: Input_Col_Fwd, causal,
                  dropout_p, softmax_scale, window_size, alibi_slopes) -> tuple:
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

@pytest.mark.parametrize("mbs", [MBS])
@pytest.mark.parametrize("S", [S])
@pytest.mark.parametrize("Nh", [Nh])
@pytest.mark.parametrize("D", [D])
@pytest.mark.parametrize("dtype", [DTYPES])
@pytest.mark.parametrize("dropout_p", [0])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("window_size", [(-1, -1)])
@pytest.mark.parametrize("alibi_slopes", [None])
def test_cuda_graph_with_flashattn(
    mbs, S, Nh, D, dtype,
    dropout_p: float, causal: bool, 
    window_size: tuple, alibi_slopes: torch.Tensor
):
    Q = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    K = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    V = torch.empty((mbs, S, Nh, D), dtype=dtype, device='cuda')
    sm_scale = Q.shape[-1] ** (-0.5)
    inp_row, inp_col = Input_Row_Fwd(Q), Input_Col_Fwd(K, V)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = fwd_comp_func(
            inp_row, inp_col, causal,
            dropout_p, sm_scale, window_size, alibi_slopes
        )
    # print(f'outs: {outs[0].data.shape}, {outs[1].data.shape}')
    g.replay()
    # print(f'outs: {outs[0].data.shape}, {outs[1].data.shape}')


def test_cuda_graph_with_add():
    a = torch.tensor([1, 2], dtype=torch.int32, device='cuda')
    b = torch.tensor([3, 4], dtype=torch.int32, device='cuda')
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        c = a + b
    # print(f'c: {c}')
    g.replay()
    assert c.allclose(a + b)
    # print(f'c: {c}')
    
    a.copy_(torch.tensor([- 1, - 2], dtype=torch.int32, device='cuda'))
    g.replay()
    assert c.allclose(a + b)
    # print(f'c: {c}')

