import torch

a = torch.tensor(1, device='cuda')
# WarnUp before cuda graph capture
for _ in range(11):
    b = a + a
# Graph Capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    b = a + a
# Torch.profiler
with torch.profiler.profile():
    g.replay()
    torch.cuda.synchronize()
