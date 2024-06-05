import torch

def add(a, b):
    a = a + b
    a = torch.rand((3, 4))

a = torch.zeros((3, 4))

add(a, 1)

print(f'a: {a}')