import torch

# GLOBAL VARS
FULL_DTYPE = torch.float32

ncclcomm_dict = {}
cpu_group_dict = {}

def is_exist_global_var(key: str):
    return key in globals().keys()

def get_global_var(key: str):
    assert key in globals().keys(), f'Invalid key: {key}'
    return globals()[key]

def set_global_var(key: str, value):
    globals()[key] = value
