import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import numpy as np
from search_algo.global_vars import *

def get_qo_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    qo_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    qo_schedule_table.fill(TASK_STATUS.EMPTY.value) 
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    qo_schedule_table[i, j, k, l] = S_map[i, l]
    return qo_schedule_table

def get_kv_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    kv_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    kv_schedule_table -= 1  # -1 means not used
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    kv_schedule_table[i, j, k, l] = S_map[i, k]
    return kv_schedule_table

def get_cc_optimal_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    assert split_degrees[0] == split_degrees[1]
    assert split_degrees[2] == split_degrees[3] == 1
    if split_degrees[0] == 4:
        if causal:
            cc_schedule_table = np.array([[[
                [0, -1, -1, -1],
                [1,  1, -1, -1],
                [0,  0,  2, -1],
                [1,  3,  2,  3],
            ]]], dtype=np.int32)
            cc_schedule_table = np.array([[[
                [0, -1, -1, -1],
                [1,  1, -1, -1],
                [0,  0,  2, -1],
                [3,  1,  2,  3],
            ]]], dtype=np.int32)
            cc_schedule_table = np.array([[[
                [0, -1, -1, -1],
                [1,  1, -1, -1],
                [1,  2,  2, -1],
                [0,  3,  0,  3],
            ]]], dtype=np.int32)
        else:
            cc_schedule_table = np.array([[[
                [0,  1,  0,  1],
                [0,  1,  0,  1],
                [2,  3,  2,  3],
                [2,  3,  2,  3],
            ]]], dtype=np.int32)
    elif split_degrees[0] == 5:
        if causal:
            cc_schedule_table = np.array([[[
                [ 0, -1, -1, -1, -1],
                [ 0,  1, -1, -1, -1],
                [ 1,  1,  2, -1, -1],
                [ 0,  4,  2,  3, -1],
                [ 3,  4,  2,  3,  4],
            ]]], dtype=np.int32)
        else:
            raise NotImplementedError()
    elif split_degrees[0] == 3:
        if causal:
            cc_schedule_table = np.array([[[
                [ 0, -1, -1],
                [ 0,  1, -1],
                [ 0,  1,  2],
            ]]], dtype=np.int32)
        else:
            raise NotImplementedError()
    return cc_schedule_table
