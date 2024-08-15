
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from queue import PriorityQueue
from heapq import heappush, heappop, heappushpop
from search_algo.global_vars import *
import math
import regex as re
import numpy as np
from typing import Optional

def get_factors(n: int):
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

class FinitePriorityQueue():
    """finite min heap
    """
    def __init__(self, maxsize: int, pack_func, unpack_func):
        self.maxsize = maxsize
        self.queue = []
        self.pack_func = pack_func
        self.unpack_func = unpack_func
    
    def reset(self):
        self.queue = []

    def push(self, item):
        if len(self.queue) < self.maxsize:
            heappush(self.queue, self.pack_func(item))
        else:
            heappushpop(self.queue, self.pack_func(item))

    def pop(self):
        return self.unpack_func(heappop(self.queue))

    def __len__(self):
        return len(self.queue)
    
def convert_profile_data_to_map(profile_list):
    profile_map = {}
    for i in range(len(profile_list)):
        map_key = tuple(profile_list[i][0])
        assert map_key not in profile_map.keys()
        profile_map[map_key] = np.array(profile_list[i][1]) / 1e6    # [fwd/bwd], (s)
    return profile_map

# # Helper function to pretty-print message sizes
# def convert_throughput(size_bytes, round_=3):
#     if size_bytes == 0:
#         return "0B"
#     size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
#     i = int(math.floor(math.log(size_bytes, BYTE_MULTPLE_DOWN)))
#     p = math.pow(BYTE_MULTPLE_DOWN, i)
#     s = round(size_bytes / p, round_)
#     return "%s %s" % (s, size_name[i])

def convert_throughput_to_B(size: float, unit: str):
    size_name = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]
    assert unit in size_name, f'Invalid unit: {unit}'
    return size * pow(BYTE_MULTPLE_DOWN, size_name.index(unit)) # ?B -> B
    
def convert_profile_data_to_comm_map(file_name: str, num_gpus_div: int):
    profile_map = {}    # Bytes -> GB/s
    # SIZE 8192, REAL_BD 403.402 MB/s, BD/PAIR 201.701 MB/s, time 0.0041 s, comm_vol 1.638 MB
    pat1 = re.compile(r'^SIZE (\d+),.*?BD/PAIR (\d*(\.\d*)?) ([A-Z]*)/s.*$')
    # SIZE 131072, REAL_BD 16.653 GB/s, time 0.0013 s, comm_vol 20.972 MB
    pat2 = re.compile(r'^SIZE (\d+),.*?REAL_BD (\d*(\.\d*)?) ([A-Z]*)/s.*$')

    with open(file_name, 'r') as f:
        for line in f.readlines():
            res = pat1.match(line)
            if res is None:
                res = pat2.match(line)
            if res is None:
                continue
            profile_map[(int(res.group(1)),)] = convert_throughput_to_B(float(res.group(2)), res.group(4)) \
                                                / pow(BYTE_MULTPLE_DOWN, 3) / num_gpus_div
    # print(f'profile_map: {profile_map}')
    return profile_map

def convert_node_profile_data_to_comp_map(file_name: Optional[str], local_size: int):
    # map_key: ((Sq, Skv), (Nhq, Nhg), bs, D, causal) -> Time[fwd/bwd]  # S per GPU !!!
    if file_name is None:
        return {}
    profile_map = {} 
    # fob = SP = S = Nh = bs = D = causal = None
    # cur_map_key = None
    
# fob=0, plan_paths: ['/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=1_X=8_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=2_X=4_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=4_X=2_dim=0.pkl', '/home/zhaijidong/yhy/llm/EasyContext/search_algo/execution_plans/intra_SP8_fob=0/SP=8_fob=0_Y=8_X=1_dim=0.pkl']

# SP=(1,8),S=(24576,524288),Nh=(1,1),bs=1,D=128,causal=False:
# # ring_flash_attn_func, fwd
# mfu: 70.723 Tflops/s, hfu: 70.723 Tflops/s, 171.527 iter/s, 5.830e-03 s/iter, (0.068, 0.000, 0.117) sec
# # zigzag_ring_flash_attn_func, fwd
# mfu: 114.705 Tflops/s, hfu: 114.705 Tflops/s, 278.196 iter/s, 3.595e-03 s/iter, (0.041, 0.000, 0.072) sec
# # stripe_flash_attn_func, fwd
# mfu: 64.222 Tflops/s, hfu: 64.222 Tflops/s, 155.759 iter/s, 6.420e-03 s/iter, (0.069, 0.000, 0.128) sec
# # orchestrated_attn_func
# mfu: 120.809 Tflops/s, hfu: 120.809 Tflops/s, 146.500 iter/s, 6.826e-03 s/iter, (0.098, 0.000, 0.027) sec
# mfu: 129.937 Tflops/s, hfu: 129.937 Tflops/s, 157.569 iter/s, 6.346e-03 s/iter, (0.095, 0.000, 0.025) sec
# mfu: 101.96 Tflops/s, hfu: 101.96 Tflops/s, 123.643 iter/s, 8.088e-03 s/iter, (0.098, 0.000, 0.032) sec
# mfu: 148.332 Tflops/s, hfu: 148.332 Tflops/s, 179.877 iter/s, 5.559e-03 s/iter, (0.097, 0.000, 0.022) sec
# # orchestrated_attn_func fused
# mfu: 194.822 Tflops/s, hfu: 194.822 Tflops/s, 236.253 iter/s, 4.233e-03 s/iter, (0.090, 0.000, 0.017) sec
# mfu: 161.504 Tflops/s, hfu: 161.504 Tflops/s, 195.850 iter/s, 5.106e-03 s/iter, (0.091, 0.000, 0.020) sec
# mfu: 137.849 Tflops/s, hfu: 137.849 Tflops/s, 167.164 iter/s, 5.982e-03 s/iter, (0.094, 0.000, 0.024) sec
# mfu: 121.863 Tflops/s, hfu: 121.863 Tflops/s, 147.778 iter/s, 6.767e-03 s/iter, (0.099, 0.000, 0.027) sec

    baselines = ['ring_flash_attn_func', 'zigzag_ring_flash_attn_func', 'stripe_flash_attn_func']
    skip_lines = 0
    pat0 = re.compile(r'^fob=(\d).*$')
    pat1 = re.compile(r'^SP=\((\d+),(\d+)\),S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat2 = re.compile(r'^.*iter/s, (-?(\d+(?:\.\d+)?(?:e[+-]\d+)?)) s/iter,.*$')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if skip_lines > 0:
                skip_lines -= 1
                continue
            for baseline in baselines:  # [NOTE]: Results of baseline models are error !!!
                if baseline in line:
                   skip_lines = 1
                #    print(f'baseline: {baseline}')
                   continue 
            res = pat2.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                assert cur_map_key is not None and fob is not None
                if cur_map_key not in profile_map:
                    profile_map[cur_map_key] = np.empty((2,), dtype=np.float32)
                    profile_map[cur_map_key].fill(np.inf)
                # print(f'cur_map_key: {cur_map_key}, fob: {fob}, {float(res.group(1))}')
                profile_map[cur_map_key][fob] = min(profile_map[cur_map_key][fob], float(res.group(1)))
                continue
            res = pat1.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}, {res.group(2)}, {res.group(3)}, {res.group(4)}, {res.group(5)}, {res.group(6)}, {res.group(7)}, {res.group(8)}, {res.group(9)}')
                SP = (int(res.group(1)), int(res.group(2)))
                tot_sp = SP[0] * SP[1]
                S = (int(res.group(3)) // tot_sp, int(res.group(4)) // tot_sp)
                # S = (int(res.group(3)), int(res.group(4)))  # just for analysis !!!
                Nh = (int(res.group(5)), int(res.group(6)))
                bs = int(res.group(7))
                D = int(res.group(8))
                causal = res.group(9) == 'True'
                cur_map_key = (SP, S, Nh, bs, D, causal)
                # print(f'pat1, fob={fob}, cur_map_key: {cur_map_key}')
                continue
            res = pat0.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                fob = int(res.group(1))
                # print(f'fob: {fob}')
                continue
            
    # print(f'{profile_map}')
    return profile_map

def select_best_schedule_in_node_profile_data(file_name: str, local_size: int):
    # map_key: ((Sq, Skv), (Nhq, Nhg), bs, D, causal) -> Time[fwd/bwd]  # S per GPU !!!
    profile_map = {} # map_key -> [(Y,X,fused,Time), (Y,X,fused,Time)] # F/B
    # YXs = [(1, 8), (2, 4), (4, 2), (8, 1)]
    YXs = []
    for y in range(1, local_size + 1):
        if local_size % y == 0:
            YXs.append((y, local_size // y))
    # print(f'YXs: {YXs}')
    
    baselines = ['ring_flash_attn_func', 'zigzag_ring_flash_attn_func', 'stripe_flash_attn_func']
    skip_lines = 0
    fused = 0
    YX_id = None
    pat0 = re.compile(r'^fob=(\d).*$')
    pat1 = re.compile(r'^.*SP=\((\d+),(\d+)\),.*S=\((\d+),(\d+)\),Nh=\((\d+),(\d+)\),bs=(\d+),D=(\d+),causal=(True|False).*$')
    pat2 = re.compile(r'^.*iter/s, (-?(\d+(?:\.\d+)?(?:e[+-]\d+)?)) s/iter,.*$')
    pat3 = re.compile(r'^.*fused.*$')
    # non causal
    with open(file_name, 'r') as f:
        for line in f.readlines():
            if skip_lines > 0:
                skip_lines -= 1
                continue
            for baseline in baselines:  # [NOTE]: Results of baseline models are error !!!
                if baseline in line:
                   skip_lines = 1
                #    print(f'baseline: {baseline}')
                   continue 
            res = pat2.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                assert cur_map_key is not None and fob is not None
                if cur_map_key not in profile_map:
                    profile_map[cur_map_key] = [None] * 2
                # print(f'cur_map_key: {cur_map_key}, fob: {fob}, {float(res.group(1))}')
                cur_time = float(res.group(1))
                if profile_map[cur_map_key][fob] is None or cur_time < profile_map[cur_map_key][fob][- 1]:
                    profile_map[cur_map_key][fob] = YXs[YX_id] + (fused, cur_time)
                YX_id += 1
                continue
            res = pat3.match(line)
            if res:
                assert YX_id == None or YX_id == len(YXs)
                YX_id = 0
                fused = 1   # Convert to fused
                continue
            res = pat1.match(line)
            if res: # new map_key
                # print(f'res: {res.group(0)}, {res.group(1)}, {res.group(2)}, {res.group(3)}, {res.group(4)}, {res.group(5)}, {res.group(6)}, {res.group(7)}, {res.group(8)}, {res.group(9)}')
                SP = (int(res.group(1)), int(res.group(2)))
                tot_sp = SP[0] * SP[1]
                S = (int(res.group(3)) // tot_sp, int(res.group(4)) // tot_sp)
                # S = (int(res.group(3)), int(res.group(4)))  # just for analysis !!!
                Nh = (int(res.group(5)), int(res.group(6)))
                bs = int(res.group(7))
                D = int(res.group(8))
                causal = res.group(9) == 'True'
                cur_map_key = (SP, S, Nh, bs, D, causal)
                assert YX_id == None or YX_id == len(YXs)
                YX_id = 0
                fused = 0   # default: nonfused is ahead of fused
                # print(f'pat1, fob={fob}, cur_map_key: {cur_map_key}')
                continue
            res = pat0.match(line)
            if res:
                # print(f'res: {res.group(0)}, {res.group(1)}')
                fob = int(res.group(1))
                # print(f'fob: {fob}')
                continue
            
    # causal
    # print(f'profile_map: {profile_map}')
    return profile_map

