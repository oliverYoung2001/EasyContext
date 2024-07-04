from queue import PriorityQueue
from heapq import heappush, heappop, heappushpop
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.global_vars import *
import math
import regex as re

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
    
def convert_profile_data_to_map(profile_list):
    profile_map = {}
    for i in range(len(profile_list)):
        map_key = tuple(profile_list[i][0])
        assert map_key not in profile_map.keys()
        profile_map[map_key] = profile_list[i][1]    # [fwd/bwd], (us)
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
    return size * pow(BYTE_MULTPLE_DOWN, size_name.index(unit))
    
def convert_profile_data_to_comm_map(file_name: str, GPU_NUM: int):
    profile_map = {}    # Bytes -> GB/s
    # SIZE 8192, REAL_BD 671.646 MB/s, time 0.0098 s, comm_vol 6.554 MB
    pat = re.compile(r'^SIZE (\d+), REAL_BD (\d*(\.\d*)?) ([A-Z]*)/s.*$')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            res = pat.match(line)
            if res:
                # print(f'res: {res.group(1), res.group(2), res.group(4)}')
                profile_map[int(res.group(1))] = convert_throughput_to_B(float(res.group(2)), res.group(4))  / pow(BYTE_MULTPLE_DOWN, 3) / GPU_NUM
    # print(f'profile_map: {profile_map}')
    return profile_map
