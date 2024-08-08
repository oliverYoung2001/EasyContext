import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, \
                                      get_profile_data, get_init_schedule_list, get_cc_optimal_schedule
import math
import numpy as np

def main():
    SP = (8, 8)
    m_config = get_profile_data(SP)
    # fb_ratio = []
    # for k, v in m_config.comp_profile_maps[0].profile_map.items():
    #     # print(f'{k}: {v[1]/ v[0]}')
    #     fb_ratio.append((v[1]/ v[0], k))
    # fb_ratio.sort(reverse=True)
    # # print(fb_ratio)
    # for v in fb_ratio:
    #     print(f'{v[1]}: {v[0]}')    # 1.3~15.3
    
    S_BOUND = [256, 64 * 1024 // 4]  # lower-bound and upper-bound
    S_base = [1 << logS for logS in range(int(math.log2(S_BOUND[0])), int(math.log2(S_BOUND[1])) + 1)]
    batch_degrees = [(1, 2), (2, 1), (1, 3), (3, 1), (1, 4), (2, 2), (4, 1)]
    fobs = [
        0,
        1,
    ]
    Nhs = [
        1, 
        32,
    ]
    causal = False
    SP = (1, 8)
    D = 128
    # for fob in fobs:
    profile_map = m_config.comp_profile_maps[0].profile_map # Inter_Comp_Profile_Map
    for Nh in Nhs:
        # print(f'Nh: {Nh}')
        for S in S_base:
            Sq = Skv = S
            map_key = (SP, (Sq, Skv), (Nh, Nh), 1, D, causal)
            T_base = profile_map[map_key]
            Ts = []
            T_multiples = []
            print(f'Nh: {Nh}, S: {S}, T_base: {T_base}', flush=True)
            for bd in batch_degrees:
                map_key = (SP, (Sq * bd[0], Skv * bd[1]), (Nh, Nh), 1, D, causal)
                T_ = profile_map[map_key]
                Ts.append(T_)
                T_multiples.append(T_ / T_base)
                num_blocks = math.prod(bd)
                speedup = num_blocks / (T_ / T_base)
                print(f'bd: {bd}, T_: {T_}, T_/T_base: {T_ / T_base}, speedup: {speedup}, ratio: {1 / speedup}', flush=True)
            # print(f'S: {S}, T_base: {T_base}, Ts: {Ts}, T_multiples: {T_multiples}', flush=True)
        
        



if __name__ == '__main__':
    main()