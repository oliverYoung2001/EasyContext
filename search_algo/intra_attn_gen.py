import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Machine_Config, Dist_Attn_Config, \
    get_profile_data, get_init_schedule_list, create_schedule
from search_algo.dependent_graph import Dependent_Graph
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle
import numpy as np
from functools import partial


def get_configs():
    SP0, SP1 = 1, 2
    Sq = Skv = 2 * 1024   # 2k
    SP0, SP1 = 1, 4
    Sq = Skv = 4 * 1024   # 4k
    SP0, SP1 = 1, 8
    Sq = Skv = 16 * 1024   # 16k
    Sq = Skv = 8 * 1024   # 8k
    
    Nhq = Ng = 32
    bs = 1
    D = 128
    causal = False
    # causal = True
    return Dist_Attn_Config((SP0, SP1), (Sq, Skv), (Nhq, Ng), bs, D, causal)

def get_block_schedule_table(split_degrees: list, S_map: np.ndarray, causal: bool, X):
    assert len(split_degrees) == 4
    assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
    assert split_degrees[0] == split_degrees[1] and split_degrees[0] % X == 0
    Y = split_degrees[0] // X
    block_schedule_table = np.zeros((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
    block_schedule_table -= 1  # -1 means not used
    for i in range(split_degrees[2]):   # split_bs
        for j in range(split_degrees[3]):   # split_Nh
            for k in range(split_degrees[0]):   # split_Sq
                for l in range(split_degrees[1]):   # split_Skv
                    if causal and k < l:
                        continue
                    block_schedule_table[i, j, k, l] = S_map[i, k // X * X + l % X]
    return block_schedule_table

def create_plan(da_config: Dist_Attn_Config, m_config: Machine_Config, X, fob, first_dim) -> Execution_Plan:
    tot_sp = da_config.tot_sp
    # Create Schedule:
    split_degrees = [tot_sp, tot_sp, 1, 1]
    S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
    S_map[:] = np.arange(tot_sp)
    get_schedule_table_func = partial(get_block_schedule_table, X=X)
    schedule =  create_schedule(da_config, m_config, split_degrees, S_map, get_schedule_table_func)
    # Create Dependent Graph:
    d_graph = Dependent_Graph(schedule, fob, 1) # Intra-machine
    # Create Execution Plan:
    plan = Execution_Plan(d_graph, 0, False)
    # Generate Manual Plan:
    plan.generate_manual_plan(tot_sp, X, first_dim=first_dim)
    return plan

def write_plan(execute_plan: Execution_Plan, prefix: str):
    # dump plan
    plan_name = execute_plan.get_plan_name()
    plan_file = f'{prefix}/{plan_name}.pkl'
    with open(plan_file, 'wb') as f:
        pickle.dump(execute_plan, f)
    # load plan
    with open(plan_file, 'rb') as f:
        execute_plan_loaded = pickle.load(f)
    execute_plan_loaded.print_lp_result()

def main():
    da_config = get_configs()
    m_config = get_profile_data()
    tot_sp = da_config.SP[0] * da_config.SP[1]
    par_dir = f'{os.path.dirname(__file__)}/execution_plans/intra_SP{da_config.SP[1]}'
    os.makedirs(par_dir, exist_ok=True)
    for X in range(1, tot_sp + 1):
        if tot_sp % X != 0:
            continue
        if X == 1 or X == tot_sp:
            plan = create_plan(da_config, m_config, X, fob=0, first_dim=0)
            write_plan(plan, prefix=par_dir)
        else:
            for first_dim in range(1):  # [TODO]: Support first_dim == 1
                plan = create_plan(da_config, m_config, X, fob=0, first_dim=first_dim)
                write_plan(plan, prefix=par_dir)
    
if __name__ == '__main__':
    main()