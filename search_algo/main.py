import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Evaluation_Configs, \
                                      get_profile_data, get_init_schedule_list
from search_algo.dependent_graph import Dependent_Graph
from search_algo.execute_plan import Execution_Plan
from search_algo.global_vars import *
import pickle

def get_exp_config():
    plan_type = 'automatic'
    # plan_type = 'maunal'
    # plan_type = 'ablation1'
    MAX_QUEUE_SIZE = 100
    fob = 0
    return Evaluation_Configs(plan_type, MAX_QUEUE_SIZE, fob)

def get_configs():
    SP0, SP1 = 1, 4
    Sq = Skv = 4 * 1024   # 4k
    # SP0, SP1 = 1, 8
    # Sq = Skv = 16 * 1024   # 16k
    # Sq = Skv = 8 * 1024   # 8k
    
    Nhq = Ng = 32
    bs = 1
    D = 128
    causal = False
    causal = True
    return Dist_Attn_Config((SP0, SP1), (Sq, Skv), (Nhq, Ng), bs, D, causal)

def show_schedule(schedule: Dist_Attn_Schedule, fob, name):
    print(f'{name}, schedule:\n{schedule.schedule_table}', flush=True)
    print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')
    if schedule.split_degrees[0] != schedule.da_config.SP[1] or schedule.split_degrees[1] != schedule.da_config.SP[1]:
        return
    
def main():
    exp_config = get_exp_config()
    da_config = get_configs()
    print(f'plan_name: {da_config.get_plan_name(fob=0)}', flush=True)
    m_config = get_profile_data()
    init_schedule_list = get_init_schedule_list(da_config, m_config)
    search_engine = Search_Engine(exp_config, da_config, m_config, init_schedule_list)
    search_engine.search_optimal_schedules()
    SCHEDULE_UNIQUE_ID = get_global_var('SCHEDULE_UNIQUE_ID')
    # print(f'tot schedules: {SCHEDULE_UNIQUE_ID}')
    # fwd
    fob = exp_config.fob
    print(f'[INFO]: fwd schedules')
    print(f'schedule_squeue_num[fwd]: {len(search_engine.schedule_queues[fob])}')
    par_dir = f'{os.path.dirname(__file__)}/execution_plans/SP{da_config.SP}_S{da_config.S}'
    if exp_config.plan_type == 'ablation1':
        par_dir = f'{par_dir}_{exp_config.plan_type}'
    os.makedirs(par_dir, exist_ok=True)
    # return
    # for schedule in init_schedule_list:
    #     show_schedule(schedule, fob, 'init')

    QUEUE_LEN = len(search_engine.schedule_queues[fob])
    for _ in range(len(search_engine.schedule_queues[fob])):
        schedule = search_engine.schedule_queues[fob].pop()
        show_schedule(schedule, fob, f'alg{_}')
        d_graph = Dependent_Graph(schedule, fob, 1) # Intra-machine
        execute_plan = Execution_Plan(d_graph, fob, exp_config.plan_type)
        execute_plan.print_lp_result()
        # if _ == 1:  # qo schedule
        #     example_schedule = schedule
        # if _ == 2:  # kv schedule
        #     example_schedule = schedule
        # if _ == 3:  # real example
        #     example_schedule = schedule
        # dump plan
        plan_name = execute_plan.get_plan_name()
        plan_file = f'{par_dir}/{plan_name}_alg{_}.pkl'
        print(f'plan_file: {plan_file}')
        with open(plan_file, 'wb') as f:
            pickle.dump(execute_plan, f)
    return
    d_graph = Dependent_Graph(example_schedule, fob, 1) # Intra-machine
    execute_plan = Execution_Plan(d_graph, fob)
    # dump plan
    plan_name = execute_plan.get_plan_name()
    plan_file = f'{os.path.dirname(__file__)}/execution_plans/{plan_name}.pkl'
    with open(plan_file, 'wb') as f:
        pickle.dump(execute_plan, f)
    # load plan
    with open(plan_file, 'rb') as f:
        execute_plan_loaded = pickle.load(f)
    execute_plan_loaded.print_lp_result()
    
if __name__ == '__main__':
    main()