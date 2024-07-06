import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Search_Engine, get_configs, get_profile_data, get_init_schedule_list
from search_algo.dependent_graph import Dependent_Graph
from search_algo.execute_engine import Execute_Engine
from search_algo.global_vars import *

def main():
    da_config = get_configs()
    m_config = get_profile_data()
    init_schedule_list = get_init_schedule_list(da_config, m_config)
    search_engine = Search_Engine(da_config, m_config, init_schedule_list)
    search_engine.search_optimal_schedules()
    SCHEDULE_UNIQUE_ID = get_global_var('SCHEDULE_UNIQUE_ID')
    # print(f'tot schedules: {SCHEDULE_UNIQUE_ID}')
    # fwd
    fob = 0
    print(f'[INFO]: fwd schedules')
    for _ in range(len(search_engine.schedule_queues[fob])):
        schedule = search_engine.schedule_queues[fob].pop()
        if _ == 1:  # qo schedule
            example_schedule = schedule
        # if _ == 2:  # kv schedule
        #     example_schedule = schedule
        print(f'schedule:\n{schedule.schedule_table}', flush=True)
        print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()}, get_absolute_cc_time:\n{schedule.get_absolute_cc_time()}')
    d_graph = Dependent_Graph(example_schedule, fob, 1) # Intra-machine
    execute_engine = Execute_Engine(d_graph, fob)
    
    
if __name__ == '__main__':
    main()