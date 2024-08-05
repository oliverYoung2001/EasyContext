import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from search_algo.search_engine import Search_Engine, Dist_Attn_Schedule, Dist_Attn_Config, Machine_Config, \
                                        TASK_STATUS, get_profile_data, get_init_schedule_list
from search_algo.utils import convert_profile_data_to_map, FinitePriorityQueue, convert_profile_data_to_comm_map
from search_algo.global_vars import *
import numpy as np
from functools import reduce
import copy
import math

def get_configs():
    SP0, SP1 = 1, 2
    Sq = Skv = 2 * 1024   # 2k
    SP0, SP1 = 1, 3
    Sq = Skv = 3 * 1024   # 3k
    # SP0, SP1 = 1, 4
    # Sq = Skv = 4 * 1024   # 4k
    # SP0, SP1 = 1, 5
    # Sq = Skv = 5 * 1024   # 5k
    # SP0, SP1 = 1, 8
    # Sq = Skv = 16 * 1024   # 16k
    # Sq = Skv = 8 * 1024   # 8k
    
    Nhq = Ng = 32
    bs = 1
    D = 128
    causal = False
    causal = True
    return Dist_Attn_Config((SP0, SP1), (Sq, Skv), (Nhq, Ng), bs, D, causal)

class Brute_Force_Search_Engine():
    def __init__(self, da_config: Dist_Attn_Config, m_config: Machine_Config, init_schedule_list: list):
        self.da_config = da_config
        self.m_config = m_config
        self.tot_sp = reduce(lambda x,y:x*y, self.da_config.SP)
        # return
        self.init_schedule_list = init_schedule_list
        for schedule in self.init_schedule_list:
            schedule.get_ub_cc_units_constrain()
        # self.ub = np.empty((2, 2), dtype=np.float32)    # [fwd/bwd][Comp/Comm]
        # self.ub = self.ub.fill(np.inf)
        
        # Now only support split_degrees = [tot_sp, tot_sp, 1, 1], S_map = np.arange(tot_sp)
        self.split_degrees = [self.tot_sp, self.tot_sp, 1, 1]
        
        # # comp/comm ub
        # self.ub_cc_units = np.max(np.array([schedule.ub_cc_units_constrain for schedule in self.init_schedule_list]), axis=0)   # [fwd/bwd][Comp/Comm(units)]
        
        # # # extra comp ub (not a real ub in some case !!!)
        # # alignment between different split_degrees
        # self.ub_cc_units[:, 0] = np.minimum(
        #     self.ub_cc_units[:, 0], 
        #     np.max(np.array([schedule.ub_comp_units * np.prod(self.split_degrees) / np.prod(schedule.split_degrees)
        #                      for schedule in self.init_schedule_list]), axis=0)
        # )
        # print(f'ub_cc_units: {self.ub_cc_units}')
        # # extra comm ub (real ub)
        # self.ub[:, 1] = np.max(np.array([schedule.ub_comm_units for schedule in self.init_schedule_list]), axis=0)
        
        # self.MAX_QUEUE_SIZE = 100
        self.MAX_QUEUE_SIZE = 100000000
        self.schedule_queues = [None, None]  # [fwd/bwd]  
        
        # extra comp ub (not a real ub in some case !!!)
        self.ub_comp = int(math.ceil((self.tot_sp - 1) / 2))
        print(f'ub_comp: {self.ub_comp}')
    
    def reset_before_search(self):
        # Constrain: x = unpack_func(pack_func(x))
        def pack_func(schedule):
            SCHEDULE_UNIQUE_ID = get_global_var('SCHEDULE_UNIQUE_ID')
            SCHEDULE_UNIQUE_ID += 1
            set_global_var('SCHEDULE_UNIQUE_ID', SCHEDULE_UNIQUE_ID)
            fob = self.fob
            print(f'SCHEDULE_UNIQUE_ID: {SCHEDULE_UNIQUE_ID}')
            print(f'schedule:\n{schedule.schedule_table}', flush=True)
            # print(f'fob: {fob}, get_e2e_time(): {schedule.get_e2e_time()[fob]:.3e}, get_absolute_cc_time:{schedule.get_absolute_cc_time()[fob]}')
            # print(f'fob: {fob}, get_relative_cc_time:{schedule.get_relative_cc_time()[fob]}')
            print(f'fob: {fob}, get_tot_comm_units: {schedule.get_tot_comm_units()[fob][0]}')
            schedule.get_relative_cc_time()
            balanced_r_cc_time = schedule.balanced_r_cc_time[fob]
            r_cc_time = schedule.r_cc_time[fob]
            print(f'get_relative_cc_time:\n{r_cc_time}')
            print(f'get_balanced_relative_cc_time: {np.max(balanced_r_cc_time[1:])}, {np.max(balanced_r_cc_time[0])}\n{balanced_r_cc_time}')
            return (- schedule.get_tot_comm_units()[self.fob][0], SCHEDULE_UNIQUE_ID, schedule)
        def unpack_func(q_item):
            return q_item[2]
        
        # Initialize self.schedule_queues[self.fob]:
        schedule_queue = FinitePriorityQueue(self.MAX_QUEUE_SIZE, pack_func, unpack_func)
        for schedule in self.init_schedule_list:    # add all initial schedules into priority queue
            schedule_queue.push(schedule)
        self.schedule_queues[self.fob] = schedule_queue
        
        S_map = np.empty((self.split_degrees[2], min(self.split_degrees[0], self.split_degrees[1])), dtype=np.int32)
        S_map[:] = np.arange(self.tot_sp)

        self.cur_schedule = Dist_Attn_Schedule(self.da_config, self.m_config, self.split_degrees, S_map)
        self.cur_cc_units = np.zeros((2, 3, self.tot_sp), dtype=np.float32) # [fwd/bwd][Comp/Comm_in/Comm_out][SP]
        
        # For initialized diagonal elements' comp workload
        for g in range(self.tot_sp):
            self.cur_cc_units[:, 0, g] = np.sum(self.cur_schedule.schedule_table == g)
        assert np.prod(self.cur_cc_units[:, 0, :] == 1) == 1
    
    def get_next_pos(self, cur_pos: tuple) -> tuple:
        """_summary_

        Args:
            cur_pos (list): current position, [split_degree[2], split_degree[3], split_degree[0], split_degree[1]]

        Returns:
            _type_: [split_degree[2], split_degree[3], split_degree[0], split_degree[1]]
        """
        if cur_pos[3] + 1 < self.split_degrees[1]:
            return (cur_pos[0], cur_pos[1], cur_pos[2], cur_pos[3] + 1)
        if cur_pos[2] + 1 < self.split_degrees[0]:
            return (cur_pos[0], cur_pos[1], cur_pos[2] + 1, 0)
        if cur_pos[1] + 1 < self.split_degrees[3]:
            return (cur_pos[0], cur_pos[1] + 1, 0, 0)
        if cur_pos[0] + 1 < self.split_degrees[2]:
            return (cur_pos[0] + 1, 0, 0, 0)
        return None
    
    def is_end(self, cur_pos: tuple):
        return cur_pos is None

    def get_next_unsettled_pos(self, cur_pos: tuple):
        next_pos = self.get_next_pos(cur_pos)
        while not self.is_end(next_pos) and self.cur_schedule.schedule_table[next_pos] != TASK_STATUS.UNSETTLED.value:
            next_pos = self.get_next_pos(next_pos)
        return next_pos
    
    def apply_pruning_passed(self, cur_pos: tuple, g: int):
        # # pruning strategy 1: comp
        # if self.cur_cc_units[self.fob, 0, g] + 1 >= self.ub_cc_units[self.fob, 0]:
        #     return False
        # pruning strategy 2: comp
        if self.cur_cc_units[self.fob, 0, g] > self.ub_comp:
            return False
        return True
          
    def update_cur_status(self, cur_pos: tuple, g: int):
        self.cur_schedule.schedule_table[cur_pos] = g
        self.cur_cc_units[self.fob, 0, g] += 1
        pass
    
    def restore_cur_status(self, cur_pos: tuple, g: int):
        self.cur_schedule.schedule_table[cur_pos] = TASK_STATUS.UNSETTLED.value
        self.cur_cc_units[self.fob, 0, g] -= 1
        pass
    
    def brute_force_search(self, cur_pos: tuple):
        """_summary_

        Args:
            cur_pos (list): current position, [split_degree[2], split_degree[3], split_degree[0], split_degree[1]]
        """
        # print(f'cur_pos: {cur_pos}')
        if self.is_end(cur_pos):
            new_schedule = copy.deepcopy(self.cur_schedule)
            self.schedule_queues[self.fob].push(new_schedule)
            # raise Exception()
            # exit(0)
            return
        # if cur_pos == (0, 0, 7, 0):
        #     print(f'cur_pos: {cur_pos}', flush=True)
        assert self.cur_schedule.schedule_table[cur_pos] == TASK_STATUS.UNSETTLED.value
        next_pos = self.get_next_unsettled_pos(cur_pos)
        
        for g in range(self.tot_sp):    # fill in gpu_id
            if not self.apply_pruning_passed(cur_pos, g):
                continue
            self.update_cur_status(cur_pos, g)
            self.brute_force_search(next_pos)
            self.restore_cur_status(cur_pos, g)
        
    def search_optimal_schedules(self):
    #     # [NOTE]: both support fwd & bwd
    #     assert self.split_degrees[0] == self.split_degrees[1]   # Sq_split == Skv_split
    #     assert self.split_degrees[2] == self.split_degrees[3] == 1  # [NOTE]: now only support bs_split == Nh_split == 1
        # [TODO]: support bs_split == 2
        # search for fwd
        self.fob = 0    # fwd
        self.reset_before_search()
        # try:
        if True:
            self.brute_force_search(self.get_next_unsettled_pos((0, 0, 0, 0)))
        # except Exception as e:
        #     pass
        
        # search for bwd
        self.fob = 1    # bwd
        self.reset_before_search()
        # self.brute_force_search(self.get_next_unsettled_pos((0, 0, 0, 0)))
   
if __name__ == '__main__':
    da_config = get_configs()
    m_config = get_profile_data()
    tot_sp = reduce(lambda x,y:x*y, da_config.SP)
    split_degrees = (tot_sp, tot_sp, 1, 1)
    # initialize schedule
    S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
    S_map[:] = np.arange(tot_sp)

    cur_schedule = Dist_Attn_Schedule(da_config, m_config, split_degrees, S_map)
    search_engine = Brute_Force_Search_Engine(da_config, m_config, [])
    search_engine.search_optimal_schedules()
    
    
    