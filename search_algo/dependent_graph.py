import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, FlashAttn_Profile_Map, Machine_Config
import numpy as np
    
class Cuda_Kernel():
    def __init__(self, key: tuple, m_config: Machine_Config, type: str):
        self.key = key
        self.m_config = m_config
        self.type = type
        self.precursors = set()
        self.successors = set()
    
    def add_precursor(self, precursor):
        self.precursors.add(precursor)
    
    def add_successor(self, successor):
        self.successors.add(successor)
    
    def is_empty(self, fob):
        return self.time[fob] <= 0
    
    def add_edge(self, v, fob):
        # check whether self or v is empty
        if self.is_empty(fob) or v.is_empty(fob):
            return
        self.add_successor(v)
        v.add_precursor(self)
                
class Comp_Kernel(Cuda_Kernel):
    def __init__(self, key: tuple, m_config: Machine_Config, comp_map_key: tuple, hierarchy: int):
        # dict keys: (b_id, h_id, r_id, c_id, gpuid)
        super().__init__(key, m_config, 'comp')
        # flashattn profile map_key:
        self.comp_map_key = comp_map_key
        # kernel time
        self.time = m_config.comp_profile_maps[hierarchy].get_comp_time_from_map_key(comp_map_key)    # [fwd/bwd]
    
    
class Comm_Kernel(Cuda_Kernel):
    def __init__(self, key: tuple, m_config: Machine_Config, comm_raw_map_key: tuple, units: np.ndarray, hierarchy: int):
        # dict keys: (b_id, h_id, r/c_id, send, recv, i/o, r/c)
        super().__init__(key, m_config, 'comm')
        # Bytes of data to send/recv
        self.comp_raw_map_key = comm_raw_map_key
        assert units.shape == (2,)
        self.units = units  # [fwd/bwd]
        self.hierarchy = hierarchy
        # kernel time
        self.time = np.array([
            m_config.comm_profile_maps[hierarchy].get_comm_time_from_map_key((comm_raw_map_key[0] * unit,))
            for unit in units
        ])    # [fwd/bwd]
        


class Dependent_Graph():
    def __init__(self, schedule: Dist_Attn_Schedule, fob: bool):
        # [NOTE]: only support star tree of broadcase/reduce here !!!
        # build dependent graph from schedule_table
        
        self.schedule = schedule
        self.da_config = schedule.da_config
        self.m_config = schedule.m_config
        self.split_degrees = schedule.split_degrees
        self.fob = fob  # fwd or bwd
        self.tot_sp = schedule.tot_sp
        self.hierarchy = hierarchy = schedule.da_config.hierarchy
        # self.root_kernel = Cuda_Kernel()
        # comp: (b_id, h_id, r_id, c_id, gpuid) -> Cuda_Kernel
        # comm: (b_id, h_id, r/c_id, send, recv, i/o, r/c) -> Cuda_Kernel
        self.kernel_dict = {}
        
        # step1: Build Comp Kernel
        comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        if schedule.schedule_table[i, j, k, l] >= 0:    # Valid comp kernel
                            comp_key = (i, j, k, l, schedule.schedule_table[i, j, k, l])
                            assert comp_key not in self.kernel_dict.keys()
                            self.kernel_dict[comp_key] = Comp_Kernel(comp_key, schedule.m_config, comp_map_key, hierarchy)
        # step2: Build Comm Kernel
        assert schedule.split_degrees[0] == schedule.split_degrees[1] # [NOTE]: now only support Sq_split == Skv_split !!!
        comm_raw_map_key = schedule.m_config.comm_profile_maps[hierarchy].get_comm_map_key(schedule.da_config, [1, 1], schedule.split_degrees)
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                # row
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    cur_g_id = schedule.S_map[i, k]
                    dst_set = set()
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0 and dst_g_id != cur_g_id and dst_g_id not in dst_set:
                            dst_set.add(dst_g_id)
                            # input row broadcast
                            comm_key = (i, j, k, cur_g_id, dst_g_id, 'i', 'r')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_row, hierarchy)
                            # output row reduce
                            comm_key = (i, j, k, dst_g_id, cur_g_id, 'o', 'r')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_row, hierarchy)
                # col
                for l in range(schedule.split_degrees[1]):  # split_Skv
                    cur_g_id = schedule.S_map[j, l]
                    dst_set = set()
                    for k in range(schedule.split_degrees[0]):  # split_Sq
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0 and dst_g_id != cur_g_id and dst_g_id not in dst_set:
                            dst_set.add(dst_g_id)
                            # input col broadcast
                            comm_key = (i, j, l, cur_g_id, dst_g_id, 'i', 'c')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_inp_col, hierarchy)
                            # output col reduce
                            comm_key = (i, j, l, dst_g_id, cur_g_id, 'o', 'c')
                            assert comm_key not in self.kernel_dict.keys()
                            self.kernel_dict[comm_key] = Comm_Kernel(comm_key, schedule.m_config, comm_raw_map_key, schedule.u_out_col, hierarchy)
        # [NOTE]: every nonempty kernel in self.kernel_dict should be launched by Execute_Engine
        
        # step3: Build dependences between kernels, differentiate fwd and bwd !!!
        # comp kernel centric
        for i in range(schedule.split_degrees[2]):   # split_bs
            for j in range(schedule.split_degrees[3]):   # split_Nh
                for k in range(schedule.split_degrees[0]):   # split_Sq
                    for l in range(schedule.split_degrees[1]):   # split_Skv
                        dst_g_id = schedule.schedule_table[i, j, k, l]
                        if dst_g_id >= 0:    # Valid comp kernel
                            comp_key = (i, j, k, l, dst_g_id)
                            comp_kernel = self.kernel_dict[comp_key]
                            cur_g_id = schedule.S_map[i, k]
                            if dst_g_id != cur_g_id:
                                # input row broadcast
                                comm_key = (i, j, k, cur_g_id, dst_g_id, 'i', 'r')
                                self.kernel_dict[comm_key].add_edge(comp_kernel, fob)
                                # output row reduce
                                comm_key = (i, j, k, dst_g_id, cur_g_id, 'o', 'r')
                                comp_kernel.add_edge(self.kernel_dict[comm_key], fob)
                            
                            cur_g_id = schedule.S_map[i, l]
                            if dst_g_id != cur_g_id:
                                # input col broadcast
                                comm_key = (i, j, l, cur_g_id, dst_g_id, 'i', 'c')
                                self.kernel_dict[comm_key].add_edge(comp_kernel, fob)
                                # output col reduce
                                comm_key = (i, j, l, dst_g_id, cur_g_id, 'o', 'c')
                                comp_kernel.add_edge(self.kernel_dict[comm_key], fob)
        
        
                            
        
