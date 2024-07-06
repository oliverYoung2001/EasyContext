import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.global_vars import *
from search_algo.utils import convert_profile_data_to_map, FinitePriorityQueue, convert_profile_data_to_comm_map
from functools import reduce
import numpy as np
from enum import Enum
import copy
import json

class TASK_STATUS(Enum):
    EMPTY = - 1
    UNSETTLED = - 2

class Dist_Attn_Config():
    def __init__(self, SP, S, Nh, bs, D, causal):
        self.SP = SP
        self.S = S
        self.Nh = Nh
        self.bs = bs
        self.D = D
        self.causal = causal
    
    def get_plan_name(self, fob=True):
        return f'S={self.S}_SP={self.SP}_causal={self.causal}_fob={fob}_b={self.bs}_Nh={self.Nh}_D={self.D}'



class FlashAttn_Profile_Map():
    def __init__(self, profile_map):
        self.profile_map = profile_map
    
    def merge_comp_map_key(self, map_key0: tuple, map_key1: tuple, merge_dim: int) -> tuple:
        # map_key: (min(Sq, Skv), bs, Nh, D, Sq/Skv, causal)
        assert not (map_key0[5] or map_key1[5]), "Can't merge two map_key when one is causal"
        # merge_dim: consistent with split_degrees (0, 1, 2, 3) -> (Sq, Skv, bs, Nh)
        assert merge_dim in range(0, 4), "Merge dim is out of range [0, 3] which is consistent with [Sq, Skv, bs, Nh]"
        if merge_dim in [0, 1]:  # Sq, Skv
            for d in [1, 2, 3, 5]:
                assert map_key0[d] == map_key1[d], f"map_key0[{d}] != map_key1[{d}]"
            S0 = (map_key0[0] * map_key0[4], map_key0[0]) if map_key0[4] >= 1 else (map_key0[0], map_key0[0] / map_key0[4]) # Sq, Skv
            S1 = (map_key1[0] * map_key1[4], map_key1[0]) if map_key1[4] >= 1 else (map_key1[0], map_key1[0] / map_key1[4]) # Sq, Skv
            assert S0[merge_dim ^ 1] == S0[merge_dim ^ 1], f"S0[{merge_dim ^ 1}] != S1[{merge_dim ^ 1}]"
            S01 = (S0[0] + S1[0], S0[1]) if merge_dim == 0 else (S0[0], S0[1] + S1[1])
            
            assert S01[0] / S01[1] in [0.25, 0.5, 1, 2, 4], \
                f"Current profile data doesn't contain this Q/K Sequence ratio: {S01[0] / S01[1]}"
            return (min(S01[0], S01[1]), map_key0[1], map_key0[2], map_key0[3], S01[0] / S01[1], map_key0[5])
        else:
            for d in range(5):
                if d != merge_dim - 1:
                    assert map_key0[d] == map_key1[d], f"map_key0[{d}] != map_key1[{d}]"
            map_key_merged = list(map_key0)
            map_key_merged[merge_dim - 1] += map_key1[merge_dim - 1]
            return tuple(map_key_merged)
                
    def get_comp_map_key(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> tuple:
        assert len(batch_degrees) == 2  # Q, KV
        assert len(split_degrees) == 4  # Sq, Skv, bs, Nh
        # Example for key:
        # [
        #     4096,
        #     1,
        #     32,
        #     128,
        #     1,
        #     false
        # ],
        tot_sp = reduce(lambda x,y:x*y, da_config.SP)
        Sq_split = da_config.S[0] * batch_degrees[0] // split_degrees[0]
        Skv_split = da_config.S[1] * batch_degrees[1] // split_degrees[1]
        bs_split = da_config.bs // split_degrees[2]
        Nh_split = da_config.Nh[0] // split_degrees[3]
        assert Sq_split / Skv_split in [0.25, 0.5, 1, 2, 4], \
            f"Current profile data doesn't contain this Q/K Sequence ratio: {Sq_split / Skv_split}"
        assert da_config.Nh[0] == da_config.Nh[1], \
            f"Current profile data doesn't contain this GQA: (Nh={da_config.Nh[0]}, Ng={da_config.Nh[1]})"
        # [TODO]: differentiate causal and noncausal
        map_key = (min(Sq_split, Skv_split), bs_split, Nh_split, da_config.D, Sq_split / Skv_split, False)
        return map_key
    
    def get_comp_time_from_map_key(self, map_key: tuple) -> np.ndarray:
        assert map_key in self.profile_map.keys(), f"Key {map_key} not found in profile_map"
        return self.profile_map[map_key]    # [fwd/bwd], (s)
    
    def get_comp_time(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> np.ndarray:
        map_key = self.get_comp_map_key(da_config, batch_degrees, split_degrees)
        return self.get_comp_time_from_map_key(map_key)    

class Comm_Profile_Map():
    def __init__(self, profile_map):
        self.profile_map = profile_map
    
    def merge_comm_map_key(self, map_key0: tuple, map_key1: tuple) -> tuple:
        return (map_key0[0] + map_key1[0])
    
    def get_comm_map_key(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> tuple:
        # [NOTE]: Now only support inter-machine communication
        assert(len(batch_degrees) == 2) # Q, KV
        tot_sp = reduce(lambda x,y:x*y, da_config.SP)
        Sq = da_config.S[0] * batch_degrees[0] // tot_sp
        # Skv = da_config.S[1] * batch_degrees[1] // tot_sp
        return (int(Sq * (da_config.bs / split_degrees[2]) * (da_config.Nh[0] / split_degrees[3]) * da_config.D * 2),)    # B
    
    def get_comm_time_from_map_key(self, map_key: tuple) -> float:
        if map_key[0] <= 0:
            return 0
        assert map_key in self.profile_map.keys(), f"Key {map_key[0]} not found in comm_profile_map"
        return map_key[0] / pow(BYTE_MULTPLE_DOWN, 3) / self.profile_map[map_key]    # s
    
    def get_comm_time(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> float:
        comm_map_key = self.get_comm_map_key(da_config, batch_degrees, split_degrees)
        return self.get_comm_time_from_map_key(comm_map_key)

       
class Machine_Config():
    def __init__(self, BW, flashattn_profile_map: dict, inter_comm_profile_map: dict, intra_comm_profile_map: dict):
        self.BW = BW
        self.flashattn_profile_map = FlashAttn_Profile_Map(flashattn_profile_map)
        self.comm_profile_maps = [Comm_Profile_Map(inter_comm_profile_map), Comm_Profile_Map(intra_comm_profile_map)]    # inter/intra-machine
    
    def merge_comm_map_key(self, map_key0: tuple, map_key1: tuple) -> tuple:
        return (map_key0[0] + map_key1[0])
    
    def get_comm_map_key(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> tuple:
        # [NOTE]: Now only support inter-machine communication
        assert(len(batch_degrees) == 2) # Q, KV
        tot_sp = reduce(lambda x,y:x*y, da_config.SP)
        Sq = da_config.S[0] * batch_degrees[0] // tot_sp
        # Skv = da_config.S[1] * batch_degrees[1] // tot_sp
        return (Sq * (da_config.bs / split_degrees[2]) * (da_config.Nh[0] / split_degrees[3]) * da_config.D * 2,)    # B
    
    def get_comm_time_from_map_key(self, map_key: tuple) -> float:
        return map_key[0] / pow(BYTE_MULTPLE_DOWN, 3) / self.BW[1]    # Intra-Machine, GB/s
    
    def get_comm_time(self, da_config: Dist_Attn_Config, batch_degrees: list, split_degrees: list) -> float:
        comm_map_key = self.get_comm_map_key(da_config, batch_degrees, split_degrees)
        return self.get_comm_time_from_map_key(comm_map_key)


    
class Dist_Attn_Schedule():
    def __init__(self, da_config: Dist_Attn_Config, m_config: Machine_Config, 
                 split_degrees: list, S_map: np.ndarray, schedule_table: np.ndarray = None):
        self.da_config = da_config
        self.m_config = m_config
        assert len(split_degrees) == 4
        self.split_degrees = split_degrees   # Sq, Skv, bs, Nh
        assert max(split_degrees[0], split_degrees[1]) % min(split_degrees[0], split_degrees[1]) == 0
        assert split_degrees[0] == split_degrees[1] # [NOTE]: now only support Sq_split == Skv_split !!!
        assert S_map.shape == (split_degrees[2], min(split_degrees[0], split_degrees[1]))
        if schedule_table is None:  # initialize schedule_table
            schedule_table = np.empty((split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1]), dtype=np.int32)
            schedule_table.fill(TASK_STATUS.UNSETTLED.value)
            if da_config.causal:
                for k in range(split_degrees[0]):   # Sq
                    schedule_table[:, :, k, k + 1:] = TASK_STATUS.EMPTY.value
            for k in range(split_degrees[0]):
                schedule_table[:, :, k, k] = np.expand_dims(S_map[:, k], axis=1)    # we can set the diagonal elements directly
        
        for k in range(split_degrees[0]):   # we should assert diagonal elements first
            for k in range(split_degrees[0]):
                assert np.prod(schedule_table[:, :, k, k] == np.expand_dims(S_map[:, k], axis=1)) == 1
                
        assert schedule_table.shape == (split_degrees[2], split_degrees[3], split_degrees[0], split_degrees[1])
        self.tot_sp = reduce(lambda x,y:x*y, self.da_config.SP)
        self.schedule_table = schedule_table
        self.S_map = S_map
        
        
        Sq_split = self.da_config.S[0] // self.split_degrees[0]
        Skv_split = self.da_config.S[1] // self.split_degrees[1]
        kv_unit_ratio = Skv_split / Sq_split * self.da_config.Nh[1] / self.da_config.Nh[0]  # for crossattn and GQA
        # comm units count: fwd, bwd
        self.u_inp_row = np.array([1, 3])    # (q), (q, o, do)
        self.u_inp_col = np.array([2, 2]) * kv_unit_ratio    # (k, v), (k, v)
        self.u_out_row = np.array([1, 1])    # (o), (dq)
        self.u_out_col = np.array([0, 2]) * kv_unit_ratio    # (), (dk, dv)
    
    def update_comm_units_pool(self, comm_units_pool: dict, self_mask: np.ndarray, 
                               units: np.ndarray, split_dim: int, is_out: int, g: int):
        is_out = int(is_out)
        assert units.shape == (2,)
        split_degrees = self.split_degrees
        for i in range(split_degrees[2]):
            for j in range(split_degrees[3]):
                for k in range(split_degrees[split_dim]):
                    if self_mask[i, 0, k]:  # row or col of GPU_g
                        sp_set = np.unique(np.take(self.schedule_table[i, j], indices=k, axis=split_dim))
                        for invalid_task_status in TASK_STATUS:   # remove invalid task status
                            if invalid_task_status.value in sp_set:
                                sp_set = sp_set[sp_set != invalid_task_status.value]
                        if g not in sp_set: # add current g
                            sp_set = np.append(sp_set, g)
                            sp_set = np.sort(sp_set)
                        if sp_set.size <= 2:  # only root or 1 child
                            continue
                        pool_key = tuple(sp_set)
                        if pool_key not in comm_units_pool.keys():
                            comm_units_pool[pool_key] = np.zeros((2, 2), dtype=np.float32)
                        comm_units_pool[pool_key][:, is_out] += units
    
    def balance_comm_units_pool(self, comm_units_pool: dict, r_cc_time: np.ndarray):
        """_summary_

        Args:
            comm_units_pool (dict): # {sp_set -> comm_units[fwd/bwd][Comm_in/Comm_out]}
            r_cc_time (np.ndarray): [fwd/bwd][Comp/Comm_in/Comm_out][SP]

        Returns:
            _type_: [fwd/bwd][Comp/Comm_in/Comm_out][SP]
        """
        # broadcast or reduce: pouring water model
        # [NOTE]: not a optimal algorithm, just a heuristic algorithm
        tot_comm_units = r_cc_time[:, 1:].sum(axis=-1) # [fwd/bwd][Comm_in/Comm_out]
        print(f'r_cc_time:\n{r_cc_time[:, 1:, :]}')
        for sp_set, comm_units in comm_units_pool.items():
            tot_comm_units += comm_units * (len(sp_set) - 2)
        print(f'tot_comm_units:\n{tot_comm_units}')
        # input and output comm units need to be the same
        assert (np.absolute(tot_comm_units[:, 0] - tot_comm_units[:, 1]) / tot_comm_units[:, 0]).sum() < 2e-6
        
        self.ave_comm_units = np.max(tot_comm_units, axis=-1) / self.tot_sp    # [fwd/bwd]
        ub_comm_units = np.maximum(self.ave_comm_units, np.max(r_cc_time[:, 1:, :], axis=(-2,-1)))   # [fwd/bwd]
        print(f'ub_comm_units (before):\n{ub_comm_units}')
        comm_units_pool = dict(sorted(comm_units_pool.items(), key=lambda x: len(x[0]), reverse=False)) # sort by sp_set size (small -> large)
        balanced_r_comm_time = r_cc_time[:, 1:, :]  # a slice of r_cc_time, [fwd/bwd][Comm_in/Comm_out][SP]
        for sp_set, comm_units in comm_units_pool.items():
            # sp_set: tuple
            # comm_units: [fwd/bwd][Comm_in/Comm_out]
            assert len(sp_set) > 2
            # step1: pouring water to glass up to ub
            for g in sp_set:
                for d in [0, 1]:    # 0 stands for input, 1 stands for output
                    if np.sum(balanced_r_comm_time[:, d, g] < ub_comm_units) > 0:
                        offset = np.minimum(ub_comm_units - balanced_r_comm_time[:, d, g], comm_units[:, d])
                        balanced_r_comm_time[:, d, g] += offset
                        comm_units[:, d] -= offset
            if comm_units.sum() <= 0:
                continue
            print(f'[OVERFLOW] {sp_set}: {comm_units}')
            # step2: pouring water left evenly to all glasses
            balanced_r_comm_time[:, :, list(sp_set)] += np.expand_dims(comm_units / len(sp_set), axis=-1)
            # step3: update ub
            ub_comm_units = np.max(balanced_r_comm_time, axis=(-2,-1))
        print(f'ub_comm_units (after):\n{ub_comm_units}')
        self.ub_comm_units = ub_comm_units
        return r_cc_time    # modified r_cc_time
    
    def get_relative_cc_time(self) -> np.ndarray: # np.[fwd/bwd][Comp/Comm_in/Comm_out][SP]
        # [TODO]: support sth like ring comm !
        # comp unit: (Sq_split, Skv_split)
        # comm unit: Skv_split / Sq_split for cross_attn (kv, dkv); Nh[1] / Nh[0] for GQA (kv, dkv)
        # print(f'schedule:\n{self.schedule_table}')
        if hasattr(self, 'r_cc_time'):
            return self.r_cc_time, self.balanced_r_cc_time
        
        r_cc_time = np.zeros((2, 3, self.tot_sp), dtype=np.float32)
        # comm units count: fwd, bwd
        u_inp_row = self.u_inp_row
        u_inp_col = self.u_inp_col
        u_out_row = self.u_out_row
        u_out_col = self.u_out_col
        
        split_degrees = self.split_degrees
        comm_units_pool = {}  # {sp_set -> comm_units[fwd/bwd][Comm_in/Comm_out]}
        
        # comp units
        for g in range(self.tot_sp):
            r_cc_time[:, 0, g] = np.sum(self.schedule_table == g)
        self.ub_comp_units = np.max(r_cc_time[:, 0, :], axis=-1)    # [fwd/bwd]
        # print(f'ub_comp_units:\n{self.ub_comp_units}')
        # comm units
        for g in range(self.tot_sp):
            Sq_self_mask = np.expand_dims( # [split_degrees[2], 1, split_degrees[0]]
                np.repeat(
                    self.S_map, 
                    split_degrees[0] // min(split_degrees[0], split_degrees[1]), 
                    axis=-1
                ) == g,
                axis=-2
            )
            Sq_other_mask = Sq_self_mask ^ 1
            Skv_self_mask = np.expand_dims( # [split_degrees[2], 1, split_degrees[1]]
                np.repeat(
                    self.S_map,
                    split_degrees[1] // min(split_degrees[0], split_degrees[1]), 
                    axis=-1
                ) == g, # ! Skv_self_mask
                axis=-2
            )
            Skv_other_mask = Skv_self_mask ^ 1
            Sq_self_unequal_num = np.logical_and(
                np.logical_or.reduce(
                    np.logical_and(self.schedule_table != g, self.schedule_table >= 0), 
                    axis=-1
                ),
                Sq_self_mask
            ).sum()
            Skv_self_unequal_num = np.logical_and(
                np.logical_or.reduce(
                    np.logical_and(self.schedule_table != g, self.schedule_table >= 0),
                    axis=-2
                ),
                Skv_self_mask
            ).sum()
            Sq_other_equal_num = np.logical_and(
                np.logical_or.reduce(self.schedule_table == g, axis=-1),
                Sq_other_mask
            ).sum()
            Skv_other_equal_num = np.logical_and(
                np.logical_or.reduce(self.schedule_table == g, axis=-2),
                Skv_other_mask
            ).sum()
            
            # input, in: fix
                # row: (q), (q, o, do)
            r_cc_time[:, 1, g] += u_inp_row * Sq_other_equal_num
                # col: (k, v), (k, v)
            r_cc_time[:, 1, g] += u_inp_col * Skv_other_equal_num
                
            # input, out: roots of 2 broadcast (row + col), modeling as a pool
                # row: (q), (q, o, do), only root
            r_cc_time[:, 2, g] += u_inp_row * Sq_self_unequal_num
                # pool
            self.update_comm_units_pool(comm_units_pool, Sq_self_mask, u_inp_row, split_dim=0, is_out=True, g=g)
                # col: (k, v), (k, v), only root
            r_cc_time[:, 2, g] += u_inp_col * Skv_self_unequal_num
                # pool
            self.update_comm_units_pool(comm_units_pool, Skv_self_mask, u_inp_col, split_dim=1, is_out=True, g=g)

            
            # output, in: roots of 2 reduce (row + col), modeling as a pool
                # row: (o), (dq), only root
            r_cc_time[:, 1, g] += u_out_row * Sq_self_unequal_num
                # pool
            self.update_comm_units_pool(comm_units_pool, Sq_self_mask, u_out_row, split_dim=0, is_out=False, g=g)
                # col: (), (dk, dv), only root
            r_cc_time[:, 1, g] += u_out_col * Skv_self_unequal_num
                # pool
            self.update_comm_units_pool(comm_units_pool, Skv_self_mask, u_out_col, split_dim=1, is_out=False, g=g)

            # output, out: fix
                # row: (o), (dq)
            r_cc_time[:, 2, g] += u_out_row * Sq_other_equal_num
                # col: (), (dk, dv)
            r_cc_time[:, 2, g] += u_out_col * Skv_other_equal_num

        balanced_r_cc_time = self.balance_comm_units_pool(comm_units_pool, r_cc_time)
        self.r_cc_time = r_cc_time
        self.balanced_r_cc_time = balanced_r_cc_time
        return r_cc_time, balanced_r_cc_time    # (Sq / sq_split * bs / bs_split * Nh[0] / Nh_split * D * 2B) for comm
    
    def get_absolute_cc_time(self) -> np.ndarray:
        if hasattr(self, 'ub_cc_time'):
            return self.ub_cc_time
        _, cc_time = self.get_relative_cc_time()
        assert cc_time.shape == (2, 3, self.da_config.SP[0] * self.da_config.SP[1])
        self.unit_comp_time = self.m_config.flashattn_profile_map.get_comp_time(self.da_config, [1, 1], self.split_degrees) # (s), [fwd/bwd]
        self.unit_comm_time = self.m_config.get_comm_time(self.da_config, [1, 1], self.split_degrees)    # (s), scalar
        self.ub_cc_time = np.empty((2, 2), dtype=np.float32)    # [fwd/bwd][Comp/Comm]
        self.ub_cc_time[:, 0] = self.ub_comp_units * self.unit_comp_time # (s), [fwd/bwd]
        self.ub_cc_time[:, 1] = self.ub_comm_units * self.unit_comm_time # (s), [fwd/bwd]
        # print(f'self.ub_comp_units:\n{self.ub_comp_units}')
        # print(f'self.ub_comm_units:\n{self.ub_comm_units}')
        # print(f'self.ub_cc_time:\n{self.ub_cc_time}')
        return self.ub_cc_time  # [fwd/bwd][Comp/Comm]
        # cc_time[0, 0] *= unit_comp_time[0]    # (us), fwd, Comp
        # cc_time[1, 0] *= unit_comp_time[1]    # (us), bwd, Comp
        # cc_time[:, 1:] *= unit_comm_time      # (s), fwd/bwd, Comm_in/Comm_out
        # Convert us to s for comp
        # cc_time[:, 0] /= 1e6
        # return cc_time
    
    def get_e2e_time(self):
        if hasattr(self, 'e2e_time'):
            return self.e2e_time
        self.get_absolute_cc_time()
        self.e2e_time = np.max(self.ub_cc_time, axis=-1)    # [fwd/bwd]
        return self.e2e_time

    def get_ub_cc_units_constrain(self) -> np.ndarray:
        if hasattr(self, 'ub_cc_units_constrain'):
            return self.ub_cc_units_constrain
        # step1: get e2e_time
        self.get_e2e_time()
        # step2: calc ub_cc_units_constrain from e2e_time
        self.ub_cc_units_constrain = np.empty((2, 2), dtype=np.float32)    # [fwd/bwd][Comp/Comm]
        self.ub_cc_units_constrain[:, 0] = self.e2e_time / self.unit_comp_time
        self.ub_cc_units_constrain[:, 1] = self.e2e_time / self.unit_comm_time
        return self.ub_cc_units_constrain   # [fwd/bwd][Comp/Comm(units)]


class Search_Engine():
    def __init__(self, da_config: Dist_Attn_Config, m_config: Machine_Config, init_schedule_list: list):
        self.da_config = da_config
        self.m_config = m_config
        self.tot_sp = reduce(lambda x,y:x*y, self.da_config.SP)
        self.init_schedule_list = init_schedule_list
        for schedule in self.init_schedule_list:
            schedule.get_ub_cc_units_constrain()
        self.ub = np.empty((2, 2), dtype=np.float32)    # [fwd/bwd][Comp/Comm]
        self.ub = self.ub.fill(np.inf)
        
        # Now only support split_degrees = [tot_sp, tot_sp, 1, 1], S_map = np.arange(tot_sp)
        self.split_degrees = [self.tot_sp, self.tot_sp, 1, 1]
        
        # comp/comm ub
        self.ub_cc_units = np.max(np.array([schedule.ub_cc_units_constrain for schedule in self.init_schedule_list]), axis=0)   # [fwd/bwd][Comp/Comm(units)]
        
        # # extra comp ub (not a real ub in some case !!!)
        # alignment between different split_degrees
        self.ub_cc_units[:, 0] = np.minimum(
            self.ub_cc_units[:, 0], 
            np.max(np.array([schedule.ub_comp_units * np.prod(self.split_degrees) / np.prod(schedule.split_degrees)
                             for schedule in self.init_schedule_list]), axis=0)
        )
        print(f'ub_cc_units: {self.ub_cc_units}')
        # # extra comm ub (real ub)
        # self.ub[:, 1] = np.max(np.array([schedule.ub_comm_units for schedule in self.init_schedule_list]), axis=0)
        
        self.MAX_QUEUE_SIZE = 100
        self.schedule_queues = [None, None]  # [fwd/bwd]        
    
    def reset_before_search(self):
        # Constrain: x = unpack_func(pack_func(x))
        def pack_func(schedule):
            SCHEDULE_UNIQUE_ID = get_global_var('SCHEDULE_UNIQUE_ID')
            SCHEDULE_UNIQUE_ID += 1
            set_global_var('SCHEDULE_UNIQUE_ID', SCHEDULE_UNIQUE_ID)
            # print(f'SCHEDULE_UNIQUE_ID: {SCHEDULE_UNIQUE_ID}')
            # print(f'schedule:\n{schedule.schedule_table}', flush=True)
            # print(f'fob: {self.fob}, get_e2e_time(): {schedule.get_e2e_time()}, get_absolute_cc_time:\n{schedule.get_absolute_cc_time()}')
            return (- schedule.get_e2e_time()[self.fob], SCHEDULE_UNIQUE_ID, schedule)
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
        # pruning strategy 1:
        if self.cur_cc_units[self.fob, 0, g] + 1 >= self.ub_cc_units[self.fob, 0]:
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
        if self.is_end(cur_pos):
            new_schedule = copy.deepcopy(self.cur_schedule)
            self.schedule_queues[self.fob].push(new_schedule)
            exit(0)
            return
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
        # self.brute_force_search(self.get_next_unsettled_pos((0, 0, 0, 0)))
        
        # search for bwd
        self.fob = 1    # bwd
        self.reset_before_search()
        # self.brute_force_search(self.get_next_unsettled_pos((0, 0, 0, 0)))


def get_configs():
    SP0, SP1 = 1, 4
    Sq = Skv = 16 * 1024   # 16k
    Sq = Skv = 8 * 1024   # 8k
    Sq = Skv = 4 * 1024   # 4k
    Nhq = Ng = 32
    bs = 1
    D = 128
    causal = False
    causal = True
    return Dist_Attn_Config((SP0, SP1), (Sq, Skv), (Nhq, Ng), bs, D, causal)

def get_profile_data():
    BW = (12.5, 215)   # Inter-Machine, Intra-Machine, GB/s, bidirectional
    PROFILE_FILE_NAME = './prof_data/time_flashattn_ratio.json'
    INTER_COMM_FIlE_NAME = './prof_data/cb_16_g3018-9.log'
    INTRA_COMM_FIlE_NAME = './prof_data/cb_8_g3028.log'
    with open(PROFILE_FILE_NAME, 'r') as f:
        profile_data = json.load(f)
    assert 'flash_attn' in profile_data.keys(), 'flash_attn not found in profile_data'
    
    return Machine_Config(BW, convert_profile_data_to_map(profile_data['flash_attn']), \
            convert_profile_data_to_comm_map(INTER_COMM_FIlE_NAME, 16),
            convert_profile_data_to_comm_map(INTRA_COMM_FIlE_NAME, 8),
        )

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

def create_schedule(da_config, m_config, split_degrees: list, S_map: np.ndarray, schedule_table_func):
    return Dist_Attn_Schedule(da_config, m_config, split_degrees, S_map, \
                              schedule_table_func(split_degrees, S_map, da_config.causal))
    
def get_init_schedule_list(da_config, m_config):
    # [NOTE]: 
    # 1. Now only support single topology level, i.e. level0 which means inter-machine topology
    # [TODO]:
    # 1. Support split bs and Nh.
    # 2. Support 2 level topology schedule.
    # 3. Support intellegent batch (along Sq, Skv, bs, Nh)
    # 4. Support cc overlap
    # 5. Support profiling for comm time
    # 6. differentiate causal block and noncausal block
    tot_sp = reduce(lambda x,y:x*y, da_config.SP)
    if not da_config.causal:        # optimal
        split_degrees = [tot_sp, tot_sp, 1, 1]
        S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
        S_map[:] = np.arange(tot_sp)
        qo_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_qo_schedule_table)
        kv_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_kv_schedule_table)
        return [qo_schedule, kv_schedule]
    else:
        if da_config.bs % 2 == 0:   # optimal
            split_degrees = [tot_sp, tot_sp, 2, 1]
            S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
            S_map[0] = np.arange(tot_sp)
            S_map[1] = np.arange(tot_sp - 1, - 1, - 1)
            qo_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_qo_schedule_table)
            kv_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_kv_schedule_table)
            return [qo_schedule, kv_schedule]
        else:   # not optimal
            split_degrees = [tot_sp, tot_sp, 1, 1]
            S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
            S_map[:] = np.arange(tot_sp)
            qo_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_qo_schedule_table)
            kv_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_kv_schedule_table)
            
            split_degrees = [tot_sp * 2, tot_sp * 2, 1, 1]
            S_map = np.empty((split_degrees[2], min(split_degrees[0], split_degrees[1])), dtype=np.int32)
            S_map[:] = np.concatenate((np.arange(tot_sp), np.arange(tot_sp - 1, - 1, - 1)))
            zigzag_kv_schedule = create_schedule(da_config, m_config, split_degrees, S_map, get_kv_schedule_table)
            
            # return [qo_schedule, kv_schedule]
            return [qo_schedule, kv_schedule, zigzag_kv_schedule]
