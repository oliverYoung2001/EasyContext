import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.search_engine import Dist_Attn_Schedule, Dist_Attn_Config, Machine_Config, Evaluation_Configs
from search_algo.utils import get_factors
from search_algo.dependent_graph import Dependent_Graph, Comp_Kernel
from search_algo.execute_plan import Execution_Plan
import numpy as np
import copy
import itertools

class Comm_Rebuild_Engine():   # (Broadcast/reduce, row/col)
    def __init__(self):
        pass

class Graph_Substitution():
    def __init__(self):
        pass
    
class Graph_Transformation():
    def __init__(self):
        pass


class Comp_Fusion_Transformation(Graph_Transformation):
    def __init__(self, sub, ids_tuple: tuple):
        super().__init__()
        self.sub = sub
        assert len(ids_tuple) == len(sub.shape)
        for dim in range(len(sub.shape)):
            assert len(ids_tuple[dim]) == sub.shape[dim]
            assert isinstance(ids_tuple[dim], np.ndarray)
        self.ids_tuple = ids_tuple
        self.ids_set = set()
        assert len(ids_tuple) == 2
        for x in ids_tuple[0]:
            for y in ids_tuple[1]:
                self.ids_set.add((x, y))
    
    def apply_on_d_graph(self, d_graph: Dependent_Graph):
        # [TODO]: apply transformation on d_graph
        schedule = d_graph.schedule
        hierarchy = d_graph.hierarchy
        ids_tuple = self.ids_tuple
        ids_set = self.ids_set
        comp_map_key = schedule.m_config.comp_profile_maps[hierarchy].get_comp_map_key(schedule.da_config, [len(ids_tuple[0]), len(ids_tuple[1])], schedule.split_degrees)
        
        # Collect comp kernels to be fused
        old_kernels = []
        b_id = h_id = 0
        gpuid = None
        for xy in ids_set:
            x = xy[0]
            y = xy[1]
            if gpuid is None:
                gpuid = schedule.schedule_table[b_id, h_id, x, y]
            else:
                assert gpuid == schedule.schedule_table[b_id, h_id, x, y]
            comp_key = (b_id, h_id, x, y, gpuid)
            old_kernels.append(d_graph.kernel_dict[comp_key])
        
        # Create new comp kernel in d_graph.kernel_dict
        comp_key = (b_id, h_id, tuple(ids_tuple[0]), tuple(ids_tuple[1]), gpuid)
        new_kernel = Comp_Kernel(comp_key, schedule.m_config, comp_map_key, hierarchy)
        d_graph.kernel_dict[comp_key] = new_kernel
        
        # Update precursors and successors for all kernels
        for old_kernel in old_kernels:
            # for new kernel
            new_kernel.precursors.update(old_kernel.precursors)
            new_kernel.successors.update(old_kernel.successors)
            # for Comm kernels
            for precursor in old_kernel.precursors:
                precursor.successors.remove(old_kernel)
                precursor.add_successor(new_kernel)
            for successor in old_kernel.successors:
                successor.precursors.remove(old_kernel)
                successor.add_precursor(new_kernel)
            del d_graph.kernel_dict[old_kernel.key]
        
class Comp_Fusion_Substitution(Graph_Substitution):
    def  __init__(self, shape: tuple) -> None:
        super().__init__()
        assert len(shape) == 2
        self.shape = shape
    
    def dfs_lines(self, x_id: int, x_ids: list, y_ids: np.ndarray):
        if len(x_ids) == self.shape[0]:
            # Select every group of self.shape[1] y_ids in y_set and add (x_list, y_list) to self.cur_trans
            for selected_y_ids in itertools.combinations(y_ids, self.shape[1]):
                self.cur_trans.append(Comp_Fusion_Transformation(self, (np.array(x_ids), np.array(selected_y_ids))))
            return
        if x_id >= self.bool_schedule_table.shape[0]:
            return
        # not select x_id
        self.dfs_lines(x_id + 1, x_ids, y_ids)
        
        cur_y_ids = np.where(self.bool_schedule_table[x_id])[0] # np.ndarray
        new_y_ids = np.intersect1d(y_ids, cur_y_ids)
        if len(new_y_ids) < self.shape[1]:
            return
        # select x_id
        x_ids.append(x_id)
        self.dfs_lines(x_id + 1, x_ids, new_y_ids)
        x_ids.pop()
        
        
        
    def findall_trans_in_d_graph(self, d_graph: Dependent_Graph, hierarchy_sp: int) -> list:
        schedule_table = d_graph.schedule.schedule_table
        split_degrees = d_graph.split_degrees
        assert split_degrees[2] == split_degrees[3] == 1, "Not support bs_split or Nh_split > 1 !!!"
        schedule_table_sp = schedule_table[0, 0]    # (split_degrees[0], split_degrees[1]) <==> (Sq_split, Skv_split)
        assert schedule_table_sp.shape == tuple(split_degrees[: 2]), f"Error: {schedule_table_sp.shape} != {split_degrees[: 2]}"
        trans = []
        causal = False
        
        # Adssign invalid values to diagonal of schedule table when causal !!!
        if schedule_table_sp[0, split_degrees[1] - 1] < 0: # causal
            causal = True
            assert split_degrees[0] == split_degrees[1]
            table_diagonal = copy.deepcopy(np.diagonal(schedule_table_sp))
            schedule_table_sp[np.diag_indices_from(schedule_table_sp)] = - 1
            
        for _ in range(hierarchy_sp):
            trans.append([])
        # Enumerate the top line of transformations
        
        for sp_id in range(hierarchy_sp):
            self.bool_schedule_table = schedule_table_sp == sp_id
            self.cur_trans = trans[sp_id]
            self.dfs_lines(0, [], np.arange(hierarchy_sp))
        
        # Retore diagonal values of schedule table
        if causal: # causal
            schedule_table_sp[np.diag_indices_from(schedule_table_sp)] = table_diagonal
        
        return trans
    
        

        
class Graph_Transformation_Engine():    # batch engine
    # input: d_graph
    # output: d_graph
    # [NOTE]: each position in schedule table cannot be fused more than ones !!!
    def __init__(self, exp_config: Evaluation_Configs, da_config: Dist_Attn_Config, m_config: Machine_Config):
        self.exp_config = exp_config
        self.hierarchy = da_config.hierarchy
        self.tot_sp = da_config.tot_sp
        self.hierarchy_sp = da_config.SP[self.hierarchy]
        
        # type1: comp fusion substitutions
        self.comp_unit_ub = self.hierarchy_sp // 2  # 4 -> 2, 5 -> 2, 8 -> 4
        # self.ub_factors = get_factors(self.comp_unit_ub)
        self.comp_fusion_shapes = []
        for x in range(1, int(self.comp_unit_ub ** 0.5) + 1):
            for y in range(x, self.comp_unit_ub // x + 1):
                if x == 1 and y == 1:
                    continue
                if x * y <= self.comp_unit_ub:
                    self.comp_fusion_shapes.append((x, y))
                if x != y:
                    self.comp_fusion_shapes.append((y, x))
        self.comp_fusion_shapes.sort(key=lambda x: (x[0] * x[1], x[1]), reverse=True)
        print(f'comp_fusion_shapes: {self.comp_fusion_shapes}')
        self.comp_fusion_subs = [Comp_Fusion_Substitution(shape) for shape in self.comp_fusion_shapes]
        
        # type2: comm fusion substitutions [TODO]
        self.comm_fusion_subs = []
        
        # substitutions dictionary
        self.subs_dict = {
            'comp_fusion': self.comp_fusion_subs,
            'comm_fusion': self.comm_fusion_subs,
        }
        
    def apply_transformation(self):
        pass
    
    def print_trans(self):
        print(f'All transformations:', flush=True)
        for sp_id in range(self.hierarchy_sp):
            print(f'{sp_id}:', flush=True)
            for tran in self.trans_sp[sp_id]:
                print(f'{tran.ids_tuple}', flush=True)
        
    def get_all_transformations(self):
        '''
        transformations are concrete substitutions with positions on a concrete graph
        '''
        assert hasattr(self, 'd_graph'), 'No d_graph assigned !!!'
        d_graph = self.d_graph
        self.trans_sp = [] # different comp modules
        for _ in range(self.hierarchy_sp):
            self.trans_sp.append([])
        for sub in self.subs_dict['comp_fusion']:
            sub_trans = sub.findall_trans_in_d_graph(d_graph, self.hierarchy_sp)
            for sp_id in range(self.hierarchy_sp):
                self.trans_sp[sp_id] += sub_trans[sp_id]
        self.trans_all = []
        for trans in self.trans_sp:
            self.trans_all += trans
        self.print_trans()
    
    def dfs_trans(self, trans_all_id: int, selected_trans: list, fused_pos: set):
        if trans_all_id >= len(self.trans_all):
            if len(selected_trans) == 0:
                return
            # Apply transformations on d_graph and assess the performance
            new_d_graph = copy.deepcopy(self.d_graph)   # apply transformations on new_d_graph
            # print(f'd_graph: {self.d_graph}, new_d_graph: {new_d_graph}')
            for tran in selected_trans:
                tran.apply_on_d_graph(new_d_graph)
            # print trans
            print(f'Selected Transformations: ', end='', flush=True)
            for tran in selected_trans:
                print(f'{tuple(tran.ids_tuple)} ', end='', flush=True)
            print(flush=True)
            execute_plan = Execution_Plan(new_d_graph, self.exp_config.fob, plan_type=self.exp_config.plan_type)
            execute_plan.print_lp_result()
            return
        # not select
        self.dfs_trans(trans_all_id + 1, selected_trans, fused_pos)
        
        # select
        if len(fused_pos & self.trans_all[trans_all_id].ids_set) == 0:   # not conflict
            new_fused_pos = fused_pos | self.trans_all[trans_all_id].ids_set
            selected_trans.append(self.trans_all[trans_all_id])
            self.dfs_trans(trans_all_id + 1, selected_trans, new_fused_pos)
            selected_trans.pop()
        
        
    def transform(self, d_graph: Dependent_Graph):
        self.d_graph = d_graph
        self.get_all_transformations()
        
        self.dfs_trans(0, [], set())