import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.dependent_graph import Dependent_Graph
import pulp
import regex as re
import random
import json

class Execution_Plan(): # input: kernel streams of gpus
    def __init__(self, d_graph: Dependent_Graph, fob: bool):
        self.d_graph = d_graph
        self.da_config = d_graph.da_config
        self.m_config = d_graph.m_config
        self.split_degrees = d_graph.split_degrees
        self.fob = fob  # fwd or bwd
        self.tot_sp = d_graph.tot_sp
        print(f'schedule:\n{d_graph.schedule.schedule_table}', flush=True)
        print(f'fob: {fob}, get_e2e_time(): {d_graph.schedule.get_e2e_time()}, get_absolute_cc_time:\n{d_graph.schedule.get_absolute_cc_time()}', flush=True)
        self.generate_execution_plan()
    
    def get_plan_name(self):
        da_config = self.da_config
        return da_config.get_plan_name(self.fob)
        
    def generate_execution_plan(self):
        fob = self.fob
        d_graph = self.d_graph
        TOT_TIME_UP = d_graph.schedule.get_e2e_time()[fob] * 1000
        print(f'TOT_TIME_UP: {TOT_TIME_UP}')
        # Using LP to optimize the execution order
        mylp = pulp.LpProblem(f"Cuda_Stream_Events", pulp.LpMinimize)
        # Variables
        self.valid_kernels = []
        v_id = 0
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                self.valid_kernels.append(v)
                v.id = v_id
                v_id += 1
                v.start_time = pulp.LpVariable(f'start_time[{v.id}]', cat=pulp.const.LpContinuous, lowBound=0)
        end_time = pulp.LpVariable(f'end_time', cat=pulp.const.LpContinuous, lowBound=0)
        # Constraints
        # 1. stream exclusive
        stream_kernel_lists = {}  # (tot_sp, 3) -> list, 3 stands for comp, send, recv
        for g in range(self.tot_sp):
            for s in range(3):
                stream_kernel_lists[(g, s)] = []
        for v in self.valid_kernels:
            if v.type == 'comp':
                stream_kernel_lists[(v.key[-1], 0)].append(v)
            elif v.type == 'comm':
                stream_kernel_lists[(v.key[3], 1)].append(v)
                stream_kernel_lists[(v.key[4], 2)].append(v)
        # SEED = 2
        # random.seed(SEED)
        # print(f'SEED: {SEED}')
        # for kernel_list in stream_kernel_lists.values():    # random shuffle
        #     kernel_list = random.shuffle(kernel_list)
        overlap_set = set()
        for kernel_list in stream_kernel_lists.values():
            for i in range(len(kernel_list)):
                for j in range(i + 1, len(kernel_list)):
                    # mylp += kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time or \
                    #         kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time
                    # mylp += (kernel_list[i].start_time + kernel_list[i].time[fob] - kernel_list[j].start_time) * \
                    #         (kernel_list[j].start_time + kernel_list[j].time[fob] - kernel_list[i].start_time) <= 0
                    id0 = kernel_list[i].id
                    id1 = kernel_list[j].id
                    overlap_key = (min(id0, id1), max(id0, id1))
                    if overlap_key in overlap_set:
                        continue
                    overlap_set.add(overlap_key)
                    overlap_ij = pulp.LpVariable(f"overlap_{overlap_key[0]}_{overlap_key[1]}", cat='Binary')
                    mylp += kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time + TOT_TIME_UP * overlap_ij
                    mylp += kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time + TOT_TIME_UP * (1 - overlap_ij)
                    
        
        # 2. kernel dependences
        for u in self.valid_kernels:
            for v in u.successors:
                mylp += v.start_time >= u.start_time + u.time[fob]
        
        # 3. end_time
        for u in self.valid_kernels:
            mylp += end_time >= u.start_time + u.time[fob]
        
        # Objective
        mylp += end_time
        
        # Solve
        solver = pulp.getSolver('GUROBI')
        mylp.solve()
        self.mylp = mylp
        self.stream_kernel_lists = stream_kernel_lists
        
        # post process
        
        # sort all kernels in each stream according to start_time
        for g in range(self.tot_sp):
            for s in range(3):
                self.stream_kernel_lists[(g, s)].sort(key=lambda x: x.start_time.value())
    
        return mylp
    
    def print_lp_result(self):
        fob = self.fob
        d_graph = self.d_graph
        mylp = self.mylp
        for v in d_graph.kernel_dict.values():
            if not v.is_empty(fob):
                print(f'{v.key}: {v.start_time.value():.3e}, {v.time[fob]:.3e}, {(v.start_time.value() + v.time[fob]):.3e}')
        
        print(f'Streams:')
        for g in range(self.tot_sp):
            for s in range(3):
                print(f"gpu{g}, {['comp', 'send', 'recv'][s]}: {len(self.stream_kernel_lists[(g, s)])}")
                for v in self.stream_kernel_lists[(g, s)]:
                    print(f'{v.key}: {v.start_time.value():.3e}, {v.time[fob]:.3e}, {(v.start_time.value() + v.time[fob]):.3e}')
        print(f'objective={pulp.value(mylp.objective):.3e}')
        
            