import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
from search_algo.dependent_graph import Dependent_Graph
import pulp

class Execute_Engine(): # input: kernel streams of gpus
    def __init__(self, d_graph: Dependent_Graph, fob: bool):
        self.d_graph = d_graph
        self.da_config = d_graph.da_config
        self.m_config = d_graph.m_config
        self.split_degrees = d_graph.split_degrees
        self.fob = fob  # fwd or bwd
        self.tot_sp = d_graph.tot_sp
        self.generate_cuda_stream_events()
        
    def generate_cuda_stream_events(self):
        fob = self.fob
        d_graph = self.d_graph
        # Using LP to optimize the execution order
        mylp = pulp.LpProblem(f"Cuda_Stream_Events", pulp.LpMinimize)
        # Variables
        self.valid_kernels = []
        v_id = 0
        for v in d_graph.kernel_dict.values():
            if not v.is_empty[fob]:
                self.valid_kernels.append(v)
                v.id = v_id
                v_id += 1
                v.start_time = pulp.LpVariable(f'start_time[{v.id}]', cat=pulp.const.LpContinuous, lowBound=0)
        end_time = pulp.LpVariable(f'end_time', cat=pulp.const.LpContinuous, lowBound=0)
        # Constraints
        # 1. stream exclusive
        stream_kernel_lists = {}  # (tot_sp, 3) -> list, 3 stands for comp, send, recv
        for g in self.tot_sp:
            for s in range(3):
                stream_kernel_lists[(g, s)] = []
        for v in self.valid_kernels:
            if v.type == 'comp':
                stream_kernel_lists[(v.key[-1], 0)].append(v)
            elif v.type == 'comm':
                stream_kernel_lists[(v.key[2], 1)].append(v)
                stream_kernel_lists[(v.key[3], 2)].append(v)
        for kernel_list in stream_kernel_lists.values():
            for i in range(len(kernel_list)):
                for j in range(i + 1, len(kernel_list)):
                    mylp += kernel_list[i].start_time + kernel_list[i].time[fob] <= kernel_list[j].start_time or \
                            kernel_list[j].start_time + kernel_list[j].time[fob] <= kernel_list[i].start_time
        
        # 2. kernel dependences
        for u in self.valid_kernels:
            for v in u.seccessors:
                mylp += v.start_time >= u.start_time + u.time[fob]
        
        # 3. end_time
        for u in self.valid_kernels:
            mylp += end_time >= u.start_time + u.time[fob]
        
        # Objective
        mylp += end_time
        
        # Solve
        solver = pulp.getSolver('GUROBI')
        mylp.solve()
        
        pass
    