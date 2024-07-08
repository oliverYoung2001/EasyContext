import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from collections.abc import Iterable
from functools import partial
from search_algo.execute_plan import Execution_Plan
import math

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

class Integrated_Data():
    def __init__(self):
        pass
    
class Input_Row_Fwd(Integrated_Data):
    def __init__(self, Q):
        self.data = Q
        self.Q = self.data
    
    @classmethod
    def from_idata(cls, other):
        data = torch.empty_like(other.data)
        Q = data.view_as(other.Q)
        return cls(Q)

class Input_Col_Fwd(Integrated_Data):
    def __init__(self, K, V):
        self.data = torch.cat([K.flatten(), K.flatten()], dim=0)
        self.K = self.data[: K.numel()].view_as(K)
        self.V = self.data[K.numel() :].view_as(V)
    
    @classmethod
    def from_idata(cls, other):
        data = torch.empty_like(other.data)
        K = data[: other.K.numel()].view_as(other.K)
        V = data[other.K.numel():].view_as(other.V)
        return cls(K, V)

class Output_Row_Fwd(Integrated_Data):
    def __init__(self, O, lse): # [mbs, Sq, Nh, D], [mbs, Sq, Nh, 1]
        self.data = torch.cat([O.flatten(), lse.flatten()], dim=0)
        self.O = self.data[: O.numel()].view_as(O)
        self.lse = self.data[O.numel() :].view_as(lse)
    
    @classmethod
    def from_execution_plan(cls, execution_plan: Execution_Plan, x: torch.Tensor):
        da_config = execution_plan.da_config
        split_degrees = execution_plan.split_degrees    # (Sq, Skv, bs, Nh)
        O_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh // split_degrees[3], da_config.D)
        lse_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh // split_degrees[3], 1)
        O_numel, lse_numel = math.prod(O_shape), math.prod(lse_shape)
        data = torch.empty(O_numel + lse_numel, dtype=x.dtype, device=x.device)
        O = data[: O_numel].view(O_shape)
        lse = data[O_numel:].view(lse_shape)
        return cls(O, lse)
        
        
    def reduce(self, other):
        # def _update_out_and_lse(
        #     out: torch.Tensor,
        #     lse: torch.Tensor,
        #     block_out: torch.Tensor,
        #     block_lse: torch.Tensor,
        # ) -> Tuple[torch.Tensor, torch.Tensor]:
        #     new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
        #     out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out
        #     lse = new_lse
        #     return out, lse
        # [TODO]: complete here !!!
        # old_lse = self.lse
        # self.lse = self.lse + torch.log(1 + torch.exp(other.lse - self.lse))    # inplace ?
        # self.O = torch.exp(old_lse - self.lse) * self.O + torch.exp(other.lse - self.lse) * other.O # inplace ?
        pass
        

class Output_Col_Fwd(Integrated_Data):
    def __init__(self):
        self.data = torch.empty(0)
    
    def reduce(self, other):
        pass

class Input_Row_Bwd(Integrated_Data):
    def __init__(self, Q, dO, D, lse):
        self.data = torch.cat([Q.flatten(), dO.flatten(), D.flatten(), lse.flatten()], dim=0)
        self.Q = self.data[: Q.numel()].view_as(Q)
        self.dO = self.data[Q.numel(): Q.numel() + dO.numel()].view_as(dO)
        self.D = self.data[Q.numel() + dO.numel(), - lse.numel()].view_as(D)
        self.lse = self.data[- lse.numel():].view_as(lse)

class Input_Col_Bwd(Integrated_Data):
    def __init__(self, K, V):
        self.data = torch.cat([K.flatten(), K.flatten()], dim=0)
        self.K = self.data[: K.numel()].view_as(K)
        self.V = self.data[K.numel() :].view_as(V)

class Output_Row_Bwd(Integrated_Data):
    def __init__(self, dQ):
        self.data = dQ
        self.dQ = self.data
    
    def reduce(self, other):
        self.data.add_(other.data)  # inplace operation

class Output_Col_Bwd(Integrated_Data):
    def __init__(self, dK, dV):
        self.data = torch.cat([dK.flatten(), dV.flatten()], dim=0)
        self.dK = self.data[: dK.numel()].view_as(dK)
        self.dV = self.data[dK.numel() :].view_as(dV)
    
    def reduce(self, other):
        self.data.add_(other.data)


class IntraComm:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        if 'intra' not in process_groups.keys():
            for i in range(self.world_size // self.local_size):
                ranks = range(i * self.local_size, (i + 1) * self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['intra'] = group
        self.process_groups = process_groups
        
    def send(self, dst: int, idata: Integrated_Data):
        dist.send(idata.data, self.node_id * self.local_size + dst, group=self.process_groups['intra'])
    
    def recv(self, src: int, idata: Integrated_Data = None) -> Integrated_Data:
        dist.recv(idata.data, self.node_id * self.local_size + src, group=self.process_groups['intra'])
        return idata


