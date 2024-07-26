import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from collections.abc import Iterable
from functools import partial
from search_algo.execute_plan import Execution_Plan
import math
from tests.distributed.device_communicators.pynccl import PyNcclCommunicator
from .global_vars import *
from typing import overload, List, Union

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
    def __init__(self, K, V, data: Optional[torch.Tensor] = None):
        if data is not None:
            self.data = data
        else:
            self.data = torch.cat([K.flatten(), K.flatten()], dim=0)
        self.K = self.data[: K.numel()].view_as(K)
        self.V = self.data[K.numel() :].view_as(V)
    
    @classmethod
    def from_idata(cls, other):
        data = torch.empty_like(other.data)
        K = data[: other.K.numel()].view_as(other.K)
        V = data[other.K.numel():].view_as(other.V)
        return cls(K, V, data)

class Output_Row_Fwd(Integrated_Data):
    def __init__(self, O, lse: Optional[torch.Tensor] = None): # [mbs, Sq, Nh, D], [mbs, Sq, Nh, 1]
        # [NOTE]: simplify the data structure, not include lse !!!
        # self.data = torch.cat([O.flatten(), lse.flatten()], dim=0)
        # self.O = self.data[: O.numel()].view_as(O)
        # self.lse = self.data[O.numel() :].view_as(lse)
        self.data = O
        self.O = self.data
        self.lse = None
    
    @classmethod
    def from_execution_plan(cls, execution_plan: Execution_Plan, x: torch.Tensor):
        da_config = execution_plan.da_config
        split_degrees = execution_plan.split_degrees    # (Sq, Skv, bs, Nh)
        O_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh[0] // split_degrees[3], da_config.D)  # b, Sq, Nhq, D
        # lse_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh[0] // split_degrees[3], 1)
        # O_numel, lse_numel = math.prod(O_shape), math.prod(lse_shape)
        # data = torch.empty(O_numel + lse_numel, dtype=x.dtype, device=x.device)
        # O = data[: O_numel].view(O_shape)
        # lse = data[O_numel:].view(lse_shape)
        # return cls(O, lse)
        O_numel = math.prod(O_shape)
        data = torch.empty(O_numel, dtype=x.dtype, device=x.device)
        O = data.view(O_shape)
        return cls(O)
        
        
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
    def __init__(self, Q, dO, D, lse, data: Optional[torch.Tensor] = None):    # [NOTE]: optimized version of Backward for distributed senario
        '''
        D: torch.float32, [mbs, Nh, Sq]
        '''
        if data is not None:
            self.data = data
        else:
            self.data = torch.cat([Q.flatten(), dO.flatten(), D.flatten().view(Q.dtype), lse.flatten()], dim=0)
        self.Q = self.data[: Q.numel()].view_as(Q)
        self.dO = self.data[Q.numel(): Q.numel() + dO.numel()].view_as(dO)
        self.D = self.data[Q.numel() + dO.numel(): - lse.numel()].view(D.dtype).view_as(D)
        self.lse = self.data[- lse.numel():].view_as(lse)
    
    @classmethod
    def from_idata(cls, other):
        data = torch.empty_like(other.data)
        Q = data[: other.Q.numel()].view_as(other.Q)
        dO = data[other.Q.numel(): other.Q.numel() + other.dO.numel()].view_as(other.dO)
        D = data[other.Q.numel() + other.dO.numel(): - other.lse.numel()].view(other.D.dtype).view_as(other.D)
        lse = data[- other.lse.numel():].view_as(other.lse)
        return cls(Q, dO, D, lse, data)
    
    # def __init__(self, Q, dO, O, lse, data: Optional[torch.Tensor] = None):      # normal version
    #     if data is not None:
    #         self.data = data
    #     else:
    #         self.data = torch.cat([Q.flatten(), dO.flatten(), O.flatten(), lse.flatten()], dim=0)
    #     self.Q = self.data[: Q.numel()].view_as(Q)
    #     self.dO = self.data[Q.numel(): Q.numel() + dO.numel()].view_as(dO)
    #     self.O = self.data[Q.numel() + dO.numel(): - lse.numel()].view_as(O)
    #     self.lse = self.data[- lse.numel():].view_as(lse)

    # @classmethod
    # def from_idata(cls, other):
    #     data = torch.empty_like(other.data)
    #     Q = data[: other.Q.numel()].view_as(other.Q)
    #     dO = data[other.Q.numel(): other.Q.numel() + other.dO.numel()].view_as(other.dO)
    #     O = data[other.Q.numel() + other.dO.numel(): - other.lse.numel()].view_as(other.O)
    #     lse = data[- other.lse.numel():].view_as(other.lse)
    #     return cls(Q, dO, O, lse, data)
    
class Input_Col_Bwd(Integrated_Data):
    def __init__(self, K, V, data: Optional[torch.Tensor] = None):
        if data is not None:
            self.data = data
        else:
            self.data = torch.cat([K.flatten(), K.flatten()], dim=0)
        self.K = self.data[: K.numel()].view_as(K)
        self.V = self.data[K.numel() :].view_as(V)
    
    @classmethod
    def from_idata(cls, other):
        data = torch.empty_like(other.data)
        K = data[: other.K.numel()].view_as(other.K)
        V = data[other.K.numel():].view_as(other.V)
        return cls(K, V, data)

class Output_Row_Bwd(Integrated_Data):
    def __init__(self, dQ):
        self.data = dQ
        self.dQ = self.data
    
    @classmethod
    def from_execution_plan(cls, execution_plan: Execution_Plan, x: torch.Tensor):
        da_config = execution_plan.da_config
        split_degrees = execution_plan.split_degrees    # (Sq, Skv, bs, Nh)
        dQ_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh[0] // split_degrees[3], da_config.D)  # b, Sq, Nhq, D
        dQ_numel = math.prod(dQ_shape)
        data = torch.empty(dQ_numel, dtype=x.dtype, device=x.device)
        dQ = data.view(dQ_shape)
        return cls(dQ)
    
    def reduce(self, other):
        pass
        # self.data.add_(other.data)  # inplace operation

class Output_Col_Bwd(Integrated_Data):
    def __init__(self, dK, dV, data: Optional[torch.Tensor] = None):
        if data is not None:
            self.data = data
        else:
            self.data = torch.cat([dK.flatten(), dV.flatten()], dim=0)
        self.dK = self.data[: dK.numel()].view_as(dK)
        self.dV = self.data[dK.numel() :].view_as(dV)
    
    @classmethod
    def from_execution_plan(cls, execution_plan: Execution_Plan, x: torch.Tensor):
        da_config = execution_plan.da_config
        split_degrees = execution_plan.split_degrees    # (Sq, Skv, bs, Nh)
        dK_shape = dV_shape = (da_config.bs // split_degrees[2], da_config.S[0] // split_degrees[0], da_config.Nh[0] // split_degrees[3], da_config.D)  # b, Sq, Nhq, D
        dK_numel = math.prod(dK_shape)
        dV_numel = math.prod(dV_shape)
        data = torch.empty(dK_numel + dV_numel, dtype=x.dtype, device=x.device)
        dK = data[: dK_numel].view(dK_shape)
        dV = data[dK_numel: ].view(dV_shape)
        return cls(dK, dV, data)
    
    def reduce(self, other):
        pass
        # self.data.add_(other.data)


class IntraComm:
    def __init__(self, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        
    def send(self, dst: int, idata: Integrated_Data, stream: torch.cuda.Stream, ncclcomm: PyNcclCommunicator) -> None:
        # cur_stream = torch.cuda.current_stream()
        # print_rank_0(f'send, stream: {stream.cuda_stream}, cur_stream: {cur_stream.cuda_stream}')
        # with torch.cuda.stream(stream):
        # print(f'rank{self.rank}, peer{self.node_id * self.local_size + dst}, send In, {idata.data.numel()} !!!', flush=True)
        
        global_dst = self.node_id * self.local_size + dst
        ncclcomm.send(idata.data, self.rank < global_dst, stream)
        
        # dist.send(idata.data, self.node_id * self.local_size + dst, group=self.process_groups['intra'])
        # print(f'rank{self.rank}, send Out !!!', flush=True)
    
    def recv(self, src: int, idata: Integrated_Data, stream: torch.cuda.Stream, ncclcomm: PyNcclCommunicator) -> Integrated_Data:
        # cur_stream = torch.cuda.current_stream()
        # print_rank_0(f'recv, stream: {stream.cuda_stream}, cur_stream: {cur_stream.cuda_stream}')
        # with torch.cuda.stream(stream):
        # print(f'rank{self.rank}, peer{self.node_id * self.local_size + src}, recv In, {idata.data.numel()} !!!', flush=True)
        
        global_src = self.node_id * self.local_size + src
        ncclcomm.recv(idata.data, self.rank < global_src, stream)
        
        # dist.recv(idata.data, self.node_id * self.local_size + src, group=self.process_groups['intra'])
        # print(f'rank{self.rank}, recv Out !!!', flush=True)
        return idata


