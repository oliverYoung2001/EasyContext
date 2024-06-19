import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from typing import Optional, Tuple
from collections.abc import Iterable
from functools import partial

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # block_out = block_out.to(torch.float32)
    # block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse

def update_out_and_lse(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        # out = block_out.to(torch.float32)
        out = block_out
        lse = block_lse
        # lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif True:  # HACK
        return out, lse
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse

def wait_recursively(reqs):
    if reqs is None:
        return
    for req in reqs:
        if hasattr(req, 'wait'):
            req.wait()
        elif isinstance(req, Iterable):
            wait_recursively(req)
        else:
            raise Exception("Unknown type of reqs in wait_recursively !!!")

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
        self.send_rank = self.node_id * self.local_size + (self.local_rank + 1) % self.local_size
        self.recv_rank = self.node_id * self.local_size + (self.local_rank - 1) % self.local_size
        self._ops = []
        self._reqs = None
        
    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor

        # recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self.process_groups['intra'])
        # send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self.process_groups['intra'])
        # self._ops.append(recv_op)
        # self._ops.append(send_op)
        
        # optimized:
        recv_op = partial(dist.irecv, res, self.recv_rank, group=self.process_groups['intra'])
        send_op = partial(dist.isend, to_send, self.send_rank, group=self.process_groups['intra'])
        if self.local_rank == 0:
            self._ops.append(send_op)
            self._ops.append(recv_op)
        else:
            self._ops.append(recv_op)
            self._ops.append(send_op)
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        if len(self._ops) > 0:
            # self._reqs = dist.batch_isend_irecv(self._ops)
            self._reqs = [op() for op in self._ops]
            self._ops = []

    def wait(self):
        wait_recursively(self._reqs)
        self._reqs = None

inter_comm_stream = None
class InterComm:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        assert self.world_size % self.local_size == 0
        self.node_num = self.world_size // self.local_size
        if 'world' not in process_groups.keys():
            process_groups['world'] = None
        if 'inter' not in process_groups.keys():
            for i in range(self.local_size):
                ranks = range(i, self.world_size, self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['inter'] = group
        
        self.process_groups = process_groups       
        self._ops = []
        self._reqs = None
        global inter_comm_stream
        inter_comm_stream = torch.cuda.Stream(torch.cuda.current_device())     
    
    def send_recv_q(self, step, q: torch.Tensor, _q: Optional[torch.Tensor] = None):
        offset = step // 2 + 1 if step % 2 == 0 else self.node_num - (step // 2 + 1)
        send_node_id = self.node_id - offset
        recv_node_id = self.node_id + offset
        
        if 0 <= send_node_id < self.node_num:
            q = q.contiguous()
            send_op = dist.P2POp(dist.isend, q, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_op)
        if 0 <= recv_node_id < self.node_num:
            # if _q is None:
            #     _q = torch.empty_like(q)
            recv_op = dist.P2POp(dist.irecv, _q, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_op)
        else:
            return None
        return _q

    def send_recv_o(self, step, o: torch.Tensor, lse: torch.Tensor, _o: Optional[torch.Tensor] = None, _lse: Optional[torch.Tensor] = None):
        # [TODO]: combine o and lse
        offset = step // 2 + 1 if step % 2 == 0 else self.node_num - (step // 2 + 1)
        send_node_id = self.node_id + offset
        recv_node_id = self.node_id - offset
        
        if 0 <= send_node_id < self.node_num:
            o = o.contiguous()
            send_o_op = dist.P2POp(dist.isend, o, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_o_op)
            lse = lse.contiguous()
            send_lse_op = dist.P2POp(dist.isend, lse, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_lse_op)
            
        if 0 <= recv_node_id < self.node_num:
            assert(_o is not None and _lse is not None)
            recv_o_op = dist.P2POp(dist.irecv, _o, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_o_op)
            recv_lse_op = dist.P2POp(dist.irecv, _lse, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_lse_op)
        else:
            return None, None
        return _o, _lse
    
    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        if len(self._ops) > 0:
            global inter_comm_stream
            with torch.cuda.stream(inter_comm_stream):
                self._reqs = dist.batch_isend_irecv(self._ops)
            self._ops = []
    
    def wait(self):
        wait_recursively(self._reqs)
        self._reqs = None

class OverlappedInterComm:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        self.rank = PROC_INFO['rank']
        self.world_size = PROC_INFO['world_size']
        self.local_rank = PROC_INFO['local_rank']
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        assert self.world_size % self.local_size == 0
        self.node_num = self.world_size // self.local_size
        if 'world' not in process_groups.keys():
            process_groups['world'] = None
        if 'inter' not in process_groups.keys():
            for i in range(self.local_size):
                ranks = range(i, self.world_size, self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['inter'] = group
        
        self.process_groups = process_groups       
        self._ops = []
        self._reqs = None
        global inter_comm_stream
        inter_comm_stream = torch.cuda.Stream(torch.cuda.current_device())     
    
    def send_recv_q(self, step, q: torch.Tensor, _q: Optional[torch.Tensor] = None):
        # offset = step // 2 + 1 if step % 2 == 0 else self.node_num - (step // 2 + 1)
        if step >= self.node_num - 1:
            return None
        offset = (step >> 2 << 1) + (step & 1) + 1 if step & 2 == 0 \
                else self.node_num - ((step >> 2 << 1) + (step & 1) + 1)
        send_node_id = self.node_id - offset
        recv_node_id = self.node_id + offset
        
        if 0 <= send_node_id < self.node_num:
            q = q.contiguous()
            send_op = dist.P2POp(dist.isend, q, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_op)
        if 0 <= recv_node_id < self.node_num:
            recv_op = dist.P2POp(dist.irecv, _q, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_op)
        else:
            return None
        return _q

    def send_recv_o(self, step, o: torch.Tensor, lse: torch.Tensor, _o: Optional[torch.Tensor] = None, _lse: Optional[torch.Tensor] = None):
        # [TODO]: combine o and lse
        step -= 2
        if step < 0:
            return None, None
        offset = (step >> 2 << 1) + (step & 1) + 1 if step & 2 == 0 \
                else self.node_num - ((step >> 2 << 1) + (step & 1) + 1)
        send_node_id = self.node_id + offset
        recv_node_id = self.node_id - offset
        
        if 0 <= send_node_id < self.node_num:
            o = o.contiguous()
            send_o_op = dist.P2POp(dist.isend, o, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_o_op)
            lse = lse.contiguous()
            send_lse_op = dist.P2POp(dist.isend, lse, send_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(send_lse_op)
            
        if 0 <= recv_node_id < self.node_num:
            assert(_o is not None and _lse is not None)
            recv_o_op = dist.P2POp(dist.irecv, _o, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_o_op)
            recv_lse_op = dist.P2POp(dist.irecv, _lse, recv_node_id * self.local_size + self.local_rank, group=self.process_groups['inter'])
            self._ops.append(recv_lse_op)
        else:
            return None, None
        return _o, _lse
    
    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        if len(self._ops) > 0:
            global inter_comm_stream
            with torch.cuda.stream(inter_comm_stream):
                self._reqs = dist.batch_isend_irecv(self._ops)
            self._ops = []
    
    def wait(self):
        wait_recursively(self._reqs)
        self._reqs = None
