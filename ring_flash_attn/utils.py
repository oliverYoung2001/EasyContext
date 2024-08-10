from typing import Optional, Tuple

import torch
import torch.distributed as dist
from functools import partial

__all__ = ["update_out_and_lse", "RingComm"]

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        # if torch.distributed.get_rank() == 0:
        if torch.distributed.get_rank() == torch.distributed.get_world_size() - 1:
            print(message, flush=True)
    else:
        print(message, flush=True)

# @torch.jit.script       # [TODO]: it will lead to massive fallback_function in torch.profile which degrade performance **massively**, why ?
def _update_out_and_lse_old(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return out, lse
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))
    # print_rank_0(f'out: {out.shape}, lse: {lse.shape}, block_out: {block_out.shape}, block_lse: {block_lse.shape}')

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse


# @torch.jit.script # [TODO]: it will lead to massive fallback_function in torch.profile which degrade performance **massively**, why ?
def _update_out_and_lse(
    out: torch.Tensor,
    lse: torch.Tensor,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # block_out = block_out.to(torch.float32)   # [NOTE]: why we need to translate it to float32 ?
    block_lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)

    new_lse = lse + torch.log(1 + torch.exp(block_lse - lse))

    out = torch.exp(lse - new_lse) * out + torch.exp(block_lse - new_lse) * block_out

    lse = new_lse
    return out, lse

def update_out_and_lse_old(
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
    slice_=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out is None:
        if slice_ is not None:
            raise RuntimeError("first update_out_and_lse should not pass slice_ args")
        out = block_out.to(torch.float32)
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    # elif True:  # HACK
    #     return out, lse
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse_old(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse_old(out, lse, block_out, block_lse)
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
        lse = block_lse.transpose(-2, -1).unsqueeze(dim=-1)
    elif slice_ is not None:
        slice_out, slice_lse = out[slice_], lse[slice_]
        slice_out, slice_lse = _update_out_and_lse(
            slice_out, slice_lse, block_out, block_lse
        )
        out[slice_], lse[slice_] = slice_out, slice_lse
    else:
        out, lse = _update_out_and_lse(out, lse, block_out, block_lse)
    return out, lse


@torch.jit.script
def flatten_varlen_lse(lse, cu_seqlens):
    new_lse = []
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse.append(lse[i, :, : end - start])
    return torch.cat(new_lse, dim=1)


@torch.jit.script
def unflatten_varlen_lse(lse, cu_seqlens, max_seqlen: int):
    num_seq = len(cu_seqlens) - 1
    num_head = lse.shape[-2]
    new_lse = torch.empty(
        (num_seq, max_seqlen, num_head, 1), dtype=torch.float32, device=lse.device
    )
    for i in range(num_seq):
        start, end = cu_seqlens[i], cu_seqlens[i + 1]
        new_lse[i, : end - start] = lse[start:end]
    return new_lse.squeeze(dim=-1).transpose(1, 2).contiguous()

send_stream = None
recv_stream = None
class RingCommOld:
    def __init__(self, process_groups: dist.ProcessGroup, PROC_INFO: dict = None):
        if 'world' not in process_groups.keys():
            process_groups['world'] = None
        self._process_group = process_groups['world']
        self._ops = []
        self.rank = dist.get_rank(self._process_group)
        self.world_size = dist.get_world_size(self._process_group)
        self._reqs = None

        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        self.tag = 0    # distinguish k and v

        if self._process_group is not None:
            self.send_rank = dist.get_global_rank(self._process_group, self.send_rank)
            self.recv_rank = dist.get_global_rank(self._process_group, self.recv_rank)
        global send_stream, recv_stream
        device = torch.cuda.current_stream().device
        if send_stream is None:
            send_stream = torch.cuda.Stream(device)
        if recv_stream is None:
            recv_stream = torch.cuda.Stream(device)
            # print(f'rank{self.rank}: device: {device}, send_stream: {send_stream}, recv_stream: {recv_stream}', flush=True)


    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # if torch.distributed.get_rank() == 0:
        #     print(f'to_send: {to_send.shape}, {to_send.dtype}', flush=True) # [mbs, S / sp, Nh, D], bf16
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
            
        # # [NOTE]: _coalescing_manager in batch_isend_irecv has some performance issues !!!
        # recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_group)
        # send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_group)
        # self._ops.append(recv_op)
        # self._ops.append(send_op)
        
        # optimized:
        recv_op = partial(dist.irecv, res, self.recv_rank, group=self._process_group)
        send_op = partial(dist.isend, to_send, self.send_rank, group=self._process_group)
        # recv_op = partial(dist.recv, res, self.recv_rank, group=self._process_group)
        # send_op = partial(dist.send, to_send, self.send_rank, group=self._process_group)
        global send_stream, recv_stream
        recv_op.stream = recv_stream
        send_op.stream = send_stream
        if self.rank == 0:
            self._ops.append(send_op)
            self._ops.append(recv_op)
        else:
            self._ops.append(recv_op)
            self._ops.append(send_op)
        
        return res

    def commit(self):
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        # self._reqs = dist.batch_isend_irecv(self._ops)
        # self._reqs = [op() for op in self._ops]
        self._reqs = []
        for op in self._ops:
            # print(f'rank{self.rank}: op: {op.func}, stream: {op.stream}', flush=True)
            with torch.cuda.stream(op.stream):
            # if True:
                self._reqs.append(op())

    def wait(self):
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        for req in self._reqs:
            req.wait()
        global send_stream, recv_stream
        torch.cuda.current_stream().wait_stream(send_stream)
        torch.cuda.current_stream().wait_stream(recv_stream)
        self._reqs = None
        self._ops = []
        self.tag = 0

class RingCommNew:
    def __init__(self, process_groups: dict, PROC_INFO: dict = None):
        if 'world' not in process_groups.keys():
            process_groups['world'] = None
        self._ops = []
        self._ops_scatter = []
        self._ops_gather = []
        self._ops_2 = []
        self.rank = dist.get_rank(process_groups['world'])
        self.world_size = dist.get_world_size(process_groups['world'])
        self.local_size = PROC_INFO['tasks_per_node']
        self.node_id = PROC_INFO['nodeid']
        self._reqs = None
        
        if 'local' not in process_groups.keys():
            for i in range(self.world_size // self.local_size):
                ranks = range(i * self.local_size, (i + 1) * self.local_size)
                group = dist.new_group(ranks)
                if self.rank in ranks:
                    process_groups['local'] = group
        if 'comm_stream' not in process_groups.keys():
            process_groups['comm_stream'] = torch.cuda.Stream(device=PROC_INFO['local_rank'])


        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1) % self.world_size
        if process_groups['world'] is not None:
            self.send_rank = dist.get_global_rank(process_groups['world'], self.send_rank)
            self.recv_rank = dist.get_global_rank(process_groups['world'], self.recv_rank)
        
        self._process_groups = process_groups
        self.PROC_INFO = PROC_INFO

    def send_recv(
        self, to_send: torch.Tensor, recv_tensor: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        PAR_DIM = 1
        if recv_tensor is None:
            res = torch.empty_like(to_send)
        else:
            res = recv_tensor
        # print(f'to_send: {to_send.shape}, local_size: {self.local_size}') # [mbs, S / sp, Nh, D]
        # [TODO]: optimize memory copy below !!!
        
        if self.PROC_INFO['local_rank'] != 0:
            # recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_groups['world'])
            recv_op = dist.P2POp(dist.irecv, res, self.recv_rank, group=self._process_groups['local'])
            self._ops.append(recv_op)
            
        if self.PROC_INFO['local_rank'] + 1 != self.local_size:
            # send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_groups['world'])
            send_op = dist.P2POp(dist.isend, to_send, self.send_rank, group=self._process_groups['local'])
            self._ops.append(send_op)

        scatter_list = None
        gather_list = None
        if self.PROC_INFO['local_rank'] + 1 == self.local_size:
            assert to_send.shape[PAR_DIM] % self.local_size == 0
            scatter_list = list(torch.split(to_send, to_send.shape[PAR_DIM] // self.local_size, dim=PAR_DIM))
        if self.PROC_INFO['local_rank'] == 0:
        # if True:
            gather_list = list(torch.split(res, res.shape[PAR_DIM] // self.local_size, dim=PAR_DIM))
        
        scattered_tensor_shape = list(to_send.shape)
        scattered_tensor_shape[PAR_DIM] //= self.local_size
        scattered_tensor = torch.empty(scattered_tensor_shape, dtype=to_send.dtype, device=to_send.device)
        to_gather_tensor = torch.empty_like(scattered_tensor)
        src_rank = (self.rank // self.local_size + 1) * self.local_size - 1
        dst_rank = self.rank // self.local_size * self.local_size
        # print(f'rank{self.rank}, local_rank{dist.get_rank(group=self._process_groups["local"])}, sd_rank: {src_rank} {dst_rank}, scatter_list is None: {scatter_list is None}', flush=True)
        
        # p2p for scatter
        if self.PROC_INFO['local_rank'] + 1 == self.local_size:
            for i in range(self.local_size - 1):
                self._ops_scatter.append(
                    dist.P2POp(dist.isend, scatter_list[i], self.node_id * self.local_size + i, group=self._process_groups['local'])
                )
        else:
            self._ops_scatter.append(
                dist.P2POp(dist.irecv, scattered_tensor, (self.node_id + 1) * self.local_size - 1, group=self._process_groups['local'])
            )
        self._ops_2 = [
            # partial(dist.scatter, scattered_tensor, scatter_list, src=src_rank, group=self._process_groups['local'], async_op=True),
            partial(dist.batch_isend_irecv, [
                dist.P2POp(dist.isend, scattered_tensor, (self.rank + self.local_size) % self.world_size, group=self._process_groups['world']),
                dist.P2POp(dist.irecv, to_gather_tensor, (self.rank - self.local_size + self.world_size) % self.world_size, group=self._process_groups['world']),
            ]),
            # partial(dist.gather, to_gather_tensor, gather_list, dst=dst_rank, group=self._process_groups['local'], async_op=True),
            # partial(dist.all_gather, tensor_list=gather_list, tensor=to_gather_tensor, group=self._process_groups['local'], async_op=True),
        ]
        
        # p2p for gather
        if self.PROC_INFO['local_rank'] == 0:
            for i in range(1, self.local_size):
                self._ops_gather.append(
                    dist.P2POp(dist.irecv, gather_list[i], self.node_id * self.local_size + i, group=self._process_groups['local'])
                )
        else:
            self._ops_gather.append(
                dist.P2POp(dist.isend, to_gather_tensor, self.node_id * self.local_size, group=self._process_groups['local'])
            )
        if self.PROC_INFO['local_rank'] == 0:
            res = torch.cat(gather_list, dim=PAR_DIM)
            # recv_tensor = res
        
        return res

    def commit(self):
        # print(f'rank{self.rank}: commit start !!!', flush=True)
        if self._reqs is not None:
            raise RuntimeError("commit called twice")
        self._reqs = []
        # with torch.cuda.stream(self._process_groups['comm_stream']):
        # 9.8s (no workload)
        if len(self._ops) > 0: # 9.8s
            self._reqs.append(dist.batch_isend_irecv(self._ops))
        if len(self._ops_scatter) > 0:  # 9.8s
            self._reqs.append(dist.batch_isend_irecv(self._ops_scatter))
        for op in self._ops_2:      # 13.0s
            self._reqs.append(op())
        if len(self._ops_gather) > 0:   # 9.8s
            self._reqs.append(dist.batch_isend_irecv(self._ops_gather))
        # print(f'rank{self.rank}: commit done !!!', flush=True)
            
    def wait_recursively(self, reqs):
        for req in reqs:
            if hasattr(req, 'wait'):
                req.wait()
            elif isinstance(req, list):
                self.wait_recursively(req)
            else:
                raise Exception("Unknown type of reqs in wait_recursively !!!")
            
    def wait(self):
        # print(f'rank{self.rank}: wait start !!!', flush=True)
        if self._reqs is None:
            raise RuntimeError("wait called before commit")
        self.wait_recursively(self._reqs)
        self._reqs = None
        self._ops = []
        self._ops_scatter = []
        self._ops_gather = []
        self._ops_2 = []
        # print(f'rank{self.rank}: wait done !!!', flush=True)


RingComm = RingCommOld
# RingComm = RingCommNew
