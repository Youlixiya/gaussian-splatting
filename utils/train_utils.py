import torch
from torch import nn

class Queue(nn.Module):
    def __init__(self, h, w, f=16, device='cuda'):
        super().__init__()
        self.h = h
        self.w = w
        self.f = f
        self.device = device
        self.register_buffer('data', torch.zeros(h, w, f,device=device))
        self.register_buffer('ptrs', torch.zeros(h, dtype=torch.long, device=device))
    
    def append(self, index, data):
        ptr = self.ptrs[index]
        data_num = data.shape[0]
        if ptr + data_num > self.w:
            self.data[index, ptr:] = data[:self.w-ptr]
            self.data[index, :data_num-self.w+ptr] = data[self.w-ptr:]
        else:
            self.data[index, ptr : ptr + data_num] = data
        ptr = (ptr + data_num) % self.w 
        self.ptrs[index] = ptr
    
    def mean(self, dim=1):
        assert dim < 3, 'given dim must < 3.'
        return self.data.mean(dim=dim)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]
    
class Cache(nn.Module):
    def __init__(self, h, w, f=16, device='cuda'):
        super().__init__()
        self.h = h
        self.w = w
        self.f = f
        self.device = device
        self.register_buffer('data', torch.zeros(h, w, f,device=device))
        # self.register_buffer('ptrs', torch.zeros(h, dtype=torch.long, device=device))
    
    def append(self, index, data):
        
        self.data[:, index, :] = data
    
    def mean(self, dim=1):
        assert dim < 3, 'given dim must < 3.'
        return self.data.mean(dim=dim)

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        return self.data[idx]