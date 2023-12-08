import torch
from torch import nn
import torch.nn.functional as F

class Codebook(nn.Module):
    def __init__(self, N, embedding_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(N, embedding_dim)
    
    def forward(self, x):
        return self.embedding_layer(x)

class SematicTable(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.table = []
    
    def __len__(self):
        return len(self.table)
    
    def is_empty(self):
        return self.table == []
    
    def append(self, embedding):
        self.table.append(embedding)
    
    def get_cosine_similarities(self, embedding):
        return F.cosine_similarity(embedding.unsqueeze(0), torch.stack(self.table), dim=1)

class Layer(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 norm=nn.GroupNorm,
                 act=nn.GELU):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, out_dim, 1)
        self.norm = norm(out_dim //4, out_dim)
        self.act = act()
        
    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 out_dim,
                 layers=3,
                 norm=nn.GroupNorm,
                 act=nn.GELU):
        super().__init__()
        assert layers >=3
        modules = []
        modules.append(Layer(input_dim, hidden_dim, norm, act))
        for _ in range(layers -1 ):
            modules.append(Layer(hidden_dim, hidden_dim, norm, act))
        modules.append(Layer(hidden_dim, out_dim, nn.Identity, nn.Identity))
        self.mlp = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.mlp(x)


if __name__ == '__main__':
    mlp = MLP(16, 32, 1)
    a = torch.randn((1, 16, 1000, 808))
    print(mlp(a).shape)