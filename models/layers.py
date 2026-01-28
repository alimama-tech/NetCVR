import torch
import numpy as np

class FusedAttention(torch.nn.Module):
    def __init__(self, dim):
        super(FusedAttention,self).__init__()

        self.dim = dim
        self.q_layer = torch.nn.Linear(dim, dim)
        self.k_layer = torch.nn.Linear(dim, dim)
        self.v_layer = torch.nn.Linear(dim, dim)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self,input):
        Q = self.q_layer(input)
        K = self.k_layer(input)
        V = self.v_layer(input)
        attention = (Q*K).sum(-1)/(self.dim**0.5)
        attention = self.softmax(attention)
        outputs = (attention.unsqueeze(-1)*V).sum(1)
        return outputs