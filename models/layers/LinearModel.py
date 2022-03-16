import torch
import torch.nn as nn

class LinearModel(nn.Module):

    def __init__(self, linear_layer, hidden_dim=2048, target_size=1):

        super(LinearModel, self).__init__()
        
        self.target_size = target_size
        
        self.linear_layer = linear_layer
        self.dense = nn.Linear(hidden_dim, target_size)
        if target_size > 1:
            self.out = nn.LogSoftmax(dim=-1)
        else:
            self.out = nn.Sigmoid()
        
    def forward(self, x):
        x = x[0]
        x = self.linear_layer(x)
        return self.out(self.dense(x))