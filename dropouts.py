import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandardDropout(nn.Module):
    def __init__(self, cfg):
        super(StandardDropout, self).__init__()
        self.prob = cfg.DROPOUT_PROB

    def forward(self, input):
        if not self.training:
            return input
        prob = torch.ones(input.shape).to(device) * (1-self.prob)
        q = torch.tensor(1/(1-self.prob)).to(device)
        mask = (torch.bernoulli(prob) * q)
        return mask * input

class RNNDrop(nn.Module):
    def __init__(self, cfg):
        super(RNNDrop, self).__init__()
        self.prob = cfg.DROPOUT_PROB
        # self.seq_len = sequence_len
    
    def forward(self, input):
        if not self.training:
            return input
        prob = torch.ones((input.size(0), 1, input.size(2))).to(device) * (1-self.prob)
        q = torch.tensor(1/(1-self.prob)).to(device)
        mask = torch.bernoulli(prob).to(device) * q
        full_mask = torch.ones(input.shape).to(device) * mask
        return full_mask * input

class WeightedDrop(nn.Module):
    def __init__(self, cfg):
        super(WeightedDrop, self).__init__()

    def forward(self, input):
        if not self.training:
            return input
        prob = torch.ones((input.size(0), 1, input.size(2))).to(device) * torch.sigmoid(input)
        q = 1/(1-prob)
        mask = torch.bernoulli(prob).to(device) * q
        return mask * input

class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        super(Standout, self).__init__()
        # self.pi = last_layer.weight

        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()

    def forward(self, previous, current, p=0.5, deterministic=False):
        if not self.training:
            return current
        # Function as in page 3 of paper: Variational Dropout
        # self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.p = self.nonlinearity(self.alpha * current + self.beta)
        self.mask = torch.bernoulli(1-self.p)
        q = 1/(1-self.p)
        # Deterministic version as in the paper
        return q * self.mask * current