import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StandardDropout(nn.Module):
    def __init__(self, drop_prob):
        super(StandardDropout, self).__init__()
        self.prob = drop_prob

    def forward(self, input):
        prob = torch.ones(input.shape).to(device) * (1-self.prob)        
        q = torch.tensor(1/(1-self.prob)).to(device)
        mask = (torch.bernoulli(prob).to(device) * q)
        return mask * input

class RNNDrop(nn.Module):
    def __init__(self, prob, per_step=False):
        super(RNNDrop, self).__init__()
        self.prob = prob
        self.per_step = per_step
    
    def reset_mask(self, mask_dim):
        # Generaate mask for every freature for all sampls
        prob = torch.ones(mask_dim).to(device) * (1-self.prob)
        q = torch.tensor(1/(1-self.prob)).to(device)
        self.mask = torch.bernoulli(prob).to(device) * q

    def forward(self, input):  
        # if not per step then make the same mask for every step in seq
        if not self.per_step:
            self.reset_mask((input.shape[0], input.shape[-1]))
            self.mask = torch.unsqueeze(self.mask, dim=1)
        return input * self.mask

class WeightedDrop(nn.Module):
    def __init__(self, keep_high_magnitude=True):
        super(WeightedDrop, self).__init__()

        self.keep_high_magnitude = keep_high_magnitude

    def forward(self, input):
        ones = torch.ones(input.shape).to(device) 
        prob = ones *  torch.softmax(torch.abs(input), dim=-1)
        # take 1-p if keep low magnitude neurons
        if not self.keep_high_magnitude:
            prob = 1-prob    
        q = 1/(prob)
        mask = torch.bernoulli(prob).to(device) * q
        return mask * input

# https://github.com/mabirck/adaptative-dropout-pytorch/blob/master/layers.py
class Standout(nn.Module):
    def __init__(self, last_layer, alpha=0.5, beta=1):  
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()

    def forward(self, previous, current):
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)

        mask = Variable(torch.Tensor(self.p.size()).uniform_(0,1)).to(device) < self.p
        mask = mask.type(torch.FloatTensor).to(device)

        return (1/self.p) * mask * current

class GradBasedDropout(nn.Module):
    def __init__(self, input_dim):
        super(GradBasedDropout, self).__init__()
        self.keep_prob = torch.ones(input_dim).to(device)
    
    def forward(self, x):
        keep_prob = torch.ones(x.shape).to(device) * torch.unsqueeze(self.keep_prob, dim=0) # for all batches same probability
        mask = torch.bernoulli(keep_prob) * 1/keep_prob
        return mask * x
