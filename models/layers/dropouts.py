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
        self.p = torch.clip(self.p, min=0.00001, max=1)
        mask = Variable(torch.Tensor(self.p.size()).uniform_(0,1)).to(device) < self.p
        mask = mask.type(torch.FloatTensor).to(device)

        return (1/self.p) * mask * current

class GradBasedDropout(nn.Module):
    def __init__(self, input_dim, drop_prob):
        super(GradBasedDropout, self).__init__()
        self.keep_prob = torch.ones(input_dim).to(device)
        self.drop_prob = drop_prob
    
    def update_keep_prob(self, grad, method):
        ## The idea is to keep neurons with higher gradients stay
        ## and drop neurons with low gradients
        
        if method == "TANH":
            self.keep_prob = torch.tanh(torch.abs(grad).sum(dim=-1))
            if self.drop_prob is not None:
                # scale to 1-DROP_PROB to 1 if specified, else left as 0 to 1  range
                self.keep_prob = (self.keep_prob * self.drop_prob) + (1-self.drop_prob)
        elif method == "ABS_NORM":
            grad = torch.abs(grad).sum(dim=-1)
            self.keep_prob = (grad - torch.min(grad))/(torch.max(grad) - torch.min(grad) + 1e-7)
            if self.drop_prob is not None:
                # scale to 1-DROP_PROB to 1 if specified, else left as 0 to 1  range
                self.keep_prob = (self.keep_prob * self.drop_prob) + (1-self.drop_prob)
        elif method == "NORM":
            grad = grad.sum(dim=-1)
            self.keep_prob = (grad - torch.min(grad))/(torch.max(grad) - torch.min(grad) + 1e-7)
            if self.drop_prob is not None:
                # scale to 1-DROP_PROB to 1 if specified, else left as 0 to 1  range
                self.keep_prob = (self.keep_prob * self.drop_prob) + (1-self.drop_prob)
        
        

    def forward(self, x):
        keep_prob = torch.ones(x.shape).to(device) * torch.unsqueeze(self.keep_prob, dim=0) # for all batches same probability
        keep_prob = torch.clip(keep_prob, min=0.00001, max=1)
        mask = torch.bernoulli(keep_prob) * 1/keep_prob
        return mask * x