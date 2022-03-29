import torch
import torch.nn as nn
from torch.autograd import Variable

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class StandardDropout(nn.Module):
    """
    Stanard Dropout layer that does inverted dropout.
    """

    def __init__(self, drop_prob):
        """
        Initializae dropout layer with a droput rate.

        @param drop_prob : Float value which marks the probability of dropping
                           each neuron.
        """
        super(StandardDropout, self).__init__()
        self.prob = drop_prob

    def forward(self, input):
        """
        Forward pass of the layer.

        @param input : Input tensor of the layer.

        @return Tensor with standard dropout applied on the input tensor.
        """
        prob = torch.ones(input.shape).to(device) * (1-self.prob)        
        q = torch.tensor(1/(1-self.prob)).to(device)
        mask = (torch.bernoulli(prob).to(device) * q)
        return mask * input


class RNNDrop(nn.Module):
    """
    RNNDrop implemnetation. 
    """

    def __init__(self, drop_prob):
        """
        Initializae dropout layer with a droput rate.

        @param drop_prob : Float value which marks the probability of dropping
                           each neuron.
        """
        super(RNNDrop, self).__init__()
        self.prob = drop_prob
    
    def reset_mask(self, mask_dim):
        """
        Given dimension creates a dropout mask for the input.
        Expected to be called at the beginning of each forward pass
        in BLSTM. So that each batch uses unique masks while
        mask of a single sequence stays same.

        @param mask_dim : Tuple of int specifying dimension of the mask.
        """
        prob = torch.ones(mask_dim).to(device) * (1-self.prob)
        q = torch.tensor(1/(1-self.prob)).to(device)
        self.mask = torch.bernoulli(prob).to(device) * q
        
    def forward(self, input):
        """
        Forward pass of RNN drop

        @param input : Input tensor

        @return Tensor after applying the dropout mask on the input values.
        """
        return self.mask * input


# https://github.com/mabirck/adaptative-dropout-pytorch/blob/master/layers.py
class Standout(nn.Module):
    """
    Taking Marco Birck's implementation and modifying to do inverted
    standout.
    """
    
    def __init__(self, last_layer, alpha=0.5, beta=1):
        """
        Initialize pi with last layer weights and hyper parameter values.

        @param last_layer : Last Linear layer, dropout is appplied on the output of that layer.
        @param alpha      : Hyper parameter alpha, defaulted to 0.5
        @param beta       : Hyper parameter beta, defaulted to 1.0
        """
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()

    def forward(self, previous, current):
        """
        Forward pass of standout.

        @param previous : Input tensor of previous layer 
        @param current  : Output tensor of previous layer

        @return Tensor where dropout mask determined using previous layer's input and
                applied on output of previoius layer.
        """
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.p = torch.clip(self.p, min=0.00001, max=1)
        mask = Variable(torch.Tensor(self.p.size()).uniform_(0,1)).to(device) < self.p
        mask = mask.type(torch.FloatTensor).to(device)

        return (1/self.p) * mask * current


class GradBasedDropout(nn.Module):
    """
    Gradient Based Dropout layer.
    """
    
    def __init__(self, input_dim, drop_prob):
        """
        Initializes Grad Based Dropout layer.

        @param input_dim : Dimenson of keep probability (same as dimenson of layer)
        @param drop_prob : Dropout rate. Can be None as well when scaling to 0-1 range.
        """
        super(GradBasedDropout, self).__init__()
        self.keep_prob = torch.ones(input_dim).to(device)
        self.drop_prob = drop_prob
    
    def update_keep_prob(self, grad, method):
        """
        Given gradient and method scale gradients to a value within
        0 - 1. If method is TANH absolute of grad is taken and tanh
        function is applied. If ABS_NORM then absolute of gradient
        is taken and scaled within a 0-1 range.

        If dropout rate is not None then value is scaled to
        (1-drop_prob) to 1. So probabily of keeping neuron is higher.
        """
        grad = torch.abs(grad).sum(dim=-1)

        if method == "TANH":
            self.keep_prob = torch.tanh(grad)
        elif method == "ABS_NORM":
            self.keep_prob = (grad - torch.min(grad))/(torch.max(grad) - torch.min(grad) + 1e-7)
        
        if self.drop_prob is not None:
            self.keep_prob = (self.keep_prob * self.drop_prob) + (1-self.drop_prob)

    def forward(self, x):
        """
        Forward pass of gradient based dropout.

        @param x : Input tensor
        
        @return Tensor after applying dropout mask on the input tensor.
        """
        # for all batches same probability
        keep_prob = torch.ones(x.shape).to(device) * torch.unsqueeze(self.keep_prob, dim=0) 
        keep_prob = torch.clip(keep_prob, min=0.00001, max=1)
        mask = torch.bernoulli(keep_prob) * 1/keep_prob
        return mask * x
