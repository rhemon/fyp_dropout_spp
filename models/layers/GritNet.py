import torch
import torch.nn as nn

class GritNet(nn.Module):
    """
    GritNet architecture taken from Kim et al.'s work.
    Paper: https://arxiv.org/abs/1804.07405
    """
    
    def __init__(self, blstm_layer, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1):
        """
        Initialize layers of GritNet.
        BLSTM layer is taken as argument so that model with diffrent dropout can use
        the same base code. Where the layer passed should initialize with the different
        dropout methods.

        @param blstm_layer   : PyTorch Module object. Expected to be BLSTM layer (with or without dropout).
        @param event_dim     : Integer value. Defaulted to 1113, the dimension of the one hot vector.
        @param embedding_dim : Integer value. Defaulted to 2048, the dimension of embedding for each event
                               time-delta.
        @param hidden_dim    : Output dimension of the BLSTM layer. Defaulted to 128
        @param target_size   : Target size. When BINARY its 1, when its GRADE its 5.
        """
        super(GritNet, self).__init__()
        
        self.event_dim = event_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        
        self.embeddings = nn.Embedding(event_dim, embedding_dim)

        self.blstm = blstm_layer

        self.dense = nn.Linear(hidden_dim*2, target_size)

        if target_size > 1:
            self.out = nn.LogSoftmax(dim=-1)
        else:
            self.out = nn.Sigmoid()
        
    def forward(self, inputs):
        """
        Forward pass of GritNet

        @param inputs : Tuple<Tensor> where first tensor is event's index and second
                        tensor is time delta's indexes.

        @return Model's output tensor.
        """
        event_x, event_time_x = inputs
        event_embedding = self.embeddings(event_x)
        time_embedding = self.embeddings(event_time_x)
        x = torch.cat([event_embedding, time_embedding], axis=-1)
        x = self.blstm(x)
        gmp_output = torch.max(x, 1, keepdim=False)[0] 
        return self.out(self.dense(gmp_output))
