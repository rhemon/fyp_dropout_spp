import torch.nn as nn

class LinearModel(nn.Module):
    """
    LinearModel implementation.
    """

    def __init__(self, linear_layer, hidden_dim=2048, target_size=1):
        """
        LinearModel constructor.

        @param linear_layer : Expected to be a Linear Layer with or without dropout.
        @param hidden_dim   : Output dimension of the hidden linear layer.
        @param target_size  : Output dimenson of Linear Model. 1 if BINARY, 5 for GRADE and 10
                              for NUMBERS.
        """
        super(LinearModel, self).__init__()
        
        self.target_size = target_size
        
        self.linear_layer = linear_layer
        self.dense = nn.Linear(hidden_dim, target_size)

        if target_size > 1:
            self.out = nn.LogSoftmax(dim=-1)
        else:
            self.out = nn.Sigmoid()
        
    def forward(self, inputs):
        """
        Forward pass through LinearModel

        @param inputs : Tuple of Tensor. Expected to be only one Tensor. It it is a tuple
                        due to the way train code is written.
        
        @return LinearModel's output tensor.
        """
        x = inputs[0]
        x = self.linear_layer(x)
        return self.out(self.dense(x))
