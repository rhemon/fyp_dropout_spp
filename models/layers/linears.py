import torch.nn as nn

from models.layers.dropouts import StandardDropout, Standout, GradBasedDropout


class LinearWithNoDropout(nn.Module):
    """
    Linear layer with no dropout.
    """

    def __init__(self, input_dim, hidden_dim):
        """
        @param input_dim  : Input dimension of the linear layer.
        @param hidden_dim : Output dimension of the linear layer.
        """
        super(LinearWithNoDropout, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        Forward pass of the layer.
        
        @param inputs : Input tensor of the layer

        @return Output tensor of the layer.
        """
        return self.activation(self.fc(inputs))


class LinearWithDropout(nn.Module):
    """
    Linear layer with standard inverted dropout.
    """

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        """
        @param input_dim    : Input dimension of the linear layer.
        @param hidden_dim   : Output dimension of the linear layer.
        @param dropout_prob : Dropout rate for the layer.
        """
        super(LinearWithDropout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = StandardDropout(dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        Forward pass of the layer.
        
        @param inputs : Input tensor of the layer

        @return Output tensor of the layer.
        """
        x = self.activation(self.fc(inputs))
        if self.training:
            return self.drop(x)
        return x


class LinearWithStandout(nn.Module):
    """
    Linear layer with standout.
    """
    
    def __init__(self, input_dim, hidden_dim):
        """
        @param input_dim    : Input dimension of the linear layer.
        @param hidden_dim   : Output dimension of the linear layer.
        """
        super(LinearWithStandout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = Standout(self.fc)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        Forward pass of the layer.
        
        @param inputs : Input tensor of the layer

        @return Output tensor of the layer.
        """
        x = self.activation(self.fc(inputs))
        if self.training:
            return self.drop(inputs, x)
        return x

class LinearWithGradBasedDropout(nn.Module):
    """
    Linear layer with gradient based dropout.
    """
    
    def __init__(self, input_dim, hidden_dim, dropout_prob):
        """
        @param input_dim    : Input dimension of the linear layer.
        @param hidden_dim   : Output dimension of the linear layer.
        @param dropout_prob : Dropout rate for the layer.
        """
        super(LinearWithGradBasedDropout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = GradBasedDropout(hidden_dim, dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, inputs):
        """
        Forward pass of the layer.
        
        @param inputs : Input tensor of the layer

        @return Output tensor of the layer.
        """
        x = self.activation(self.fc(inputs))
        if self.training:
            return self.drop(x)
        return x