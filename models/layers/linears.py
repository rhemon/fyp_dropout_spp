import torch
import torch.nn as nn


from models.layers.dropouts import StandardDropout, Standout, GradBasedDropout


class LinearWithDropout(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(LinearWithDropout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = StandardDropout(dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.drop(self.activation(self.fc(x)))

class LinearWithStandout(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(LinearWithStandout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = Standout(self.fc)
        (dropout_prob)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.drop(x, self.activation(self.fc(x)))

class LinearWithGradBasedDropout(nn.Module):

    def __init__(self, input_dim, hidden_dim, dropout_prob):
        super(LinearWithGradBasedDropout, self).__init__()

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.drop = GradBasedDropout(hidden_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.drop(self.activation(self.fc(x)))

