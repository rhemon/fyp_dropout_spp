import torch

from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithGradBasedDropout
from models.layers.LinearModel import  LinearModel


class LinearModelGradBasedDrop(LinearModelNoDropout):
    """
    LinearModel with gradient based dropout.
    """

    def __init__(self, cfg, train_path, input_dim, dataprocessor, **kwargs):
        """
        Model constructor. Intializes prob_method and grads.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        @param hidden_dim    : Linear layer's output dimension. Defaulted to 2048.
        """
        super(LinearModelGradBasedDrop, self).__init__(cfg, train_path, input_dim, dataprocessor, **kwargs)
        self.prob_method = cfg.PROB_METHOD
        
    def set_model(self, input_dim, hidden_dim, target_size):
        """
        Initialize LinearModel with a linear layer and gradient based dropout.

        @param input_dim   : Input dimension of the linear layer.
        @param hidden_dim  : Output dimension of the linear layer.
        @param target_size : Output dimension of LinearModel.
        """
        layer = LinearWithGradBasedDropout(input_dim, hidden_dim, self.drop_prob)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)

    def update_per_iter(self):
        """
        Overwriting update_per_iter method to update prob 
        at each iteration.
        """
        cur_grads = self.model.linear_layer.fc.weight.grad.detach()
        self.model.linear_layer.drop.update_keep_prob(cur_grads, self.prob_method)
        