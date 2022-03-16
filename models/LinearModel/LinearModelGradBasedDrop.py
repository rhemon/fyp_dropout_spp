

import torch
from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithGradBasedDropout
from models.layers.LinearModel import  LinearModel

class LinearModelGradBasedDrop(LinearModelNoDropout):
    
    def __init__(self, cfg, train_path, input_dim,**kwargs):
        super(LinearModelGradBasedDrop, self).__init__(cfg, train_path, input_dim, **kwargs)
        self.prob_method = cfg.PROB_METHOD
    
    def set_model(self, input_dim, hidden_dim, target_size):
        layer = LinearWithGradBasedDropout(input_dim, hidden_dim, self.drop_prob)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)

    def backward_custom_updates(self):
        self.model.linear_layer.drop.update_keep_prob(self.model.linear_layer.fc.weight.grad.detach(), self.prob_method)
        