

import torch
from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithGradBasedDropout
from models.layers.LinearModel import  LinearModel

class LinearModelGradBasedDrop(LinearModelNoDropout):
    
    def __init__(self, cfg, train_path, input_dim, dataprocessor, **kwargs):
        super(LinearModelGradBasedDrop, self).__init__(cfg, train_path, input_dim, dataprocessor, **kwargs)
        self.prob_method = cfg.PROB_METHOD
        self.grads = None
    
    def set_model(self, input_dim, hidden_dim, target_size):
        layer = LinearWithGradBasedDropout(input_dim, hidden_dim, self.drop_prob)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)

    def update_per_iter(self):
        cur_grads = self.model.linear_layer.fc.weight.grad.detach()
        if self.grads is None:
            self.grads = cur_grads
        else:
            new_grads = torch.max(torch.abs(self.grads), torch.abs(cur_grads))
            new_grads = new_grads * (((new_grads == torch.abs(self.grads)) * torch.sign(self.grads)).int() | ((new_grads == torch.abs(cur_grads)) * torch.sign(cur_grads)).int())
            self.grads = new_grads

    def update_per_epoch(self):
        self.model.linear_layer.drop.update_keep_prob(self.grads, self.prob_method)
        self.grards = None