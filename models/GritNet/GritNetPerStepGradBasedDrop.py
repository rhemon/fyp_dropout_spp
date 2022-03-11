

import torch
from models.GritNet.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepGradBasedDrop(GritNetNoDropout):
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.prob_method = cfg.PROB_METHOD

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "GradBasedDrop", self.drop_prob).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)

    def get_keep_prob(self, grad):
        if self.prob_method == "TANH":
            normed_grad = torch.tanh(torch.abs(grad).sum(dim=-1))
        elif self.prob_method == "NORM":
            grad = torch.abs(grad).sum(dim=-1)
            normed_grad = (grad - torch.min(grad))/torch.max(grad)
        else:
            raise Exception("Method for determining probability from gradient unclear")
        
        # preventing formation of nan
        normed_grad = torch.clamp(normed_grad, min=0.00001, max=1)
        
        return normed_grad

    def backward_custom_updates(self):
        self.model.blstm.dropout_fcell_ih.keep_prob = self.get_keep_prob(self.model.blstm.forward_cell_ih.weight.grad.detach())
        self.model.blstm.dropout_fcell_hh.keep_prob = self.get_keep_prob(self.model.blstm.forward_cell_hh.weight.grad.detach())
        self.model.blstm.dropout_bcell_ih.keep_prob = self.get_keep_prob(self.model.blstm.backward_cell_ih.weight.grad.detach())
        self.model.blstm.dropout_bcell_hh.keep_prob = self.get_keep_prob(self.model.blstm.backward_cell_hh.weight.grad.detach())
    