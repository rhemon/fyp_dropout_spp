

import torch
from models.GritNet.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepGradBasedDrop(GritNetNoDropout):
    
    def __init__(self, cfg, train_path, input_dim, **kwargs):
        super().__init__(cfg, train_path, input_dim, **kwargs)
        self.prob_method = cfg.PROB_METHOD

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        print(event_dim)
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "GradBasedDrop", self.drop_prob).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)


    def backward_custom_updates(self):
        self.model.blstm.dropout_fcell_ih.update_keep_prob(self.model.blstm.forward_cell_ih.weight.grad.detach(), self.prob_method)
        self.model.blstm.dropout_fcell_hh.update_keep_prob(self.model.blstm.forward_cell_hh.weight.grad.detach(), self.prob_method)
        self.model.blstm.dropout_bcell_ih.update_keep_prob(self.model.blstm.backward_cell_ih.weight.grad.detach(), self.prob_method)
        self.model.blstm.dropout_bcell_hh.update_keep_prob(self.model.blstm.backward_cell_hh.weight.grad.detach(), self.prob_method)
    