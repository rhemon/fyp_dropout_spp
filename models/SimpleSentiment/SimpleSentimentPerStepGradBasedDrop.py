

import torch
from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment

class SimpleSentimentPerStepGradBasedDrop(SimpleSentimentNoDropout):
    
    def __init__(self, cfg, train_path, input_dim, dataprocessor, **kwargs):
        super().__init__(cfg, train_path, input_dim, dataprocessor, **kwargs)
        self.prob_method = cfg.PROB_METHOD
        
        self.fih_grads = None
        self.fhh_grads = None
        self.bih_grads = None
        self.bhh_grads = None

    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "GradBasedDrop", self.drop_prob).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)

    def max_by_magnitude(self, old_grads, cur_grads):
        new_grads = torch.max(torch.abs(old_grads), torch.abs(cur_grads))
        new_grads = new_grads * (((new_grads == torch.abs(old_grads)) * torch.sign(old_grads)).int() | ((new_grads == torch.abs(cur_grads)) * torch.sign(cur_grads)).int())
        return new_grads

    def update_per_iter(self):
        cur_fih_grads = self.model.blstm.forward_cell_ih.weight.grad.detach()
        cur_fhh_grads = self.model.blstm.forward_cell_hh.weight.grad.detach()
        cur_bih_grads = self.model.blstm.backward_cell_ih.weight.grad.detach()
        cur_bhh_grads = self.model.blstm.backward_cell_hh.weight.grad.detach()
        if self.fih_grads is None:
            self.fih_grads = cur_fih_grads
            self.fhh_grads = cur_fhh_grads
            self.bih_grads = cur_bih_grads
            self.bhh_grads = cur_bhh_grads
        else:
            self.fih_grads = self.max_by_magnitude(self.fih_grads, cur_fih_grads)
            self.fhh_grads = self.max_by_magnitude(self.fhh_grads, cur_fhh_grads)
            self.bih_grads = self.max_by_magnitude(self.bih_grads, cur_bih_grads)
            self.bhh_grads = self.max_by_magnitude(self.bhh_grads, cur_bhh_grads)
        
    def update_per_epoch(self):
        self.model.blstm.dropout_fcell_ih.update_keep_prob(self.fih_grads, self.prob_method)
        self.model.blstm.dropout_fcell_hh.update_keep_prob(self.fhh_grads, self.prob_method)
        self.model.blstm.dropout_bcell_ih.update_keep_prob(self.bih_grads, self.prob_method)
        self.model.blstm.dropout_bcell_hh.update_keep_prob(self.bhh_grads, self.prob_method)
        
        self.fih_grads = None
        self.fhh_grads = None
        self.bih_grads = None
        self.bhh_grads = None