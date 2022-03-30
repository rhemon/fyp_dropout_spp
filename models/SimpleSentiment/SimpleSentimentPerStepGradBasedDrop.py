import torch

from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment


class SimpleSentimentPerStepGradBasedDrop(SimpleSentimentNoDropout):
    """
    SimpleSentimet with gradient based dropout.
    """
    
    def __init__(self, cfg, train_path, input_dim, dataprocessor, **kwargs):
        """
        Model constructor. Intialize prob method and grads.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        """
        super().__init__(cfg, train_path, input_dim, dataprocessor, **kwargs)
        self.prob_method = cfg.PROB_METHOD

    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        """
        Initialize SimpleSentiment model with just PerStepBLSTM layer using gradient 
        based dropout.
        
        @param vocab_size     : Event dimension, defaulted to 290713.
        @param embedding_dim : Embedding dimension. Defaulted to 200.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "GradBasedDrop", self.drop_prob).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)

    def update_per_iter(self):
        """
        Overwriting update_per_iter method to update prob
        at each iteration.
        """
        cur_fih_grads = self.model.blstm.forward_cell_ih.weight.grad.detach()
        cur_fhh_grads = self.model.blstm.forward_cell_hh.weight.grad.detach()
        cur_bih_grads = self.model.blstm.backward_cell_ih.weight.grad.detach()
        cur_bhh_grads = self.model.blstm.backward_cell_hh.weight.grad.detach()
        
        self.model.blstm.dropout_fcell_ih.update_keep_prob(cur_fih_grads, self.prob_method)
        self.model.blstm.dropout_fcell_hh.update_keep_prob(cur_fhh_grads, self.prob_method)
        self.model.blstm.dropout_bcell_ih.update_keep_prob(cur_bih_grads, self.prob_method)
        self.model.blstm.dropout_bcell_hh.update_keep_prob(cur_bhh_grads, self.prob_method)
