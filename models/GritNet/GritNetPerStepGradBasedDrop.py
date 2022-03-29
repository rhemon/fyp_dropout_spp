import torch
from models.GritNet.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet


class GritNetPerStepGradBasedDrop(GritNetNoDropout):
    """
    GritNet with gradient based dropout.
    """
    
    def __init__(self, cfg, train_path, input_dim, dataprocessor, **kwargs):
        """
        Model constructor. Initializes prob method that specifies how to scale
        the gradient to a value within 0 to 1.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        """
        super().__init__(cfg, train_path, input_dim, dataprocessor, **kwargs)
        self.prob_method = cfg.PROB_METHOD

        self.fih_grads = None
        self.fhh_grads = None
        self.bih_grads = None
        self.bhh_grads = None

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        """
        Initialize GritNet model with just PerStepBLSTM layer using GradBasedDrop.
        
        @param event_dim     : Event dimension, defaulted to 1113.
        @param embedding_dim : Embedding dimension. Defaulted to 2048.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "GradBasedDrop", self.drop_prob).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)

    def max_by_magnitude(self, old_grads, cur_grads):
        """
        Returns new max gradients by magnitude.

        @param old_grads : Old tracked gradients
        @param cur_grads : Current gradients of the network.

        @return Tensor with new max gradients.
        """
        new_grads = torch.max(torch.abs(old_grads), torch.abs(cur_grads))
        new_grads = (new_grads * (((new_grads == torch.abs(old_grads)) * torch.sign(old_grads)).int() | 
                                    ((new_grads == torch.abs(cur_grads)) * torch.sign(cur_grads)).int()))
        return new_grads

    def update_per_iter(self):
        """
        Overwriting update_per_iter method to update max gradient
        at each iteration.
        """
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
        """
        Ovewriting update_per_epoch method to update keep prob 
        by using the max graident.
        """
        self.model.blstm.dropout_fcell_ih.update_keep_prob(self.fih_grads, self.prob_method)
        self.model.blstm.dropout_fcell_hh.update_keep_prob(self.fhh_grads, self.prob_method)
        self.model.blstm.dropout_bcell_ih.update_keep_prob(self.bih_grads, self.prob_method)
        self.model.blstm.dropout_bcell_hh.update_keep_prob(self.bhh_grads, self.prob_method)
        
        # Reset grads
        self.fih_grads = None
        self.fhh_grads = None
        self.bih_grads = None
        self.bhh_grads = None
