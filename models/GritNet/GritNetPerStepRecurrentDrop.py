from models.GritNet.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet


class GritNetPerStepRecurrentDrop(GritNetNoDropout):
    """
    GritNet with Recurrent dropoout.
    """

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        """
        Initialize GritNet model with just PerStepBLSTM layer with Recurrent
        dropout.
        
        @param event_dim     : Event dimension, defaulted to 1113.
        @param embedding_dim : Embedding dimension. Defaulted to 2048.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")

        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "RecurrentDropout", self.drop_prob).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)
