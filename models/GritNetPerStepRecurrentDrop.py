

from models.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepRecurrentDrop(GritNetNoDropout):
    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size, batch_size):
        
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
            
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "RecurrentDropout", self.drop_prob, batch_size).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size, batch_size).to(self.device)
