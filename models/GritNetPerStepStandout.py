

from models.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepStandout(GritNetNoDropout):
    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size, batch_size):
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "Standout", self.drop_prob, batch_size).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size, batch_size).to(self.device)
