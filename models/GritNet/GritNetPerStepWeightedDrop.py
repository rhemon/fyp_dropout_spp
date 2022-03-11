

from models.GritNet.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepWeightedDrop(GritNetNoDropout):
    
    def __init__(self, cfg, **kwargs):
        self.keep_high_magnitude = cfg.KEEP_HIGH_MAGNITUDE
        super(GritNetPerStepWeightedDrop, self).__init__(cfg, **kwargs)
        
    
    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "WeightedDrop", self.drop_prob, self.keep_high_magnitude).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)
