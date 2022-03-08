

from models.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.GritNet import  GritNet

class GritNetPerStepWeightedDrop(GritNetNoDropout):
    
    def __init__(self, cfg, train_path, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1, batch_size=32):
        self.keep_high_magnitude = cfg.KEEP_HIGH_MAGNITUDE
        super(GritNetPerStepWeightedDrop, self).__init__(cfg, train_path, event_dim, embedding_dim, hidden_dim, target_size, batch_size)
        
    
    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size, batch_size):
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, "WeightedDrop", self.drop_prob, batch_size, self.keep_high_magnitude).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size, batch_size).to(self.device)
