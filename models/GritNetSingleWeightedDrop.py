

from models.GritNetNoDropout import GritNetNoDropout
from models.layers.lstms import WeightedDropAfterBLSTM
from models.layers.GritNet import  GritNet

class GritNetSingleWeightedDrop(GritNetNoDropout):
    def __init__(self, cfg, train_path, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1, batch_size=32):
        self.keep_high_magnitude = cfg.KEEP_HIGH_MAGNITUDE
        super(GritNetSingleWeightedDrop, self).__init__(cfg, train_path, event_dim, embedding_dim, hidden_dim, target_size, batch_size)
        

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size, batch_size):
        blstm_layer = WeightedDropAfterBLSTM(embedding_dim*2, hidden_dim, bidirectional=True, batch_first=True, keep_high_magnitude=self.keep_high_magnitude).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size, batch_size).to(self.device)
