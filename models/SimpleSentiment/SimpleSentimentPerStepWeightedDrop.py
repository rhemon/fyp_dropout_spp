

from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment

class SimpleSentimentPerStepWeightedDrop(SimpleSentimentNoDropout):
    
    def __init__(self, cfg, **kwargs):
        self.keep_high_magnitude = cfg.KEEP_HIGH_MAGNITUDE
        super(SimpleSentimentPerStepWeightedDrop, self).__init__(cfg, **kwargs)
        
    
    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "WeightedDrop", self.drop_prob, self.keep_high_magnitude).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)
