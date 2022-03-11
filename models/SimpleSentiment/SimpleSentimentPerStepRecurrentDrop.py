

from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment

class SimpleSentimentPerStepRecurrentDrop(SimpleSentimentNoDropout):
    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
            
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "RecurrentDropout", self.drop_prob).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)
