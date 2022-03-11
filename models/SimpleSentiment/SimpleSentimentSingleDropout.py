

from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import DropoutAfterBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment

class SimpleSentimentSingleDropout(SimpleSentimentNoDropout):
    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
        blstm_layer = DropoutAfterBLSTM(self.drop_prob, embedding_dim, hidden_dim, bidirectional=True, batch_first=True).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)
