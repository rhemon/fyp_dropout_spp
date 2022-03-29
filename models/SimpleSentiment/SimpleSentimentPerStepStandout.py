from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import PerStepBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment


class SimpleSentimentPerStepStandout(SimpleSentimentNoDropout):
    """
    SimpleSentiment with standout.
    """
    
    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        """
        Initialize SimpleSentiment model with just PerStepBLSTM layer with standout..
        
        @param vocab_size     : Event dimension, defaulted to 290713.
        @param embedding_dim : Embedding dimension. Defaulted to 200.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, "Standout", self.drop_prob).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)
