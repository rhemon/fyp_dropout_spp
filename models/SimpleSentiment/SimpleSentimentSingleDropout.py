from models.SimpleSentiment.SimpleSentimentNoDropout import SimpleSentimentNoDropout
from models.layers.lstms import DropoutAfterBLSTM
from models.layers.SimpleSentiment import  SimpleSentiment


class SimpleSentimentSingleDropout(SimpleSentimentNoDropout):
    """
    SimpleSentimet with single standard dropout.
    """
    
    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        """
        Initialize SimpleSentiment model with just DroputAfterBLSTM.
        
        @param vocab_size     : Event dimension, defaulted to 290713.
        @param embedding_dim : Embedding dimension. Defaulted to 200.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
        blstm_layer = DropoutAfterBLSTM(self.drop_prob, embedding_dim, hidden_dim).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)
