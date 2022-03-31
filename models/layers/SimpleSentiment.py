import torch
import torch.nn as nn

class SimpleSentiment(nn.Module):
    """
    SimpleSentiment implementation.
    """
    
    def __init__(self, blstm_layer, vocab_size=335507, embedding_dim=200, hidden_dim=128, target_size=1, seq_len=50):
        """
        Initialize layers of SimpleSentiment.
        BLSTM layer is taken as argument so that model with diffrent dropout can use
        the same base code. Where the layer passed should initialize with the different
        dropout methods.

        @param blstm_layer   : PyTorch Module object. Expected to be BLSTM layer (with or without dropout).
        @param vocab_size    : Integer value. Defaulted to 335507, the total number of words.
        @param embedding_dim : Integer value. Defaulted to 200, the dimension of embedding for each event
                               time-delta.
        @param hidden_dim    : Output dimension of the BLSTM layer. Defaulted to 128
        @param target_size   : Target size. Defaulted to 1 as for Sentiment its only binary classification.
        @param seq_len       : Sequence lenght. Defaulted to 50 for Sentiment140 dataset.
        """
        super(SimpleSentiment, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
        self.blstm = blstm_layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(seq_len*hidden_dim*2, target_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        """
        Forward pass through SimpleSentiment model.

        @param text : Input tensor which is tokenzied text.
        
        @return SimpleSentiment model's output.
        """
        text = text[0]
        x = self.embeddings(text)
        x = self.blstm(x)
        x = self.flatten(x)        
        return self.sigmoid(self.dense(x))
