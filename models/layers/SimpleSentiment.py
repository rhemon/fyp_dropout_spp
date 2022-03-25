import torch
import torch.nn as nn

class SimpleSentiment(nn.Module):

    def __init__(self, blstm_layer, vocab_size=290713, embedding_dim=200, hidden_dim=128, target_size=1, seq_len=50):
        
        super(SimpleSentiment, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        
        self.embeddings = nn.Embedding(vocab_size+1, embedding_dim)
        self.blstm = blstm_layer
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(seq_len*hidden_dim*2, target_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        text = text[0]
        x = self.embeddings(text)
        
        x = self.blstm(x)
        
        x = self.flatten(x)        
        return self.sigmoid(self.dense(x))
