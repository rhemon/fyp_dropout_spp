import torch
import torch.nn as nn
import torch.nn.functional as F

# Setting devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GritNet(nn.Module):

    def __init__(self, cfg, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1, batch_size=32, dropout=None):
        
        super(GritNet, self).__init__()
        
        self.event_dim = event_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.batch_size = batch_size
        
        self.embeddings = nn.Embedding(event_dim, embedding_dim)

        self.blstm = nn.LSTM(embedding_dim*2, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = dropout
        self.dense = nn.Linear(hidden_dim*2, target_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, event_x, event_time_x, lens):
        
        batch_size, seq_len = event_x.size()
#         print(event_x.size())
        
        event_embedding = self.embeddings(event_x)
#         print(event_embedding.size())
        time_embedding = self.embeddings(event_time_x)
        
        x = torch.cat([event_embedding, time_embedding], axis=-1)
        
        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        
        blstm_output, _ = self.blstm(packed_x, (h0, c0))
        
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(blstm_output, batch_first=True)

        if self.dropout is not None:
            x = self.dropout(x)

        gmp_output = torch.max(x, 1, keepdim=False)[0] # batch_size, 256 ? 

        return self.sigmoid(self.dense(gmp_output))


