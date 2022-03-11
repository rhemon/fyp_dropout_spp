import torch
import torch.nn as nn

class GritNet(nn.Module):

    def __init__(self, blstm_layer, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1):
        
        super(GritNet, self).__init__()
        
        self.event_dim = event_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        
        self.embeddings = nn.Embedding(event_dim, embedding_dim)
        
        self.blstm = blstm_layer

        self.dense = nn.Linear(hidden_dim*2, target_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, X):
        event_x, event_time_x, lens = X
        event_embedding = self.embeddings(event_x)
        time_embedding = self.embeddings(event_time_x)
        
        x = torch.cat([event_embedding, time_embedding], axis=-1)
        
        x = self.blstm((x, lens))

        gmp_output = torch.max(x, 1, keepdim=False)[0] # batch_size, 256 ? 

        return self.sigmoid(self.dense(gmp_output))