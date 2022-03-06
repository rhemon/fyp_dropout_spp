import torch
import torch.nn as nn
import torch.nn.functional as F
from dropouts import Standout

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

        if cfg.DROPOUT == "Standout" or cfg.DROPOUT == "GradBasedDropout":
            raise Exception("Standout and Grad Based Droput should be used with GritNetWithCustomBLSTM model")

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


class GritNetWithCustomBLSTM(nn.Module):

    def __init__(self, cfg=None, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1, batch_size=32, dropout=None):
        
        super(GritNetWithCustomBLSTM, self).__init__()
        
        self.event_dim = event_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.batch_size = batch_size
        
        self.embeddings = nn.Embedding(event_dim, embedding_dim)

        self.dense_ih_forward = nn.Linear(embedding_dim*2, 4*hidden_dim)
        self.dense_hh_forward = nn.Linear(hidden_dim, 4*hidden_dim)

        self.dense_ih_backward = nn.Linear(embedding_dim*2, 4*hidden_dim)
        self.dense_hh_backward = nn.Linear(hidden_dim, 4*hidden_dim)

        if cfg.DROPOUT == "Standout":
            self.forward_standout_ih = Standout(self.dense_ih_forward)
            self.forward_standout_hh = Standout(self.dense_hh_forward)
            self.backward_standout_ih = Standout(self.dense_ih_backward)
            self.backward_standout_hh = Standout(self.dense_hh_backward)
            self.dropout_method = "standout"
        elif cfg.DROPOUT == "GradBasedDropout":
            self.forward_ih_keep_prob = torch.ones((4*hidden_dim)).to(device)
            self.forward_hh_keep_prob =  torch.ones((4*hidden_dim)).to(device)
            self.backward_ih_keep_prob = torch.ones((4*hidden_dim)).to(device)
            self.backward_hh_keep_prob = torch.ones((4*hidden_dim)).to(device)
            self.dropout_method = "gradbased"
        else:
            self.dropout_method = ""
            if dropout is not None:
                print("Trying to use a dropout method that is not supported for custom lstm implementation. Training will continue with no dropout")

        self.dense = nn.Linear(hidden_dim*2, target_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, event_x, event_time_x, lens):
        
        event_embedding = self.embeddings(event_x)
        time_embedding = self.embeddings(event_time_x)
        
        x = torch.cat([event_embedding, time_embedding], axis=-1)

        lstm_inputs = x.data.unbind(1)
        reverse_lstm_inputs = lstm_inputs[::-1]
        
        forward_lstm_outputs = []
        backward_lstm_outputs = []

        forward_h0, forward_c0 = (torch.zeros(x.size(0), self.hidden_dim).to(device), torch.zeros(x.size(0), self.hidden_dim).to(device))
        backward_h0, backward_c0 = (torch.zeros(x.size(0), self.hidden_dim).to(device), torch.zeros(x.size(0), self.hidden_dim).to(device))

        for i in range(len(lstm_inputs)):
            forward_gates_ih = self.dense_ih_forward(lstm_inputs[i])
            forward_gates_hh = self.dense_hh_forward(forward_h0)
            backward_gates_ih = self.dense_ih_backward(reverse_lstm_inputs[i])
            backward_gates_hh = self.dense_hh_backward(backward_h0)

            if self.training:
                if self.dropout_method == "standout":
                    forward_gates_ih = self.forward_standout_ih(lstm_inputs[i], forward_gates_ih)
                    forward_gates_hh = self.forward_standout_hh(forward_h0, forward_gates_hh)
                    backward_gates_ih = self.backward_standout_ih(reverse_lstm_inputs[i], backward_gates_ih)
                    backward_gates_hh = self.backward_standout_hh(backward_h0, backward_gates_hh)
                elif self.dropout_method == "gradbased":
                    forward_gates_ih = forward_gates_ih * 1/self.forward_ih_keep_prob
                    forward_gates_hh = forward_gates_hh * 1/self.forward_hh_keep_prob
                    backward_gates_ih = backward_gates_ih * 1/self.backward_ih_keep_prob
                    backward_gates_ih = backward_gates_ih * 1/self.backward_hh_keep_prob

            forward_gates = forward_gates_ih + forward_gates_hh 
            backward_gates = backward_gates_ih + backward_gates_hh

            forward_ingate, forward_forgetgate, forward_cellgate, forward_outgate = forward_gates.chunk(4, dim=1)
            backward_ingate, backward_forgetgate, backward_cellgate, backward_outgate = backward_gates.chunk(4, dim=1)

            forward_ingate = torch.sigmoid(forward_ingate)
            forward_forgetgate = torch.sigmoid(forward_forgetgate)
            forward_cellgate = torch.tanh(forward_cellgate)
            forward_outgate = torch.sigmoid(forward_outgate)

            backward_ingate = torch.sigmoid(backward_ingate)
            backward_forgetgate = torch.sigmoid(backward_forgetgate)
            backward_cellgate = torch.tanh(backward_cellgate)
            backward_outgate = torch.sigmoid(backward_outgate)
            
            forward_c0 = (forward_forgetgate * forward_c0) + (forward_ingate * forward_cellgate)
            forward_h0 = forward_outgate * torch.tanh(forward_c0)

            backward_c0 = (backward_forgetgate * backward_c0) + (backward_ingate * backward_cellgate)
            backward_h0 = backward_outgate * torch.tanh(backward_c0)

            forward_lstm_outputs.append(forward_h0)
            backward_lstm_outputs.append(backward_h0)
        
        forward_lstm_outputs = torch.stack(forward_lstm_outputs, dim=1)
        backward_lstm_outputs = torch.stack(backward_lstm_outputs, dim=1)

        x = torch.cat([forward_lstm_outputs, backward_lstm_outputs], dim=-1)

        gmp_output = torch.max(x, 1, keepdim=False)[0] 

        return self.sigmoid(self.dense(gmp_output))


