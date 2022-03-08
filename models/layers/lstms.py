
from models.layers.dropouts import StandardDropout, RNNDrop, Standout, WeightedDrop, GradBasedDropout
import torch
import torch.nn as nn


# Setting devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, bidirectional, batch_first):
        super(BLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.blstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=batch_first)

    def forward(self, inputs):
        x, lens = inputs

        packed_x = torch.nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(2, x.size(0), self.hidden_dim).to(device)
        
        blstm_output, _ = self.blstm(packed_x, (h0, c0))
        
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(blstm_output, batch_first=True)

        return x

class DropoutAfterBLSTM(nn.Module):
    def __init__(self, dropout_prob, embedding_dim, hidden_dim, bidirectional, batch_first):
        super(DropoutAfterBLSTM, self).__init__()

        self.blstm = BLSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=batch_first)
        self.dropout = StandardDropout(dropout_prob)
    
    def forward(self, inputs):

        x = self.blstm(inputs)
        if self.training:
            x = self.dropout(x)
        return x

class RNNDropAfterBLSTM(nn.Module):
    def __init__(self, dropout_prob, embedding_dim, hidden_dim, bidirectional, batch_first, batch_size=32):
        super(RNNDropAfterBLSTM, self).__init__()

        self.blstm = BLSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=batch_first)
        self.dropout = RNNDrop(dropout_prob)
    
    def forward(self, inputs):

        x = self.blstm(inputs)
        if self.training:
            x = self.dropout(x)
        return x

class WeightedDropAfterBLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, bidirectional, batch_first, keep_high_magnitude):
        super(WeightedDropAfterBLSTM, self).__init__()

        self.blstm = BLSTM(embedding_dim, hidden_dim, bidirectional=bidirectional, batch_first=batch_first)
        self.dropout = WeightedDrop(keep_high_magnitude)
    
    def forward(self, inputs):
        x = self.blstm(inputs)
        if self.training:
            x = self.dropout(x)
        return x

class PerStepBLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, dropout_method, drop_prob=0.2, batch_size=32, keep_high_magnitude=True):
        super(PerStepBLSTM, self).__init__()

        self.hidden_dim = hidden_dim

        self.forward_in_ih = nn.Linear(embedding_dim, hidden_dim)
        self.forward_forget_ih = nn.Linear(embedding_dim, hidden_dim)
        self.forward_cell_ih = nn.Linear(embedding_dim, hidden_dim)
        self.forward_out_ih = nn.Linear(embedding_dim, hidden_dim)

        self.forward_in_hh = nn.Linear(hidden_dim, hidden_dim)
        self.forward_forget_hh = nn.Linear(hidden_dim, hidden_dim)
        self.forward_cell_hh = nn.Linear(hidden_dim, hidden_dim)
        self.forward_out_hh = nn.Linear(hidden_dim, hidden_dim)
                
        self.backward_in_ih = nn.Linear(embedding_dim, hidden_dim)
        self.backward_forget_ih = nn.Linear(embedding_dim, hidden_dim)
        self.backward_cell_ih = nn.Linear(embedding_dim, hidden_dim)
        self.backward_out_ih = nn.Linear(embedding_dim, hidden_dim)

        self.backward_in_hh = nn.Linear(hidden_dim, hidden_dim)
        self.backward_forget_hh = nn.Linear(hidden_dim, hidden_dim)
        self.backward_cell_hh = nn.Linear(hidden_dim, hidden_dim)
        self.backward_out_hh = nn.Linear(hidden_dim, hidden_dim)

        self.setup_dropout_layer(dropout_method, drop_prob, keep_high_magnitude)

    def setup_dropout_layer(self, dropout_method, drop_prob, keep_high_magnitude):
        self.rnn_drop = False
        self.recurrent_drop = False
        self.magnitude_drop = False
        self.h0_drop = False
        self.standout_drop = False
        self.grad_drop = False

        if dropout_method == "Standard":
            self.dropout = StandardDropout(drop_prob)
            self.h0_drop = True
        elif dropout_method == "RNNDrop":
            self.dropout = RNNDrop(drop_prob, True)
            self.cellstate_drop = True
        elif dropout_method == "RecurrentDropout":
            self.dropout = StandardDropout(drop_prob)
            self.recurrent_drop = True
        elif dropout_method == "Standout":
            self.dropout_fcell_ih = Standout(self.forward_cell_ih)
            self.dropout_fcell_hh = Standout(self.forward_cell_hh)
            self.dropout_bcell_ih = Standout(self.backward_cell_ih)
            self.dropout_bcell_hh = Standout(self.backward_cell_hh)
            self.standout_drop = True
        elif dropout_method == "GradBasedDrop":
            self.dropout_fcell_ih = GradBasedDropout(self.hidden_dim)
            self.dropout_fcell_hh = GradBasedDropout(self.hidden_dim)
            self.dropout_bcell_ih = GradBasedDropout(self.hidden_dim)
            self.dropout_bcell_hh = GradBasedDropout(self.hidden_dim)
            self.grad_drop = True
        elif dropout_method == "WeightedDrop":
            self.dropout = WeightedDrop(keep_high_magnitude)
            self.magnitude_drop = True

    def forward(self, x):
        x, _ = x
        lstm_inputs = x.unbind(1)
        reverse_lstm_inputs = lstm_inputs[::-1]
        
        forward_lstm_outputs = []
        backward_lstm_outputs = []

        forward_h0, forward_c0 = (torch.zeros(x.size(0), self.hidden_dim).to(device), torch.zeros(x.size(0), self.hidden_dim).to(device))
        backward_h0, backward_c0 = (torch.zeros(x.size(0), self.hidden_dim).to(device), torch.zeros(x.size(0), self.hidden_dim).to(device))

        if self.training and self.rnn_drop:
            self.dropout.reset_mask((x.size(0), self.hidden_dim))

        for i in range(len(lstm_inputs)):
            forward_step_input = lstm_inputs[i]

            forward_ingate = self.forward_in_ih(forward_step_input) + self.forward_in_hh(forward_h0)
            forward_forgetgate = self.forward_forget_ih(forward_step_input) + self.forward_forget_hh(forward_h0)
            forward_outgate = self.forward_out_ih(forward_step_input) + self.forward_out_hh(forward_h0)
            forward_cell_ih = self.forward_cell_ih(forward_step_input)
            forward_cell_hh = self.forward_cell_hh(forward_h0)

            backward_step_input = reverse_lstm_inputs[i]
            backward_ingate = self.backward_in_ih(backward_step_input) + self.backward_in_hh(backward_h0)
            backward_forgetgate = self.backward_forget_ih(backward_step_input) + self.backward_forget_hh(backward_h0)
            backward_outgate = self.backward_out_ih(backward_step_input) + self.backward_out_hh(backward_h0)
            backward_cell_ih = self.backward_cell_ih(backward_step_input)
            backward_cell_hh = self.backward_cell_hh(backward_h0)
            
            if self.training:
                if self.standout_drop:
                    forward_cell_ih = self.dropout_fcell_ih(forward_step_input, forward_cell_ih)
                    forward_cell_hh = self.dropout_fcell_hh(forward_h0, forward_cell_hh)
                    backward_cell_ih = self.dropout_bcell_ih(backward_step_input, backward_cell_ih)
                    backward_cell_hh = self.dropout_bcell_hh(backward_h0, backward_cell_hh)
                elif self.grad_drop:
                    forward_cell_ih = self.dropout_fcell_ih(forward_cell_ih)
                    forward_cell_hh = self.dropout_fcell_hh(forward_cell_hh)
                    backward_cell_ih = self.dropout_bcell_ih(backward_cell_ih)
                    backward_cell_hh = self.dropout_bcell_hh(backward_cell_hh)
            
            forward_cellgate = forward_cell_ih + forward_cell_hh
            backward_cellgate = backward_cell_ih + backward_cell_hh

            if self.training and self.magnitude_drop:
                forward_cellgate = self.dropout(forward_cellgate)
                backward_cellgate = self.dropout(backward_cellgate)

            forward_ingate = torch.sigmoid(forward_ingate)
            forward_forgetgate = torch.sigmoid(forward_forgetgate)
            forward_cellgate = torch.tanh(forward_cellgate)
            forward_outgate = torch.sigmoid(forward_outgate)

            backward_ingate = torch.sigmoid(backward_ingate)
            backward_forgetgate = torch.sigmoid(backward_forgetgate)
            backward_cellgate = torch.tanh(backward_cellgate)
            backward_outgate = torch.sigmoid(backward_outgate)

            if self.training and self.recurrent_drop:
                forward_cellgate = self.dropout(forward_cellgate)
                backward_cellgate = self.dropout(backward_cellgate)
            
            forward_c0 = (forward_forgetgate * forward_c0) + (forward_ingate * forward_cellgate)
            if self.training and self.rnn_drop:
                forward_c0 = self.dropout(forward_c0) 

            forward_h0 = forward_outgate * torch.tanh(forward_c0)
                
            backward_c0 = (backward_forgetgate * backward_c0) + (backward_ingate * backward_cellgate)
            if self.training and self.rnn_drop:
                backward_c0 = self.dropout(backward_c0)
            
            backward_h0 = backward_outgate * torch.tanh(backward_c0)
            
            if self.training and self.h0_drop:
                forward_h0 = self.dropout(forward_h0)
                backward_c0 = self.dropout(backward_h0)

            forward_lstm_outputs.append(forward_h0)
            backward_lstm_outputs.append(backward_h0)
        
        forward_lstm_outputs = torch.stack(forward_lstm_outputs, dim=1)
        backward_lstm_outputs = torch.stack(backward_lstm_outputs, dim=1)

        x = torch.cat([forward_lstm_outputs, backward_lstm_outputs], dim=-1)

        return x
