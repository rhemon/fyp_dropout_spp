import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import matplotlib.pyplot as plt

from models.layers.lstms import BLSTM 
from models.layers.GritNet import GritNet
from models.BaseModel import BaseModel

class GritNetNoDropout(BaseModel):

    def __init__(self, cfg, train_path, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1):
        super(GritNetNoDropout, self).__init__(cfg, train_path)
        
        self.set_model(event_dim, embedding_dim, hidden_dim, target_size)

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        blstm_layer = BLSTM(embedding_dim*2, hidden_dim, bidirectional=True, batch_first=True).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)

    def predict(self, X, threshold=0.5):
        x_test, l_test = X
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,x_test.shape[0], 32):
                y_preds.append(self.model((x_test[i:min(x_test.shape[0],i+32), :, 0], x_test[i:min(x_test.shape[0],i+32),:,1], l_test[i:min(x_test.shape[0],i+32)])))

        return torch.squeeze(torch.cat(y_preds,dim=0).cpu() > threshold, 1)

    def fit(self, X, y_train):
        x_train, l_train = X

        train_dataset = torch.utils.data.TensorDataset(x_train[:,:,0], x_train[:,:,1], l_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)