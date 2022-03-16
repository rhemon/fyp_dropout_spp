import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import matplotlib.pyplot as plt

from models.layers.lstms import BLSTM 
from models.layers.LinearModel import LinearModel
from models.BaseModel import BaseModel

class LinearModelNoDropout(BaseModel):

    def __init__(self, cfg, train_path, input_dim, dataprocessor, hidden_dim=2048):
        super(LinearModelNoDropout, self).__init__(cfg, train_path, dataprocessor)
        
        target_size = 1
        if cfg.OUTPUT_TYPE == "GRADE":
            target_size = 5
        elif cfg.OUTPUT_TYPE == "NUMBERS":
            target_size = 10
        self.set_model(input_dim, hidden_dim, target_size)

    def set_model(self, input_dim, hidden_dim, target_size):
        layer = nn.Linear(input_dim, hidden_dim).to(self.device)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)

    def predict(self, X, threshold=0.5):
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,X.shape[0], self.batch_size):
                y_preds.append(self.model((X[i:min(X.shape[0], i+self.batch_size)],)))

        if y_preds[0].shape[-1] > 1:
            yp = torch.argmax(torch.cat(y_preds, dim=0), dim=1).cpu()
            return yp

        return torch.squeeze(torch.cat(y_preds,dim=0).cpu() > threshold, 1)

    def fit(self, X, y_train):
        train_dataset = torch.utils.data.TensorDataset(X, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
