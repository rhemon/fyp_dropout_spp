
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import matplotlib.pyplot as plt

from models.layers.lstms import BLSTM 
from models.layers.GritNet import GritNet
from models.BaseModel import BaseModel

class GritNetNoDropout(BaseModel):

    def __init__(self, cfg, train_path, event_dim=1113, embedding_dim=2048, hidden_dim=128, target_size=1, batch_size=32):
        super(GritNetNoDropout, self).__init__(cfg, train_path)
        self.batch_size = batch_size
        
        try:
            self.drop_prob = cfg.DROP_PROB
        except AttributeError:
            self.drop_prob = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.set_model(event_dim, embedding_dim, hidden_dim, target_size, batch_size)

        self.lr = cfg.LR
        self.total_epoch = cfg.EPOCH
        self.loss_fn = getattr(nn, cfg.LOSS)()
        self.optimizer = getattr(optim, cfg.OPTIMIZER)(self.model.parameters(), lr=self.lr)

        self.CPT_EVERY = cfg.CPT_EVERY
        # Setting devices


    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size, batch_size):
        blstm_layer = BLSTM(embedding_dim*2, hidden_dim, bidirectional=True, batch_first=True).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size, batch_size).to(self.device)

    def backward_custom_updates(self):
        pass

    def train(self, train_loader):
        ep = 1
        cpt_track_step = 0
        self.model.train()
        all_losses = []
        epoch_losses = []

        for epoch in range(self.total_epoch):
            batch_losses = []
            for batch_idx, (event_x, event_time_x, lens, result) in enumerate(train_loader):
                event_x = event_x.to(self.device)
                event_time_x = event_time_x.to(self.device)
                # lens = lens.to(self.device)
                result = result.to(self.device)
        #         result = result.type(torch.LongTensor)
                scores = self.model(event_x, event_time_x, lens).squeeze(1)
                loss = self.loss_fn(scores, result)
                all_losses.append(loss.item())
                batch_losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()

                self.backward_custom_updates()
                
                self.optimizer.step()
                checkpoint_name = self.checkpoint_folder / Path("checkpoint_e"+str(epoch)+"_b"+str(batch_idx)+".pt")
                cpt_track_step += 1
                if cpt_track_step % self.CPT_EVERY == 0:
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': loss,
                        }, checkpoint_name)
                    print(f'epoch: {epoch + 1} step: {batch_idx + 1}/{len(train_loader)} loss: {loss}')
                    cpt_track_step = 0
            ep += 1
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        checkpoint_name = self.checkpoint_folder / Path("model.pt")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    }, checkpoint_name)

        plt.figure(0)          
        plt.plot([i for i in range(len(all_losses))], all_losses)
        plt.savefig(self.checkpoint_folder / Path("all_losses.png"))
        
        plt.figure(1)
        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.savefig(self.checkpoint_folder / Path("epoch_losses.png"))
        

    def predict(self, X, threshold=0.5):
        x_test, l_test = X
        x_test = x_test.to(self.device)
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,x_test.shape[0], 32):
                y_preds.append(self.model(x_test[i:min(x_test.shape[0],i+32), :, 0], x_test[i:min(x_test.shape[0],i+32),:,1], l_test[i:min(x_test.shape[0],i+32)]))

        return torch.squeeze(torch.cat(y_preds,dim=0).cpu() > threshold, 1)

    def fit(self, X, y_train):
        x_train, l_train = X

        train_dataset = torch.utils.data.TensorDataset(x_train[:,:,0], x_train[:,:,1], l_train, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
