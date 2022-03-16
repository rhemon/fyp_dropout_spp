
import datetime
from tabnanny import check

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

import matplotlib.pyplot as plt
import shutil

class BaseModel:

    def __init__(self, cfg, checkpoint_folder):

        self.checkpoint_folder = checkpoint_folder

        self.batch_size = cfg.BATCH_SIZE
        
        try:
            self.drop_prob = cfg.DROP_PROB
        except AttributeError:
            self.drop_prob = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.lr = cfg.LR
        self.total_epoch = cfg.EPOCH
        self.loss_fn_name = cfg.LOSS
        self.optim_name = cfg.OPTIMIZER
        self.PRINT_EVERY = cfg.PRINT_EVERY
        
    def set_model(self, **kwargs):
        raise Exception("Not Implemented")

    def backward_custom_updates(self):
        pass

    def train(self, train_loader):
        
        self.loss_fn = getattr(nn, self.loss_fn_name)()
        self.optimizer = getattr(optim, self.optim_name)(self.model.parameters(), lr=self.lr)

        ep = 1
        print_track_step = 0
        self.model.train()
        all_losses = []
        epoch_losses = []

        for epoch in range(self.total_epoch):
            batch_losses = []
            for batch_idx, example, in enumerate(train_loader):
                
                result = example[-1]
                scores = self.model(example[:-1]).squeeze(1)
                loss = self.loss_fn(scores, result)
                all_losses.append(loss.item())
                batch_losses.append(loss.item())
                self.optimizer.zero_grad()
                loss.backward()

                self.backward_custom_updates()
                
                self.optimizer.step()
                print_track_step += 1
                if print_track_step % self.PRINT_EVERY == 0:
                    print(f'epoch: {epoch + 1} step: {batch_idx + 1}/{len(train_loader)} loss: {loss}')
                    print_track_step = 0

            print(f'epoch: {epoch + 1} loss: {sum(batch_losses)/len(batch_losses)}')
            print_track_step = 0
            checkpoint_name = self.checkpoint_folder / Path("checkpoint_e"+str(epoch)+".pt")
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        }, checkpoint_name)
            ep += 1
            epoch_losses.append(sum(batch_losses)/len(batch_losses))
        checkpoint_name = self.checkpoint_folder / Path("model.pt")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'all_losses': all_losses,
                    'epoch_losses': epoch_losses,
                    }, checkpoint_name)

        plt.figure(0)          
        plt.plot([i for i in range(len(all_losses))], all_losses)
        plt.savefig(self.checkpoint_folder / Path("all_losses.png"))
        
        plt.figure(1)
        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.savefig(self.checkpoint_folder / Path("epoch_losses.png"))
        

    def predict(self, X, threshold=0.5):
        pass

    def fit(self, X, y_train):
        pass