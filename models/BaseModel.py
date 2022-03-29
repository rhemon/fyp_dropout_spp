import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim


class BaseModel:
    """
    BaseModel implementing commnon functionalities
    of all models.
    """
    
    def __init__(self, cfg, checkpoint_folder, dataprocessor):
        """
        Model constructor.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param dataprocessor : Data Loader
        """
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
        self.dataprocessor = dataprocessor
        
    def set_model(self, **kwargs):
        """
        Set model must be implemented in inhertied classes. Expected
        to intialize the neural network model here.
        """
        raise Exception("Not Implemented")

    def update_per_iter(self):
        """
        Called at the end of every iteration in train loop. Expected to 
        be used by models using gradient based dropout.
        """
        pass
    
    def update_per_epoch(self):
        """
        Called at the end of every epoch in train loop. Expected to 
        be used by models using gradient based dropout.
        """
        pass

    def train(self, train_loader):
        """
        Train loop.
        
        @param train_loader : DataLoader object model fits on.
        """
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

                self.update_per_iter()
                
                self.optimizer.step()
                print_track_step += 1
                if print_track_step % self.PRINT_EVERY == 0:
                    print(f'epoch: {epoch + 1} step: {batch_idx + 1}/{len(train_loader)} loss: {loss}')
                    print_track_step = 0
            
            self.update_per_epoch()
            
            # per epoch checkpoint
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
        
        # Final checkpoint save
        checkpoint_name = self.checkpoint_folder / Path("model.pt")
        torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'all_losses': all_losses,
                    'epoch_losses': epoch_losses,
                    }, checkpoint_name)

        # Plot loss over time
        plt.figure(0)          
        plt.plot([i for i in range(len(all_losses))], all_losses)
        plt.savefig(self.checkpoint_folder / Path("all_losses.png"))
        
        plt.figure(1)
        plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
        plt.savefig(self.checkpoint_folder / Path("epoch_losses.png"))

    def predict(self, X, threshold=0.5):
        """
        Predict method to be implemneted by inherited
        classes.
        @param X         : input tensor
        @param threshold : Threshold to use for binary classification.

        @return Tensor with predicted class values.
        """
        pass

    def fit(self, X, y_train):
        """
        Fit method to be implemented by inherited classes.
        
        @param X       : Input tensors
        @param y_train : Target tensors
        """
        pass
