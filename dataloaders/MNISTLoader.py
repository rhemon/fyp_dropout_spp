import torch
import torchvision.datasets as datasets
from torchvision.transforms import transforms 

from dataloaders.BaseLoader import BaseLoader

class MNISTLoader(BaseLoader):

    def __init__(self, cfg, checkpoint_folder=None):
        super(MNISTLoader, self).__init__(cfg)

        self.train_data = datasets.MNIST(root='raw_data_sets/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='raw_data_sets/', train=False, transform=transforms.ToTensor(), download=True)


    def create_dataset(self):
        
        x_train, y_train = self.train_data.data, self.train_data.targets
        x_test, y_test = self.test_data.data, self.test_data.targets

        x_train = x_train.view(x_train.shape[0], -1).type(torch.FloatTensor)
        x_test = x_test.view(x_test.shape[0], -1).type(torch.FloatTensor)

        return torch.cat([x_train, x_test]).to(self.device), torch.cat([y_train, y_test]).type(torch.LongTensor).to(self.device)
    
    def load_dataset(self, test_split_ratio=0.2):
        train, test = super().load_dataset(test_split_ratio)
        return train[0], train[1], test[0], test[1]