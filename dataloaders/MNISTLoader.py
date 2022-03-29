import torch
import torchvision.datasets as datasets
from torchvision.transforms import transforms 

from dataloaders.BaseLoader import BaseLoader


class MNISTLoader(BaseLoader):
    """
    MNISTLoader extending from BaseLoader to load
    MNIST dataset from PyTorch's datasets.
    """

    def __init__(self, cfg, checkpoint_folder=None):
        """
        MNISTLoader Constructor. Sets up base attributes and loads train and test MNIST
        dataset from PyTorch datasets.

        @param cfg               : SimpleNamespace object created from json object 
        @param checkpoint_folder : None or Path object specifying path of foldre where checkpoints are saved.
        """
        super(MNISTLoader, self).__init__(cfg)

        self.train_data = datasets.MNIST(root='raw_data_sets/', train=True, transform=transforms.ToTensor(), download=True)
        self.test_data = datasets.MNIST(root='raw_data_sets/', train=False, transform=transforms.ToTensor(), download=True)

    def create_dataset(self):
        """
        Flattens the 2D images into 1D and combines PyTorch's
        train and test MNIST dataset into 1. Letting BaseLoader's
        load_dataset handle the train test split for this project.

        @return Tuple<Tensor> where first is the inpput tensor and second is the
                target tensor labeled from 0-9. Loaded into GPU if available.
        """
        x_train, y_train = self.train_data.data, self.train_data.targets
        x_test, y_test = self.test_data.data, self.test_data.targets

        x_train = x_train.view(x_train.shape[0], -1).type(torch.FloatTensor)
        x_test = x_test.view(x_test.shape[0], -1).type(torch.FloatTensor)

        return torch.cat([x_train, x_test]).to(self.device), torch.cat([y_train, y_test]).type(torch.LongTensor).to(self.device)
