import torch

from models.layers.linears import LinearWithNoDropout 
from models.layers.LinearModel import LinearModel
from models.BaseModel import BaseModel


class LinearModelNoDropout(BaseModel):
    """
    LinearModel with no dropout.
    """

    def __init__(self, cfg, train_path, input_dim, dataprocessor, hidden_dim=2048):
        """
        Model constructor.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        @param hidden_dim    : Linear layer's output dimension. Defaulted to 2048.
        """
        super(LinearModelNoDropout, self).__init__(cfg, train_path, dataprocessor)
        
        target_size = 1
        if cfg.OUTPUT_TYPE == "GRADE":
            target_size = 5
        elif cfg.OUTPUT_TYPE == "NUMBERS":
            target_size = 10
        self.set_model(input_dim, hidden_dim, target_size)

    def set_model(self, input_dim, hidden_dim, target_size):
        """
        Initialize LinearModel with a linear layer and no dropout.

        @param input_dim   : Input dimension of the linear layer.
        @param hidden_dim  : Output dimension of the linear layer.
        @param target_size : Output dimension of LinearModel.
        """
        layer = LinearWithNoDropout(input_dim, hidden_dim).to(self.device)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)

    def predict(self, X, threshold=0.5):
        """
        Predict class for given input using given threshold for Binary classification.
        For multi class argmax is used to determine the class.

        @param X         : Input tensor, expected to be 2D tensor. Batch size x Input dim.
        @param threshold : Defaulted to 0.5. If model's output higher than threshold then considered pass.
                           Used only for binary classification.

        @return Tensor with predicted class values.
        """
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
        """
        Given input and target values, train model on it.

        @param X       : Input tensor.
        @param y_train : Target tensor.
        """
        train_dataset = torch.utils.data.TensorDataset(X, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
