import torch

from models.layers.lstms import PerStepBLSTM 
from models.layers.GritNet import GritNet
from models.BaseModel import BaseModel


class GritNetNoDropout(BaseModel):
    """
    GritNet model with no dropout.
    """

    def __init__(self, cfg, train_path, input_dim, dataprocessor, event_dim=1113, embedding_dim=2048, hidden_dim=128, **kwargs):
        """
        Model constructor.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        @param event_dim     : Event dimension, defaulted to 1113.
        @param embedding_dim : Embedding dimension. Defaulted to 2048.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        """
        super(GritNetNoDropout, self).__init__(cfg, train_path, dataprocessor)

        target_size = 1
        if cfg.OUTPUT_TYPE == "GRADE":
            target_size = 5

        self.set_model(event_dim, embedding_dim, hidden_dim, target_size)

    def set_model(self, event_dim, embedding_dim, hidden_dim, target_size):
        """
        Initialize GritNet model with just PerStepBLSTM layer.
        
        @param event_dim     : Event dimension, defaulted to 1113.
        @param embedding_dim : Embedding dimension. Defaulted to 2048.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        blstm_layer = PerStepBLSTM(embedding_dim*2, hidden_dim, None).to(self.device)
        self.model = GritNet(blstm_layer, event_dim, embedding_dim, hidden_dim, target_size).to(self.device)

    def predict(self, X, threshold=0.5):
        """
        Predict class for given input using given threshold for Binary classification.
        For multi class argmax is used to determine the class.

        @param X         : Input tensor combining both event and time-delta indexes.
        @param threshold : Defaulted to 0.5. If model's output higher than threshold then considered pass.
                           Used only for binary classification.

        @return Tensor with predicted class values.
        """
        x_test = X
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,x_test.shape[0], 32):
                y_preds.append(self.model((x_test[i:min(x_test.shape[0],i+32), :, 0], 
                                           x_test[i:min(x_test.shape[0],i+32),:,1])))
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
        x_train = X

        train_dataset = torch.utils.data.TensorDataset(x_train[:,:,0], x_train[:,:,1], y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
