

from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithDropout
from models.layers.LinearModel import  LinearModel

class LinearModelDropout(LinearModelNoDropout):
    def set_model(self, input_dim, hidden_dim, target_size):
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
        layer = LinearWithDropout(input_dim, hidden_dim, self.drop_prob)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)
