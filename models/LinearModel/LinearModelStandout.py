

from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithStandout
from models.layers.LinearModel import  LinearModel

class LinearModelStandout(LinearModelNoDropout):
    def set_model(self, input_dim, hidden_dim, target_size):
        layer = LinearWithStandout(input_dim, hidden_dim)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)
