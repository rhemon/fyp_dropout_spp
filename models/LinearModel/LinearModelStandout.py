from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithStandout
from models.layers.LinearModel import  LinearModel


class LinearModelStandout(LinearModelNoDropout):
    """
    LinearModel with standout.
    """

    def set_model(self, input_dim, hidden_dim, target_size):
        """
        Initialize LinearModel with a linear layer and standout.

        @param input_dim   : Input dimension of the linear layer.
        @param hidden_dim  : Output dimension of the linear layer.
        @param target_size : Output dimension of LinearModel.
        """
        layer = LinearWithStandout(input_dim, hidden_dim)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)
