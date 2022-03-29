from models.LinearModel.LinearModelNoDropout import LinearModelNoDropout
from models.layers.linears import LinearWithDropout
from models.layers.LinearModel import  LinearModel


class LinearModelDropout(LinearModelNoDropout):
    """
    LinearModel with standard dropout.
    """

    def set_model(self, input_dim, hidden_dim, target_size):
        """
        Initialize LinearModel with a linear layer and standard dropout.

        @param input_dim   : Input dimension of the linear layer.
        @param hidden_dim  : Output dimension of the linear layer.
        @param target_size : Output dimension of LinearModel.
        """
        if self.drop_prob is None:
            raise Exception("DROP_PROB not specified in the config")
        layer = LinearWithDropout(input_dim, hidden_dim, self.drop_prob)
        self.model = LinearModel(layer, hidden_dim, target_size).to(self.device)
