import torch

from models.layers.lstms import PerStepBLSTM 
from models.layers.SimpleSentiment import SimpleSentiment
from models.BaseModel import BaseModel


class SimpleSentimentNoDropout(BaseModel):
    """
    SimpleSentiment with no droput
    """

    def __init__(self, cfg, train_path, input_dim, dataprocessor, vocab_size=335507, embedding_dim=200, hidden_dim=128, target_size=1, seq_len=50, **kwargs):
        """
        Model constructor.

        @param cfg           : SimpleNamespace object which is the configuraiton intialized from json file.
        @param train_path    : Path to checkpoint folder
        @param input_dim     : Model's input dimension
        @param dataprocessor : Data Loader
        @param vocab_size    : Vocab dimension, defaulted to 335507.
        @param embedding_dim : Embedding dimension. Defaulted to 200.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Model output size. Defaulted to 1.
        @param seq_len       : Sequence Length. Defaulted to 50.
        """
        super(SimpleSentimentNoDropout, self).__init__(cfg, train_path, dataprocessor)
        
        self.set_model(vocab_size, embedding_dim, hidden_dim, target_size, seq_len)

    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        """
        Initialize SimpleSentiment model with just PerStepBLSTM layer.
        
        @param vocab_size     : Event dimension, defaulted to 335507.
        @param embedding_dim : Embedding dimension. Defaulted to 200.
        @param hidden_dim    : BLSTM output dimension. Defaulted to 128.
        @param target_size   : Output dimension of the model.
        """
        blstm_layer = PerStepBLSTM(embedding_dim, hidden_dim, None).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)

    def predict(self, X, threshold=0.5):
        """
        Predict class for given input using given threshold.

        @param X         : Input tensor is expected to be tokenized text.
        @param threshold : Defaulted to 0.5. If model's output higher than threshold then considered pass.

        @return Tensor with predicted class values.
        """
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,X.shape[0], 512):
                y_preds.append(self.model((X[i:min(X.shape[0], i+512)],)))

        return torch.squeeze(torch.cat(y_preds,dim=0).cpu() > threshold, 1)

    def fit_embedding_weights(self):
        """
        Load GloVe embedding weights.
        
        Following Reyes's article loading pre-trained GloVe embedding
        weights in to SimpleSentiment's embedding layer.
        https://medium.com/@karyrs1506/sentiment-analysis-on-tweets-with-lstm-22e3bbf93a61
        """
        vocab = self.dataprocessor.tokenizer.word_index
        with open('raw_data_sets/Sentiment140/glove.twitter.27B.200d.txt', 'rt', encoding='utf-8') as f:
            content = f.read().strip().split('\n')
        glove = {}
        for line in content:
            word_embedding = line.split(' ')
            glove[word_embedding[0]] = [float(each_embed) for each_embed in word_embedding[1:]]
        
        embedding_weights = torch.zeros((len(vocab)+1, 200))
        for word, i in vocab.items():
            embeds = glove.get(word)
            if embeds is not None:
                embedding_weights[i] = torch.tensor(glove.get(word))
        
        self.model.embeddings.weight = torch.nn.Parameter(embedding_weights.to(self.device))

    def fit(self, X, y_train):
        """
        Given input and target values, train model on it.
        Before training loads GloVe Embedding weights.

        @param X       : Input tensor.
        @param y_train : Target tensor.
        """
        self.fit_embedding_weights()

        train_dataset = torch.utils.data.TensorDataset(X, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
