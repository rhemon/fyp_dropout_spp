import torch

from models.layers.lstms import BLSTM 
from models.layers.SimpleSentiment import SimpleSentiment
from models.BaseModel import BaseModel

class SimpleSentimentNoDropout(BaseModel):

    def __init__(self, cfg, train_path, input_dim, dataprocessor, vocab_size=290713, embedding_dim=64, hidden_dim=128, target_size=1, seq_len=50, **kwargs):
        super(SimpleSentimentNoDropout, self).__init__(cfg, train_path, dataprocessor)
        
        self.set_model(vocab_size, embedding_dim, hidden_dim, target_size, seq_len)

    def set_model(self, vocab_size, embedding_dim, hidden_dim, target_size, seq_len):
        blstm_layer = BLSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True).to(self.device)
        self.model = SimpleSentiment(blstm_layer, vocab_size, embedding_dim, hidden_dim, target_size, seq_len).to(self.device)

    def predict(self, X, threshold=0.5):
        y_preds = []

        with torch.no_grad():
            self.model.eval()    
            for i in range(0,X.shape[0], self.batch_size):
                y_preds.append(self.model((X[i:min(X.shape[0], i+self.batch_size)],)))

        return torch.squeeze(torch.cat(y_preds,dim=0).cpu() > threshold, 1)

    def fit_embedding_weights(self):
        vocab = self.dataprocessor.word_index
        with open('raw_data_sets\\Sentiment140\\glove.twitter.27B.200d.txt', 'rt', encoding='utf-8') as f:
            content = f.read().strip().split('\n')
        glove = {}
        for line in content:
            word_embedding = line.split(' ')
            glove[word_embedding[0]] = [float(each_embed) for each_embed in word_embedding[1:]]
        
        embedding_weights = torch.zeros((len(vocab)+1, 200))
        for word, i in vocab.items():
            embedding_weights[i] = torch.tensor(glove.get(word))
        
        self.model.embeddings.weight = embedding_weights

    def fit(self, X, y_train):
        
        self.fit_embedding_weights()

        train_dataset = torch.utils.data.TensorDataset(X, y_train)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size)

        self.train(train_loader)
