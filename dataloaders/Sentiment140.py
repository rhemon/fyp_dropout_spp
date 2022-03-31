from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import torch

from dataloaders.BaseLoader import BaseLoader 


class Sentiment140(BaseLoader):
    """
    Sentiment140 dataset loader. Extended from BaseLoader.

    Follows Karina Reyes's article on Sentiment Analysis on Tweet using LSTM
    to clean up the dataset and tokenize. (set_tokenizer and create_dataset)
    https://medium.com/@karyrs1506/sentiment-analysis-on-tweets-with-lstm-22e3bbf93a61
 
    """

    def __init__(self, cfg, checkpoint_folder):
        """
        Sentiment140 Constructor. Sets up base attributes and tokenizer path.

        @param cfg               : SimpleNamespace object created from json object 
        @param checkpoint_folder : None or Path object specifying path of foldre where checkpoints are saved.
        """
        super(Sentiment140, self).__init__(cfg, checkpoint_folder)
        
        # pre-cleaned data by following Reyes's code (code for this is in the data set notebook)
        # https://medium.com/@karyrs1506/sentiment-analysis-on-tweets-with-lstm-22e3bbf93a61
        self.data = pd.read_csv("raw_data_sets/Sentiment140/cleaned_data.csv", encoding="utf-8")[['target', 'text']]
        self.data['text'] = self.data['text'].astype("str")

        try:
            self.tokenizer_path = cfg.TOKENIZER_PATH
        except AttributeError:
            self.tokenizer_path = None        
    
    def set_tokenizer(self, text):
        """
        Set tokenizer by loading existing tokenzier if path provided.
        Else create new tokenizer and fit on the text provided. Saved in the 
        raw_data_sets/Sentiment140/ folder so that it can be reused later.

        @param text : Array of text from the DataFrame loaded from csv.
        """
        if self.tokenizer_path is not None:
            with open(self.tokenizer_path, 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        else:
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_texts(text)
            with open("raw_data_sets/Sentiment140/" + self.checkpoint_folder.stem + ".pickle", 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    def create_dataset(self):
        """
        Creates dataset from the cleaned version. Uses the tokenizer
        to convert text to sequence and pads them.
        Converts 0, 4 label to 0, 1 label.

        @return Tuple<Tensor> where first is the input tensor and second is
                the target tensor.
        """
        
        encoder = LabelEncoder()
        encoder.fit(self.data['target'].to_list())

        y = encoder.transform(self.data['target'].to_list())
        
        y = y.reshape(-1, 1)

        print("Tokenizing text...")
        self.set_tokenizer(self.data['text'])

        max_length = max([len(s.split()) for s in self.data['text']])
        x = pad_sequences(self.tokenizer.texts_to_sequences(self.data['text']),
                                maxlen = max_length)
        print("...Tokenized")

        return (torch.tensor(x).type(torch.LongTensor).to(self.device), 
                torch.tensor(y).type(torch.FloatTensor).to(self.device).squeeze())
