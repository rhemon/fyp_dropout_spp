from tabnanny import check
import torch
import pandas as pd
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from pathlib import Path

from dataloaders.BaseLoader import BaseLoader 

class Sentiment140(BaseLoader):

    def __init__(self, cfg, checkpoint_folder):
        super(Sentiment140, self).__init__(cfg, checkpoint_folder)
        
        self.data = pd.read_csv("raw_data_sets/Sentiment140/cleaned_data.csv", encoding="utf-8")[['target', 'text']]
        self.data['text'] = self.data['text'].astype("str")

        try:
            self.tokenizer_path = cfg.TOKENIZER_PATH
        except AttributeError:
            self.tokenizer_path = None        

    
    def get_tokenizer(self, text):

        # load if config specified a file
        if self.tokenizer_path is not None:
            with open(self.tokenizer_path, 'rb') as handle:
                return pickle.load(handle)
        
        # create a tokenizer
        tokenizer = Tokenizer()
        # fit the tokenizer in the train text
        tokenizer.fit_on_texts(text)

        # saving tokenizer
        with open("raw_data_sets/Sentiment140/" + self.checkpoint_folder.stem + ".pickle", 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        return tokenizer
    
    def create_dataset(self):
        
        # create a label encoder
        encoder = LabelEncoder()
        # enconde labels (0 or 1) in train data
        encoder.fit(self.data['target'].to_list())

        # transform labels in y_train and y_test data to the encoded ones
        y = encoder.transform(self.data['target'].to_list())
        
        # reshape y_train and y_test data
        y = y.reshape(-1, 1)

        print("Tokenizing text...")
        tokenizer = self.get_tokenizer(self.data['text'])

        # get max length of the train data
        max_length = max([len(s.split()) for s in self.data['text']])

        # pad sequences in x_train data set to the max length
        x = pad_sequences(tokenizer.texts_to_sequences(self.data['text']),
                                maxlen = max_length)
        print("...Tokenized")

        print("x shape: ", x.shape)
        print("y shape:", y.shape)

        return (torch.tensor(x).type(torch.LongTensor).to(self.device), 
                torch.tensor(y).type(torch.FloatTensor).to(self.device).squeeze())
            
    def load_dataset(self, test_split_ratio=0.2):
        train, test = super().load_dataset(test_split_ratio)
        return train[0], train[1], test[0], test[1]