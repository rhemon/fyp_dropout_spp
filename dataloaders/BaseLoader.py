from xml.dom.minidom import Attr
import numpy as np
import pandas as pd
import datetime
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 


class BaseLoader:
    """
    Class to laod KEATS dataset as OHV input, although actualy output is essentially indexes of the onehot vector position
    as data will be used in a GritNet model that uses Embedding layer.

    Takes in columns type and output type from cfg
    Columns type to specify which fields to use. Output type Binary or multivariative (support not added yet) 

    """    

    def __init__(self, cfg, checkpoint_folder=None):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_method = cfg.LOAD_METHOD
        try:
            self.class_sizes = cfg.CLASS_SPLIT
        except AttributeError:
            self.class_sizes = None
        
        
        self.checkpoint_folder = checkpoint_folder


    def create_dataset(self):
        pass
    
    def split_class_sets(self, all_class_sets, test_split_ratio=0.2):
        train_sets = []
        test_sets = []
        for class_sets in all_class_sets:
            split = train_test_split(*class_sets, test_size=test_split_ratio, random_state=0)
            train_split = []
            test_split = []
            for i in range(0, len(split), 2):
                train_split.append(split[i])
                test_split.append(split[i+1])
            train_sets.append(train_split)
            test_sets.append(test_split)
        return train_sets, test_sets
            
    def oversample(self, all_class_sets, max_sample_per_class):
        for i in range(len(all_class_sets)):
            num_to_duplicate = max_sample_per_class - all_class_sets[i][0].shape[0]
            if num_to_duplicate > 0:
                indexes_to_duplicate = np.random.choice(np.arange(all_class_sets[i][0].shape[0]), size=num_to_duplicate)
                for j in range(len(all_class_sets[i])):
                    all_class_sets[i][j] = torch.cat([all_class_sets[i][j], all_class_sets[i][j][indexes_to_duplicate]])
        return all_class_sets
    
    def combine_classes(self, all_class_sets, controlled=False):
        dataset = [[] for i in range(len(all_class_sets[0]))]
        
        for i, class_set in enumerate(all_class_sets):
            if controlled and self.class_sizes is not None:
                class_size = self.class_sizes[i] 
                indexes = np.random.choice(np.arange(class_set[0].shape[0]), size=class_size)
            else:
                indexes = np.arange(class_set[0].shape[0])

            for j, each in enumerate(class_set):
                dataset[j].append(each[indexes])
        
        for i in range(len(dataset)):
            dataset[i] = torch.cat(dataset[i])

        return dataset

    def load_dataset(self, test_split_ratio=0.2):
        datasets = self.create_dataset() ## [x, y]
        classes = list(datasets[-1].unique())
        
        ## for each class seprate [x,y] in all_class_sets
        all_class_sets = [] 
        max_sample_per_class = 0
        for each_class in classes:
            class_sets = []
            for each_dataset in datasets:
                class_samples = each_dataset[datasets[-1] == each_class]
                class_sets.append(class_samples)
            all_class_sets.append(class_sets)
            
            if class_sets[0].shape[0] > max_sample_per_class:
                max_sample_per_class = class_sets[0].shape[0]

        train_sets, test_sets = self.split_class_sets(all_class_sets, test_split_ratio)
        
        if self.load_method == "OVERSAMPLE_NON_MAX_TRAIN":
            max_sample_per_class = int(max_sample_per_class * (1-test_split_ratio))
            train_sets = self.oversample(train_sets[:], max_sample_per_class)

        dataset_train = shuffle(*self.combine_classes(train_sets, True), random_state=0)
        dataset_test = shuffle(*self.combine_classes(test_sets), random_state=0)

        print("Train:", dataset_train[0].shape[0])
        print("Test:", dataset_test[0].shape[0])
        print("Max sample per class:", max_sample_per_class)

        return dataset_train, dataset_test
