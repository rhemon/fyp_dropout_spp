import numpy as np
import pandas as pd
import datetime
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 


class KEATSBase:
    """
    Class to laod KEATS dataset as OHV input, although actualy output is essentially indexes of the onehot vector position
    as data will be used in a GritNet model that uses Embedding layer.

    Takes in columns type and output type from cfg
    Columns type to specify which fields to use. Output type Binary or multivariative (support not added yet) 

    TODO: Further breakdown to support BOW representation.
    

    """    

    def __init__(self, cfg):

        self.START_TIME = datetime.datetime(2021, 4, 1, 0, 0)
        self.TIME_OHV_LENGTH = 788
        try:
            self.HOUR_PER_INDEX = cfg.HOUR_PER_INDEX
        except AttributeError:
            self.HOUR_PER_INDEX = 6

        self.logs = pd.read_csv('raw_data_sets/KEATS Dataset/KEATS_logs.csv')
        self.marks = pd.read_csv('raw_data_sets/KEATS Dataset/finalMarksv8.csv')
        self.output_type = cfg.OUTPUT_TYPE

        self.logs['Time'] = pd.to_datetime(self.logs['Time'], dayfirst=True)

        self.columns = []
        event_contexts = self.logs['Event context']
        component = self.logs['Component']
        event_name = self.logs['Event name']
        
        self.load_method = cfg.LOAD_METHOD

        columns_type = cfg.COLUMNS_TYPE

        if "ALL" in columns_type:
            self.columns = ["Event context", "Component", "Event name"]
            self.unique_events = sorted(list((event_contexts + '-' + component + '-'+ event_name).unique()))
        elif "CONTEXT" in columns_type:
            self.columns = ["Event context"]
            self.unique_events = sorted(list(event_contexts.unique()))
        elif "NAME" in columns_type:
            self.columns = ["Event name"]
            self.unique_events = sorted(list(event_name.unique()))
        elif "CN" in columns_type:
            self.columns = ["Event context", "Event name"]
            self.unique_events = sorted(list((event_contexts + '-'+ event_name).unique()))
        else:
            raise Exception(f"Unsupported {columns_type} representation of KEATS dataset")
            
    def convert_time_to_index(self, event_time):
        delta = event_time - self.START_TIME
        hours = (delta.total_seconds())/3600.0
        index = int(hours//self.HOUR_PER_INDEX)
        return index

    def convert_event_to_index(self, event):
        if event not in self.unique_events:
            raise Exception("Event", event, "passed doesn't seem to exist in the UNIQUE EVENT array")
        return self.unique_events.index(event)    

    def get_final_mark(self, sid):
        if self.output_type == "BINARY":
            target = torch.zeros((1,))
            target[0] = int((self.marks[self.marks['Id'] == sid ].get('Final') >= 40).item())
        else:
            raise Exception(f"Unsupported output type: {self.output_type}")
        return target

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
    
    def combine_classes(self, all_class_sets):
        dataset = [[] for i in range(len(all_class_sets[0]))]
        for class_set in all_class_sets:
            for i, each in enumerate(class_set):
                dataset[i].append(each)
        
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
        
        if self.load_method == "OVERSAMPLE_NON_MAX_TRAIN":
            train_sets, test_sets = self.split_class_sets(all_class_sets, test_split_ratio)
            train_sets = self.oversample(train_sets[:], int(max_sample_per_class*(1-test_split_ratio)))
        elif self.load_method == "OVERSAMPLE_NON_MAX":
            all_class_sets = self.oversample(all_class_sets[:], max_sample_per_class)
            train_sets, test_sets = self.split_class_sets(all_class_sets, test_split_ratio)
        
        dataset_train = shuffle(*self.combine_classes(train_sets), random_state=0)
        dataset_test = shuffle(*self.combine_classes(test_sets), random_state=0)

        print("Train:", dataset_train[0].shape[0])
        print("Test:", dataset_test[0].shape[0])

        return dataset_train, dataset_test
