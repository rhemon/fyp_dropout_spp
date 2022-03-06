from operator import index
import numpy as np
import pandas as pd
import datetime
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 


class OneHotKeatsDatasetProcessor:
    """
    Class to laod KEATS dataset as OHV input, although actualy output is essentially indexes of the onehot vector position
    as data will be used in a GritNet model that uses Embedding layer.

    Takes in columns type and output type from cfg
    Columns type to specify which fields to use. Output type Binary or multivariative (support not added yet) 

    TODO: Further breakdown to support BOW representation.
    TODO: add mode to apply balancing techniques

    """    

    def __init__(self, cfg):

        self.START_TIME = datetime.datetime(2021, 4, 1, 0, 0)
        self.TIME_OHV_LENGTH = 788
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

        if columns_type == "OHV_ALL":
            self.columns = ["Event context", "Component", "Event name"]
            self.unique_events = sorted(list((event_contexts + '-' + component + '-'+ event_name).unique()))
        elif columns_type == "OHV_CONTEXT":
            self.columns = ["Event context"]
            self.unique_events = sorted(list(event_contexts.unique()))
        elif columns_type == "OHV_NAME":
            self.columns = ["Event name"]
            self.unique_events = sorted(list(event_name.unique()))
        elif columns_type == "OHV_CN":
            self.columns = ["Event context", "Event name"]
            self.unique_events = sorted(list((event_contexts + '-'+ event_name).unique()))
        else:
            raise Exception(f"Unsupported {columns_type} representation of KEATS dataset")
            
    def convert_time_to_index(self, event_time):
        delta = event_time - self.START_TIME
        hours = (delta.total_seconds())/3600.0
        index = int(hours//6)
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

    def create_onehot_record(self, sid):
        """
        Creates record for each student, <number_of_activities, 2> first index is the index of event
        second is the index of time.
        """
        student_activities = self.logs[self.logs['Id'] == sid]
        if len(student_activities) == 0:
            return None
        student_activities = student_activities.sort_values(by=["Time"]).reset_index(drop=True)

        # For now only taking final mark in consideration
        target = self.get_final_mark(sid)
        
        index_input = torch.zeros((len(student_activities), 2))
        
        for i, each_activity in student_activities.iterrows():
            event = ''
            for each_column in self.columns:
                event = event + each_activity[each_column] + '-'
            event = event[:-1]
            event_time = each_activity['Time']
            event_index = self.convert_event_to_index(event)
            event_time_index = len(self.unique_events) + self.convert_time_to_index(event_time)
            
            index_input[i, 0] = event_index
            index_input[i, 1] = event_time_index
        
        return (index_input, target)

    def create_onehot_dataset(self):
        """
            Pads every record with max time step length, and stores lenght of each record separately
            so that it can be used when training.
        """
        ids = self.marks['Id'].unique()

        indexes = []
        targets = []
        max_t_step = 0
        for i in ids:
            step = self.create_onehot_record(i)
            if step is not None:
                ii, y = step
                indexes.append(ii)
                if ii.size(0) > max_t_step:
                    max_t_step = ii.size(0)
                targets.append(y)
        lengths = [] 
        inputs = torch.zeros((len(indexes), max_t_step, 2))
        lengths = torch.zeros((len(indexes),))
        for i, ii in enumerate(indexes):
            inputs[i, :ii.size(0), :] = ii[:, :] + 1 # 0 should be padding
            lengths[i] = ii.size(0)

        targets = torch.cat(targets, axis=0)
        
        return inputs, lengths, targets


    def load_dataset(self, test_split_ratio=0.2):
        
        if self.load_method == "ORIGINAL":
            x_train, x_test, l_train, l_test, y_train, y_test = self.prep_dataset_original()
        elif self.load_method == "OVERSAMPLE_NON_MAX":
            x_train, x_test, l_train, l_test, y_train, y_test = self.prep_dataset_w_oversample_non_max()
        else:
            raise Exception(f"Support method for {self.load_method} not added yet!")

        x_train, l_train, y_train = shuffle(x_train, l_train, y_train, random_state=0)
        x_test, l_test, y_test = shuffle(x_test, l_test, y_test, random_state=0)

        train_dataset = torch.utils.data.TensorDataset(x_train[:,:,0], x_train[:,:,1], l_train, y_train)
        

        test_dataset = torch.utils.data.TensorDataset(x_test[:,:,0], x_test[:,:,1], l_test, y_test)
        # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1)

        return train_dataset, test_dataset   

    def prep_dataset_original(self, test_split_ratio=0.2):
        """
        Returns a dataset loader that also separates the indexes as two separate input
        so can be treated separately by the model.
        """
        datasets = self.create_onehot_dataset()
        classes = list(datasets[-1].unique())

        x_train = []
        x_test = []
        l_train = []
        l_test = []
        y_train = []
        y_test = []

        for each_class in classes:
            class_sets = []
            for each_dataset in datasets:
                class_samples = each_dataset[datasets[-1] == each_class]
                class_sets.append(class_samples)
                
            split_train_test_set = train_test_split(*class_sets, test_size=test_split_ratio, random_state=0)
            x_train.append(split_train_test_set[0])
            x_test.append(split_train_test_set[1])
            l_train.append(split_train_test_set[2])
            l_test.append(split_train_test_set[3])
            y_train.append(split_train_test_set[4])
            y_test.append(split_train_test_set[5])
            
        x_train = torch.cat(x_train)
        x_test = torch.cat(x_test)
        l_train = torch.cat(l_train)
        l_test = torch.cat(l_test)
        y_train = torch.cat(y_train)
        y_test = torch.cat(y_test)

        print(f"Train size: {x_train.shape[0]}\nTest Size: {x_test.shape[0]}")

        return x_train, x_test, l_train, l_test, y_train, y_test

    def prep_dataset_w_oversample_non_max(self, test_split_ratio=0.2):
        datasets = self.create_onehot_dataset()
        classes = list(datasets[-1].unique())

        x_train = []
        x_test = []
        l_train = []
        l_test = []
        y_train = []
        y_test = []

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
        
        for i in range(len(all_class_sets)):
            num_to_duplicate = max_sample_per_class - all_class_sets[i][0].shape[0]
            if num_to_duplicate > 0:
                indexes_to_duplicate = np.random.choice(np.arange(all_class_sets[i][0].shape[0]), size=num_to_duplicate)
                for j in range(len(all_class_sets[i])):
                    all_class_sets[i][j] = torch.cat([all_class_sets[i][j], all_class_sets[i][j][indexes_to_duplicate]])
        
        for class_sets in all_class_sets:
            split_train_test_set = train_test_split(*class_sets, test_size=test_split_ratio, random_state=0)
            x_train.append(split_train_test_set[0])
            x_test.append(split_train_test_set[1])
            l_train.append(split_train_test_set[2])
            l_test.append(split_train_test_set[3])
            y_train.append(split_train_test_set[4])
            y_test.append(split_train_test_set[5])
            
        x_train = torch.cat(x_train)
        x_test = torch.cat(x_test)
        l_train = torch.cat(l_train)
        l_test = torch.cat(l_test)
        y_train = torch.cat(y_train)
        y_test = torch.cat(y_test)

        print(f"Train size: {x_train.shape[0]}\nTest Size: {x_test.shape[0]}")
        
        return x_train, x_test, l_train, l_test, y_train, y_test
        