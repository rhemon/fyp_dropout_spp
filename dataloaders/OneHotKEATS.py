import numpy as np
import pandas as pd
import datetime
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 

from dataloaders.KEATSBase import KEATSBase

class OneHotKEATS(KEATSBase):
    """
    Class to laod KEATS dataset as OHV input, although actualy output is essentially indexes of the onehot vector position
    as data will be used in a GritNet model that uses Embedding layer.

    Takes in columns type and output type from cfg
    Columns type to specify which fields to use. Output type Binary or multivariative (support not added yet) 
    """    

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

    def create_dataset(self):
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
        if targets.max()>1:
            targets = targets.type(torch.LongTensor)
        return inputs, lengths, targets

    def load_dataset(self, test_split_ratio=0.2):
        dataset_train, dataset_test = super().load_dataset(test_split_ratio)
        # return ((dataset_train[0].type(torch.LongTensor).to(self.device), 
        #         dataset_train[1]), 
        #         dataset_train[2].to(self.device), 
        #         (dataset_test[0].type(torch.LongTensor).to(self.device), 
        #         dataset_test[1]), 
        #         dataset_test[2].to(self.device)) 
        return dataset_train[0].type(torch.LongTensor).to(self.device), dataset_train[2].to(self.device), dataset_test[0].type(torch.LongTensor).to(self.device), dataset_test[2].to(self.device)