from xml.dom.minidom import Attr
import numpy as np
import pandas as pd
import datetime
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from dataloaders.BaseLoader import BaseLoader 


class KEATSBase(BaseLoader):
    """
    Class to laod KEATS dataset as OHV input, although actualy output is essentially indexes of the onehot vector position
    as data will be used in a GritNet model that uses Embedding layer.

    Takes in columns type and output type from cfg
    Columns type to specify which fields to use. Output type Binary or multivariative (support not added yet) 

    """    

    def __init__(self, cfg, checkpoint_folder=None):
        super(KEATSBase, self).__init__(cfg)
        self.START_TIME = datetime.datetime(2021, 4, 1, 0, 0)
        self.TIME_OHV_LENGTH = 788
        try:
            self.HOUR_PER_INDEX = cfg.HOUR_PER_INDEX
        except AttributeError:
            self.HOUR_PER_INDEX = 6

        self.logs = pd.read_csv('raw_data_sets/KEATS Dataset/KEATS_logs.csv')
        self.marks = pd.read_csv('raw_data_sets/KEATS Dataset/finalMarksv8.csv')
        self.set_grades(self.marks['Final'])
        self.output_type = cfg.OUTPUT_TYPE

        self.logs['Time'] = pd.to_datetime(self.logs['Time'], dayfirst=True)

        self.columns = []
        event_contexts = self.logs['Event context']
        component = self.logs['Component']
        event_name = self.logs['Event name']

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
    
    def set_grades(self, marks):
        self.grades = torch.zeros(marks.shape)
        
        self.grades[marks >= 40] = 1
        self.grades[marks >= 50] = 2
        self.grades[marks >= 60] = 3
        self.grades[marks >= 70] = 4

        return self.grades

    def get_final_mark(self, sid):
        if self.output_type == "BINARY":
            target = torch.zeros((1,))
            target[0] = int((self.marks[self.marks['Id'] == sid ].get('Final') >= 40).item())
        elif self.output_type == "GRADE":
            return self.grades[self.marks['Id'] == sid]
        else: 
            raise Exception(f"Unsupported output type: {self.output_type}")
        return target

