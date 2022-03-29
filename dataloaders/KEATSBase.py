import datetime
import pandas as pd
import torch

from dataloaders.BaseLoader import BaseLoader 


class KEATSBase(BaseLoader):
    """
    Class to laod KEATS dataset.
    Implements base methods that are common for both
    OHV and BoW representation.
    """    

    def __init__(self, cfg, checkpoint_folder=None):
        """
        KEATSBase Constructor. Sets up base attributes and output type.

        @param cfg               : SimpleNamespace object created from json object 
        @param checkpoint_folder : None or Path object specifying path of foldre where checkpoints are saved.
        """
        super(KEATSBase, self).__init__(cfg)
        self.START_TIME = datetime.datetime(2021, 4, 1, 0, 0)
        self.TIME_OHV_LENGTH = 788
        self.HOUR_PER_INDEX = 6

        self.logs = pd.read_csv('raw_data_sets/KEATS Dataset/KEATS_logs.csv')
        self.logs['Time'] = pd.to_datetime(self.logs['Time'], dayfirst=True)

        self.marks = pd.read_csv('raw_data_sets/KEATS Dataset/finalMarksv8.csv')
        self.set_grades(self.marks['Final']) 
        
        self.output_type = cfg.OUTPUT_TYPE
        
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
        """
        Convert time to a index value marking the position of it in the time-delta
        vector.

        @param event_time : DateTime object which is the time stamp of the activity.

        @return int value which is the time-delta vector's index.
        """
        delta = event_time - self.START_TIME
        hours = (delta.total_seconds())/3600.0
        index = int(hours//self.HOUR_PER_INDEX)
        return index

    def convert_event_to_index(self, event):
        """
        Convert event to index. Returns the index at which event is located
        in the unique_events list.

        @param event : String object describing the activity.

        @return int value which is the unique events vector's index.
        """
        if event not in self.unique_events:
            raise Exception("Event", event, "passed doesn't seem to exist in the UNIQUE EVENT array")
        return self.unique_events.index(event)    
    
    def set_grades(self, marks):
        """
        Convert marks passed to grades.

        @param marks: Array of final marks

        @return Tensor of grades labeled 0-4.
        """
        self.grades = torch.zeros(marks.shape)
        
        self.grades[marks >= 40] = 1
        self.grades[marks >= 50] = 2
        self.grades[marks >= 60] = 3
        self.grades[marks >= 70] = 4

        return self.grades

    def get_final_mark(self, sid):
        """
        For given student id return final mark. If output type
        set to BINARY returns 0 (fail) or 1 (pass). Otherwise
        returns grade (labeled between 0-4 (fail - first class)).

        @param sid: Student id
        
        @return Tensor which is a single target value for the model.
        """
        if self.output_type == "BINARY":
            target = torch.zeros((1,))
            target[0] = int((self.marks[self.marks['Id'] == sid ].get('Final') >= 40).item())
        elif self.output_type == "GRADE":
            return self.grades[self.marks['Id'] == sid]
        else: 
            raise Exception(f"Unsupported output type: {self.output_type}")
        
        return target
