import torch

from dataloaders.KEATSBase import KEATSBase

class BoWKEATS(KEATSBase):

    def create_bow_record(self, sid):
        student_activities = self.logs[self.logs['Id'] == sid]

        target = self.get_final_mark(sid)
        
        bow = torch.zeros((len(self.unique_events),))   

        for _, each_activity in student_activities.iterrows():    
            event = ''
            for each_col in self.columns:
                event = event + each_activity[each_col] + '-'
            event = event[:-1]
            event_index = self.convert_event_to_index(event)
            bow[event_index] +=  1.0
        
        return (bow, target)
        
    def create_dataset(self):
        
        ids = self.marks['Id'].unique()
        
        bows = []
        targets = []
        for sid in ids:
            step = self.create_bow_record(sid)
            if step is not None:
                bows.append(step[0])
                targets.append(step[1])
        
        bows = torch.stack(bows, axis=0)
        targets = torch.cat(targets, axis=0)
        
        return bows, targets
    
    def load_dataset(self, test_split_ratio=0.2):
        dataset_train, dataset_test = super().load_dataset(test_split_ratio)
        return *dataset_train, *dataset_test