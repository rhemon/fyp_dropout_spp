import torch

from dataloaders.KEATSBase import KEATSBase


class BoWKEATS(KEATSBase):
    """
    Bag of Words (BoW) representation of KEATS dataset.
    Inherited from KEATSBase and overwrites
    create_dataset implemnetation.
    """

    def create_bow_record(self, sid):
        """
        Return records of a single student in BoW representation.

        @param sid: Student ID 

        @return Tuple<Tensor> where it is two tensor, first is the 1D input
                tensor and second is the target tensor with a single value.
        """
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
        """
        Create dataset over written to loop through each student
        and generate the BoW representation of the complete dataset.

        @return Tuple<Tensor> where first is the BoW input tensor and second
                is the target tensor. Both tensor are loaded onto GPU if
                available.
        """
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
        
        if targets.max()>1:
            targets = targets.type(torch.LongTensor)
        return bows.to(self.device), targets.to(self.device)
    