import torch

from dataloaders.KEATSBase import KEATSBase


class OneHotKEATS(KEATSBase):
    """
    OHV representation of KEATS dataset.
    Extends from KEATSBase overwriting the implementation of 
    create_dataset.
    """    

    def create_onehot_record(self, sid):
        """
        Creates record for each student, <number_of_activities, 2> first index 
        is the index of event second is the index of time-delta.

        @param sid : Student Id.
        
        @return Tuple<Tensor> where first is the OHV input and second tensor is
                target value.
        """
        student_activities = self.logs[self.logs['Id'] == sid]
        if len(student_activities) == 0:
            return None
        student_activities = student_activities.sort_values(by=["Time"]).reset_index(drop=True)

        target = self.get_final_mark(sid)
        
        index_input = torch.zeros((len(student_activities), 2))
        
        for i, each_activity in student_activities.iterrows():
            event = ''
            for each_column in self.columns:
                event = event + each_activity[each_column] + '-'
            event = event[:-1]
            event_time = each_activity['Time']
            event_index = self.convert_event_to_index(event)

            # OHV representation combines the vector into single vector
            # so index is added on top of the total number of unique
            # events
            event_time_index = len(self.unique_events) + self.convert_time_to_index(event_time)
            
            index_input[i, 0] = event_index
            index_input[i, 1] = event_time_index
        
        return (index_input, target)

    def create_dataset(self):
        """
        Returns OHV representation of the KEATS dataset.
        Pads every record with max time step length, and stores lenght of each record separately
        so that it can be used when training.

        @return Tuple<Tensor> first tensor the input tensors and second
                is the target tensors.
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
        
        inputs = torch.zeros((len(indexes), max_t_step, 2))
        for i, ii in enumerate(indexes):
            # 0 should be padding
            inputs[i, :ii.size(0), :] = ii[:, :] + 1 

        targets = torch.cat(targets, axis=0)
        
        # for grades conver to LongTensor instead of FloatTensor
        # to work with NLLLoss
        if targets.max()>1:
            targets = targets.type(torch.LongTensor)
        
        return inputs.type(torch.LongTensor).to(self.device), targets.to(self.device)
