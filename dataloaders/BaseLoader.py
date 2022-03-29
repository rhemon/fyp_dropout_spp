import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle 
import torch


class BaseLoader:
    """
    BaseLoader class that sets some of the root functionalties
    that are common throughout all the loaders.

    Specifically the load_dataset method that performs the split
    for both class distribution and train-test. 
    """    

    def __init__(self, cfg, checkpoint_folder=None):
        """
        BaseLoader Constructor. Sets up load method, class sizes if specified in 
        configuration else set to None.
        
        And sets checkpoint folder to passed value.

        @param cfg               : SimpleNamespace object created from json object 
        @param checkpoint_folder : None or Path object specifying path of foldre where checkpoints are saved.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.load_method = cfg.LOAD_METHOD
        try:
            self.class_sizes = cfg.CLASS_SPLIT
        except AttributeError:
            self.class_sizes = None
        
        self.checkpoint_folder = checkpoint_folder

    def create_dataset(self):
        """
        Not implemented in Base class. Expected to be implemented in loaders
        that are used when training models.
        """
        pass
    
    def split_class_sets(self, all_class_sets, test_split_ratio=0.2):
        """
        Split data and target of each class into train and test set.
        By default, split ration is 80-20 for train-test.

        @param all_class_sets     : A list of list of tensors (List<List<Tensor>>) where each 
                                    List<Tensor> is tensors of a single class.
        @param test_split_ration  : Float, default to 0.2. Percentage of each class that goes
                                    into test set.

        @return Tuple<List<List<Tensor>>> where the tuple has two List<List<Tensor>> (train, test).
        """
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
        """
        Oversample data of each class to match max_sample_per_class

        @param all_class_sets       : A list of list of tensors (List<List<Tensor>>) where each 
                                      List<Tensor> is tensors of a single class.
        @param max_sample_per_class : A int value, where number of examples per class is
                                      sampled to match it.

        @return List<List<Tensor>> where each Tensor is oversampled to have 
                max_sample_per_class examples.
        """
        for i in range(len(all_class_sets)):
            num_to_duplicate = max_sample_per_class - all_class_sets[i][0].shape[0]
            if num_to_duplicate > 0:
                indexes_to_duplicate = np.random.choice(np.arange(all_class_sets[i][0].shape[0]), size=num_to_duplicate)
                for j in range(len(all_class_sets[i])):
                    all_class_sets[i][j] = torch.cat([all_class_sets[i][j], all_class_sets[i][j][indexes_to_duplicate]])
        return all_class_sets
    
    def combine_classes(self, all_class_sets, controlled=False):
        """
        Combine classes List<List<Tensor>> into single List<Tensor>.
        If controlled set to true then class sizes to sample number of examples
        specified in the configuration.

        @param all_class_sets : A list of list of tensors (List<List<Tensor>>) where each 
                                List<Tensor> is tensors of a single class.
        @param controlled     : Boolean value. Whether to sample according to specified
                                split or not.
        
        @return List<Tensor> combining all classes into single Tensor (normally just x, y).
        """
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
        """
        Load dataset and oversample and split accordingly if needed. If oversample set by 
        configaration then examples in train set are oversampled to have equal 
        distribution.Also won trian set if class split is specified then it is used
        to sample examples per class accordingly. 

        Test set set is kept same. Random state is set to 0so for all datasets 
        the split will be always same. Making comparision of results fair.

        @param test_split_ration : Floating value (default 0.2) specifying how much of
                                   full dataset goes into test set.
        
        @return Tuple<Tensor> where normally expected to be 4 tensors. 
                X_train, y_train, X_test, y_test
        """
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

        return *dataset_train, *dataset_test
