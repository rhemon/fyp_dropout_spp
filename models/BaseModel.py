import datetime
import shutil
from pathlib import Path

class BaseModel:

    def __init__(self, cfg, train_path):
        
        model_desc = Path(cfg.MODEL)
        # if path passed is model_cpts then its on model evaluation mode, dont recreate files
        if "model_cpts" in str(train_path):
            self.checkpoint_folder = train_path
        else:
            time_stamp =  datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
            self.checkpoint_folder = Path("model_cpts") / model_desc / Path(time_stamp)
            self.checkpoint_folder.mkdir(parents=True, exist_ok=True)
            shutil.copy(train_path, self.checkpoint_folder)

