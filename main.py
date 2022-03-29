
import argparse
import datetime
import importlib
import json
import numpy as np
import os
import shutil
from pathlib import Path
import torch
from types import SimpleNamespace

from evaluators import get_evaluation_methods

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--train-path", help="Path to training json file")
parser.add_argument("-md", "--model-dir", help="Path to checkpoint folder for which evaluation is executed.")

# Seed 
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


# Setting devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device", device)

def evaluate(cfg, model, X_train, X_test, y_train, y_test, fname="results.txt"):
    evaluation_methods = get_evaluation_methods(cfg)
    evaluation_result = "Train\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_train)
        evaluation_result += each_method(y_preds, y_train.cpu())
    evaluation_result += "\n\nTest\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_test)
        evaluation_result += each_method(y_preds, y_test.cpu())
    
    with open(model.checkpoint_folder / Path(fname), "w") as result_file:
        result_file.writelines(evaluation_result)

def get_checkpoint_folder(cfg):
    model_desc = Path(cfg.MODEL)
    
    time_stamp =  datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    checkpoint_folder = Path("model_cpts") / Path(cfg.DATASET_PROCESSOR) / Path(cfg.OUTPUT_TYPE) / model_desc / Path(time_stamp)
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    shutil.copy(train_path, checkpoint_folder)
    
    return checkpoint_folder

if __name__ == '__main__':

    args = parser.parse_args()
    train_mode = True

    # Based argument type load configuration accordingly and set train mode.
    # Specifying model dir will mean to only run evaluation by loading
    # last checkpoint.
    if args.train_path:
        with open(args.train_path) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        
        train_path = Path(args.train_path)
        checkpoint_dir = get_checkpoint_folder(cfg)
    elif args.model_dir:
        checkpoint_dir = Path(args.model_dir)
        config_file = [file for file in os.listdir(checkpoint_dir) if file.endswith("json")][0]
        config_file = Path(checkpoint_dir / Path(config_file))
        with open(config_file) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        train_mode = False   
    else:
        raise Exception("No configuration provided")

    dataprocessor = getattr(importlib.import_module(f"dataloaders.{cfg.DATASET_PROCESSOR}"), cfg.DATASET_PROCESSOR)(cfg, checkpoint_dir)
    X_train, y_train, X_test, y_test = dataprocessor.load_dataset()
    model = getattr(importlib.import_module(f"models.{cfg.MODEL_DIR}.{cfg.MODEL}"), cfg.MODEL)(
                        cfg, checkpoint_dir, input_dim=X_train.shape[-1], dataprocessor=dataprocessor)
    
    if train_mode:
        model.fit(X_train, y_train)
        fname = 'results.txt'
    else:
        model.model.load_state_dict(torch.load(checkpoint_dir/Path("model.pt"))['model_state_dict'])
        fname = 'evaluation.txt'
    
    evaluate(cfg, model, X_train, X_test, y_train, y_test, fname)
