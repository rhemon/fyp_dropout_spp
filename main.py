
import argparse
from distutils.command.config import config
import json
from tabnanny import check
from types import SimpleNamespace
import importlib
import torch
from pathlib import Path
from evaluators import get_evaluation_methods
import os
import numpy as np
import datetime
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--train-path", help="Path to training json file")
parser.add_argument("-md", "--model-dir", help="Path to model file")

# Seed 
torch.manual_seed(123)
torch.cuda.manual_seed(123)
np.random.seed(123)


# Setting devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device", device)

def evaluate(cfg, model, X_train, X_test, y_train, y_test):
    evaluation_methods = get_evaluation_methods(cfg)
    evaluation_result = "Train\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_train)
        evaluation_result += each_method(y_preds, y_train.cpu())
    evaluation_result += "\n\nTest\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_test)
        evaluation_result += each_method(y_preds, y_test.cpu())
    
    with open(model.checkpoint_folder / Path("results.txt"), "w") as result_file:
        result_file.writelines(evaluation_result)

def get_checkpoint_folder(cfg):
    model_desc = Path(cfg.MODEL)
    # if path passed is model_cpts then its on model evaluation mode, dont recreate files
    if "model_cpts" in str(train_path):
        checkpoint_folder = train_path
    else:
        time_stamp =  datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        checkpoint_folder = Path("model_cpts") / Path(cfg.DATASET_PROCESSOR) / Path(cfg.OUTPUT_TYPE) / model_desc / Path(time_stamp)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(train_path, checkpoint_folder)
    
    return checkpoint_folder

if __name__ == '__main__':

    args = parser.parse_args()

    if args.train_path:
        with open(args.train_path) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        
        try:
            folder = cfg.MODEL_DIR
        except:
            # defaulted to GritNet as during GritNet experiments
            # json did not include MODEL_DIR
            folder = "GritNet"

        train_path = Path(args.train_path)
        checkpiont_dir = get_checkpoint_folder(cfg)
        
        dataprocessor = getattr(importlib.import_module(f"dataloaders.{cfg.DATASET_PROCESSOR}"), cfg.DATASET_PROCESSOR)(cfg, checkpiont_dir)
        
        X_train, y_train, X_test, y_test = dataprocessor.load_dataset()

        model = getattr(importlib.import_module(f"models.{folder}.{cfg.MODEL}"), cfg.MODEL)(cfg, checkpiont_dir, input_dim=X_train.shape[-1])

        
        model.fit(X_train, y_train)
        
        evaluate(cfg, model, X_train, X_test, y_train, y_test)
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        config_file = [file for file in os.listdir(model_dir) if file.endswith("json")][0]
        config_file = Path(model_dir / Path(config_file))

        with open(config_file) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        
        model = getattr(importlib.import_module(f"models.{cfg.MODEL}"), cfg.MODEL)(cfg, model_dir)
        dataprocessor = getattr(importlib.import_module(f"dataloaders.{cfg.DATASET_PROCESSOR}"), cfg.DATASET_PROCESSOR)(cfg)

        X_train, y_train, X_test, y_test = dataprocessor.load_dataset()
        evaluate(cfg, model, X_train, X_test, y_train, y_test)
    else:
        print("No training configuration provided")
