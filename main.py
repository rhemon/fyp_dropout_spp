
import argparse
from distutils.command.config import config
import json
from types import SimpleNamespace
import importlib
import torch
from pathlib import Path
from evaluators import get_evaluation_methods
import os
import numpy as np

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

def evaluate(cfg, model, X_train, X_test):
    evaluation_methods = get_evaluation_methods(cfg)
    evaluation_result = "Train\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_train)
        evaluation_result += each_method(y_preds, y_train)
    evaluation_result += "\n\nTest\n"
    for each_method in evaluation_methods:
        y_preds = model.predict(X_test)
        evaluation_result += each_method(y_preds, y_test)
    
    with open(model.checkpoint_folder / Path("results.txt"), "w") as result_file:
        result_file.writelines(evaluation_result)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.train_path:
        with open(args.train_path) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        
        train_path = Path(args.train_path)
        model = getattr(importlib.import_module(f"models.{cfg.MODEL}"), cfg.MODEL)(cfg, train_path)
        dataprocessor = getattr(importlib.import_module(f"dataloaders.{cfg.DATASET_PROCESSOR}"), cfg.DATASET_PROCESSOR)(cfg)

        X_train, y_train, X_test, y_test = dataprocessor.load_dataset()
        
        model.fit(X_train, y_train)
        
        evaluate(cfg, model, X_train, X_test)
    elif args.model_dir:
        model_dir = Path(args.model_dir)
        config_file = [file for file in os.listdir(model_dir) if file.endswith("json")][0]
        config_file = Path(model_dir / Path(config_file))

        with open(config_file) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        
        model = getattr(importlib.import_module(f"models.{cfg.MODEL}"), cfg.MODEL)(cfg, model_dir)
        dataprocessor = getattr(importlib.import_module(f"dataloaders.{cfg.DATASET_PROCESSOR}"), cfg.DATASET_PROCESSOR)(cfg)

        X_train, y_train, X_test, y_test = dataprocessor.load_dataset()
        evaluate(cfg, model, X_train, X_test)
    else:
        print("No training configuration provided")
