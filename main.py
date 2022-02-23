from pathlib import Path
import argparse
import json
from types import SimpleNamespace
import importlib
from tqdm import tqdm
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import shutil

from model_utils import train, predict
from evaluators import get_evaluation_methods

parser = argparse.ArgumentParser()
parser.add_argument("-tp", "--train-path", help="Path to training json file")


# Setting devices
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Detected device", device)

if __name__ == '__main__':

    args = parser.parse_args()

    if args.train_path:
        with open(args.train_path) as cfg_file:
            cfg = json.load(cfg_file, object_hook=lambda d: SimpleNamespace(**d))
        

        model = getattr(importlib.import_module('models'), cfg.MODEL)(cfg).to(device)
        dataprocessor = getattr(importlib.import_module('dataloaders'), cfg.DATASET_PROCESSOR)(cfg)

        train_dataset, test_dataset = dataprocessor.load_dataset()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

        lr = cfg.LR
        total_epoch = cfg.EPOCH
        loss_fn = getattr(nn, cfg.LOSS)()
        optimizer = getattr(optim, cfg.OPTIMIZER)(model.parameters(), lr=lr)


        model_desc = Path(cfg.MODEL + "_" + cfg.LOSS)
        time_stamp =  datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        checkpoint_folder = model_desc / Path(time_stamp)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.train_path), checkpoint_folder)
        
        train(cfg, total_epoch, model, loss_fn, optimizer, checkpoint_folder, device, train_loader)

        train_preds, train_targets = predict(model, train_dataset)
        test_preds, test_targets = predict(model, test_dataset)
        evaluation_methods = get_evaluation_methods(cfg)
        evaluation_result = "Train\n"
        for each_method in evaluation_methods:
            evaluation_result += each_method(train_preds, train_targets)
        evaluation_methods += "\n\nTest\n"
        for each_method in evaluation_methods:
            evaluation_result += each_method(test_preds, test_targets)
        
        with open(checkpoint_folder / Path("results.txt"), "w") as result_file:
            result_file.writelines(evaluation_result)
        

        # Train set evaluation
        

    else:
        print("No training configuration provided")