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

        train_loader, test_loader = dataprocessor.load_dataset()

        lr = cfg.LR
        total_epoch = cfg.EPOCH
        loss_fn = getattr(nn, cfg.LOSS)()
        optimizer = getattr(optim, cfg.OPTIMIZER)(model.parameters(), lr=lr)


        model_desc = Path(cfg.MODEL + "_" + cfg.LOSS)
        time_stamp =  datetime.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        checkpoint_folder = model_desc / Path(time_stamp)
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        shutil.copy(Path(args.train_path), checkpoint_folder)
        ep = 1
        cpt_track_step = 0
        for epoch in range(total_epoch):
            for batch_idx, (event_x, event_time_x, lens, result) in enumerate(train_loader):
                event_x = event_x.type(torch.LongTensor).to(device)
                event_time_x = event_time_x.type(torch.LongTensor).to(device)
                # lens = lens.to(device)
                result = result.to(device)
        #         result = result.type(torch.LongTensor)
                scores = model(event_x, event_time_x, lens).squeeze(1)
                loss = loss_fn(scores, result)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                checkpoint_name = checkpoint_folder / Path(str(epoch)+"_"+str(batch_idx)+".pt")
                cpt_track_step += 1
                if cpt_track_step % cfg.CPT_EVERY == 0:
                    torch.save({
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        }, checkpoint_name)
                    print(f'epoch: {epoch + 1} step: {batch_idx + 1}/{len(train_loader)} loss: {loss}')
                    cpt_track_step = 0
            ep += 1
    else:
        print("No training configuration provided")