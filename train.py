import torch
from pathlib import Path


def train(cfg, total_epoch, model, loss_fn, optimizer, checkpoint_folder, device, train_loader):
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