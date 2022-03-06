import torch
from pathlib import Path
import matplotlib.pyplot as plt

def train(cfg, total_epoch, model, loss_fn, optimizer, checkpoint_folder, device, train_loader):
    ep = 1
    cpt_track_step = 0
    model.train()
    all_losses = []
    epoch_losses = []

    for epoch in range(total_epoch):
        batch_losses = []
        for batch_idx, (event_x, event_time_x, lens, result) in enumerate(train_loader):
            event_x = event_x.type(torch.LongTensor).to(device)
            event_time_x = event_time_x.type(torch.LongTensor).to(device)
            # lens = lens.to(device)
            result = result.to(device)
    #         result = result.type(torch.LongTensor)
            scores = model(event_x, event_time_x, lens).squeeze(1)
            loss = loss_fn(scores, result)
            all_losses.append(loss.item())
            batch_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()

            if cfg.DROPOUT == "GradBasedDropout":
                model.forward_ih_keep_prob = torch.sigmoid(model.dense_ih_forward.weight.grad.detach().sum(dim=1)).to(device)
                model.forward_hh_keep_prob =  torch.sigmoid(model.dense_hh_forward.weight.grad.detach().sum(axis=1)).to(device)
                model.backward_ih_keep_prob = torch.sigmoid(model.dense_ih_backward.weight.grad.detach().sum(axis=1)).to(device)
                model.backward_hh_keep_prob = torch.sigmoid(model.dense_hh_backward.weight.grad.detach().sum(axis=1)).to(device)
            
            optimizer.step()
            checkpoint_name = checkpoint_folder / Path("checkpoint_e"+str(epoch)+"_b"+str(batch_idx)+".pt")
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
        epoch_losses.append(sum(batch_losses)/len(batch_losses))
    checkpoint_name = checkpoint_folder / Path("model.pt")
    torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_name)

    plt.figure(0)          
    plt.plot([i for i in range(len(all_losses))], all_losses)
    plt.savefig(checkpoint_folder / Path("all_losses.png"))
    
    plt.figure(1)
    plt.plot([i for i in range(len(epoch_losses))], epoch_losses)
    plt.savefig(checkpoint_folder / Path("epoch_losses.png"))
    

def predict(model, dataset, device, threshold=0.5):
    y_preds = []
    targets = []
    with torch.no_grad():
        model.eval()
        
        for each_record in dataset:
            each_record = [torch.unsqueeze(each, 0) for each in each_record]
            each_record[0] = each_record[0].type(torch.LongTensor).to(device)
            each_record[1] = each_record[1].type(torch.LongTensor).to(device)
            pred = model(*each_record[:3])
            y_preds.append(pred)
            targets.append(each_record[-1])

    return torch.squeeze(torch.cat(y_preds).cpu() > threshold, 1), torch.cat(targets).cpu()