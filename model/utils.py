import h5py
from os import path
from tqdm import tqdm

import wandb
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from ignite.metrics import SSIM

class TopomapsToCortex(Dataset):
    def __init__(self, dataset_path):
        super(TopomapsToCortex, self).__init__()
        self.data = h5py.File(dataset_path, 'r')
        self.filenames = list(self.data.keys())

        self.snr_hashmap = np.zeros((len(self.filenames) // 2), dtype=int)
        self.lbl_hashmap = np.zeros((len(self.filenames) // 2), dtype=int)
        self.subj_hashmap = np.empty((len(self.filenames) // 2), dtype=object)

        for filename in self.filenames:
            if filename.startswith("cortex_views"):
                i = int(filename.split("_")[2].split("-")[-1])
                self.lbl_hashmap[i] = int(filename.split("_")[-2].split("-")[-1])
                self.subj_hashmap[i] = filename.split("_")[-1].split("-")[-1]
            else:
                i = int(filename.split("_")[1].split("-")[-1])
                self.snr_hashmap[i] = int(filename.split("_")[-2].split("-")[-1])

    def __len__(self):
        return len(self.filenames) // 2

    def __getitem__(self, i):
        topomaps_key = f'topomaps_eeg-{i}_snr-{self.snr_hashmap[i]}_subj-{self.subj_hashmap[i]}'
        cortex_views_key = f'cortex_views_dip-{i}_lbls-{self.lbl_hashmap[i]}_subj-{self.subj_hashmap[i]}'

        topomaps = np.array(self.data[topomaps_key])
        cortex_views = np.array(self.data[cortex_views_key])

        topomaps = torch.from_numpy(topomaps)
        cortex_views = torch.from_numpy(cortex_views)

        return topomaps, cortex_views

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
    
class LearningRateWarmup(object):
    def __init__(self, optimizer, warmup_iteration, target_lr, after_scheduler=None):
        self.optimizer = optimizer
        self.warmup_iteration = warmup_iteration
        self.target_lr = target_lr
        self.after_scheduler = after_scheduler
        self.step(1)

    def warmup_learning_rate(self, cur_iteration):
        warmup_lr = self.target_lr * (float(cur_iteration) / float(self.warmup_iteration))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = warmup_lr

    def step(self, cur_iteration):
        if cur_iteration <= self.warmup_iteration:
            self.warmup_learning_rate(cur_iteration)
        else:
            self.after_scheduler.step(cur_iteration - self.warmup_iteration)

    def load_state_dict(self, state_dict):
        self.after_scheduler.load_state_dict(state_dict)

OPTIMIZERS = ('Adagrad', 'Adam', 'AdamW', 'Adamax', 'SGD')
LOSSES = ("MSELoss", "L1Loss")

def predict_val(make_predictions, out_dir, model, val_loader, device, loss_mask):

    print('\nCreating archive with predictions validation set ...\n')
    num_batches=len(val_loader)
    with h5py.File(path.join(out_dir, f'{make_predictions}.h5'), 'w') as out_file:
        with torch.no_grad():
            for batch_id, (X, y) in enumerate(val_loader):
                X = X.float().to(device)
                pred = model(X)

                X = torch.squeeze(X).cpu().numpy()
                pred = torch.squeeze(pred*loss_mask).cpu().numpy()
                y = torch.squeeze(y).cpu().numpy()

                out_file.create_dataset(f'val_input_{str(batch_id)}', data=X,
                            shape=X.shape, dtype=np.float32)
                out_file.create_dataset(f'val_output_{str(batch_id)}', data=pred,
                            shape=pred.shape, dtype=np.float32)
                out_file.create_dataset(f'val_true_{str(batch_id)}', data=y,
                            shape=y.shape, dtype=np.float32)
                
    print(f'\nMade {num_batches} batches predictions for val data.')


def train_one_epoch(train_loader, model, optimizer, loss_func, device, loss_mask):
    
    model.train(True)

    running_train_loss = 0
    for image, label in train_loader:

        image, label = image.to(device), label.to(device)
        image = image.float()
        label = label.float()

        optimizer.zero_grad()
        outputs = model(image)
        
        loss = loss_func((outputs*loss_mask), label)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

    train_loss = running_train_loss / len(train_loader)

    return train_loss

def train(model, loss_func, train_loader, val_loader, optimizer, scheduler,
          current_epoch, epochs, out_dir, model_name, device, loss_mask):
    
    metric = SSIM(data_range=1.0, device=device)
    early_stopper = EarlyStopper(patience=20, min_delta=0.00001)

    for epoch in tqdm(range(current_epoch, current_epoch+epochs)):
        
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_func, device, loss_mask)
        
        # scheduler.step(current_epoch+1)
        scheduler.step()
        metric.reset()
        model.train(False)

        running_val_loss = 0
        running_val_metric = 0
        for image, label in val_loader:

            image, label = image.to(device), label.to(device)
            image = image.float()
            label = label.float()

            with torch.no_grad():
                val_outputs = model(image)

            loss = loss_func((val_outputs*loss_mask), label)
            running_val_loss += loss.item()

            metric.update((label, (val_outputs*loss_mask)))
            running_val_metric += metric.compute()

        val_loss = running_val_loss / len(val_loader)
        val_metric = running_val_metric / len(val_loader)

        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "val_ssim": val_metric})
        if (epoch + 1) % 1 == 0:
            print(f'\nEPOCH {epoch + 1}:')
            print('Train loss: {}; Val loss: {}'.format(train_loss, val_loss))
        
        if early_stopper.early_stop(val_loss):
            print(f'\nEaraly stopping at {epoch+1} epoch')             
            break

        if (epoch + 1) % 5 == 0:
            checkpoint_path = path.join(out_dir, f'{model_name}_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
        

