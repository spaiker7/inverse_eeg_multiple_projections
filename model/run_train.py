import argparse
import yaml
from datetime import datetime

from pathlib import Path
import wandb
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR

from models import AttentionUNet
from utils import *

seed = torch.Generator()
seed.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'\nUsing {device} device \n')

def main():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--out_dir', type=str, default="./runs/",
                         help='the directory for checkpoints and log files')
    parser.add_argument('--dataset_path', type=str,
                         help='path to h5 dataset archive')
    parser.add_argument('--model', type=str,
                         help='model name: ["UNet"]')    
    parser.add_argument('--batch_size', type=int, default=4,
                        help='batch size for training and validation set')
    parser.add_argument('--train_part', type=float, default=0.95,
                        help='the portion used for training (the remaining goes to validation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        help='naming of the default pytorch optimizer like "Adam", "SGD", etc.')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='L2 regularization for learnable parameters')
    parser.add_argument('--loss_func', type=str, default="MSELoss",
                        help='naming of the default pytorch loss function like "L1Loss","MSELoss", etc.')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='if not empty - the path to the saved checkpoint')
    parser.add_argument('--make_predictions', type=str, default='',
                        help='if not empty - the path to saving h5 archive with predictions')
    parser.add_argument('--loss_mask', type=str, default='',
                        help='if not empty - calculate loss by the path to given mask.npy')
    
    args = parser.parse_args()
    args_dict = vars(args)

    with open('config.yml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        
    args_dict.update(config)

    print(f'Run name: {args_dict["run_name"]}\n ')

    if args.model == 'AttentionUNet':
        model = AttentionUNet(**config['AttentionUNet'])
    else:
        raise NotImplementedError
    
    num_total_million_params = round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 2)
    print(f"\nModel: {args_dict['model']} \nTraining {model.__class__.__name__} with {num_total_million_params}M params\n")

    model = nn.DataParallel(model)
    model = model.to(device)

    dataset = TopomapsToCortex(args_dict['dataset_path'])
    train_part = round(len(dataset) * args_dict['train_part'])
 
    train_dataset, val_dataset = torch.utils.data.random_split(
                                        dataset,
                                        [train_part, len(dataset) - train_part],
                                        generator=seed)

    train_loader = DataLoader(train_dataset, batch_size=args_dict['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args_dict['batch_size'], shuffle=False)
    print(f'\nDataset size: {len(dataset)}\nTrain: {len(train_dataset)}\nTest: {len(val_dataset)}')
    print(f'\nInput tensor shape: {next(iter(train_loader))[0].shape} \nOutput tensor shape: {next(iter(train_loader))[1].shape} \n')

    if args_dict['optimizer'] in OPTIMIZERS:
        optimizer = getattr(torch.optim, args_dict['optimizer'])(model.parameters(),
                            lr=args_dict['lr'], weight_decay=args_dict['weight_decay'])
    else:
        raise NotImplementedError
    
    if args_dict['checkpoint'].strip():
        print('\nLoading checkpoint...\n')
        checkpoint = torch.load(args_dict['checkpoint'])
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch']
    else:
        current_epoch = 0

    # scheduler_cosine = CosineAnnealingLR(optimizer, args_dict["epochs"] - args_dict["warmup_epochs"], last_epoch=current_epoch-1)
    # scheduler = LearningRateWarmup(optimizer, args_dict["warmup_epochs"], target_lr=lr, after_scheduler=scheduler_cosine)
    scheduler = ExponentialLR(optimizer, gamma=0.95, last_epoch=current_epoch-1)

    if args_dict['loss_func'] in LOSSES:
        loss_func = getattr(nn, args_dict['loss_func'])()
        loss_func = loss_func.to(device)
    else:
        raise NotImplementedError
    
    if args_dict['loss_mask'].strip():
        mask = np.load(args_dict['loss_mask'])['arr_0']
    else:
        mask = 1
    mask = torch.tensor(mask, dtype=torch.float32).to(device)

    outdir = Path(args_dict['out_dir'])
    chkpts_dir = outdir / 'chkpts'
    preds_dir = outdir / 'preds'
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=True)
        chkpts_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

    wandb.login()
    wandb.init(
        project="eeg-inverse-problem",
        config=args_dict,
        name=args_dict['run_name']
    )
    
    train(
        model, loss_func, train_loader, val_loader,
        optimizer, scheduler,
        current_epoch=current_epoch,
        epochs=args_dict['epochs'],
        out_dir=chkpts_dir,
        model_name=f'{args_dict["model"]}_{datetime.now().strftime("%Y-%m-%d-%H")}',
        device=device,
        loss_mask=mask
        )

    if args_dict['make_predictions'].strip():
        predict_val(
            args_dict['make_predictions'],
            path.join(args_dict['out_dir'], 'preds'),
            model, val_loader, 
            device=device,
            loss_mask=mask,
            )
    
if __name__ == "__main__":
    main()