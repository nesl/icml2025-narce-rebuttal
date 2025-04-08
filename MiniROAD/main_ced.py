import torch
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import argparse
import yaml
import os
import os.path as osp
from utils import get_logger
from model import build_model
from datasets import build_data_loader
from criterions import build_criterion
from trainer import build_trainer, build_eval
from utils import *
import numpy as np
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

class CEDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return data, label

    def __len__(self):
        return len(self.data)


parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/miniroad_ced.yaml')
parser.add_argument('--eval', type=str, default=None)
parser.add_argument('--amp', action='store_true')
parser.add_argument('--tensorboard', action='store_true')
parser.add_argument('--lr_scheduler', action='store_true')
parser.add_argument('--no_rgb', action='store_true')
parser.add_argument('--no_flow', action='store_true')
parser.add_argument('--dataset', type=int, help='Dataset size', choices=[2000, 4000, 6000, 8000, 10000])
parser.add_argument('--testset', type=str, help='Test dataset', choices=['3min', '5min', '15min', '30min'])
parser.add_argument('--seed',  type=int, help='Random seed') #0, 17, 1243, 3674, 7341, 53, 97, 103, 191, 99719

args = parser.parse_args()

# combine argparse and yaml
opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
opt.update(vars(args))
cfg = opt

set_seed(args.seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.eval != None:
    identifier = f'{cfg["model"]}_{cfg["feature_pretrained"]}_flow{not cfg["no_flow"]}_{args.dataset}_s{args.seed}_{args.testset}'
else:
    identifier = f'{cfg["model"]}_{cfg["feature_pretrained"]}_flow{not cfg["no_flow"]}_{args.dataset}_s{args.seed}'
result_path = create_outdir(osp.join(cfg['output_path'], identifier))
logger = get_logger(result_path)
logger.info(cfg)




batch_size = 256
n_epochs = 5000
""" Load datasets """


if args.eval != None:
    test_data_file = './datasets/new_CE_dataset/new_ce{}_test_data.npy'.format(args.testset)
    test_label_file = './datasets/new_CE_dataset/new_ce{}_test_labels.npy'.format(args.testset)
else:
    train_data_file = './datasets/new_CE_dataset/new_ce5min_train_data_{}.npy'.format(args.dataset)
    train_label_file = './datasets/new_CE_dataset/new_ce5min_train_labels_{}.npy'.format(args.dataset)
    val_data_file = './datasets/new_CE_dataset/new_ce5min_val_data.npy'
    val_label_file = './datasets/new_CE_dataset/new_ce5min_val_labels.npy'
    test_data_file = './datasets/new_CE_dataset/new_ce5min_test_data.npy'
    test_label_file = './datasets/new_CE_dataset/new_ce5min_test_labels.npy'

    ce_train_data = np.load(train_data_file)
    ce_train_labels = np.load(train_label_file)
    ce_val_data = np.load(val_data_file)
    ce_val_labels = np.load(val_label_file)

    ce_train_dataset = CEDataset(ce_train_data, ce_train_labels)
    ce_train_loader = DataLoader(ce_train_dataset, 
                                batch_size=batch_size, 
                                shuffle=True, 
                                )
    ce_val_dataset = CEDataset(ce_val_data, ce_val_labels)
    ce_val_loader = DataLoader(ce_val_dataset,
                                batch_size=batch_size,
                                shuffle=False, 
                                )
    print(train_data_file)
    print(ce_train_data.shape, ce_train_labels.shape, ce_val_data.shape, ce_val_labels.shape)


ce_test_data = np.load(test_data_file)
ce_test_labels = np.load(test_label_file)

print(ce_test_data.shape, ce_test_labels.shape)


ce_test_dataset = CEDataset(ce_test_data, ce_test_labels)
ce_test_loader = DataLoader(ce_test_dataset, 
                            batch_size=batch_size, 
                            shuffle=False
                            )

testloader = ce_test_loader
model = build_model(cfg, device)
evaluate = build_eval(cfg)
if args.eval != None:
    train_identifier = f'{cfg["model"]}_{cfg["feature_pretrained"]}_flow{not cfg["no_flow"]}_{args.dataset}_s{args.seed}'
    train_result_path = create_outdir(osp.join(cfg['output_path'], train_identifier))
    model_path=osp.join(train_result_path, 'ckpts', 'best.pth')
    # model.load_state_dict(torch.load(args.eval))
    model.load_state_dict(torch.load(model_path))
    f1 = evaluate(model, testloader, logger)
    logger.info(f'{cfg["task"]} result: {f1*100:.2f} {cfg["metric"]}')
    exit()
    
summary(model)


trainloader = ce_train_loader
criterion = build_criterion(cfg, device)
train_one_epoch = build_trainer(cfg)
optim = torch.optim.AdamW if cfg['optimizer'] == 'AdamW' else torch.optim.Adam
optimizer = optim([{'params': model.parameters(), 'initial_lr': cfg['lr']}],
                    lr=cfg['lr'], weight_decay=cfg["weight_decay"])

scheduler = build_lr_scheduler(cfg, optimizer, len(trainloader)) if args.lr_scheduler else None
writer = SummaryWriter(osp.join(result_path, 'runs')) if args.tensorboard else None
scaler = torch.cuda.amp.GradScaler() if args.amp else None
total_params = sum(p.numel() for p in model.parameters())

logger.info(f'Dataset: {cfg["data_name"]},  Model: {cfg["model"]}')    
logger.info(f'lr:{cfg["lr"]} | Weight Decay:{cfg["weight_decay"]} | Window Size:{cfg["window_size"]} | Batch Size:{cfg["batch_size"]}') 
logger.info(f'Total epoch:{cfg["num_epoch"]} | Total Params:{total_params/1e6:.1f} M | Optimizer: {cfg["optimizer"]}')
logger.info(f'Output Path:{result_path}')

best_f1, best_epoch = 0, 0
for epoch in range(1, cfg['num_epoch']+1):
    epoch_loss = train_one_epoch(trainloader, model, criterion, optimizer, scaler, epoch, writer, scheduler=scheduler)
    # trainloader.dataset._init_features()
    f1 = evaluate(model, testloader, logger)
    if f1 > best_f1:
        best_f1 = f1
        best_epoch = epoch
        torch.save(model.state_dict(), osp.join(result_path, 'ckpts', 'best.pth'))
    logger.info(f'Epoch {epoch} F1: {f1:.2f} | Best F1: {best_f1*100:.2f} at epoch {best_epoch}, iter {epoch*cfg["batch_size"]*len(trainloader)} | train_loss: {epoch_loss/len(trainloader):.4f}, lr: {optimizer.param_groups[0]["lr"]:.7f}')
    
os.rename(osp.join(result_path, 'ckpts', 'best.pth'), osp.join(result_path, 'ckpts', f'best.pth'))