import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import json
from abc import *
from pathlib import Path
import argparse
import os
import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import copy
import logging
from datetime import datetime
from torch.optim import lr_scheduler
import random

from trainers.ali_trainer_esdfmRF import AliPretrainEsdfmRF_PLETrainer
from dataloaders.ali_odl_dataloader_esdfmRF import get_ali_pretrain_dataloader
from models.ali_pretrain_esdfmRF_PLE import PretrainEsdfmRF_PLE
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/ali/processed_data_expand.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/mnt/nfs/luomingxuan/airbench/main_exp3/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Ali', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=3, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=3, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=0.01, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=0.01, help='refund wait window size (days)')
    parser.add_argument('--stream_wait_window', type=int, default=0.1, help='stream wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=10, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=17, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=24, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="esdfmRF_PLE_pretrain", help='[esdfmRF_PLE_pretrain]')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--embed_dim', type=int, default=32, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    parser.add_argument('--device_idx', type=str, default='0', help='Device index')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-2, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--model_save_pth', type=str, default='/mnt/nfs/luomingxuan/airbench/main_exp3/', help='Model save pth')
    parser.add_argument('-reg_loss_decay',type=float,default=1e-4,help='Regularization loss decay coefficient')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    train_loader, test_loader = get_ali_pretrain_dataloader(args)
    device = torch.device(f"cuda:{args.device_idx}" if torch.cuda.is_available() else "cpu")
    model = PretrainEsdfmRF_PLE(args).to(device)
    pretrain_trainer = AliPretrainEsdfmRF_PLETrainer(args, model, train_loader, test_loader)
    pretrain_trainer.train()
