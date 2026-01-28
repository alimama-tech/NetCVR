import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import math
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
import torch.nn.functional as F
from mx_utils.metrics import auc_score,nll_score,prauc_score,pcoc_score,stable_log1pex,nll_score_split
import pandas as pd

class AliPretrainEsdfmRFTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)

    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()


    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)




            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            net_cvr_loss.backward()
            self.optimizer.step()

            loss = cvr_loss + net_cvr_loss


            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, cvr_outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")

        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_ShareEmbTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)

    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()


    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)




 

            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            loss = cvr_loss + net_cvr_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, cvr_outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")

        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_PLETrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)

    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()


    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)

            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            net_cvr_loss.backward()
            self.optimizer.step()

            loss = cvr_loss + net_cvr_loss 



            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, cvr_outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")

        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_PLE_DelayTimeAwareTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)

    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def extract_hour_from_timestamp(self, ts, invalid_value=-1):
        """
        将相对于 2025-05-01 00:00:00 的秒级时间戳转换为“一天中的小时” (0-23)
        
        Args:
            ts: Tensor of timestamps in seconds from '2025-05-01 00:00:00'
            invalid_value: value indicating missing timestamp (e.g., -1)

        Returns:
            hour: Tensor of hours (0-23), with invalid_value mapped to -1
        """
        is_invalid = (ts == invalid_value)
        
        ts_safe = torch.where(is_invalid, torch.tensor(0, device=ts.device), ts)
        
        hour = (ts_safe // 3600) % 24
        
        hour = torch.where(is_invalid, torch.tensor(-1, dtype=hour.dtype, device=hour.device), hour)
        
        return hour.long()

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)
            click_hour = self.extract_hour_from_timestamp(click_ts)
            pay_hour = self.extract_hour_from_timestamp(pay_ts)
            click_hour = click_hour.to(self.device)
            pay_hour = pay_hour.to(self.device)

            pay_delay_time_aware_labels = (pay_ts - click_ts) / 3600
            pay_delay_time_aware_labels = torch.log(pay_delay_time_aware_labels + 1)
            pay_delay_time_aware_mask = (pay_labels == 1)

            refund_delay_time_aware_labels = (refund_ts - click_ts) / 3600
            refund_delay_time_aware_labels = torch.log(refund_delay_time_aware_labels + 1)
            refund_delay_time_aware_mask = (pay_labels == 1) & (net_pay_labels == 0)

            
            self.optimizer.zero_grad()
            cvr_delay_time_aware_outputs = self.model.cvr_delay_time_aware(features)
            cvr_delay_time_aware_loss = F.mse_loss(cvr_delay_time_aware_outputs[pay_delay_time_aware_mask],pay_delay_time_aware_labels[pay_delay_time_aware_mask])
            cvr_delay_time_aware_loss.backward()
            self.optimizer.step()
            

            
            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features,click_hour)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)

            cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            rfr_delay_time_aware_outputs = self.model.rfr_delay_time_aware(features)
            rfr_delay_time_aware_loss = F.mse_loss(rfr_delay_time_aware_outputs[refund_delay_time_aware_mask],refund_delay_time_aware_labels[refund_delay_time_aware_mask])
            rfr_delay_time_aware_loss.backward()
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features,click_hour)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            

            net_cvr_loss.backward()
            self.optimizer.step()

            loss = cvr_loss + net_cvr_loss 

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features,click_hour)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None

    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                click_ts = batch['click_ts'].to(self.device)
                pay_ts = batch['pay_ts'].to(self.device)

                click_hour = self.extract_hour_from_timestamp(click_ts)
                pay_hour = self.extract_hour_from_timestamp(pay_ts)
                click_hour = click_hour.to(self.device)
                pay_hour = pay_hour.to(self.device)

                cvr_outputs,net_cvr_outputs = Recmodel.predict(features,click_hour)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())


        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_PLE_v2_Trainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss

    def cos_sim_loss(self,emb1, emb2):
        norm_emb1 = F.normalize(emb1.mean(dim=0, keepdim=True), p=2, dim=1)
        norm_emb2 = F.normalize(emb2.mean(dim=0, keepdim=True), p=2, dim=1)
        cos_sim = (norm_emb1 * norm_emb2).sum()
        return cos_sim.abs()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def hsic_loss(self ,emb1, emb2, sigma=1.0):
        def rbf_kernel(X, sigma):
            X2 = (X ** 2).sum(1, keepdim=True)
            dist = X2 + X2.T - 2 * X @ X.T
            return torch.exp(-dist / (2 * sigma ** 2))

        K = rbf_kernel(emb1, sigma)
        L = rbf_kernel(emb2, sigma)

        H = torch.eye(K.size(0), device=K.device) - 1.0 / K.size(0)
        Kc = H @ K @ H
        Lc = H @ L @ H

        hsic = (Kc * Lc).sum() / (K.size(0) - 1) ** 2
        return hsic
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)
    
    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def listwise_loss(self, cvr_outputs, labels, mask, reduction='mean'):
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        _, sorted_indices = torch.sort(label_valid, descending=True)
        sorted_logits = cvr_valid[sorted_indices]

        logsumexp_from_j = torch.logcumsumexp(sorted_logits.flip(dims=[0]), dim=0).flip(dims=[0])

        per_element_loss = sorted_logits - logsumexp_from_j
        listwise_loss = -per_element_loss.sum()

        if reduction == 'mean':
            listwise_loss = listwise_loss / len(sorted_logits)
        
        return listwise_loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, pay_labels):
        corrected_cvr_logits = cvr_outputs + correction_outputs

        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        corrected_cvr_mask = torch.ones_like(pay_labels).to(self.device)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,pay_labels,corrected_cvr_mask)        
        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, pay_labels, pay_mask=None):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        pay_labels = pay_labels.view(B)
        pay_mask = torch.ones_like(pay_labels).bool()
        valid_neg_mask = (pay_mask) & (pay_labels == 0)
        if valid_neg_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = cvr_hidden_states[valid_neg_mask]
        z_neg = F.normalize(z_neg, p=2, dim=-1)
        sim_matrix = torch.mm(z_neg, z_neg.t())
        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()
        return diversity_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()


    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)



            self.optimizer.zero_grad()
            cvr_hidden_states = self.model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,pay_labels)
            cvr_cl_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            cvr_loss.backward()
            self.optimizer.step()





            loss = cvr_loss + net_cvr_loss  

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        all_metrics["Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = 0
        all_corrected_pay_preds = []
        all_corrected_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)
                

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                corrected_cvr_outputs,corrected_net_cvr_outputs = Recmodel.correction_predict(features)
                all_corrected_pay_preds.extend(corrected_cvr_outputs.cpu().numpy().tolist())
                all_corrected_net_pay_preds.extend(corrected_net_cvr_outputs.cpu().numpy().tolist())






        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")



        all_metrics["Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"Global_CORRECTED_CVR_AUC: {all_metrics['Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_AUC: {all_metrics['Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_NLL: {all_metrics['Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_NLL: {all_metrics['Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PCOC: {all_metrics['Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PCOC: {all_metrics['Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PRAUC: {all_metrics['Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PRAUC: {all_metrics['Global_CORRECTED_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_PLE_v3_Trainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss

    def cos_sim_loss(self,emb1, emb2):
        norm_emb1 = F.normalize(emb1.mean(dim=0, keepdim=True), p=2, dim=1)
        norm_emb2 = F.normalize(emb2.mean(dim=0, keepdim=True), p=2, dim=1)
        cos_sim = (norm_emb1 * norm_emb2).sum()
        return cos_sim.abs()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def hsic_loss(self ,emb1, emb2, sigma=1.0):
        def rbf_kernel(X, sigma):
            X2 = (X ** 2).sum(1, keepdim=True)
            dist = X2 + X2.T - 2 * X @ X.T
            return torch.exp(-dist / (2 * sigma ** 2))

        K = rbf_kernel(emb1, sigma)
        L = rbf_kernel(emb2, sigma)

        H = torch.eye(K.size(0), device=K.device) - 1.0 / K.size(0)
        Kc = H @ K @ H
        Lc = H @ L @ H

        hsic = (Kc * Lc).sum() / (K.size(0) - 1) ** 2
        return hsic
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)
    
    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def listwise_loss(self, cvr_outputs, labels, mask, reduction='mean'):
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        _, sorted_indices = torch.sort(label_valid, descending=True)
        sorted_logits = cvr_valid[sorted_indices]

        logsumexp_from_j = torch.logcumsumexp(sorted_logits.flip(dims=[0]), dim=0).flip(dims=[0])

        per_element_loss = sorted_logits - logsumexp_from_j
        listwise_loss = -per_element_loss.sum()

        if reduction == 'mean':
            listwise_loss = listwise_loss / len(sorted_logits)
        
        return listwise_loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, pay_labels):
        corrected_cvr_logits = cvr_outputs + correction_outputs

        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        corrected_cvr_mask = torch.ones_like(pay_labels).to(self.device)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,pay_labels,corrected_cvr_mask)        
        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, pay_labels, pay_mask=None):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        pay_labels = pay_labels.view(B)
        pay_mask = torch.ones_like(pay_labels).bool()

        valid_neg_mask = (pay_mask)
        if valid_neg_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = cvr_hidden_states[valid_neg_mask]
        z_neg = F.normalize(z_neg, p=2, dim=-1)
        sim_matrix = torch.mm(z_neg, z_neg.t())
        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()
        return diversity_loss

    def refund_cl_loss(self, net_cvr_hidden_states, pay_labels, net_pay_labels, pay_mask=None):
        B = net_cvr_hidden_states.size(0)
        net_cvr_hidden_states = net_cvr_hidden_states.view(B, -1)
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        pay_mask = (pay_labels == 1)
        valid_mask = (pay_mask) 
        if valid_mask.sum() <= 1:
            return net_cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = net_cvr_hidden_states[valid_mask]
        z_neg = F.normalize(z_neg, p=2, dim=-1)
        sim_matrix = torch.mm(z_neg, z_neg.t())
        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()
        return diversity_loss

    def cvr_cl_loss_v2(self, cvr_hidden_states_no_pay, cvr_hidden_states_delay_pay):
        """
        拉远 no-pay 和 delay-pay 用户的表征。
        使用负样本对之间的相似度作为损失，越小越好。
        """
        if cvr_hidden_states_no_pay.size(0) == 0 or cvr_hidden_states_delay_pay.size(0) == 0:
            return cvr_hidden_states_no_pay.new_tensor(0.0, requires_grad=True)

        z_no_pay = F.normalize(cvr_hidden_states_no_pay, p=2, dim=-1)
        z_delay_pay = F.normalize(cvr_hidden_states_delay_pay, p=2, dim=-1)

        sim_matrix = torch.mm(z_no_pay, z_delay_pay.t())

        contrastive_loss = sim_matrix.mean() + 1

        return contrastive_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()


    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            self.optimizer.zero_grad()
            cvr_hidden_states = self.model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,pay_labels)
            cvr_cl_loss.backward()
            self.optimizer.step()




            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            cvr_loss.backward()
            self.optimizer.step()




            loss = cvr_loss + net_cvr_loss  + cvr_cl_loss

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None


    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        all_metrics["Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = 0
        all_corrected_pay_preds = []
        all_corrected_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)
                

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                corrected_cvr_outputs,corrected_net_cvr_outputs = Recmodel.correction_predict(features)
                all_corrected_pay_preds.extend(corrected_cvr_outputs.cpu().numpy().tolist())
                all_corrected_net_pay_preds.extend(corrected_net_cvr_outputs.cpu().numpy().tolist())






        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")



        all_metrics["Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"Global_CORRECTED_CVR_AUC: {all_metrics['Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_AUC: {all_metrics['Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_NLL: {all_metrics['Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_NLL: {all_metrics['Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PCOC: {all_metrics['Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PCOC: {all_metrics['Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PRAUC: {all_metrics['Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PRAUC: {all_metrics['Global_CORRECTED_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainEsdfmRF_PLE_test_Trainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)

        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )

        
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,cvr_outputs,cvr_labels,net_cvr_outputs,net_cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        
        loss = cvr_loss + net_cvr_loss
        return loss

    def cvr_loss_fn(self,cvr_outputs,cvr_labels):
        cvr_outputs = cvr_outputs.view(-1)
        cvr_labels = cvr_labels.float()

        cvr_pos_loss = stable_log1pex(cvr_outputs)
        cvr_neg_loss = cvr_outputs + stable_log1pex(cvr_outputs)
        cvr_loss = torch.mean(cvr_pos_loss * cvr_labels + cvr_neg_loss * (1 - cvr_labels))
        loss = cvr_loss
        return loss

    def net_cvr_loss_fn(self,net_cvr_outputs,net_cvr_labels):
        net_cvr_outputs = net_cvr_outputs.view(-1)
        net_cvr_labels = net_cvr_labels.float()

        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean(net_cvr_pos_loss * net_cvr_labels + net_cvr_neg_loss * (1 - net_cvr_labels))
        loss = net_cvr_loss
        return loss
    
    def refund_loss_fn(self,net_cvr_outputs,pay_labels,net_pay_labels):
        refund_labels = (pay_labels == 1) & (net_pay_labels == 0)
        refund_labels = refund_labels.float().to(self.device)
        refund_mask = (pay_labels == 1) | (refund_labels > 0)
        refund_mask = refund_mask.float().to(self.device)
        net_cvr_outputs = net_cvr_outputs.view(-1)
        refund_labels = refund_labels.float()
        net_cvr_pos_loss = stable_log1pex(net_cvr_outputs)
        net_cvr_neg_loss = net_cvr_outputs + stable_log1pex(net_cvr_outputs)
        net_cvr_loss = torch.mean((net_cvr_pos_loss * refund_labels + net_cvr_neg_loss * (1 - refund_labels))*refund_mask)
        loss = net_cvr_loss
        return loss

    def add_refund_bpr_loss(self,cvr_logits, pay_labels, refund_labels, gamma=1.0,eps = 1e-6):
        """
        在下单用户中，让「未退款」的预测得分 > 「已退款」的得分
        使用 BPR loss 实现
        """
        pos_mask = (pay_labels == 1)
        neg_mask = (refund_labels == 1)

        pos_logits = cvr_logits[pos_mask]
        neg_logits = cvr_logits[neg_mask]

        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss

    def cos_sim_loss(self,emb1, emb2):
        norm_emb1 = F.normalize(emb1.mean(dim=0, keepdim=True), p=2, dim=1)
        norm_emb2 = F.normalize(emb2.mean(dim=0, keepdim=True), p=2, dim=1)
        cos_sim = (norm_emb1 * norm_emb2).sum()
        return cos_sim.abs()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def hsic_loss(self ,emb1, emb2, sigma=1.0):
        def rbf_kernel(X, sigma):
            X2 = (X ** 2).sum(1, keepdim=True)
            dist = X2 + X2.T - 2 * X @ X.T
            return torch.exp(-dist / (2 * sigma ** 2))

        K = rbf_kernel(emb1, sigma)
        L = rbf_kernel(emb2, sigma)

        H = torch.eye(K.size(0), device=K.device) - 1.0 / K.size(0)
        Kc = H @ K @ H
        Lc = H @ L @ H

        hsic = (Kc * Lc).sum() / (K.size(0) - 1) ** 2
        return hsic
    
    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return -(pearson_corr + 1).pow(2)
    
    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
    
    def gap_aware_cvr_loss(self, cvr_outputs, net_cvr_outputs, alpha=1.0):
        cvr_prob = torch.sigmoid(cvr_outputs)
        net_cvr_prob = torch.sigmoid(net_cvr_outputs)
        gap = cvr_prob - net_cvr_prob
        penalty = torch.relu(gap).mean()
        stacked = torch.stack([gap, cvr_prob])
        corr_matrix = torch.corrcoef(stacked)
        corr = corr_matrix[0, 1]
        reg_loss =penalty
        return alpha * reg_loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def listwise_loss(self, cvr_outputs, labels, mask, reduction='mean'):
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        _, sorted_indices = torch.sort(label_valid, descending=True)
        sorted_logits = cvr_valid[sorted_indices]

        logsumexp_from_j = torch.logcumsumexp(sorted_logits.flip(dims=[0]), dim=0).flip(dims=[0])

        per_element_loss = sorted_logits - logsumexp_from_j
        listwise_loss = -per_element_loss.sum()

        if reduction == 'mean':
            listwise_loss = listwise_loss / len(sorted_logits)
        
        return listwise_loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, pay_labels):
        corrected_cvr_logits = cvr_outputs + correction_outputs

        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        corrected_cvr_mask = torch.ones_like(pay_labels).to(self.device)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,pay_labels,corrected_cvr_mask)        
        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, pay_labels, pay_mask=None):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        pay_labels = pay_labels.view(B)
        pay_mask = torch.ones_like(pay_labels).bool()

        valid_neg_mask = (pay_mask)
        if valid_neg_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = cvr_hidden_states[valid_neg_mask]
        z_neg = F.normalize(z_neg, p=2, dim=-1)
        sim_matrix = torch.mm(z_neg, z_neg.t())
        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()
        return diversity_loss

    def refund_cl_loss(self, net_cvr_hidden_states, net_pay_labels, pay_mask=None):
        B = net_cvr_hidden_states.size(0)
        net_cvr_hidden_states = net_cvr_hidden_states.view(B, -1)
        net_pay_labels = net_pay_labels.view(B)
        pay_mask = torch.ones_like(net_pay_labels).bool()
        valid_mask = (pay_mask) 
        if valid_mask.sum() <= 1:
            return net_cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = net_cvr_hidden_states[valid_mask]
        z_neg = F.normalize(z_neg, p=2, dim=-1)
        sim_matrix = torch.mm(z_neg, z_neg.t())
        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()
        return diversity_loss

    def cvr_cl_loss_v2(self, cvr_hidden_states_no_pay, cvr_hidden_states_delay_pay):
        """
        拉远 no-pay 和 delay-pay 用户的表征。
        使用负样本对之间的相似度作为损失，越小越好。
        """
        if cvr_hidden_states_no_pay.size(0) == 0 or cvr_hidden_states_delay_pay.size(0) == 0:
            return cvr_hidden_states_no_pay.new_tensor(0.0, requires_grad=True)

        z_no_pay = F.normalize(cvr_hidden_states_no_pay, p=2, dim=-1)
        z_delay_pay = F.normalize(cvr_hidden_states_delay_pay, p=2, dim=-1)

        sim_matrix = torch.mm(z_no_pay, z_delay_pay.t())

        contrastive_loss = sim_matrix.mean() + 1

        return contrastive_loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)






            self.optimizer.zero_grad()
            net_cvr_outputs = self.model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,pay_labels,net_pay_labels)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_outputs = self.model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pay_labels)
            cvr_loss.backward()
            self.optimizer.step()




            loss = cvr_loss + net_cvr_loss  

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():

                cvr_outputs,net_cvr_outputs = self.model.predict(features)
                
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return None

    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Global_CVR_AUC"] = 0
        all_metrics["Global_NetCVR_AUC"] = 0
        all_metrics["Global_CVR_NLL"] = 0
        all_metrics["Global_NetCVR_NLL"] = 0
        all_metrics["Global_CVR_PCOC"] = 0
        all_metrics["Global_NetCVR_PCOC"] = 0
        all_metrics["Global_CVR_PRAUC"] = 0
        all_metrics["Global_NetCVR_PRAUC"] = 0
        all_pay_labels = []
        all_net_pay_labels = []
        all_pay_preds = []
        all_net_pay_preds = []

        all_metrics["Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = 0
        all_corrected_pay_preds = []
        all_corrected_net_pay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
         
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                cvr_outputs,net_cvr_outputs = Recmodel.predict(features)
                

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(cvr_outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(net_cvr_outputs.cpu().numpy().tolist())

                corrected_cvr_outputs,corrected_net_cvr_outputs = Recmodel.correction_predict(features)
                all_corrected_pay_preds.extend(corrected_cvr_outputs.cpu().numpy().tolist())
                all_corrected_net_pay_preds.extend(corrected_net_cvr_outputs.cpu().numpy().tolist())






        all_metrics["Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"Global_CVR_AUC: {all_metrics['Global_CVR_AUC']:.5f}")
        self.logger.info(f"Global_NetCVR_AUC: {all_metrics['Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CVR_NLL: {all_metrics['Global_CVR_NLL']:.5f}")
        self.logger.info(f"Global_NetCVR_NLL: {all_metrics['Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CVR_PCOC: {all_metrics['Global_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_NetCVR_PCOC: {all_metrics['Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CVR_PRAUC: {all_metrics['Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_NetCVR_PRAUC: {all_metrics['Global_NetCVR_PRAUC']:.5f}")



        all_metrics["Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(all_corrected_net_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"Global_CORRECTED_CVR_AUC: {all_metrics['Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_AUC: {all_metrics['Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_NLL: {all_metrics['Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_NLL: {all_metrics['Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PCOC: {all_metrics['Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PCOC: {all_metrics['Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"Global_CORRECTED_CVR_PRAUC: {all_metrics['Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_CORRECTED_NetCVR_PRAUC: {all_metrics['Global_CORRECTED_NetCVR_PRAUC']:.5f}")

        return None

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainInwTnRefundEsdfmRFTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,inw_logits,tn_logits,refund_labels,tn_labels,inw_refund_labels_afterRefund,delay_refund_labels_afterRefund,eps=1e-6):
    
        inwpay_logits = inw_logits.view(-1)
        tn_logits = tn_logits.view(-1)

        inwpay_labels = inw_refund_labels_afterRefund.float()
        dp_labels = delay_refund_labels_afterRefund.float()  

        tn_labels = tn_labels.float()
        refund_labels = refund_labels.float()

        tn_mask = (1 - refund_labels) + dp_labels
        tn_pos_loss = stable_log1pex(tn_logits)
        tn_neg_loss = tn_logits + stable_log1pex(tn_logits)
        tn_loss = (tn_pos_loss * tn_labels + tn_neg_loss *(1-tn_labels))
        tn_loss = (tn_loss * tn_mask).sum() / (tn_mask.sum() + eps)

        inwpay_pos_loss = stable_log1pex(inwpay_logits)
        inwpay_neg_loss = inwpay_logits + stable_log1pex(inwpay_logits)
        inwpay_loss =torch.mean(inwpay_pos_loss * inwpay_labels + inwpay_neg_loss *(1-inwpay_labels))
        loss = inwpay_loss + tn_loss
        return loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["Inw_AUC"] = 0
        all_metrics["TN_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            refund_labels = batch['refund_labels'].to(self.device)
            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            inw_refund_labels_afterRefund = batch['inw_refund_labels_afterRefund'].to(self.device)
            delay_refund_label_afterRefund = batch['delay_refund_label_afterRefund'].to(self.device)
            tn_labels = ((refund_labels==0)&(delay_refund_label_afterRefund==0)).float()
            self.optimizer.zero_grad()
            outputs = self.model(features)
            inw_logits = outputs[:,0].view(-1)
            tn_logits = outputs[:,1].view(-1)

            loss = self.loss_fn(inw_logits,tn_logits,refund_labels,tn_labels,inw_refund_labels_afterRefund,delay_refund_label_afterRefund)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                inw_probs,tn_probs = self.model.predict(features)
                inwpay_auc= auc_score(inw_refund_labels_afterRefund, inw_probs)
                all_metrics["Inw_AUC"] += inwpay_auc
                tn_auc= auc_score(tn_labels, tn_probs)
                all_metrics["TN_AUC"] += tn_auc


            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["Inw_AUC"] /= len(tqdm_dataloader)
        all_metrics["TN_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: Inw_AUC: {all_metrics['Inw_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: TN_AUC: {all_metrics['TN_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return all_metrics

    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Global_Inw_AUC"] = 0
        all_metrics["Global_Inw_NLL"] = 0
        all_metrics["Global_Inw_PCOC"] = 0
        all_metrics["Global_Inw_PRAUC"] = 0


        all_metrics["Global_TN_AUC"] = 0
        all_metrics["Global_TN_NLL"] = 0
        all_metrics["Global_TN_PCOC"] = 0
        all_metrics["Global_TN_PRAUC"] = 0
        all_inw_labels = []
        all_inw_preds = []
        all_tn_labels = []
        all_tn_preds = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                refund_labels = batch['refund_labels'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                inw_refund_labels_afterRefund = batch['inw_refund_labels_afterRefund'].to(self.device)
                delay_refund_label_afterRefund = batch['delay_refund_label_afterRefund'].to(self.device)

                tn_labels = ((refund_labels==0)&(delay_refund_label_afterRefund==0)).float()
                inw_probs,tn_probs = Recmodel.predict(features)
                all_inw_labels.extend(inw_refund_labels_afterRefund.cpu().numpy().tolist())
                all_inw_preds.extend(inw_probs.cpu().numpy().tolist())
                all_tn_labels.extend(tn_labels.cpu().numpy().tolist())
                all_tn_preds.extend(tn_probs.cpu().numpy().tolist())




        all_metrics["Global_Inw_AUC"] = auc_score(
            torch.tensor(all_inw_labels, dtype=torch.float32),
            torch.tensor(all_inw_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inw_NLL"] = nll_score(
            torch.tensor(all_inw_labels, dtype=torch.float32),
            torch.tensor(all_inw_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inw_PCOC"] = pcoc_score(
            torch.tensor(all_inw_labels, dtype=torch.float32),
            torch.tensor(all_inw_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inw_PRAUC"] = prauc_score(
            torch.tensor(all_inw_labels, dtype=torch.float32),
            torch.tensor(all_inw_preds, dtype=torch.float32)
        )

        all_metrics["Global_TN_AUC"] = auc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_NLL"] = nll_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_PCOC"] = pcoc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_PRAUC"] = prauc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )


        self.logger.info(f"Global_Inw_AUC: {all_metrics['Global_Inw_AUC']:.5f}")
        self.logger.info(f"Global_Inw_NLL: {all_metrics['Global_Inw_NLL']:.5f}")
        self.logger.info(f"Global_Inw_PCOC: {all_metrics['Global_Inw_PCOC']:.5f}")
        self.logger.info(f"Global_Inw_PRAUC: {all_metrics['Global_Inw_PRAUC']:.5f}")
        self.logger.info(f"Global_TN_AUC: {all_metrics['Global_TN_AUC']:.5f}")
        self.logger.info(f"Global_TN_NLL: {all_metrics['Global_TN_NLL']:.5f}")
        self.logger.info(f"Global_TN_PCOC: {all_metrics['Global_TN_PCOC']:.5f}")
        self.logger.info(f"Global_TN_PRAUC: {all_metrics['Global_TN_PRAUC']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
class AliPretrainInwTnPayEsdfmRFTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,inw_logits,tn_logits,pay_labels,tn_labels,inw_pay_labels_afterPay,delay_pay_labels_afterPay,eps=1e-6):
    
        inwpay_logits = inw_logits.view(-1)
        tn_logits = tn_logits.view(-1)

        inwpay_labels = inw_pay_labels_afterPay.float()
        dp_labels = delay_pay_labels_afterPay.float()  
        tn_labels = tn_labels.float()
        pay_labels = pay_labels.float()

        tn_mask = (1 - pay_labels) + dp_labels
        tn_pos_loss = stable_log1pex(tn_logits)
        tn_neg_loss = tn_logits + stable_log1pex(tn_logits)
        tn_loss = (tn_pos_loss * tn_labels + tn_neg_loss *(1-tn_labels))
        tn_loss = (tn_loss * tn_mask).sum() / (tn_mask.sum() + eps)

        inwpay_pos_loss = stable_log1pex(inwpay_logits)
        inwpay_neg_loss = inwpay_logits + stable_log1pex(inwpay_logits)
        inwpay_loss =torch.mean(inwpay_pos_loss * inwpay_labels + inwpay_neg_loss *(1-inwpay_labels))
        loss = inwpay_loss + tn_loss
        return loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["Inwpay_CVR_AUC"] = 0
        all_metrics["TN_CVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)

            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            tn_labels = ((pay_labels==0)&(delay_pay_labels_afterPay==0)).float()
            self.optimizer.zero_grad()
            outputs = self.model(features)
            inw_logits = outputs[:,0].view(-1)
            tn_logits = outputs[:,1].view(-1)
            loss = self.loss_fn(inw_logits,tn_logits,pay_labels,tn_labels,inw_pay_labels_afterPay,delay_pay_labels_afterPay)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                inw_probs,tn_probs = self.model.predict(features)

                inwpay_auc= auc_score(inw_pay_labels_afterPay, inw_probs)
                all_metrics["Inwpay_CVR_AUC"] += inwpay_auc
                tn_auc= auc_score(tn_labels, tn_probs)
                all_metrics["TN_CVR_AUC"] += tn_auc


            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["Inwpay_CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["TN_CVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: Inwpay_CVR_AUC: {all_metrics['Inwpay_CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: TN_CVR_AUC: {all_metrics['TN_CVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return all_metrics

    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Inwpay_CVR_AUC"] = 0
        all_metrics["Global_Inwpay_CVR_AUC"] = 0
        all_metrics["Global_Inwpay_CVR_NLL"] = 0
        all_metrics["Global_Inwpay_CVR_PCOC"] = 0
        all_metrics["Global_Inwpay_CVR_PRAUC"] = 0


        all_metrics["Global_TN_CVR_AUC"] = 0
        all_metrics["Global_TN_CVR_NLL"] = 0
        all_metrics["Global_TN_CVR_PCOC"] = 0
        all_metrics["Global_TN_CVR_PRAUC"] = 0
        all_inwpay_labels = []
        all_inwpay_preds = []
        all_tn_labels = []
        all_tn_preds = []
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                tn_labels = ((pay_labels==0)&(delay_pay_labels_afterPay==0)).float()
                inw_probs,tn_probs = Recmodel.predict(features)
                all_inwpay_labels.extend(inw_pay_labels_afterPay.cpu().numpy().tolist())
                all_inwpay_preds.extend(inw_probs.cpu().numpy().tolist())
                all_tn_labels.extend(tn_labels.cpu().numpy().tolist())
                all_tn_preds.extend(tn_probs.cpu().numpy().tolist())

                with torch.no_grad():
                    inwpay_cvr_auc = auc_score(inw_pay_labels_afterPay,inw_probs)
                    all_metrics["Inwpay_CVR_AUC"] += inwpay_cvr_auc


        all_metrics["Inwpay_CVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Inwpay_CVR_AUC: {all_metrics['Inwpay_CVR_AUC']:.5f}")

        all_metrics["Global_Inwpay_CVR_AUC"] = auc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_NLL"] = nll_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )

        all_metrics["Global_TN_CVR_AUC"] = auc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_CVR_NLL"] = nll_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )
        all_metrics["Global_TN_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_tn_labels, dtype=torch.float32),
            torch.tensor(all_tn_preds, dtype=torch.float32)
        )


        self.logger.info(f"Global_Inwpay_CVR_AUC: {all_metrics['Global_Inwpay_CVR_AUC']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_NLL: {all_metrics['Global_Inwpay_CVR_NLL']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_PCOC: {all_metrics['Global_Inwpay_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_PRAUC: {all_metrics['Global_Inwpay_CVR_PRAUC']:.5f}")
        self.logger.info(f"Global_TN_CVR_AUC: {all_metrics['Global_TN_CVR_AUC']:.5f}")
        self.logger.info(f"Global_TN_CVR_NLL: {all_metrics['Global_TN_CVR_NLL']:.5f}")
        self.logger.info(f"Global_TN_CVR_PCOC: {all_metrics['Global_TN_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_TN_CVR_PRAUC: {all_metrics['Global_TN_CVR_PRAUC']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliPretrainInwPayEsdfmRFTrainer(metaclass=ABCMeta):
    def __init__(self, args,model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.model=model
        self.model.to(self.device)
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def loss_fn(self,inw_logits,pay_labels,inw_pay_labels_afterPay,eps=1e-6):
    
        inwpay_logits = inw_logits.view(-1)
        inwpay_labels = inw_pay_labels_afterPay.float()
        pay_labels = pay_labels.float()

        inwpay_pos_loss = stable_log1pex(inwpay_logits)
        inwpay_neg_loss = inwpay_logits + stable_log1pex(inwpay_logits)
        inwpay_loss =torch.mean(inwpay_pos_loss * inwpay_labels + inwpay_neg_loss *(1-inwpay_labels))
        loss = inwpay_loss 
        return loss

    def train(self):
        for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
            output = self.train_one_epoch(epoch_idx)
            self.test()

    def train_one_epoch(self,epoch_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.model.train()
        tqdm_dataloader = tqdm(self.train_loader)
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["Inwpay_CVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)

            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            self.optimizer.zero_grad()
            inw_logits = self.model(features)
            loss = self.loss_fn(inw_logits,pay_labels,inw_pay_labels_afterPay)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                inw_probs= self.model.predict(features)

                inwpay_auc= auc_score(inw_pay_labels_afterPay, inw_probs)
                all_metrics["Inwpay_CVR_AUC"] += inwpay_auc


            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["Inwpay_CVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: Inwpay_CVR_AUC: {all_metrics['Inwpay_CVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")
        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1}")
        return all_metrics

    def test(self):
        self.logger.info('Testing best model with test set!')
        Recmodel = copy.deepcopy(self.model)
        Recmodel.eval()
        all_metrics = {}
        all_metrics["Inwpay_CVR_AUC"] = 0
        all_metrics["Global_Inwpay_CVR_AUC"] = 0
        all_metrics["Global_Inwpay_CVR_NLL"] = 0
        all_metrics["Global_Inwpay_CVR_PCOC"] = 0
        all_metrics["Global_Inwpay_CVR_PRAUC"] = 0



        all_inwpay_labels = []
        all_inwpay_preds = []

        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                
                inw_probs = Recmodel.predict(features)
                all_inwpay_labels.extend(inw_pay_labels_afterPay.cpu().numpy().tolist())
                all_inwpay_preds.extend(inw_probs.cpu().numpy().tolist())



        all_metrics["Global_Inwpay_CVR_AUC"] = auc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_NLL"] = nll_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )
        all_metrics["Global_Inwpay_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_inwpay_labels, dtype=torch.float32),
            torch.tensor(all_inwpay_preds, dtype=torch.float32)
        )



        self.logger.info(f"Global_Inwpay_CVR_AUC: {all_metrics['Global_Inwpay_CVR_AUC']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_NLL: {all_metrics['Global_Inwpay_CVR_NLL']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_PCOC: {all_metrics['Global_Inwpay_CVR_PCOC']:.5f}")
        self.logger.info(f"Global_Inwpay_CVR_PRAUC: {all_metrics['Global_Inwpay_CVR_PRAUC']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRFStreamTrainer(metaclass=ABCMeta):

    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)


    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
        return cvr_loss
    

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)

        stream_refund_labels = ((1-stream_pay_mask)>0) | ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = (stream_pay_labels>0) | (stream_refund_labels > 0)
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()

        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)

        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def refund_loss_fn2(self,refund_outputs,cvr_outputs,pretrain_refund_outputs,inw_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)

        stream_refund_labels = ((1-stream_pay_mask)>0) | ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = (stream_pay_labels>0) | (stream_refund_labels > 0)
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        cvr_outputs = cvr_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = ((1 - pretrain_refund_prob) * (1 + pretrain_refund_prob - inw_refund_prob)) / (1 - inw_refund_prob)

        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)

        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)


        return refund_loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}
        all_metrics["stream_model_CVR_AUC"] = 0
        all_metrics["stream_model_NetCVR_AUC"] = 0

        all_metrics["pretrained_model_CVR_AUC"] = 0
        all_metrics["pretrained_model_NetCVR_AUC"] = 0

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0


        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []

        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())

  
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, stream_cvr_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,stream_net_cvr_outputs)
                    all_metrics["stream_model_NetCVR_AUC"] += stream_model_net_cvr_auc

                    pretrained_model_cvr_auc = auc_score(pay_labels, pretrained_cvr_outputs)
                    all_metrics["pretrained_model_CVR_AUC"] += pretrained_model_cvr_auc

                    pretrained_model_net_cvr_auc = auc_score(net_pay_labels,pretrained_net_cvr_outputs)
                    all_metrics["pretrained_model_NetCVR_AUC"] += pretrained_model_net_cvr_auc

        all_metrics["stream_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["stream_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)


        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_CVR_AUC: {all_metrics['stream_model_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_NetCVR_AUC: {all_metrics['stream_model_NetCVR_AUC']:.5f}")

        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRFStreamTrainerV2(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.bpr_optimizer = self.create_bpr_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def extra_cvr_loss_fn(self,extra_cvr_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        cvr_logits = extra_cvr_outputs.view(-1)
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*stream_pay_labels + neg_loss*(1-stream_pay_labels))*stream_pay_mask)
        return cvr_loss

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn2(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
    
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)


        valid_mask = stream_pay_mask.bool()
        valid_indices = torch.where(valid_mask)[0]
        valid_cvr = cvr_prob[valid_mask]
        valid_refund = refund_prob[valid_mask]
        _, cvr_order = torch.sort(valid_cvr, descending=True)
        cvr_rank = torch.empty_like(valid_cvr).long()
        cvr_rank[cvr_order] = torch.arange(len(valid_cvr), device=cvr_prob.device)
        _, refund_order = torch.sort(valid_refund, descending=False)
        refund_rank = torch.empty_like(valid_refund).long()
        refund_rank[refund_order] = torch.arange(len(valid_refund), device=refund_prob.device)
        B_valid = len(valid_cvr)
        score = (B_valid - cvr_rank) + (B_valid - refund_rank)
        k =  int(len(score) * 0.3)
        topk_scores, topk_idx_in_valid = torch.topk(score, k=k, largest=True)
        topk_indices_in_batch = valid_indices[topk_idx_in_valid]
        corrected_labels = stream_pay_labels.clone()
        mask = (corrected_labels[topk_indices_in_batch] == 0) & (cvr_prob[topk_indices_in_batch] > 0.75)
        indices_to_correct = topk_indices_in_batch[mask]
        if len(indices_to_correct) > 0:
            corrected_labels[indices_to_correct] = 1
            stream_pay_mask[indices_to_correct] = 0
        stream_pay_labels = corrected_labels

        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_for_neg_random_neg(self,cvr_outputs, stream_pay_labels, stream_pay_mask, refund_outputs,num_pos_samples=30, num_neg_samples=50):

        mask = (stream_pay_labels == 0) & (stream_pay_mask == 1)

        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)
        
        cvr_valid = cvr_outputs[valid_indices]
        refund_valid = refund_outputs[valid_indices]

        cvr_valid_prob = torch.sigmoid(cvr_valid).detach()
        refund_valid_prob = torch.sigmoid(refund_valid).detach()

        ranking_score = cvr_valid_prob * ((1 - refund_valid_prob) ** 2)

        sorted_indices = torch.argsort(ranking_score, descending=True)
        
        num_pos = min(num_pos_samples, len(sorted_indices))
        if num_pos == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)

        pos_indices_in_valid = sorted_indices[:num_pos]
        neg_indices_in_valid = sorted_indices[num_pos:]

        if len(neg_indices_in_valid) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)

        pos_logits = cvr_valid[pos_indices_in_valid]

        neg_sampled_indices = torch.randint(
            low=0,
            high=len(neg_indices_in_valid),
            size=(num_pos, num_neg_samples),
            device=cvr_outputs.device
        )
        neg_indices_for_each_pos = neg_indices_in_valid[neg_sampled_indices]
        neg_logits = cvr_valid[neg_indices_for_each_pos]

        bpr_loss = -torch.log(torch.sigmoid(pos_logits.unsqueeze(1) - neg_logits)).mean()

        return bpr_loss

    def bpr_loss_for_neg_random_neg_v2(self, cvr_outputs, stream_pay_labels, stream_pay_mask, refund_outputs, 
                                    num_pos_samples=10, num_neg_samples=50):
        mask = (stream_pay_labels == 0) & (stream_pay_mask == 1)
        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)
        
        cvr_valid = cvr_outputs[valid_indices]
        refund_valid = refund_outputs[valid_indices]

        cvr_valid_prob = torch.sigmoid(cvr_valid).detach()
        refund_valid_prob = torch.sigmoid(refund_valid).detach()

        ranking_score = cvr_valid_prob * ((1 - refund_valid_prob)**2)

        sorted_indices = torch.argsort(ranking_score, descending=True)
        
        num_pos = min(num_pos_samples, len(sorted_indices))
        if num_pos == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)

        pos_indices_in_valid = sorted_indices[:num_pos]
        neg_indices_in_valid = sorted_indices[num_pos:]

        if len(neg_indices_in_valid) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device, requires_grad=True)

        pos_logits = cvr_valid[pos_indices_in_valid]
        neg_ranking_scores = ranking_score[neg_indices_in_valid]
        neg_weights = 1-neg_ranking_scores
        neg_weights = neg_weights / (neg_weights.sum() + 1e-8)

        num_neg_candidates = len(neg_indices_in_valid)
        sampled_neg_rel_indices = torch.multinomial(
            neg_weights,
            num_samples=num_pos * num_neg_samples,
            replacement=False
        )

        neg_selected_rel = sampled_neg_rel_indices % num_neg_candidates
        neg_selected_in_valid = neg_indices_in_valid[neg_selected_rel]

        neg_selected_in_valid = neg_selected_in_valid.view(num_pos, num_neg_samples)
        neg_logits = cvr_valid[neg_selected_in_valid]

        diff = pos_logits.unsqueeze(1) - neg_logits
        bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        return bpr_loss

    def bpr_loss_for_neg_random_neg_v3(self, cvr_outputs, stream_pay_labels, stream_pay_mask, refund_outputs, 
                                    num_pos_samples=30, num_neg_samples=50, 
                                    score_threshold=0.6):
        """
        Args:
            score_threshold (float): 伪正例的 ranking_score 必须 >= 此阈值
        """
        device = cvr_outputs.device
        stream_pay_labels = stream_pay_labels.view(-1)
        stream_pay_mask = stream_pay_mask.view(-1)
        cvr_outputs = cvr_outputs.view(-1)
        refund_outputs = refund_outputs.view(-1)

        mask = (stream_pay_labels == 0) & (stream_pay_mask == 1)
        valid_indices = torch.where(mask)[0]
        
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        cvr_valid = cvr_outputs[valid_indices]
        refund_valid = refund_outputs[valid_indices]

        cvr_valid_prob = torch.sigmoid(cvr_valid).detach()
        refund_valid_prob = torch.sigmoid(refund_valid).detach()

        ranking_score = cvr_valid_prob * ((1 - refund_valid_prob) ** 2)

        high_score_mask = ranking_score >= score_threshold
        print(torch.sum(high_score_mask))
        if not high_score_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        candidate_indices = torch.arange(len(ranking_score), device=device)[high_score_mask]
        candidate_scores = ranking_score[high_score_mask]

        sorted_in_candidate = torch.argsort(candidate_scores, descending=True)
        selected_in_candidate = sorted_in_candidate[:num_pos_samples]

        pos_candidates_in_valid = candidate_indices[selected_in_candidate]
        num_pos = len(pos_candidates_in_valid)

        if num_pos == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        remaining_mask = torch.ones(len(ranking_score), dtype=torch.bool, device=device)
        remaining_mask[pos_candidates_in_valid] = False
        neg_candidates_in_valid = torch.arange(len(ranking_score), device=device)[remaining_mask]

        if len(neg_candidates_in_valid) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        pos_logits = cvr_valid[pos_candidates_in_valid]

        neg_ranking_scores = ranking_score[neg_candidates_in_valid]
        neg_weights = 1.0 - neg_ranking_scores + 1e-6
        neg_weights = neg_weights / (neg_weights.sum() + 1e-8)

        sampled_rel_indices = torch.multinomial(
            neg_weights,
            num_samples=num_pos * num_neg_samples,
            replacement=True
        )
        neg_selected_in_valid = neg_candidates_in_valid[sampled_rel_indices]
        neg_selected_in_valid = neg_selected_in_valid.view(num_pos, num_neg_samples)
        neg_logits = cvr_valid[neg_selected_in_valid]

        diff = pos_logits.unsqueeze(1) - neg_logits
        bpr_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        return bpr_loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def cvr_loss_fn_wi_smooth(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,smooth_factor,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1).clone()
        smooth_factor = smooth_factor.view(-1)

 
        smooth_mask = (stream_pay_labels == 0)


        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def cvr_loss_fn4(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        retention_score = cvr_prob 
        stream_pay_labels = self.smooth_binary_labels_with_score(stream_pay_labels,retention_score)


        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def smooth_binary_labels_with_score(self,labels, scores, threshold=0.2,min_smooth=0.1,max_smooth=1):
        soft_labels = labels.float().clone()
        mask = (labels == 0) & (scores > threshold)
        if mask.sum() == 0:
            return soft_labels


        soft_value = min_smooth + scores[mask]
        soft_labels[mask] = soft_value

        return soft_labels
    
    def drop_and_construct_sample(self,features,pay_labels,net_pay_labels,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,threshold=0.6):
        eval_model =  copy.deepcopy(self.stream_model)
        eval_model.to(self.device)
        eval_model.eval()

        cvr_probs,net_cvr_probs = eval_model.predict(features)
        refund_logits = eval_model.net_cvr_forward(features)
        refund_probs = torch.sigmoid(refund_logits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

        valid_mask = stream_pay_mask.bool()
        valid_indices = torch.where(valid_mask)[0]
        valid_cvr = cvr_probs[valid_mask]
        valid_net_cvr = net_cvr_probs[valid_mask]
        valid_refund = refund_probs[valid_mask]
        valid_stream_pay_labels = stream_pay_labels[valid_mask]

        retention_score = valid_cvr*((1-valid_refund)**2) 
        replace_mask = (valid_stream_pay_labels == 0) & (retention_score > threshold)
        print(torch.sum(replace_mask))
        if not replace_mask.any():
            return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask
        
        negative_mask = (valid_stream_pay_labels == 0) & ~replace_mask 
        negative_indices = torch.where(negative_mask)[0]
        
        if len(negative_indices) == 0:
            return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask

        replace_indices = torch.randint(len(negative_indices), (replace_mask.sum(),))
        source_indices = negative_indices[replace_indices]

        target_indices = torch.where(replace_mask)[0]
        stream_pay_mask[valid_indices[target_indices]] = 0
        return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask

    def make_smooth_labels_for_negatives(self,smooth_factor, max_value=0.1, sharpness=2.0):
        x = smooth_factor**sharpness
        return x * max_value

    def pearson_neg_loss(self,cvr_outputs, refund_outputs):
        cvr_prob = torch.sigmoid(cvr_outputs)
        refund_prob = torch.sigmoid(refund_outputs)

        cvr_centered = cvr_prob - cvr_prob.mean()
        refund_centered = refund_prob - refund_prob.mean()

        cov = (cvr_centered * refund_centered).sum()
        std_cvr = torch.sqrt((cvr_centered ** 2).sum() + 1e-8)
        std_refund = torch.sqrt((refund_centered ** 2).sum() + 1e-8)

        pearson_corr = cov / (std_cvr * std_refund + 1e-8)

        return pearson_corr
    
    def cvr_selfsupervised_loss(self, x, stream_pay_labels, stream_pay_mask,
                            noise_std=0.01, dropout_rate=0.1,
                            mse_weight=1.0, whitening_weight=0.5):
   
        device = x.device
        labels = stream_pay_labels.view(-1)
        mask = stream_pay_mask.view(-1)

        pos_mask = (labels == 1) & (mask == 1)
        pos_indices = torch.where(pos_mask)[0]
        
        if len(pos_indices) < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        x_pos = x[pos_indices]

        cvr_logits1, cvr_logits2 = self.stream_model.cvr_pos_agument_features(
            x_pos, noise_std=noise_std, dropout_rate=dropout_rate
        )  
        mse_loss = F.mse_loss(cvr_logits1, cvr_logits2)
        z_all = torch.cat([cvr_logits1.unsqueeze(-1), cvr_logits2.unsqueeze(-1)], dim=0)

        def compute_whitening_loss(z):
            if len(z) < 2:
                return torch.tensor(0.0, device=z.device, requires_grad=True)
            z = z - z.mean(dim=0, keepdim=True)
            cov = (z.T @ z) / (z.size(0) - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return off_diag.pow(2).sum()

        whitening_loss = compute_whitening_loss(z_all)

        total_loss = mse_weight * mse_loss 
        return total_loss

    def refund_loss_fn_wi_DelayTimeBpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        bprloss = self.DelayTime_bpr_loss_weighted_neg(refund_outputs, stream_refund_labels, stream_refund_mask, delay_refund_time,5)

        total_loss = refund_loss + bprloss
        return total_loss 

    def cvr_loss_fn_wi_DelayTimeBpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.DelayTime_bpr_loss_weighted_neg(cvr_outputs, stream_pay_labels, stream_pay_mask ,delay_pay_time)

        loss= bprloss + cvr_loss 

        return loss

    def DelayTime_bpr_loss_weighted_neg(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=outputs.device)

        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]

        neg_scores = valid_outputs[neg_mask]
        neg_probs = torch.sigmoid(valid_outputs[neg_mask]).detach()

        def compute_adaptive_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w

        pos_weights = compute_adaptive_weight(pos_delay_all)

        neg_weights = (1 - neg_probs).float()
        if neg_weights.sum() <= 0:
            neg_weights = torch.ones_like(neg_weights)
        neg_weights /= neg_weights.sum()

        neg_idx = torch.multinomial(
            neg_weights,
            len(pos_scores_all) * num_neg_samples,
            replacement=True
        )
        neg_idx = neg_idx.view(len(pos_scores_all), num_neg_samples)
        neg_samples = neg_scores[neg_idx]

        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff)).mean(dim=1)

        weighted_loss = (pos_weights * per_sample_loss).sum() / pos_weights.sum()

        return weighted_loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch_DelayTimeSampleBpr(epoch_idx,day)
                
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)



            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()




            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss = cvr_loss
            cvr_loss.backward()
            self.optimizer.step()


            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1





                







            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_DelayTimeSampleBpr(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)
            delay_pay_time = pay_ts - click_ts
            delay_pay_time = delay_pay_time / 3600
            delay_pay_time = delay_pay_time.to(self.device)
            
            delay_refund_time = refund_ts - pay_ts
            delay_refund_time = delay_refund_time / 3600
            delay_refund_time = delay_refund_time.to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn_wi_DelayTimeBpr(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time)
            net_cvr_loss.backward()
            self.optimizer.step()





            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_DelayTimeBpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time)
            cvr_loss = cvr_loss
            cvr_loss.backward()
            self.optimizer.step()


            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []




        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())





        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    def create_bpr_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=1e-2, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_PLE_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def cvr_loss_fn_wi_DelayTimeBpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.DelayTime_bpr_loss_weighted_neg(cvr_outputs, stream_pay_labels, stream_pay_mask ,delay_pay_time)
        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def DelayTime_bpr_loss_weighted_neg1(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        """
        加权 BPR 损失，使用 Top-k 不确定负样本采样策略
        """
        valid_indices = torch.where(mask)[0]
        if len(valid_indices) == 0:
            return torch.tensor(0.0, device=outputs.device)

        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=outputs.device)

        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]

        neg_scores = valid_outputs[neg_mask]
        neg_probs = torch.sigmoid(neg_scores).detach()
        neg_delay = valid_delay[neg_mask]

        def compute_pos_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w

        pos_weights = compute_pos_weight(pos_delay_all)
        k_candidate = min(3000, len(neg_scores))
        uncertainty = (1 - neg_probs)
        _, topk_indices = torch.topk(uncertainty, k=k_candidate)
        candidate_uncertainty = uncertainty[topk_indices]
        candidate_weights = torch.softmax(candidate_uncertainty, dim=0)
        neg_idx_list = []
        for i in range(len(pos_scores_all)):
            idx_in_candidate = torch.multinomial(candidate_weights, num_neg_samples, replacement=True)
            global_neg_idx = topk_indices[idx_in_candidate]
            neg_idx_list.append(global_neg_idx)
        neg_idx = torch.stack(neg_idx_list)
        neg_samples = neg_scores[neg_idx]

        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean(dim=1)
        weighted_loss = (pos_weights * per_sample_loss).sum() / (pos_weights.sum() + 1e-8)
        return weighted_loss

    def DelayTime_bpr_loss_weighted_neg(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=outputs.device)

        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]

        neg_scores = valid_outputs[neg_mask]
        neg_probs = torch.sigmoid(valid_outputs[neg_mask]).detach()
        def compute_adaptive_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w
        pos_weights = compute_adaptive_weight(pos_delay_all)
        neg_weights = (1 - neg_probs).float()


        if neg_weights.sum() <= 0:
            neg_weights = torch.ones_like(neg_weights)
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(
            neg_weights,
            len(pos_scores_all) * num_neg_samples,
            replacement=True
        )
        neg_idx = neg_idx.view(len(pos_scores_all), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff)).mean(dim=1)
        weighted_loss = (pos_weights * per_sample_loss).sum() / pos_weights.sum()
        return weighted_loss

    def DelayTime_bpr_loss_weighted_neg3(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]
        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)
        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]
        neg_scores_all = valid_outputs[neg_mask]
        neg_probs_all = torch.sigmoid(neg_scores_all).detach()
        sorted_probs, sorted_indices = torch.sort(neg_probs_all, descending=True)
        last_in_top_40_percent = sorted_probs[int(len(neg_probs_all) * 0.4) - 1]
        last_in_top_50_percent = sorted_probs[int(len(neg_probs_all) * 0.4) - 1] 

        print(f"{last_in_top_50_percent.item():.4f}")
        print(neg_probs_all.max(),neg_probs_all.min())
        def compute_adaptive_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w
        pos_weights = torch.ones_like(pos_delay_all)
        neg_uncertainty = 1 - torch.abs(neg_probs_all)
        neg_weights_raw = neg_uncertainty.float().detach()
        num_neg_total = len(neg_weights_raw)
        _, sorted_indices = torch.sort(neg_weights_raw, descending=True)
        split_idx = math.ceil(num_neg_total // 10 * 4.5)
        if split_idx == 0:
            split_idx = 1
        top_hard_indices = sorted_indices[:split_idx]
        bottom_easy_indices = sorted_indices[split_idx:]

        hard_scores = neg_scores_all[top_hard_indices]
        easy_scores = neg_scores_all[bottom_easy_indices]
        hard_weights = neg_weights_raw[top_hard_indices]
        easy_weights = neg_weights_raw[bottom_easy_indices]
        def normalize_weights(w):
            if w.sum() <= 0:
                return torch.ones_like(w) / len(w)
            return w / w.sum()
        hard_weights_norm = normalize_weights(hard_weights)
        easy_weights_norm = normalize_weights(easy_weights)
        num_hard = int(num_neg_samples * 0.5)
        num_easy = num_neg_samples - num_hard
        num_hard = max(1, num_hard)
        num_easy = max(0, num_easy)
        num_pos = len(pos_scores_all)
        all_hard_samples = []
        all_easy_samples = []
        try:
            if num_hard > 0 and len(hard_weights_norm) > 0:
                hard_sampled_idx = torch.multinomial(
                    hard_weights_norm,
                    num_pos * num_hard,
                    replacement=True
                )
                hard_sampled_scores = hard_scores[hard_sampled_idx]
                all_hard_samples = hard_sampled_scores.view(num_pos, num_hard)
            if num_easy > 0 and len(easy_weights_norm) > 0:
                easy_sampled_idx = torch.multinomial(
                    easy_weights_norm,
                    num_pos * num_easy,
                    replacement=True
                )
                easy_sampled_scores = easy_scores[easy_sampled_idx]
                all_easy_samples = easy_sampled_scores.view(num_pos, num_easy)
            else:
                all_easy_samples = torch.empty(num_pos, 0, device=outputs.device)
        except RuntimeError as e:
            print(f"[Sampling Error] {e}, falling back to uniform sampling.")
            fallback_weights = torch.ones(num_neg_total, device=outputs.device)
            fallback_idx = torch.multinomial(
                fallback_weights,
                num_pos * num_neg_samples,
                replacement=True
            )
            fallback_samples = neg_scores_all[fallback_idx].view(num_pos, num_neg_samples)
            neg_samples = fallback_samples
            use_fallback = True
        else:
            use_fallback = False
        if not use_fallback:
            if all_easy_samples.size(1) == 0:
                neg_samples = all_hard_samples
            else:
                neg_samples = torch.cat([all_hard_samples, all_easy_samples], dim=1)
        else:
            neg_samples = fallback_samples
        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean(dim=1)
        weighted_loss = (pos_weights * per_sample_loss).sum() / (pos_weights.sum() + 1e-8)
        return weighted_loss

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def refund_loss_fn_wi_bpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)


        bprloss = self.bpr_loss_random_neg(refund_outputs, stream_refund_labels, stream_refund_mask, 20)

        total_loss = refund_loss + bprloss
        return total_loss   

    def refund_loss_fn_wi_DelayTimeBpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        bprloss = self.DelayTime_bpr_loss_weighted_neg(refund_outputs, stream_refund_labels, stream_refund_mask, delay_refund_time,5)

        total_loss = refund_loss + 0.1*bprloss
        return total_loss   

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch_DelayTimeSampleBpr(epoch_idx,day)
                
                
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def train_one_epoch_with_groundtruth(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None


        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            if i == 0 :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = delay_pay_labels_afterPay.clone()
                last_update_labels = torch.zeros_like(pay_labels)
            if (i > 0) & (i < 4):
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.zeros_like(pay_labels)),dim=0)
            if i == 4:
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.ones_like(pay_labels)),dim=0)
            if i == 4 :
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_groundtruth_cvr_hidden_states_with_labels_train_before.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            
            stream_pay_mask = (stream_pay_mask == 1) & (delay_pay_labels_afterPay == 0)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()
            
            loss = cvr_loss + net_cvr_loss


            if i == 4:
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_groundtruth_cvr_hidden_states_with_labels_train_after.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_DelayTimeSampleBpr(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None


        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)
            delay_pay_time = pay_ts - click_ts
            delay_pay_time = delay_pay_time / 3600
            delay_pay_time = delay_pay_time.to(self.device)
            
            delay_refund_time = refund_ts - pay_ts
            delay_refund_time = delay_refund_time / 3600
            delay_refund_time = delay_refund_time.to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn_wi_DelayTimeBpr(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time)
            # net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            
            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_DelayTimeBpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time)
            cvr_loss.backward()
            self.optimizer.step()
            
            loss = cvr_loss + net_cvr_loss

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None


        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            
            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()
            
            loss = cvr_loss + net_cvr_loss

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_for_PreExp1(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None


        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            if i != 0:
                continue

            else :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = (stream_pay_labels == 0) & (pay_labels == 1)
                last_update_labels = torch.zeros_like(pay_labels)

                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_cvr_hidden_states_with_labels_train_before_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                stream_pay_mask = (stream_pay_labels == 0) & (pay_labels == 1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
                cvr_loss.backward()
                self.optimizer.step()

                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_cvr_hidden_states_with_labels_train_wrong_dp_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()

                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                stream_pay_mask = (stream_pay_labels == 0) & (pay_labels == 1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pay_labels,stream_pay_mask)
                cvr_loss.backward()
                self.optimizer.step()


                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_cvr_hidden_states_with_labels_train_correct_dp_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()

            loss = cvr_loss 

            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_for_PreExp1_orcale(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None


        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            if i != 0:
                continue
            else :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = (stream_pay_labels == 0) & (pay_labels == 1)

                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_cvr_hidden_states_with_labels_train_oracle_before_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()

                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                stream_pay_mask = (stream_pay_labels == 0) & (pay_labels == 1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pay_labels,stream_pay_mask)
                cvr_loss = cvr_loss
                cvr_loss.backward()
                self.optimizer.step()
                
                loss = cvr_loss 



                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_stream_cvr_hidden_states_with_labels_train_oracle_after_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


                total_loss += loss.item()
                total_batches += 1

                with torch.no_grad():
                    cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                    cvr_auc = auc_score(pay_labels, cvr_outputs)
                    all_metrics["CVR_AUC"] += cvr_auc
                    net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

                tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
            mean_loss = total_loss / total_batches
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
            all_metrics["CVR_AUC"] /= total_batches
            all_metrics["NetCVR_AUC"] /= total_batches
            self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
            self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
            self.scheduler.step()
            self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

            parent_dir = os.path.dirname(self.model_pth)
            if not os.path.exists(parent_dir):
                os.makedirs(parent_dir)
            torch.save(self.stream_model.state_dict(), self.model_pth)
            self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
            return all_metrics

    def test_v2(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []






        all_metrics["stream_model_inws_Global_CVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_CVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_inws = []
        all_net_pay_labels_inws = []
        stream_model_all_pay_preds_inws = []
        stream_model_all_net_pay_preds_inws = []


        all_metrics["stream_model_dp_Global_CVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_CVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_dp = []
        all_net_pay_labels_dp = []
        stream_model_all_pay_preds_dp = []
        stream_model_all_net_pay_preds_dp = []

        all_metrics["stream_model_Global_POS_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NEG_CVR_NLL"] = 0

        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                inws_mask = (inw_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_inws.extend(pay_labels[inws_mask].cpu().numpy().tolist())
                all_net_pay_labels_inws.extend(net_pay_labels[inws_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_inws.extend(stream_cvr_outputs[inws_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_inws.extend(stream_net_cvr_outputs[inws_mask].cpu().numpy().tolist())

                dp_mask = (delay_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_dp.extend(pay_labels[dp_mask].cpu().numpy().tolist())
                all_net_pay_labels_dp.extend(net_pay_labels[dp_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_dp.extend(stream_cvr_outputs[dp_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_dp.extend(stream_net_cvr_outputs[dp_mask].cpu().numpy().tolist())

                stream_cvr_hidden_states = stream_model.get_cvr_hidden_state(features)
                stream_cvr_hidden_states = np.array(stream_cvr_hidden_states.cpu())
                pay_labels = np.array(pay_labels.cpu())
                delay_pay_labels_afterPay = np.array(delay_pay_labels_afterPay.cpu())
                hidden_state_list = [vec.tolist() for vec in stream_cvr_hidden_states]
                df = pd.DataFrame({
                    'cvr_hidden_state': hidden_state_list,
                    'pay_label': pay_labels,
                    'delay_pay_label_afterPay': delay_pay_labels_afterPay
                })

                csv_file = 'esdfmRF_ple_stream_cvr_hidden_states_with_labels.csv'
                file_exists = os.path.isfile(csv_file)

                df.to_csv(
                    csv_file,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )


        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_inws_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_inws_Global_CVR_AUC: {all_metrics['stream_model_inws_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_AUC: {all_metrics['stream_model_inws_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_NLL: {all_metrics['stream_model_inws_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_NLL: {all_metrics['stream_model_inws_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_PRAUC: {all_metrics['stream_model_inws_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_PRAUC: {all_metrics['stream_model_inws_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_AUC: {all_metrics['stream_model_dp_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_AUC: {all_metrics['stream_model_dp_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_NLL: {all_metrics['stream_model_dp_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_NLL: {all_metrics['stream_model_dp_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_PRAUC: {all_metrics['stream_model_dp_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_PRAUC: {all_metrics['stream_model_dp_Global_NetCVR_PRAUC']:.5f}")

        all_metrics["stream_model_Global_POS_CVR_NLL"],all_metrics["stream_model_Global_NEG_CVR_NLL"] = nll_score_split(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"stream_model_Global_POS_CVR_NLL: {all_metrics['stream_model_Global_POS_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NEG_CVR_NLL: {all_metrics['stream_model_Global_NEG_CVR_NLL']:.5f}")

        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []




        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())





        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_PLE_v2_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        

        self.prev_batch = None
        self.prev_up_dates = 0 
        self.prev_up_dates_flag = 5

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def refund_loss_fn_wi_bpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)


        bprloss = self.bpr_loss_random_neg(refund_outputs, stream_refund_labels, stream_refund_mask, 20)

        total_loss = refund_loss + bprloss
        return total_loss    

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, stream_pay_labels , stream_pay_mask):
    

        corrected_cvr_logits = cvr_outputs +  correction_outputs
        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_pay_mask = stream_pay_mask.view(-1)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,stream_pay_labels,stream_pay_mask)


        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, stream_pay_labels, stream_pay_mask):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        stream_pay_labels = stream_pay_labels.view(B)
        stream_pay_mask = stream_pay_mask.view(B).bool()

        valid_neg_mask = (stream_pay_mask) & (stream_pay_labels == 0)
        if valid_neg_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)
        z_neg = cvr_hidden_states[valid_neg_mask]

        z_neg = F.normalize(z_neg, p=2, dim=-1)

        sim_matrix = torch.mm(z_neg, z_neg.t())

        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()

        return diversity_loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def check_updated_model_is_good(self,batch_num,batch,epoch_idx,day_idx):
        if (batch_num == 0) & (day_idx == 0) :
            self.prev_batch = {
                'features': batch['features'].to(self.device), 
                'pay_labels': batch['pay_labels'].to(self.device),
                'net_pay_labels': batch['net_pay_labels'].to(self.device),
                'stream_pay_labels': batch['stream_pay_labels'].to(self.device),
                'stream_net_pay_labels': batch['stream_net_pay_labels'].to(self.device),
                'stream_pay_mask': batch['stream_pay_mask'].to(self.device)
            }

        updated_stream_model = copy.deepcopy(self.stream_model)
        updated_stream_model.to(self.device)
        updated_stream_model.eval()

        last_batch_features = self.prev_batch['features']
        last_batch_pay_labels = self.prev_batch['pay_labels']
        last_batch_net_pay_labels = self.prev_batch['net_pay_labels']
        last_batch_stream_pay_labels = self.prev_batch['pay_labels']
        last_batch_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels']
        last_batch_stream_pay_mask = self.prev_batch['stream_pay_mask']

        last_batch_cvr_outputs,last_batch_net_cvr_outputs = self.last_batch_stream_model.predict(last_batch_features)
        last_batch_updated_cvr_outputs,last_batch_updated_net_cvr_outputs = updated_stream_model.predict(last_batch_features)
        last_batch_cvr_outputs = last_batch_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_updated_cvr_outputs = last_batch_updated_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_stream_pay_labels = last_batch_stream_pay_labels[last_batch_stream_pay_mask.bool()]
        last_batch_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC: {last_batch_cvr_auc:.5f} - Updated CVR AUC: {last_batch_updated_cvr_auc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR PRAUC: {last_batch_cvr_prauc:.5f} - Updated CVR PRAUC: {last_batch_updated_cvr_prauc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR NLL: {last_batch_cvr_nll:.5f} - Updated CVR NLL: {last_batch_updated_cvr_nll:.5f}")
        if last_batch_updated_cvr_auc > last_batch_cvr_auc:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is improved, continue training")
            self.prev_up_dates += 1
            if self.prev_up_dates == self.prev_up_dates_flag:
                self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}  Batch {batch_num}- update calibrated model")
                self.last_batch_stream_model = copy.deepcopy(updated_stream_model)
                self.last_batch_stream_model.to(self.device)
                self.last_batch_stream_model.eval()

                curr_features = batch['features'].to(self.device)
                curr_pay_labels = batch['pay_labels'].to(self.device)
                curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
                curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
                curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
                curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)
                last_features = self.prev_batch['features'].to(self.device)
                last_pay_labels = self.prev_batch['pay_labels'].to(self.device)
                last_net_pay_labels = self.prev_batch['net_pay_labels'].to(self.device)
                last_stream_pay_labels = self.prev_batch['stream_pay_labels'].to(self.device)
                last_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels'].to(self.device)
                last_stream_pay_mask = self.prev_batch['stream_pay_mask'].to(self.device)
                alpha = 0.3
                beta = 1.0 - alpha
                n_curr = curr_features.size(0)
                n_last = last_features.size(0)
                n_sample_curr = int(n_curr * alpha)
                n_sample_curr = max(n_sample_curr, 1)
                n_sample_last = int(n_curr * beta) 
                n_sample_last = max(n_sample_last, 1)
                idx_curr = torch.randperm(n_curr)[:n_sample_curr]
                sampled_curr_features = curr_features[idx_curr]
                sampled_curr_pay_labels = curr_pay_labels[idx_curr]
                sampled_curr_net_pay_labels = curr_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
                sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]
                idx_last = torch.randperm(n_last)[:n_sample_last]
                sampled_last_features = last_features[idx_last]
                sampled_last_pay_labels = last_pay_labels[idx_last]
                sampled_last_net_pay_labels = last_net_pay_labels[idx_last]
                sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
                sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
                sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]
                mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
                mixed_pay_labels = torch.cat([sampled_last_pay_labels, sampled_curr_pay_labels], dim=0)
                mixed_net_pay_labels = torch.cat([sampled_last_net_pay_labels, sampled_curr_net_pay_labels], dim=0)
                mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
                mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
                mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)

                self.prev_batch = {
                    'features': mixed_features.to(self.device), 
                    'pay_labels': mixed_pay_labels.to(self.device),
                    'net_pay_labels': mixed_net_pay_labels.to(self.device),
                    'stream_pay_labels': mixed_stream_pay_labels.to(self.device),
                    'stream_net_pay_labels': mixed_stream_net_pay_labels.to(self.device),
                    'stream_pay_mask': mixed_stream_pay_mask.to(self.device)
                }
                self.prev_up_dates = 0
            return True

        else:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is not improved, retrain model")
            return False

    def retrain_updated_model(self,batch,retrain_step,base_alpha=0.9):
        last_batch = self.prev_batch

        curr_features = batch['features'].to(self.device)
        curr_pay_labels = batch['pay_labels'].to(self.device)
        curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
        curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
        curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
        curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)

        last_features = last_batch['features'].to(self.device)
        last_pay_labels = last_batch['pay_labels'].to(self.device)
        last_net_pay_labels = last_batch['net_pay_labels'].to(self.device)
        last_stream_pay_labels = last_batch['stream_pay_labels'].to(self.device)
        last_stream_net_pay_labels = last_batch['stream_net_pay_labels'].to(self.device)
        last_stream_pay_mask = last_batch['stream_pay_mask'].to(self.device)


        alpha = max(base_alpha - 0.1 * retrain_step, 0.3)
        beta = 1.0 - alpha

        n_curr = curr_features.size(0)
        n_last = last_features.size(0)
        n_sample_curr = int(n_curr * alpha)
        n_sample_curr = max(n_sample_curr, 1)
        n_sample_last = int(n_curr * beta) 
        n_sample_last = max(n_sample_last, 1)

        idx_curr = torch.randperm(n_curr)[:n_sample_curr]
        sampled_curr_features = curr_features[idx_curr]
        sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
        sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
        sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]

        idx_last = torch.randperm(n_last)[:n_sample_last]
        sampled_last_features = last_features[idx_last]
        sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
        sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
        sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]

        mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
        mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
        mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
        mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)
        self.optimizer.zero_grad()
        cvr_outputs = self.stream_model.cvr_forward(mixed_features)
        pretrain_cvr_outputs = self.pretrained_model.cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(mixed_features)
        inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
        tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
        cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,mixed_stream_pay_labels,mixed_stream_pay_mask)
        cvr_loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        refund_outputs = self.stream_model.net_cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(mixed_features)
        inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
        tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
        net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,mixed_stream_pay_labels,mixed_stream_net_pay_labels,mixed_stream_pay_mask)
        net_cvr_loss.backward()
        self.optimizer.step()

        loss = cvr_loss + net_cvr_loss
        return loss
    
    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn_wi_bpr(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
            cvr_cl_loss.backward()
            self.optimizer.step()


            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()








            loss = cvr_loss + net_cvr_loss

            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def test_v2(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []






        all_metrics["stream_model_inws_Global_CVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_CVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_inws = []
        all_net_pay_labels_inws = []
        stream_model_all_pay_preds_inws = []
        stream_model_all_net_pay_preds_inws = []


        all_metrics["stream_model_dp_Global_CVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_CVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_dp = []
        all_net_pay_labels_dp = []
        stream_model_all_pay_preds_dp = []
        stream_model_all_net_pay_preds_dp = []

        all_metrics["stream_model_Global_POS_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NEG_CVR_NLL"] = 0


        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                inws_mask = (inw_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_inws.extend(pay_labels[inws_mask].cpu().numpy().tolist())
                all_net_pay_labels_inws.extend(net_pay_labels[inws_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_inws.extend(stream_cvr_outputs[inws_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_inws.extend(stream_net_cvr_outputs[inws_mask].cpu().numpy().tolist())

                dp_mask = (delay_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_dp.extend(pay_labels[dp_mask].cpu().numpy().tolist())
                all_net_pay_labels_dp.extend(net_pay_labels[dp_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_dp.extend(stream_cvr_outputs[dp_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_dp.extend(stream_net_cvr_outputs[dp_mask].cpu().numpy().tolist())

                stream_cvr_hidden_states = stream_model.get_cvr_hidden_state(features)
                stream_cvr_hidden_states = np.array(stream_cvr_hidden_states.cpu())
                pay_labels = np.array(pay_labels.cpu())
                delay_pay_labels_afterPay = np.array(delay_pay_labels_afterPay.cpu())
                hidden_state_list = [vec.tolist() for vec in stream_cvr_hidden_states]
                df = pd.DataFrame({
                    'cvr_hidden_state': hidden_state_list,
                    'pay_label': pay_labels,
                    'delay_pay_label_afterPay': delay_pay_labels_afterPay
                })

                csv_file = 'esdfmRF_ple_v2_stream_cvr_hidden_states_with_labels.csv'
                file_exists = os.path.isfile(csv_file)

                df.to_csv(
                    csv_file,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_inws_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_inws_Global_CVR_AUC: {all_metrics['stream_model_inws_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_AUC: {all_metrics['stream_model_inws_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_NLL: {all_metrics['stream_model_inws_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_NLL: {all_metrics['stream_model_inws_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_PRAUC: {all_metrics['stream_model_inws_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_PRAUC: {all_metrics['stream_model_inws_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_AUC: {all_metrics['stream_model_dp_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_AUC: {all_metrics['stream_model_dp_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_NLL: {all_metrics['stream_model_dp_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_NLL: {all_metrics['stream_model_dp_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_PRAUC: {all_metrics['stream_model_dp_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_PRAUC: {all_metrics['stream_model_dp_Global_NetCVR_PRAUC']:.5f}")

        all_metrics["stream_model_Global_POS_CVR_NLL"],all_metrics["stream_model_Global_NEG_CVR_NLL"] = nll_score_split(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"stream_model_Global_POS_CVR_NLL: {all_metrics['stream_model_Global_POS_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NEG_CVR_NLL: {all_metrics['stream_model_Global_NEG_CVR_NLL']:.5f}")

        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []


        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = 0

        stream_model_all_corrected_pay_preds = []
        stream_model_all_corrected_net_pay_preds = []



        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                stream_corrected_cvr_outputs,stream_corrected_net_cvr_outputs = stream_model.correction_predict(features)
                stream_model_all_corrected_pay_preds.extend(stream_corrected_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_corrected_net_pay_preds.extend(stream_corrected_net_cvr_outputs.cpu().numpy().tolist())



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_Global_CORRECTED_CVR_AUC: {all_metrics['stream_model_Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_AUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_NLL: {all_metrics['stream_model_Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_NLL: {all_metrics['stream_model_Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PRAUC']:.5f}")
        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_PLE_v3_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        

        self.prev_batch = None
        self.prev_up_dates = 0 
        self.prev_up_dates_flag = 5



        self.buffer_max_capacity = 500
        self.historical_delay_pay_features_buffer = torch.empty(0, 128)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss
    
    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def refund_loss_fn_wi_bpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)


        bprloss = self.bpr_loss_random_neg(refund_outputs, stream_refund_labels, stream_refund_mask, 20)

        total_loss = refund_loss + bprloss
        return total_loss

    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, stream_pay_labels , stream_pay_mask):
    

        corrected_cvr_logits = cvr_outputs +  correction_outputs
        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_pay_mask = stream_pay_mask.view(-1)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,stream_pay_labels,stream_pay_mask)


        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, stream_pay_labels, stream_pay_mask):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        stream_pay_labels = stream_pay_labels.view(B)
        stream_pay_mask = stream_pay_mask.view(B).bool()

        valid_mask = (stream_pay_mask == 1) 
        if valid_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)

        z_neg = cvr_hidden_states[valid_mask]

        z_neg = F.normalize(z_neg, p=2, dim=-1)

        sim_matrix = torch.mm(z_neg, z_neg.t())

        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()

        return diversity_loss

    def refund_cl_loss(self, net_cvr_hidden_states, stream_net_pay_labels, stream_pay_mask):
        B = net_cvr_hidden_states.size(0)
        net_cvr_hidden_states = net_cvr_hidden_states.view(B, -1)
        stream_net_pay_labels = stream_net_pay_labels.view(B)
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_mask = stream_refund_mask.view(B).bool()

        valid_mask = (stream_refund_mask == 1) 
        if valid_mask.sum() <= 1:
            return net_cvr_hidden_states.new_tensor(0.0, requires_grad=True)

        z_neg = net_cvr_hidden_states[valid_mask]

        z_neg = F.normalize(z_neg, p=2, dim=-1)

        sim_matrix = torch.mm(z_neg, z_neg.t())

        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()

        return diversity_loss

    def cvr_cl_loss_v2(self, cvr_hidden_states_no_pay, cvr_hidden_states_delay_pay):
        """
        拉远 no-pay 和 delay-pay 用户的表征。
        使用负样本对之间的相似度作为损失，越小越好。
        """
        if cvr_hidden_states_no_pay.size(0) == 0 or cvr_hidden_states_delay_pay.size(0) == 0:
            return cvr_hidden_states_no_pay.new_tensor(0.0, requires_grad=True)

        z_no_pay = F.normalize(cvr_hidden_states_no_pay, p=2, dim=-1)
        z_delay_pay = F.normalize(cvr_hidden_states_delay_pay, p=2, dim=-1)

        sim_matrix = torch.mm(z_no_pay, z_delay_pay.t())

        contrastive_loss = sim_matrix.mean() + 1

        return contrastive_loss

    def DelayTime_bpr_loss_weighted_neg(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=outputs.device)

        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]

        neg_scores = valid_outputs[neg_mask]
        neg_probs = torch.sigmoid(valid_outputs[neg_mask]).detach()

        def compute_adaptive_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w

        pos_weights = compute_adaptive_weight(pos_delay_all)

        neg_weights = (1 - neg_probs).float()
        if neg_weights.sum() <= 0:
            neg_weights = torch.ones_like(neg_weights)
        neg_weights /= neg_weights.sum()

        neg_idx = torch.multinomial(
            neg_weights,
            len(pos_scores_all) * num_neg_samples,
            replacement=True
        )
        neg_idx = neg_idx.view(len(pos_scores_all), num_neg_samples)
        neg_samples = neg_scores[neg_idx]

        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff)).mean(dim=1)

        weighted_loss = (pos_weights * per_sample_loss).sum() / pos_weights.sum()

        return weighted_loss

    def refund_loss_fn_wi_DelayTimeBpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        bprloss = self.DelayTime_bpr_loss_weighted_neg(refund_outputs, stream_refund_labels, stream_refund_mask, delay_refund_time,5)

        total_loss = refund_loss + bprloss
        return total_loss 

    def cvr_loss_fn_wi_DelayTimeBpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.DelayTime_bpr_loss_weighted_neg(cvr_outputs, stream_pay_labels, stream_pay_mask ,delay_pay_time)

        loss= bprloss + cvr_loss 

        return loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch_for_PreExp1_oracle(epoch_idx,day)
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def check_updated_model_is_good(self,batch_num,batch,epoch_idx,day_idx):
        if (batch_num == 0) & (day_idx == 0) :
            self.prev_batch = {
                'features': batch['features'].to(self.device), 
                'pay_labels': batch['pay_labels'].to(self.device),
                'net_pay_labels': batch['net_pay_labels'].to(self.device),
                'stream_pay_labels': batch['stream_pay_labels'].to(self.device),
                'stream_net_pay_labels': batch['stream_net_pay_labels'].to(self.device),
                'stream_pay_mask': batch['stream_pay_mask'].to(self.device)
            }

        updated_stream_model = copy.deepcopy(self.stream_model)
        updated_stream_model.to(self.device)
        updated_stream_model.eval()

        last_batch_features = self.prev_batch['features']
        last_batch_pay_labels = self.prev_batch['pay_labels']
        last_batch_net_pay_labels = self.prev_batch['net_pay_labels']
        last_batch_stream_pay_labels = self.prev_batch['pay_labels']
        last_batch_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels']
        last_batch_stream_pay_mask = self.prev_batch['stream_pay_mask']

        last_batch_cvr_outputs,last_batch_net_cvr_outputs = self.last_batch_stream_model.predict(last_batch_features)
        last_batch_updated_cvr_outputs,last_batch_updated_net_cvr_outputs = updated_stream_model.predict(last_batch_features)
        last_batch_cvr_outputs = last_batch_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_updated_cvr_outputs = last_batch_updated_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_stream_pay_labels = last_batch_stream_pay_labels[last_batch_stream_pay_mask.bool()]
        last_batch_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC: {last_batch_cvr_auc:.5f} - Updated CVR AUC: {last_batch_updated_cvr_auc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR PRAUC: {last_batch_cvr_prauc:.5f} - Updated CVR PRAUC: {last_batch_updated_cvr_prauc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR NLL: {last_batch_cvr_nll:.5f} - Updated CVR NLL: {last_batch_updated_cvr_nll:.5f}")
        if last_batch_updated_cvr_auc > last_batch_cvr_auc:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is improved, continue training")
            self.prev_up_dates += 1
            if self.prev_up_dates == self.prev_up_dates_flag:
                self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}  Batch {batch_num}- update calibrated model")
                self.last_batch_stream_model = copy.deepcopy(updated_stream_model)
                self.last_batch_stream_model.to(self.device)
                self.last_batch_stream_model.eval()

                curr_features = batch['features'].to(self.device)
                curr_pay_labels = batch['pay_labels'].to(self.device)
                curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
                curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
                curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
                curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)
                last_features = self.prev_batch['features'].to(self.device)
                last_pay_labels = self.prev_batch['pay_labels'].to(self.device)
                last_net_pay_labels = self.prev_batch['net_pay_labels'].to(self.device)
                last_stream_pay_labels = self.prev_batch['stream_pay_labels'].to(self.device)
                last_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels'].to(self.device)
                last_stream_pay_mask = self.prev_batch['stream_pay_mask'].to(self.device)
                alpha = 0.3
                beta = 1.0 - alpha
                n_curr = curr_features.size(0)
                n_last = last_features.size(0)
                n_sample_curr = int(n_curr * alpha)
                n_sample_curr = max(n_sample_curr, 1)
                n_sample_last = int(n_curr * beta) 
                n_sample_last = max(n_sample_last, 1)
                idx_curr = torch.randperm(n_curr)[:n_sample_curr]
                sampled_curr_features = curr_features[idx_curr]
                sampled_curr_pay_labels = curr_pay_labels[idx_curr]
                sampled_curr_net_pay_labels = curr_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
                sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]
                idx_last = torch.randperm(n_last)[:n_sample_last]
                sampled_last_features = last_features[idx_last]
                sampled_last_pay_labels = last_pay_labels[idx_last]
                sampled_last_net_pay_labels = last_net_pay_labels[idx_last]
                sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
                sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
                sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]
                mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
                mixed_pay_labels = torch.cat([sampled_last_pay_labels, sampled_curr_pay_labels], dim=0)
                mixed_net_pay_labels = torch.cat([sampled_last_net_pay_labels, sampled_curr_net_pay_labels], dim=0)
                mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
                mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
                mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)

                self.prev_batch = {
                    'features': mixed_features.to(self.device), 
                    'pay_labels': mixed_pay_labels.to(self.device),
                    'net_pay_labels': mixed_net_pay_labels.to(self.device),
                    'stream_pay_labels': mixed_stream_pay_labels.to(self.device),
                    'stream_net_pay_labels': mixed_stream_net_pay_labels.to(self.device),
                    'stream_pay_mask': mixed_stream_pay_mask.to(self.device)
                }
                self.prev_up_dates = 0
            return True

        else:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is not improved, retrain model")
            return False

    def retrain_updated_model(self,batch,retrain_step,base_alpha=0.9):
        last_batch = self.prev_batch

        curr_features = batch['features'].to(self.device)
        curr_pay_labels = batch['pay_labels'].to(self.device)
        curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
        curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
        curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
        curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)

        last_features = last_batch['features'].to(self.device)
        last_pay_labels = last_batch['pay_labels'].to(self.device)
        last_net_pay_labels = last_batch['net_pay_labels'].to(self.device)
        last_stream_pay_labels = last_batch['stream_pay_labels'].to(self.device)
        last_stream_net_pay_labels = last_batch['stream_net_pay_labels'].to(self.device)
        last_stream_pay_mask = last_batch['stream_pay_mask'].to(self.device)


        alpha = max(base_alpha - 0.1 * retrain_step, 0.3)
        beta = 1.0 - alpha

        n_curr = curr_features.size(0)
        n_last = last_features.size(0)
        n_sample_curr = int(n_curr * alpha)
        n_sample_curr = max(n_sample_curr, 1)
        n_sample_last = int(n_curr * beta) 
        n_sample_last = max(n_sample_last, 1)

        idx_curr = torch.randperm(n_curr)[:n_sample_curr]
        sampled_curr_features = curr_features[idx_curr]
        sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
        sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
        sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]

        idx_last = torch.randperm(n_last)[:n_sample_last]
        sampled_last_features = last_features[idx_last]
        sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
        sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
        sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]

        mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
        mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
        mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
        mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)
        self.optimizer.zero_grad()
        cvr_outputs = self.stream_model.cvr_forward(mixed_features)
        pretrain_cvr_outputs = self.pretrained_model.cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(mixed_features)
        inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
        tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
        cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,mixed_stream_pay_labels,mixed_stream_pay_mask)
        cvr_loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        refund_outputs = self.stream_model.net_cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(mixed_features)
        inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
        tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
        net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,mixed_stream_pay_labels,mixed_stream_net_pay_labels,mixed_stream_pay_mask)
        net_cvr_loss.backward()
        self.optimizer.step()

        loss = cvr_loss + net_cvr_loss
        return loss
    
    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
            cvr_cl_loss.backward()
            self.optimizer.step()


            

            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()






            loss = cvr_loss + net_cvr_loss + cvr_cl_loss



            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_DelayTimeSampleBpr(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)
            delay_pay_time = pay_ts - click_ts
            delay_pay_time = delay_pay_time / 3600
            delay_pay_time = delay_pay_time.to(self.device)
            
            delay_refund_time = refund_ts - pay_ts
            delay_refund_time = delay_refund_time / 3600
            delay_refund_time = delay_refund_time.to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
            cvr_cl_loss.backward()
            self.optimizer.step()



            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_DelayTimeBpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time)
           
            
            cvr_loss.backward()
            self.optimizer.step()






            loss = cvr_loss + net_cvr_loss + cvr_cl_loss



            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_for_PreExp1(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            if i != 0:
                continue

            else :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = (stream_pay_labels == 0) & (pay_labels == 1)
                last_update_labels = torch.zeros_like(pay_labels)

                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_before_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()

                self.optimizer.zero_grad()
                cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
                cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
                cvr_cl_loss.backward()
                self.optimizer.step()

                

                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
                cvr_loss.backward()
                self.optimizer.step()

                loss = cvr_loss
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_wrong_dp_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()






                

                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pay_labels,stream_pay_mask)
                cvr_loss.backward()
                self.optimizer.step()

                loss = cvr_loss 
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_correct_dp_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()

            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_for_PreExp1_oracle(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        curr_stream_features = []
        curr_pay_labels = []
        curr_dp_labels = []
        last_update_labels = []

        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)

            if i != 0:
                continue

            else :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = (stream_pay_labels == 0) & (pay_labels == 1)
                last_update_labels = torch.zeros_like(pay_labels)

                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_oracle_before_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


                self.optimizer.zero_grad()
                cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
                cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
                cvr_cl_loss.backward()
                self.optimizer.step()

                

                self.optimizer.zero_grad()
                cvr_outputs = self.stream_model.cvr_forward(features)
                pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
                pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
                pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
                inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
                tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
                cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pay_labels,stream_pay_mask)
                cvr_loss.backward()
                self.optimizer.step()

                loss = cvr_loss 
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                })

                csv_file = '/home/luomingxuan.lmx/AirBench_modified/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_oracle_after_1batch.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()






            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_save_intermediate_result(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            if i == 0 :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = delay_pay_labels_afterPay.clone()
                last_update_labels = torch.zeros_like(pay_labels)
            if (i > 0) & (i < 4):
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.zeros_like(pay_labels)),dim=0)
            if i == 4:
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.ones_like(pay_labels)),dim=0)
            if i == 4 :
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_before.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
            cvr_cl_loss.backward()
            self.optimizer.step()

            





            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()






            loss = cvr_loss + net_cvr_loss + cvr_cl_loss

            if i == 4 :
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_after.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def test_v2(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []






        all_metrics["stream_model_inws_Global_CVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_CVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_inws = []
        all_net_pay_labels_inws = []
        stream_model_all_pay_preds_inws = []
        stream_model_all_net_pay_preds_inws = []


        all_metrics["stream_model_dp_Global_CVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_CVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_dp = []
        all_net_pay_labels_dp = []
        stream_model_all_pay_preds_dp = []
        stream_model_all_net_pay_preds_dp = []

        all_metrics["stream_model_Global_POS_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NEG_CVR_NLL"] = 0


        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                inws_mask = (inw_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_inws.extend(pay_labels[inws_mask].cpu().numpy().tolist())
                all_net_pay_labels_inws.extend(net_pay_labels[inws_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_inws.extend(stream_cvr_outputs[inws_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_inws.extend(stream_net_cvr_outputs[inws_mask].cpu().numpy().tolist())

                dp_mask = (delay_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_dp.extend(pay_labels[dp_mask].cpu().numpy().tolist())
                all_net_pay_labels_dp.extend(net_pay_labels[dp_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_dp.extend(stream_cvr_outputs[dp_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_dp.extend(stream_net_cvr_outputs[dp_mask].cpu().numpy().tolist())

                stream_cvr_hidden_states = stream_model.get_cvr_hidden_state(features)
                stream_cvr_hidden_states = np.array(stream_cvr_hidden_states.cpu())
                pay_labels = np.array(pay_labels.cpu())
                delay_pay_labels_afterPay = np.array(delay_pay_labels_afterPay.cpu())
                hidden_state_list = [vec.tolist() for vec in stream_cvr_hidden_states]
                df = pd.DataFrame({
                    'cvr_hidden_state': hidden_state_list,
                    'pay_label': pay_labels,
                    'delay_pay_label_afterPay': delay_pay_labels_afterPay
                })

                csv_file = 'esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels.csv'
                file_exists = os.path.isfile(csv_file)

                df.to_csv(
                    csv_file,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_inws_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_inws_Global_CVR_AUC: {all_metrics['stream_model_inws_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_AUC: {all_metrics['stream_model_inws_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_NLL: {all_metrics['stream_model_inws_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_NLL: {all_metrics['stream_model_inws_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_PRAUC: {all_metrics['stream_model_inws_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_PRAUC: {all_metrics['stream_model_inws_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_AUC: {all_metrics['stream_model_dp_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_AUC: {all_metrics['stream_model_dp_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_NLL: {all_metrics['stream_model_dp_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_NLL: {all_metrics['stream_model_dp_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_PRAUC: {all_metrics['stream_model_dp_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_PRAUC: {all_metrics['stream_model_dp_Global_NetCVR_PRAUC']:.5f}")

        all_metrics["stream_model_Global_POS_CVR_NLL"],all_metrics["stream_model_Global_NEG_CVR_NLL"] = nll_score_split(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"stream_model_Global_POS_CVR_NLL: {all_metrics['stream_model_Global_POS_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NEG_CVR_NLL: {all_metrics['stream_model_Global_NEG_CVR_NLL']:.5f}")

        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []


        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = 0

        stream_model_all_corrected_pay_preds = []
        stream_model_all_corrected_net_pay_preds = []



        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                stream_corrected_cvr_outputs,stream_corrected_net_cvr_outputs = stream_model.correction_predict(features)
                stream_model_all_corrected_pay_preds.extend(stream_corrected_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_corrected_net_pay_preds.extend(stream_corrected_net_cvr_outputs.cpu().numpy().tolist())



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_Global_CORRECTED_CVR_AUC: {all_metrics['stream_model_Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_AUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_NLL: {all_metrics['stream_model_Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_NLL: {all_metrics['stream_model_Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PRAUC']:.5f}")
        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_PLE_test_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        

        self.prev_batch = None
        self.prev_up_dates = 0 
        self.prev_up_dates_flag = 5



        self.buffer_max_capacity = 500
        self.historical_delay_pay_features_buffer = torch.empty(0, 128)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss
    
    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def barlow_twins_cross_loss(self,emb1, emb2, lambda_param=5e-3):
        B, D = emb1.shape
        
        emb1 = (emb1 - emb1.mean(dim=0)) / (emb1.std(dim=0) + 1e-6)
        emb2 = (emb2 - emb2.mean(dim=0)) / (emb2.std(dim=0) + 1e-6)
        
        c = (emb1.T @ emb2) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum() * lambda_param
        
        return on_diag + off_diag

    def off_diagonal(self,x):
        n = x.shape[0]
        return x.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()

    def frobenius_loss(self,emb1, emb2):
        B, D = emb1.shape
        
        emb1 = emb1 - emb1.mean(dim=0, keepdim=True)
        emb2 = emb2 - emb2.mean(dim=0, keepdim=True)
        
        cross_cov = (emb1.T @ emb2) / B
        
        return cross_cov.pow(2).sum()

    def negative_similarity_loss(self, emb1, emb2, temperature=0.5):
        emb1 = F.normalize(emb1, dim=1)
        emb2 = F.normalize(emb2, dim=1)

        sim = torch.sum( (emb1 * emb2), dim=1)
        sim = 1 + sim
        loss = sim.mean()

        return loss

    def correction_cvr_loss(self, cvr_outputs, correction_outputs, stream_pay_labels , stream_pay_mask):
    

        corrected_cvr_logits = cvr_outputs +  correction_outputs
        corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_pay_mask = stream_pay_mask.view(-1)

        cvr_loss = self.bpr_loss_random_neg(corrected_cvr_logits,stream_pay_labels,stream_pay_mask)


        return cvr_loss

    def cvr_cl_loss(self, cvr_hidden_states, stream_pay_labels, stream_pay_mask):
        B = cvr_hidden_states.size(0)
        cvr_hidden_states = cvr_hidden_states.view(B, -1)
        stream_pay_labels = stream_pay_labels.view(B)
        stream_pay_mask = stream_pay_mask.view(B).bool()

        valid_mask = (stream_pay_mask == 1) 
        if valid_mask.sum() <= 1:
            return cvr_hidden_states.new_tensor(0.0, requires_grad=True)

        z_neg = cvr_hidden_states[valid_mask]

        z_neg = F.normalize(z_neg, p=2, dim=-1)

        sim_matrix = torch.mm(z_neg, z_neg.t())

        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()

        return diversity_loss

    def refund_cl_loss(self, net_cvr_hidden_states, stream_net_pay_labels, stream_pay_mask):
        B = net_cvr_hidden_states.size(0)
        net_cvr_hidden_states = net_cvr_hidden_states.view(B, -1)
        stream_net_pay_labels = stream_net_pay_labels.view(B)
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_mask = stream_refund_mask.view(B).bool()

        valid_mask = (stream_refund_mask == 1) 
        if valid_mask.sum() <= 1:
            return net_cvr_hidden_states.new_tensor(0.0, requires_grad=True)

        z_neg = net_cvr_hidden_states[valid_mask]

        z_neg = F.normalize(z_neg, p=2, dim=-1)

        sim_matrix = torch.mm(z_neg, z_neg.t())

        eye_mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_without_diag = sim_matrix[~eye_mask]
        diversity_loss = sim_without_diag.mean()

        return diversity_loss

    def cvr_cl_loss_v2(self, cvr_hidden_states_no_pay, cvr_hidden_states_delay_pay):
        """
        拉远 no-pay 和 delay-pay 用户的表征。
        使用负样本对之间的相似度作为损失，越小越好。
        """
        if cvr_hidden_states_no_pay.size(0) == 0 or cvr_hidden_states_delay_pay.size(0) == 0:
            return cvr_hidden_states_no_pay.new_tensor(0.0, requires_grad=True)

        z_no_pay = F.normalize(cvr_hidden_states_no_pay, p=2, dim=-1)
        z_delay_pay = F.normalize(cvr_hidden_states_delay_pay, p=2, dim=-1)

        sim_matrix = torch.mm(z_no_pay, z_delay_pay.t())

        contrastive_loss = sim_matrix.mean() + 1

        return contrastive_loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def check_updated_model_is_good(self,batch_num,batch,epoch_idx,day_idx):
        if (batch_num == 0) & (day_idx == 0) :
            self.prev_batch = {
                'features': batch['features'].to(self.device), 
                'pay_labels': batch['pay_labels'].to(self.device),
                'net_pay_labels': batch['net_pay_labels'].to(self.device),
                'stream_pay_labels': batch['stream_pay_labels'].to(self.device),
                'stream_net_pay_labels': batch['stream_net_pay_labels'].to(self.device),
                'stream_pay_mask': batch['stream_pay_mask'].to(self.device)
            }

        updated_stream_model = copy.deepcopy(self.stream_model)
        updated_stream_model.to(self.device)
        updated_stream_model.eval()

        last_batch_features = self.prev_batch['features']
        last_batch_pay_labels = self.prev_batch['pay_labels']
        last_batch_net_pay_labels = self.prev_batch['net_pay_labels']
        last_batch_stream_pay_labels = self.prev_batch['pay_labels']
        last_batch_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels']
        last_batch_stream_pay_mask = self.prev_batch['stream_pay_mask']

        last_batch_cvr_outputs,last_batch_net_cvr_outputs = self.last_batch_stream_model.predict(last_batch_features)
        last_batch_updated_cvr_outputs,last_batch_updated_net_cvr_outputs = updated_stream_model.predict(last_batch_features)
        last_batch_cvr_outputs = last_batch_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_updated_cvr_outputs = last_batch_updated_cvr_outputs[last_batch_stream_pay_mask.bool()]
        last_batch_stream_pay_labels = last_batch_stream_pay_labels[last_batch_stream_pay_mask.bool()]
        last_batch_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_auc = auc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_prauc = prauc_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        last_batch_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_cvr_outputs)
        last_batch_updated_cvr_nll = nll_score(last_batch_stream_pay_labels, last_batch_updated_cvr_outputs)
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC: {last_batch_cvr_auc:.5f} - Updated CVR AUC: {last_batch_updated_cvr_auc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR PRAUC: {last_batch_cvr_prauc:.5f} - Updated CVR PRAUC: {last_batch_updated_cvr_prauc:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR NLL: {last_batch_cvr_nll:.5f} - Updated CVR NLL: {last_batch_updated_cvr_nll:.5f}")
        if last_batch_updated_cvr_auc > last_batch_cvr_auc:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is improved, continue training")
            self.prev_up_dates += 1
            if self.prev_up_dates == self.prev_up_dates_flag:
                self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}  Batch {batch_num}- update calibrated model")
                self.last_batch_stream_model = copy.deepcopy(updated_stream_model)
                self.last_batch_stream_model.to(self.device)
                self.last_batch_stream_model.eval()

                curr_features = batch['features'].to(self.device)
                curr_pay_labels = batch['pay_labels'].to(self.device)
                curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
                curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
                curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
                curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)
                last_features = self.prev_batch['features'].to(self.device)
                last_pay_labels = self.prev_batch['pay_labels'].to(self.device)
                last_net_pay_labels = self.prev_batch['net_pay_labels'].to(self.device)
                last_stream_pay_labels = self.prev_batch['stream_pay_labels'].to(self.device)
                last_stream_net_pay_labels = self.prev_batch['stream_net_pay_labels'].to(self.device)
                last_stream_pay_mask = self.prev_batch['stream_pay_mask'].to(self.device)
                alpha = 0.3
                beta = 1.0 - alpha
                n_curr = curr_features.size(0)
                n_last = last_features.size(0)
                n_sample_curr = int(n_curr * alpha)
                n_sample_curr = max(n_sample_curr, 1)
                n_sample_last = int(n_curr * beta) 
                n_sample_last = max(n_sample_last, 1)
                idx_curr = torch.randperm(n_curr)[:n_sample_curr]
                sampled_curr_features = curr_features[idx_curr]
                sampled_curr_pay_labels = curr_pay_labels[idx_curr]
                sampled_curr_net_pay_labels = curr_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
                sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
                sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]
                idx_last = torch.randperm(n_last)[:n_sample_last]
                sampled_last_features = last_features[idx_last]
                sampled_last_pay_labels = last_pay_labels[idx_last]
                sampled_last_net_pay_labels = last_net_pay_labels[idx_last]
                sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
                sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
                sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]
                mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
                mixed_pay_labels = torch.cat([sampled_last_pay_labels, sampled_curr_pay_labels], dim=0)
                mixed_net_pay_labels = torch.cat([sampled_last_net_pay_labels, sampled_curr_net_pay_labels], dim=0)
                mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
                mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
                mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)

                self.prev_batch = {
                    'features': mixed_features.to(self.device), 
                    'pay_labels': mixed_pay_labels.to(self.device),
                    'net_pay_labels': mixed_net_pay_labels.to(self.device),
                    'stream_pay_labels': mixed_stream_pay_labels.to(self.device),
                    'stream_net_pay_labels': mixed_stream_net_pay_labels.to(self.device),
                    'stream_pay_mask': mixed_stream_pay_mask.to(self.device)
                }
                self.prev_up_dates = 0
            return True

        else:
            self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx} Batch {batch_num} - Last Batch CVR AUC is not improved, retrain model")
            return False

    def retrain_updated_model(self,batch,retrain_step,base_alpha=0.9):
        last_batch = self.prev_batch

        curr_features = batch['features'].to(self.device)
        curr_pay_labels = batch['pay_labels'].to(self.device)
        curr_net_pay_labels = batch['net_pay_labels'].to(self.device)
        curr_stream_pay_labels = batch['stream_pay_labels'].to(self.device)
        curr_stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
        curr_stream_pay_mask = batch['stream_pay_mask'].to(self.device)

        last_features = last_batch['features'].to(self.device)
        last_pay_labels = last_batch['pay_labels'].to(self.device)
        last_net_pay_labels = last_batch['net_pay_labels'].to(self.device)
        last_stream_pay_labels = last_batch['stream_pay_labels'].to(self.device)
        last_stream_net_pay_labels = last_batch['stream_net_pay_labels'].to(self.device)
        last_stream_pay_mask = last_batch['stream_pay_mask'].to(self.device)


        alpha = max(base_alpha - 0.1 * retrain_step, 0.3)
        beta = 1.0 - alpha

        n_curr = curr_features.size(0)
        n_last = last_features.size(0)
        n_sample_curr = int(n_curr * alpha)
        n_sample_curr = max(n_sample_curr, 1)
        n_sample_last = int(n_curr * beta) 
        n_sample_last = max(n_sample_last, 1)

        idx_curr = torch.randperm(n_curr)[:n_sample_curr]
        sampled_curr_features = curr_features[idx_curr]
        sampled_curr_stream_pay_labels = curr_stream_pay_labels[idx_curr]
        sampled_curr_stream_net_pay_labels = curr_stream_net_pay_labels[idx_curr]
        sampled_curr_stream_pay_mask = curr_stream_pay_mask[idx_curr]

        idx_last = torch.randperm(n_last)[:n_sample_last]
        sampled_last_features = last_features[idx_last]
        sampled_last_stream_pay_labels = last_stream_pay_labels[idx_last]
        sampled_last_stream_net_pay_labels = last_stream_net_pay_labels[idx_last]
        sampled_last_stream_pay_mask = last_stream_pay_mask[idx_last]

        mixed_features = torch.cat([sampled_last_features, sampled_curr_features], dim=0)
        mixed_stream_pay_labels = torch.cat([sampled_last_stream_pay_labels, sampled_curr_stream_pay_labels], dim=0)
        mixed_stream_net_pay_labels = torch.cat([sampled_last_stream_net_pay_labels, sampled_curr_stream_net_pay_labels], dim=0)
        mixed_stream_pay_mask = torch.cat([sampled_last_stream_pay_mask, sampled_curr_stream_pay_mask], dim=0)
        self.optimizer.zero_grad()
        cvr_outputs = self.stream_model.cvr_forward(mixed_features)
        pretrain_cvr_outputs = self.pretrained_model.cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(mixed_features)
        inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
        tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
        cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,mixed_stream_pay_labels,mixed_stream_pay_mask)
        cvr_loss.backward()
        self.optimizer.step()

        self.optimizer.zero_grad()
        refund_outputs = self.stream_model.net_cvr_forward(mixed_features)
        pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(mixed_features)
        pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(mixed_features)
        inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
        tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
        net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,mixed_stream_pay_labels,mixed_stream_net_pay_labels,mixed_stream_pay_mask)
        net_cvr_loss.backward()
        self.optimizer.step()

        loss = cvr_loss + net_cvr_loss
        return loss
    
    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()



            delay_pay_mask = (delay_pay_labels_afterPay == 1)
            if delay_pay_mask.sum() > 0:
                cvr_delay_pay_features = features[delay_pay_mask].detach().cpu()
                if len(self.historical_delay_pay_features_buffer) == 0:
                    self.historical_delay_pay_features_buffer = cvr_delay_pay_features
                else:
                    self.historical_delay_pay_features_buffer = torch.cat([self.historical_delay_pay_features_buffer, cvr_delay_pay_features], dim=0)
                if self.historical_delay_pay_features_buffer.size(0) > self.buffer_max_capacity:
                    self.historical_delay_pay_features_buffer = self.historical_delay_pay_features_buffer[-self.buffer_max_capacity:]
            historical_delay_pay_features = self.historical_delay_pay_features_buffer.to(self.device)
            not_pay_mask = (stream_pay_labels == 0) & (stream_pay_mask == 1)
            cvr_not_pay_features = features[not_pay_mask]
            cvr_not_pay_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(cvr_not_pay_features)
            historical_delay_pay_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(historical_delay_pay_features)
            cvr_cl_loss = self.cvr_cl_loss_v2(cvr_not_pay_cvr_hidden_states,historical_delay_pay_cvr_hidden_states)
            cvr_cl_loss.backward()
            self.optimizer.step()
            

            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()






            loss = cvr_loss + net_cvr_loss + cvr_cl_loss



            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_save_intermediate_result(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)



        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)


            if i == 0 :
                curr_stream_features = features.clone()
                curr_pay_labels = pay_labels.clone()
                curr_dp_labels = delay_pay_labels_afterPay.clone()
                last_update_labels = torch.zeros_like(pay_labels)
            if (i > 0) & (i < 4):
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.zeros_like(pay_labels)),dim=0)
            if i == 4:
                curr_stream_features = torch.cat((curr_stream_features,features),dim=0)
                curr_pay_labels = torch.cat((curr_pay_labels,pay_labels),dim=0)
                curr_dp_labels = torch.cat((curr_dp_labels,delay_pay_labels_afterPay),dim=0)
                last_update_labels = torch.cat((last_update_labels,torch.ones_like(pay_labels)),dim=0)
            if i == 4 :
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_before.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_hidden_states = self.stream_model.get_cvr_hidden_state(features)
            cvr_cl_loss = self.cvr_cl_loss(cvr_hidden_states,stream_pay_labels,stream_pay_mask)
            cvr_cl_loss.backward()
            self.optimizer.step()

            





            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,pretrain_refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()






            loss = cvr_loss + net_cvr_loss + cvr_cl_loss

            if i == 4 :
                self.stream_model.eval()
                all_cvr_hidden_states = self.stream_model.get_cvr_hidden_state(curr_stream_features)
                all_cvr_hidden_states = all_cvr_hidden_states.detach().cpu().numpy()
                all_pay_labels = curr_pay_labels.cpu().numpy()
                all_dp_labels = curr_dp_labels.cpu().numpy()
                all_last_update_labels = last_update_labels.cpu().numpy()

                hidden_state_list = [vec.tolist() for vec in all_cvr_hidden_states]
                df = pd.DataFrame({
                    'all_cvr_hidden_states': hidden_state_list,
                    'all_pay_labels': all_pay_labels,
                    'all_dp_labels': all_dp_labels,
                    'all_last_update_labels': all_last_update_labels
                })

                csv_file = '/home/luomingxuan.lmx/AirBench/data/ali/analysis_scripts/样本对比可视化/esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels_train_after.csv'
                df.to_csv(
                    csv_file,
                    mode='w',
                    header=True,
                    index=False,
                )
                self.stream_model.train()


            total_loss += loss.item()
            total_batches += 1


            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def test_v2(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []






        all_metrics["stream_model_inws_Global_CVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_inws_Global_CVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_inws = []
        all_net_pay_labels_inws = []
        stream_model_all_pay_preds_inws = []
        stream_model_all_net_pay_preds_inws = []


        all_metrics["stream_model_dp_Global_CVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_dp_Global_CVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = 0
        all_pay_labels_dp = []
        all_net_pay_labels_dp = []
        stream_model_all_pay_preds_dp = []
        stream_model_all_net_pay_preds_dp = []

        all_metrics["stream_model_Global_POS_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NEG_CVR_NLL"] = 0


        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                inws_mask = (inw_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_inws.extend(pay_labels[inws_mask].cpu().numpy().tolist())
                all_net_pay_labels_inws.extend(net_pay_labels[inws_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_inws.extend(stream_cvr_outputs[inws_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_inws.extend(stream_net_cvr_outputs[inws_mask].cpu().numpy().tolist())

                dp_mask = (delay_pay_labels_afterPay == 1) | (pay_labels == 0)
                all_pay_labels_dp.extend(pay_labels[dp_mask].cpu().numpy().tolist())
                all_net_pay_labels_dp.extend(net_pay_labels[dp_mask].cpu().numpy().tolist())
                stream_model_all_pay_preds_dp.extend(stream_cvr_outputs[dp_mask].cpu().numpy().tolist())
                stream_model_all_net_pay_preds_dp.extend(stream_net_cvr_outputs[dp_mask].cpu().numpy().tolist())

                stream_cvr_hidden_states = stream_model.get_cvr_hidden_state(features)
                stream_cvr_hidden_states = np.array(stream_cvr_hidden_states.cpu())
                pay_labels = np.array(pay_labels.cpu())
                delay_pay_labels_afterPay = np.array(delay_pay_labels_afterPay.cpu())
                hidden_state_list = [vec.tolist() for vec in stream_cvr_hidden_states]
                df = pd.DataFrame({
                    'cvr_hidden_state': hidden_state_list,
                    'pay_label': pay_labels,
                    'delay_pay_label_afterPay': delay_pay_labels_afterPay
                })

                csv_file = 'esdfmRF_ple_v3_stream_cvr_hidden_states_with_labels.csv'
                file_exists = os.path.isfile(csv_file)

                df.to_csv(
                    csv_file,
                    mode='a',
                    header=not file_exists,
                    index=False,
                )



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_inws_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_inws_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_inws, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_inws, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds_dp, dtype=torch.float32)
        )
        all_metrics["stream_model_dp_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels_dp, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds_dp, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_inws_Global_CVR_AUC: {all_metrics['stream_model_inws_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_AUC: {all_metrics['stream_model_inws_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_NLL: {all_metrics['stream_model_inws_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_NLL: {all_metrics['stream_model_inws_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_inws_Global_CVR_PRAUC: {all_metrics['stream_model_inws_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_inws_Global_NetCVR_PRAUC: {all_metrics['stream_model_inws_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_AUC: {all_metrics['stream_model_dp_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_AUC: {all_metrics['stream_model_dp_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_NLL: {all_metrics['stream_model_dp_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_NLL: {all_metrics['stream_model_dp_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_dp_Global_CVR_PRAUC: {all_metrics['stream_model_dp_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_dp_Global_NetCVR_PRAUC: {all_metrics['stream_model_dp_Global_NetCVR_PRAUC']:.5f}")

        all_metrics["stream_model_Global_POS_CVR_NLL"],all_metrics["stream_model_Global_NEG_CVR_NLL"] = nll_score_split(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        self.logger.info(f"stream_model_Global_POS_CVR_NLL: {all_metrics['stream_model_Global_POS_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NEG_CVR_NLL: {all_metrics['stream_model_Global_NEG_CVR_NLL']:.5f}")

        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []


        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = 0

        stream_model_all_corrected_pay_preds = []
        stream_model_all_corrected_net_pay_preds = []



        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())


                stream_corrected_cvr_outputs,stream_corrected_net_cvr_outputs = stream_model.correction_predict(features)
                stream_model_all_corrected_pay_preds.extend(stream_corrected_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_corrected_net_pay_preds.extend(stream_corrected_net_cvr_outputs.cpu().numpy().tolist())



        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")



        all_metrics["stream_model_Global_CORRECTED_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CORRECTED_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_corrected_net_pay_preds, dtype=torch.float32)
        )

        self.logger.info(f"stream_model_Global_CORRECTED_CVR_AUC: {all_metrics['stream_model_Global_CORRECTED_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_AUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_NLL: {all_metrics['stream_model_Global_CORRECTED_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_NLL: {all_metrics['stream_model_Global_CORRECTED_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PCOC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_CVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_CORRECTED_NetCVR_PRAUC: {all_metrics['stream_model_Global_CORRECTED_NetCVR_PRAUC']:.5f}")
        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_ShareEmb_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.last_batch_stream_model = copy.deepcopy(self.stream_model)
        self.last_batch_stream_model.to(self.device)
        self.last_batch_stream_model.eval()

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)

        loss= bprloss + cvr_loss 

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss
    
    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def refund_loss_fn_wi_DelayTimeBpr(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)

        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        bprloss = self.DelayTime_bpr_loss_weighted_neg(refund_outputs, stream_refund_labels, stream_refund_mask, delay_refund_time,5)

        total_loss = refund_loss + bprloss
        return total_loss 

    def cvr_loss_fn_wi_DelayTimeBpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

  
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
              

        bprloss = self.DelayTime_bpr_loss_weighted_neg(cvr_outputs, stream_pay_labels, stream_pay_mask ,delay_pay_time)

        loss= bprloss + cvr_loss 

        return loss

    def DelayTime_bpr_loss_weighted_neg(self, outputs, labels, mask, delay_time, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        valid_outputs = outputs[valid_indices]
        label_valid = labels[valid_indices]
        valid_delay = delay_time[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=outputs.device)

        pos_scores_all = valid_outputs[pos_mask]
        pos_delay_all = valid_delay[pos_mask]

        neg_scores = valid_outputs[neg_mask]
        neg_probs = torch.sigmoid(valid_outputs[neg_mask]).detach()

        def compute_adaptive_weight(delay):
            median = torch.median(delay)
            std = torch.std(delay)
            scale = std if std > 1 else 1.0
            raw_w = torch.sigmoid((median - delay) / scale)
            min_w = 1
            return min_w + raw_w

        pos_weights = compute_adaptive_weight(pos_delay_all)

        neg_weights = (1 - neg_probs).float()
        if neg_weights.sum() <= 0:
            neg_weights = torch.ones_like(neg_weights)
        neg_weights /= neg_weights.sum()

        neg_idx = torch.multinomial(
            neg_weights,
            len(pos_scores_all) * num_neg_samples,
            replacement=True
        )
        neg_idx = neg_idx.view(len(pos_scores_all), num_neg_samples)
        neg_samples = neg_scores[neg_idx]

        diff = pos_scores_all.unsqueeze(1) - neg_samples
        per_sample_loss = -torch.log(torch.sigmoid(diff)).mean(dim=1)

        weighted_loss = (pos_weights * per_sample_loss).sum() / pos_weights.sum()

        return weighted_loss

    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch_DelayTimeSampleBpr(epoch_idx,day)
                test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")

    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)


            self.optimizer.zero_grad()

            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)


            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
            loss = cvr_loss + net_cvr_loss
            loss.backward()
            self.optimizer.step()



            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def train_one_epoch_DelayTimeSampleBpr(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)

        prev_batch = None
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            click_ts = batch['click_ts'].to(self.device)
            pay_ts = batch['pay_ts'].to(self.device)
            refund_ts = batch['refund_ts'].to(self.device)
            delay_pay_time = pay_ts - click_ts
            delay_pay_time = delay_pay_time / 3600
            delay_pay_time = delay_pay_time.to(self.device)
            
            delay_refund_time = refund_ts - pay_ts
            delay_refund_time = delay_refund_time / 3600
            delay_refund_time = delay_refund_time.to(self.device)

            self.optimizer.zero_grad()

            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn_wi_DelayTimeBpr(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,delay_refund_time)

            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_DelayTimeBpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,delay_pay_time)
            loss = cvr_loss + net_cvr_loss
            loss.backward()
            self.optimizer.step()



            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics

    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []




        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())





        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRFWoRefundBackfillStreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)


    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels  + net_cvr_neg_loss * (1 - stream_refund_labels) )* stream_refund_mask)

        return refund_loss
    




    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
            
            cvr_loss.backward()
            self.optimizer.step()







            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}
        all_metrics["stream_model_CVR_AUC"] = 0
        all_metrics["stream_model_NetCVR_AUC"] = 0

        all_metrics["pretrained_model_CVR_AUC"] = 0
        all_metrics["pretrained_model_NetCVR_AUC"] = 0

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0


        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []

        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())

  
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, stream_cvr_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,stream_net_cvr_outputs)
                    all_metrics["stream_model_NetCVR_AUC"] += stream_model_net_cvr_auc

                    pretrained_model_cvr_auc = auc_score(pay_labels, pretrained_cvr_outputs)
                    all_metrics["pretrained_model_CVR_AUC"] += pretrained_model_cvr_auc

                    pretrained_model_net_cvr_auc = auc_score(net_pay_labels,pretrained_net_cvr_outputs)
                    all_metrics["pretrained_model_NetCVR_AUC"] += pretrained_model_net_cvr_auc

        all_metrics["stream_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["stream_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)


        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_CVR_AUC: {all_metrics['stream_model_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_NetCVR_AUC: {all_metrics['stream_model_NetCVR_AUC']:.5f}")

        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRFWiRefundBiasStreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)
        self.pretrained_inw_tn_pay_model.eval()

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_pay_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)


    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels  + net_cvr_neg_loss * (1 - stream_refund_labels) )* stream_refund_mask)

        return refund_loss
    




    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask)
            
            cvr_loss.backward()
            self.optimizer.step()







            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}
        all_metrics["stream_model_CVR_AUC"] = 0
        all_metrics["stream_model_NetCVR_AUC"] = 0

        all_metrics["pretrained_model_CVR_AUC"] = 0
        all_metrics["pretrained_model_NetCVR_AUC"] = 0

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0


        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []

        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())

  
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, stream_cvr_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,stream_net_cvr_outputs)
                    all_metrics["stream_model_NetCVR_AUC"] += stream_model_net_cvr_auc

                    pretrained_model_cvr_auc = auc_score(pay_labels, pretrained_cvr_outputs)
                    all_metrics["pretrained_model_CVR_AUC"] += pretrained_model_cvr_auc

                    pretrained_model_net_cvr_auc = auc_score(net_pay_labels,pretrained_net_cvr_outputs)
                    all_metrics["pretrained_model_NetCVR_AUC"] += pretrained_model_net_cvr_auc

        all_metrics["stream_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["stream_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_CVR_AUC"] /= len(tqdm_day_dataloader)
        all_metrics["pretrained_model_NetCVR_AUC"] /= len(tqdm_day_dataloader)


        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_CVR_AUC: {all_metrics['stream_model_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_NetCVR_AUC: {all_metrics['stream_model_NetCVR_AUC']:.5f}")

        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_DPStreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_tn_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_tn_pay_model=pretrained_inw_tn_pay_model
        self.pretrained_inw_tn_pay_model.to(self.device)

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.inw_tn_pay_optimizer = self.create_optimizer(self.pretrained_inw_tn_pay_model,args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def inw_cvr_loss_fn(self,inw_cvr_outputs,stream_pay_labels,stream_pay_mask,stream_dp_mask,eps=1e-6):
    
        inw_cvr_logits = inw_cvr_outputs.view(-1)
        stream_inw_pay_mask = (stream_pay_mask == 1) & (stream_dp_mask == 0)
        stream_inw_pay_labels = stream_pay_labels[stream_inw_pay_mask]
        inw_cvr_logits = inw_cvr_logits[stream_inw_pay_mask]


        pos_loss = stable_log1pex(inw_cvr_logits)
        neg_loss = inw_cvr_logits + stable_log1pex(inw_cvr_logits)
        cvr_loss = torch.mean((pos_loss*stream_inw_pay_labels + neg_loss*(1-stream_inw_pay_labels)))

        return cvr_loss

    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
 

        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight = ((1 - cvr_prob) * (1 + cvr_prob - inwpay_prob))/ (1 - inwpay_prob)
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
        
        bprloss = self.bpr_loss_weighted_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)
        
        
        loss= bprloss + cvr_loss

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]
        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_random_neg_weighted(self, cvr_outputs, labels, stream_pay_mask, stream_dp_mask,num_neg_samples = 50):
        valid_idx = torch.where(stream_pay_mask)[0]
        
        scores = cvr_outputs[valid_idx].squeeze()
        labels_valid = labels[valid_idx]
        dp_mask_valid = stream_dp_mask[valid_idx]

        pos_idx = (labels_valid == 1)
        neg_idx = (labels_valid == 0)

        pos_scores = scores[pos_idx]
        neg_scores = scores[neg_idx]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

  
        neg_sample_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_sample_idx]

        pairwise_logits = pos_scores.unsqueeze(1) - neg_samples
        raw_loss = -torch.log(torch.sigmoid(pairwise_logits) + 1e-8)

        pos_is_dp = dp_mask_valid[pos_idx].bool()

        pos_weights = (torch.sigmoid(scores[pos_idx]))
        weighted_loss = raw_loss * pos_weights.unsqueeze(1)

        return weighted_loss.mean()

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def cvr_loss_fn_wi_hierarchical_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,stream_dp_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        stream_pay_mask = stream_pay_mask.view(-1)
        stream_dp_mask = stream_dp_mask.view(-1)
        stream_inw_mask = (stream_pay_mask == 1) & (stream_dp_mask == 0)
        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob

        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
        
        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)
        loss= cvr_loss + bprloss
 
        return loss

    def hierarchical_bpr_loss_random_neg(self,cvr_outputs, labels, inw_mask,delay_mask, num_neg_samples=1):
        valid_inw_indices = torch.where(inw_mask)[0]
        valid_delay_indices = torch.where(delay_mask)[0]
        inw_cvr_valid = cvr_outputs[valid_inw_indices]
        delay_cvr_valid = cvr_outputs[valid_delay_indices]
        inw_label_valid = labels[valid_inw_indices]
        delay_label_valid = labels[valid_delay_indices]

        pos_mask = (inw_label_valid == 1)
        neg_mask = (delay_label_valid == 1)

        pos_scores = inw_cvr_valid[pos_mask]
        neg_scores = delay_cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)


        delay_cvr_prob_valid = torch.sigmoid(delay_cvr_valid).detach()
        neg_probs = delay_cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]


        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def cl_loss_fn(self, update_before_cvr_outputs, updated_cvr_outputs,dp_valid_mask,stream_pay_labels,margin=0, lambda_prox=0.1, lambda_mean=0.1):
        before_logits = update_before_cvr_outputs.view(-1).detach()
        after_logits = updated_cvr_outputs.view(-1)
        before_probs = torch.sigmoid(before_logits)
        pos_loss = F.relu(margin + before_logits - after_logits)
        pos_term = (pos_loss * dp_valid_mask).mean()
        prox = lambda_prox * (after_logits - before_logits).pow(2).mean()
        mean_pen = lambda_mean * (torch.sigmoid(after_logits).mean() - stream_pay_labels.mean()).pow(2)

        loss = pos_term + prox + mean_pen   

        return loss



    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total




    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()
        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)


        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            stream_dp_mask = delay_pay_labels_afterPay

     


            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()


            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_pay_outputs = self.pretrained_inw_tn_pay_model(features)
            inw_pay_outputs = pretrain_inw_tn_pay_outputs[:,0].view(-1)
            tn_pay_outputs = pretrain_inw_tn_pay_outputs[:,1].view(-1)
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()








            loss = cvr_loss + net_cvr_loss 
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []

        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())









        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))
    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

class AliEsdfmRF_Inw_StreamTrainer(metaclass=ABCMeta):
    
    def __init__(self, args,pretrained_model,pretrained_inw_pay_model,pretrained_inw_tn_refund_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_inw_pay_model=pretrained_inw_pay_model
        self.pretrained_inw_pay_model.to(self.device)

        self.pretrained_inw_tn_refund_model = pretrained_inw_tn_refund_model
        self.pretrained_inw_tn_refund_model.to(self.device)
        self.pretrained_inw_tn_refund_model.eval()
        
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_inw_tn_refund_model.parameters():
            param.requires_grad = False


        self.train_loader=train_loader
        self.test_loader=test_loader
        self.epochs = args.epochs
        self.optimizer = self.create_optimizer(self.stream_model, args)
        self.inw_optimizer = self.create_optimizer(self.pretrained_inw_pay_model, args)
        self.dataset_name = args.dataset_name
        self.mode = args.mode
        self.model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}.pth")
        self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs, eta_min=args.lr)
        self.best_auc = 0

        logfile = "./log/{}-{}-{}.txt".format(
            datetime.now().strftime("%Y-%m-%d"),
            self.mode,
            self.dataset_name
        )
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s-%(message)s',
            handlers=[
                logging.FileHandler(filename=logfile),
                logging.StreamHandler()
            ]
        )
        self.logger= logging.getLogger(__name__)


    def extra_cvr_loss_fn(self,extra_cvr_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        cvr_logits = extra_cvr_outputs.view(-1)
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*stream_pay_labels + neg_loss*(1-stream_pay_labels))*stream_pay_mask)
        return cvr_loss

    def cvr_loss_fn(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):

        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        cvr_prob = pretrain_cvr_prob
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight =  (1 + cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss
    
    def cvr_loss_fn2(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
    
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)


        valid_mask = stream_pay_mask.bool()
        valid_indices = torch.where(valid_mask)[0]
        valid_cvr = cvr_prob[valid_mask]
        valid_refund = refund_prob[valid_mask]
        _, cvr_order = torch.sort(valid_cvr, descending=True)
        cvr_rank = torch.empty_like(valid_cvr).long()
        cvr_rank[cvr_order] = torch.arange(len(valid_cvr), device=cvr_prob.device)
        _, refund_order = torch.sort(valid_refund, descending=False)
        refund_rank = torch.empty_like(valid_refund).long()
        refund_rank[refund_order] = torch.arange(len(valid_refund), device=refund_prob.device)
        B_valid = len(valid_cvr)
        score = (B_valid - cvr_rank) + (B_valid - refund_rank)
        k =  int(len(score) * 0.3)
        topk_scores, topk_idx_in_valid = torch.topk(score, k=k, largest=True)
        topk_indices_in_batch = valid_indices[topk_idx_in_valid]
        corrected_labels = stream_pay_labels.clone()
        mask = (corrected_labels[topk_indices_in_batch] == 0) & (cvr_prob[topk_indices_in_batch] > 0.75)
        indices_to_correct = topk_indices_in_batch[mask]
        if len(indices_to_correct) > 0:
            corrected_labels[indices_to_correct] = 1
            stream_pay_mask[indices_to_correct] = 0
        stream_pay_labels = corrected_labels

        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def cvr_loss_fn_wi_bpr(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_logits = refund_outputs.view(-1)
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
 


        pos_weight = 1 + cvr_prob - inwpay_prob
        neg_weight = ((1 - cvr_prob) * (1 + cvr_prob - inwpay_prob))*(1 + inwpay_prob + inwpay_prob**2 + inwpay_prob**3 + inwpay_prob**4 )
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)
        
        bprloss = self.bpr_loss_random_neg(cvr_outputs, stream_pay_labels, stream_pay_mask)
        
        
        loss= bprloss + cvr_loss

        return loss

    def bpr_loss_random_neg(self,cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)

        neg_idx = torch.randint(len(neg_scores), (len(pos_scores), num_neg_samples))
        neg_samples = neg_scores[neg_idx]

        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def bpr_loss_weighted_neg(self, cvr_outputs, labels, mask, num_neg_samples=50):
        valid_indices = torch.where(mask)[0]
        cvr_valid = cvr_outputs[valid_indices]
        label_valid = labels[valid_indices]

        pos_mask = (label_valid == 1)
        neg_mask = (label_valid == 0)

        pos_scores = cvr_valid[pos_mask]
        neg_scores = cvr_valid[neg_mask]

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return torch.tensor(0.0, device=cvr_outputs.device)
        cvr_prob_valid = torch.sigmoid(cvr_valid).detach()
        pos_probs = cvr_prob_valid[pos_mask]
        neg_probs = cvr_prob_valid[neg_mask]
        neg_weights = 1 - neg_probs
        neg_weights = neg_weights.float()
        neg_weights /= neg_weights.sum()
        neg_idx = torch.multinomial(neg_weights, len(pos_scores) * num_neg_samples, replacement=True)
        neg_idx = neg_idx.view(len(pos_scores), num_neg_samples)
        neg_samples = neg_scores[neg_idx]
        loss = -torch.log(torch.sigmoid(pos_scores.unsqueeze(1) - neg_samples)).mean()
        return loss

    def cvr_loss_fn_wi_smooth(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,smooth_factor,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1).clone()
        smooth_factor = smooth_factor.view(-1)

 
        smooth_mask = (stream_pay_labels == 0)
        stream_pay_labels[smooth_mask] = self.make_smooth_labels_for_negatives(smooth_factor[smooth_mask])


        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def cvr_loss_fn4(self,cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,tn_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        inwpay_logtits = inw_pay_outputs.view(-1)
        tn_logtits = tn_pay_outputs.view(-1)
        pretrain_cvr_logits = pretrain_cvr_outputs.view(-1)
        pretrain_cvr_prob = torch.sigmoid(pretrain_cvr_logits).detach()
        refund_prob = torch.sigmoid(refund_outputs).detach()
        cvr_prob = torch.sigmoid(cvr_logits).detach()
        tn_prob = torch.sigmoid(tn_logtits).detach()
        inwpay_prob = torch.sigmoid(inwpay_logtits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)
        retention_score = cvr_prob 
        stream_pay_labels = self.smooth_binary_labels_with_score(stream_pay_labels,retention_score)


        pos_weight = 1 + pretrain_cvr_prob - inwpay_prob
        neg_weight =  (1 + pretrain_cvr_prob - inwpay_prob)*tn_prob
        pos_loss = stable_log1pex(cvr_logits)
        neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean((pos_loss*pos_weight*stream_pay_labels + neg_loss*neg_weight*(1-stream_pay_labels))*stream_pay_mask)

        return cvr_loss

    def refund_loss_fn(self,refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,eps=1e-6):
        stream_net_pay_labels = stream_net_pay_labels.float().view(-1)
        stream_pay_labels = stream_pay_labels.float().view(-1)
        stream_refund_labels = ((stream_pay_labels==1) & (stream_net_pay_labels == 0))
        stream_refund_mask = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = stream_refund_mask.float().view(-1)



        refund_outputs = refund_outputs.view(-1)
        pretrain_refund_outputs = pretrain_refund_outputs.view(-1)
        inw_refund_outputs = inw_refund_outputs.view(-1)
        tn_refund_outputs = tn_refund_outputs.view(-1)

        refund_prob = torch.sigmoid(refund_outputs).detach()
        pretrain_refund_prob = torch.sigmoid(pretrain_refund_outputs).detach()
        tn_refund_prob = torch.sigmoid(tn_refund_outputs).detach()
        inw_refund_prob = torch.sigmoid(inw_refund_outputs).detach()

        refund_pos_weight = 1 + pretrain_refund_prob - inw_refund_prob
        refund_neg_weight = (1 + pretrain_refund_prob - inw_refund_prob)*tn_refund_prob
        net_cvr_pos_loss = stable_log1pex(refund_outputs)
        net_cvr_neg_loss = refund_outputs + stable_log1pex(refund_outputs)
        refund_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels * refund_pos_weight + net_cvr_neg_loss * (1 - stream_refund_labels) * refund_neg_weight)* stream_refund_mask)

        return refund_loss
    
    def smooth_binary_labels_with_score(self,labels, scores, threshold=0.2,min_smooth=0.1,max_smooth=1):
        soft_labels = labels.float().clone()
        mask = (labels == 0) & (scores > threshold)
        if mask.sum() == 0:
            return soft_labels


        soft_value = min_smooth + scores[mask]
        soft_labels[mask] = soft_value

        return soft_labels
    
    def drop_and_construct_sample(self,features,pay_labels,net_pay_labels,stream_pay_labels,stream_net_pay_labels,stream_pay_mask,threshold=0.7):
        eval_model =  copy.deepcopy(self.stream_model)
        eval_model.to(self.device)
        eval_model.eval()

        cvr_probs,net_cvr_probs = eval_model.predict(features)
        refund_logits = eval_model.net_cvr_forward(features)
        refund_probs = torch.sigmoid(refund_logits).detach()
        stream_pay_labels = stream_pay_labels.view(-1)

        valid_mask = stream_pay_mask.bool()
        valid_indices = torch.where(valid_mask)[0]
        valid_cvr = cvr_probs[valid_mask]
        valid_net_cvr = net_cvr_probs[valid_mask]
        valid_refund = refund_probs[valid_mask]
        valid_stream_pay_labels = stream_pay_labels[valid_mask]

        retention_score = valid_cvr*((1-valid_refund)**2) 
        replace_mask = (valid_stream_pay_labels == 0) & (retention_score > threshold)
        print(torch.sum(replace_mask))
        if not replace_mask.any():
            return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask
        
        negative_mask = (valid_stream_pay_labels == 0) & ~replace_mask 
        negative_indices = torch.where(negative_mask)[0]
        
        if len(negative_indices) == 0:
            return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask

        replace_indices = torch.randint(len(negative_indices), (replace_mask.sum(),))
        source_indices = negative_indices[replace_indices]

        target_indices = torch.where(replace_mask)[0]
        features[valid_indices[target_indices]] = features[valid_indices[source_indices]]
        pay_labels[valid_indices[target_indices]] = pay_labels[valid_indices[source_indices]]
        net_pay_labels[valid_indices[target_indices]] = net_pay_labels[valid_indices[source_indices]]
        stream_pay_labels[valid_indices[target_indices]] = stream_pay_labels[valid_indices[source_indices]]
        stream_net_pay_labels[valid_indices[target_indices]] = stream_net_pay_labels[valid_indices[source_indices]]
        stream_pay_mask[valid_indices[target_indices]] = stream_pay_mask[valid_indices[source_indices]]
        return features, pay_labels, net_pay_labels, stream_pay_labels, stream_net_pay_labels, stream_pay_mask

    def make_smooth_labels_for_negatives(self,smooth_factor, max_value=0.1, sharpness=2.0):
        x = smooth_factor**sharpness
        return x * max_value

    def inw_cvr_loss_fn(self,inw_cvr_outputs,stream_pay_labels,stream_pay_mask,stream_dp_mask,eps=1e-6):
        
        inw_cvr_logits = inw_cvr_outputs.view(-1)
        stream_inw_pay_mask = (stream_pay_mask == 1) & (stream_dp_mask == 0)
        stream_inw_pay_labels = stream_pay_labels[stream_inw_pay_mask]
        inw_cvr_logits = inw_cvr_logits[stream_inw_pay_mask]


        pos_loss = stable_log1pex(inw_cvr_logits)
        neg_loss = inw_cvr_logits + stable_log1pex(inw_cvr_logits)
        cvr_loss = torch.mean((pos_loss*stream_inw_pay_labels + neg_loss*(1-stream_inw_pay_labels)))
        return cvr_loss


    def aggregate_metrics(self, metrics_list):
        total = {}
        for key in metrics_list[0].keys():
            total[key] = 0.0
        for daily_metrics in metrics_list:
            for key, value in daily_metrics.items():
                total[key] += value
        for key in total:
            total[key] /= len(metrics_list)

        return total

    def train(self):
        all_day_metrics = []
        for day in tqdm(range(len(self.train_loader)), desc="Days"):
            for epoch_idx in tqdm(range(self.epochs),desc=f"Epoch",leave=True):
                train_metrics = self.train_one_epoch(epoch_idx,day)
            test_day_metrics=self.test(day)
            all_day_metrics.append(test_day_metrics)  
        avg_metrics = self.aggregate_metrics(all_day_metrics)

        self.logger.info("==== Average Test Metrics Over All Days ====")
        for k, v in avg_metrics.items():
            self.logger.info(f"{k}: {v:.5f}")


    def train_one_epoch(self,epoch_idx,day_idx):
        """
        训练一个epoch

        Args:
            epoch_idx (int): 当前epoch的索引

        Returns:
            None

        """
        self.stream_model.train()

        total_loss = 0
        total_batches = 0
        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)

            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            stream_dp_mask = delay_pay_labels_afterPay

            self.pretrained_inw_pay_model.train()
            self.inw_optimizer.zero_grad()
            inw_pay_outputs = self.pretrained_inw_pay_model(features)
            inw_pay_loss = self.inw_cvr_loss_fn(inw_pay_outputs,stream_pay_labels,stream_pay_mask,stream_dp_mask)
            inw_pay_loss.backward()
            self.inw_optimizer.step()
            self.pretrained_inw_pay_model.eval()

            self.optimizer.zero_grad()
            refund_outputs = self.stream_model.net_cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            pretrain_inw_tn_refund_outputs = self.pretrained_inw_tn_refund_model(features)
            inw_refund_outputs = pretrain_inw_tn_refund_outputs[:,0].view(-1)
            tn_refund_outputs = pretrain_inw_tn_refund_outputs[:,1].view(-1)
            net_cvr_loss = self.refund_loss_fn(refund_outputs,pretrain_refund_outputs,inw_refund_outputs,tn_refund_outputs,stream_pay_labels,stream_net_pay_labels,stream_pay_mask)
            net_cvr_loss.backward()
            self.optimizer.step()


            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            pretrain_cvr_outputs = self.pretrained_model.cvr_forward(features)
            pretrain_refund_outputs = self.pretrained_model.net_cvr_forward(features)
            inw_pay_outputs = self.pretrained_inw_pay_model(features)
           
            cvr_loss = self.cvr_loss_fn_wi_bpr(cvr_outputs,pretrain_cvr_outputs,inw_pay_outputs,refund_outputs,stream_pay_labels,stream_pay_mask)

            cvr_loss.backward()
            self.optimizer.step()







            loss = cvr_loss + net_cvr_loss
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                cvr_outputs,net_cvr_outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, cvr_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,net_cvr_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.scheduler.step()
        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_model.state_dict(), self.model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        stream_model = copy.deepcopy(self.stream_model)

        stream_model.eval()
        self.pretrained_model.eval()
    

        all_metrics = {}

        all_metrics["stream_model_Global_CVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC"] = 0
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = 0 
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = 0



        all_pay_labels = []
        all_net_pay_labels = []
        stream_model_all_pay_preds = []
        stream_model_all_net_pay_preds = []
        pretrained_model_all_pay_preds = []
        pretrained_model_all_net_pay_preds = []




        with torch.no_grad():
            day_loader = self.test_loader.get_day_dataloader(day_idx)
            tqdm_day_dataloader = tqdm(day_loader, desc=f"Test Day {day_idx+1}", leave=False)

            for batch_idx,batch in enumerate(tqdm_day_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                stream_cvr_outputs,stream_net_cvr_outputs = stream_model.predict(features)
                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrained_model.predict(features)



                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                stream_model_all_pay_preds.extend(stream_cvr_outputs.cpu().numpy().tolist())
                stream_model_all_net_pay_preds.extend(stream_net_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy().tolist())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy().tolist())





        all_metrics["stream_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )


        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC_by_cvr_head"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL_by_cvr_head"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC_by_cvr_head"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )

        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC_by_cvr_head"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )


        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['stream_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_AUC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_NLL_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PCOC_by_cvr_head']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC_by_cvr_head: {all_metrics['pretrained_model_Global_NetCVR_PRAUC_by_cvr_head']:.5f}")

        return all_metrics

    def setup_train(self,args):
        """
        设置训练环境。

        Args:
            args (argparse.Namespace): 包含训练参数的对象。

        Returns:
            None

        """
        self.fix_random_seed_as(args)
        self.device = torch.device(f"cuda:{self.args.device_idx}" if torch.cuda.is_available() else "cpu")
        print("########################################")
        print(f"Using {self.device} device.")
        print("########################################")

    def fix_random_seed_as(self,args):
        random_seed = args.seed
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    def set_up_gpu(self,args):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.device_idx
        args.num_gpu = len(args.device_idx.split(","))

    def create_optimizer(self,model, args):
        """
        根据给定的参数创建一个优化器。

        Args:
            model (nn.Module): 要优化的模型。
            args (argparse.Namespace): 包含优化器参数的对象。

        Returns:
            optim.Optimizer: 创建的优化器。

        """
        if args.optimizer == 'AdamW':
            return optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        return optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)



def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor(0.0, device=x.device)) + torch.log1p(torch.exp(-torch.abs(x)))




