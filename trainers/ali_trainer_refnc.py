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
import torch.nn.functional as F

from mx_utils.metrics import auc_score,nll_score,prauc_score,pcoc_score,stable_log1pex



class AliPretrainReFncTrainer(metaclass=ABCMeta):
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

        print(pos_logits.sigmoid())
        print(len(pos_logits))
        print(neg_logits.sigmoid())
        print(len(neg_logits))
        print(torch.mean(pos_logits.sigmoid()))
        print(torch.mean(neg_logits.sigmoid()))
        print("#"*100)
        if len(pos_logits) == 0 or len(neg_logits) == 0:
            return 0.0

        diff = pos_logits[:, None] - neg_logits[None, :]
        bpr_loss = -torch.log(torch.sigmoid(diff + eps)).mean()

        return gamma * bpr_loss
    

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


  
    
class AliReFncStreamTrainer(metaclass=ABCMeta):

    def __init__(self, args,pretrained_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()


        for param in self.pretrained_model.parameters():
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

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s-%(message)s',
                            handlers=[logging.FileHandler(filename="./log/{}.txt".format(datetime.now().strftime("%Y-%m-%d"))),
                                      logging.StreamHandler()])
        self.logger= logging.getLogger(__name__)


    def loss_fn(self,cvr_outputs,net_cvr_outputs, stream_labels,stream_pay_mask,eps=1e-6):
        
        cvr_logits = cvr_outputs.view(-1)
        net_cvr_logits = net_cvr_outputs.view(-1)
        stream_labels = stream_labels.float().view(-1)

        cvr_pos_loss = stable_log1pex(cvr_logits)
        cvr_neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean(cvr_pos_loss * stream_labels * stream_pay_mask + cvr_neg_loss * (1 - stream_labels) * stream_pay_mask)
        net_cvr_pos_loss = stable_log1pex(net_cvr_logits)
        net_cvr_neg_loss = net_cvr_logits + stable_log1pex(net_cvr_logits)
        net_cvr_loss = torch.mean(net_cvr_pos_loss *  stream_labels + net_cvr_neg_loss *(1 - stream_labels))
        clf_loss = net_cvr_loss + cvr_loss

        return clf_loss

    def cvr_loss_fn(self,cvr_outputs, stream_labels,stream_pay_mask,eps=1e-6):
        cvr_logits = cvr_outputs.view(-1)
        stream_labels = stream_labels.float().view(-1)
        cvr_pos_loss = stable_log1pex(cvr_logits)
        cvr_neg_loss = cvr_logits + stable_log1pex(cvr_logits)
        cvr_loss = torch.mean(cvr_pos_loss * stream_labels * stream_pay_mask + cvr_neg_loss * (1 - stream_labels) * stream_pay_mask)
        clf_loss =  cvr_loss
        return clf_loss

    def net_cvr_loss_fn(self,net_cvr_outputs,stream_labels,stream_pay_mask,eps=1e-6):
        net_cvr_logits = net_cvr_outputs.view(-1)
        stream_labels = stream_labels.float().view(-1)
        net_cvr_pos_loss = stable_log1pex(net_cvr_logits)
        net_cvr_neg_loss = net_cvr_logits + stable_log1pex(net_cvr_logits)
        net_cvr_loss = torch.mean(net_cvr_pos_loss *  stream_labels  + net_cvr_neg_loss *(1 - stream_labels) )
        clf_loss = net_cvr_loss 
        return clf_loss


    def refund_loss_fn(self,net_cvr_outputs,stream_labels,stream_pay_mask,eps=1e-6):
        net_cvr_logits = net_cvr_outputs.view(-1)
        stream_labels = stream_labels.float().view(-1)
        stream_refund_labels = 1 - stream_pay_mask
        stream_refund_labels = stream_refund_labels.float().view(-1)
        stream_refund_mask = (stream_labels > 0) | (stream_refund_labels > 0)
        stream_refund_mask = stream_refund_mask.float().view(-1)
        net_cvr_pos_loss = stable_log1pex(net_cvr_logits)
        net_cvr_neg_loss = net_cvr_logits + stable_log1pex(net_cvr_logits)
        net_cvr_loss = torch.mean((net_cvr_pos_loss *  stream_refund_labels  + net_cvr_neg_loss *(1 - stream_refund_labels)) * stream_refund_mask)
        clf_loss = net_cvr_loss 
        return clf_loss
    
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
        all_metrics["NetCVR_AUC_by_CVR_head"] = 0
        day_loader = self.train_loader.get_day_dataloader(day_idx)
        tqdm_day_dataloader = tqdm(day_loader, desc=f"Train Day {day_idx}", leave=False)
        for i, batch in enumerate(tqdm_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)



            self.optimizer.zero_grad()
            cvr_outputs = self.stream_model.cvr_forward(features)
            cvr_loss = self.cvr_loss_fn(cvr_outputs,stream_pay_labels,stream_pay_mask)
            cvr_loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()
            net_cvr_outputs = self.stream_model.net_cvr_forward(features)
            net_cvr_loss = self.refund_loss_fn(net_cvr_outputs,stream_pay_labels,stream_pay_mask)
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
                net_cvr_auc_by_cvr_head = auc_score(net_pay_labels,cvr_outputs)
                all_metrics["NetCVR_AUC_by_CVR_head"] += net_cvr_auc_by_cvr_head

            tqdm_day_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC"] /= total_batches
        all_metrics["NetCVR_AUC_by_CVR_head"] /= total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC_by_CVR_head: {all_metrics['NetCVR_AUC_by_CVR_head']:.5f}")
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


        all_metrics["sum_pay_labels"] = sum(all_pay_labels)
        all_metrics["sum_net_pay_labels"] = sum(all_net_pay_labels)
        all_metrics["all_samples"] = len(all_pay_labels)
        all_metrics["mean_pay_labels"] = sum(all_pay_labels) / len(all_pay_labels)
        all_metrics["mean_net_pay_labels"] = sum(all_net_pay_labels) / len(all_net_pay_labels)
        all_metrics["sum_pay_preds"]= sum(stream_model_all_pay_preds)
        all_metrics["sum_net_pay_preds"] = sum(stream_model_all_net_pay_preds)
        all_metrics["mean_pay_preds"] = sum(stream_model_all_pay_preds) / len(stream_model_all_pay_preds)
        all_metrics["mean_net_pay_preds"] = sum(stream_model_all_net_pay_preds) / len(stream_model_all_net_pay_preds)
        self.logger.info(f"sum_pay_labels: {all_metrics['sum_pay_labels']:.5f}")
        self.logger.info(f"sum_net_pay_labels: {all_metrics['sum_net_pay_labels']:5f}")
        self.logger.info(f"all_samples: {all_metrics['all_samples']:.5f}")
        self.logger.info(f"mean_pay_labels: {all_metrics['mean_pay_labels']:.5f}")
        self.logger.info(f"mean_net_pay_labels: {all_metrics['mean_net_pay_labels']:.5f}")
        self.logger.info(f"sum_pay_preds: {all_metrics['sum_pay_preds']:.5f}")
        self.logger.info(f"sum_net_pay_preds: {all_metrics['sum_net_pay_preds']:.5f}")
        self.logger.info(f"mean_pay_preds: {all_metrics['mean_pay_preds']:.5f}")
        self.logger.info(f"mean_net_pay_preds: {all_metrics['mean_net_pay_preds']:.5f}")



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


def stable_log1pex(x):
    return -torch.minimum(x, torch.tensor(0.0, device=x.device)) + torch.log1p(torch.exp(-torch.abs(x)))


