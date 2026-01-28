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



class AliPretrainDeferTrainer(metaclass=ABCMeta):
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

    def loss_fn(self,outputs,labels):
        outputs = outputs.view(-1)
        labels = labels.float()

        pos_loss = stable_log1pex(outputs)
        neg_loss = outputs + stable_log1pex(outputs)
        loss = torch.mean(pos_loss * labels + neg_loss * (1 - labels))
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
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.loss_fn(outputs, pay_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                outputs = self.model.predict(features)
                cvr_auc = auc_score(pay_labels, outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,outputs)
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
                outputs = Recmodel.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy().tolist())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy().tolist())
                all_pay_preds.extend(outputs.cpu().numpy().tolist())
                all_net_pay_preds.extend(outputs.cpu().numpy().tolist())

                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,outputs)
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

class AliPretrainDpDeferTrainer(metaclass=ABCMeta):
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

    def loss_fn(self,outputs,labels):
        outputs = outputs.view(-1)
        labels = labels.float()

        pos_loss = stable_log1pex(outputs)
        neg_loss = outputs+stable_log1pex(outputs)
        loss = torch.mean(pos_loss*labels+neg_loss*(1-labels))
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
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["DP_CVR_AUC"] = 0
        for batch_idx,batch in enumerate(tqdm_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
            delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
            inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss =self.loss_fn(outputs, delay_pay_labels_afterPay)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1

            with torch.no_grad():
                outputs = self.model.predict(features)
                cvr_auc = auc_score(pay_labels, outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc
                dp_cvr_auc = auc_score(delay_pay_labels_afterPay,outputs)
                all_metrics["DP_CVR_AUC"] += dp_cvr_auc

            tqdm_dataloader.set_description('Epoch {}, loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        mean_loss = total_loss / total_batches
        self.logger.info(f"Epoch {epoch_idx+1} - Mean Loss: {mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["DP_CVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"Epoch {epoch_idx+1} train: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train: DP_CVR_AUC: {all_metrics['DP_CVR_AUC']:.5f}")
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
        Recmodel.load_state_dict(torch.load(self.model_pth))
        Recmodel.eval()
        all_metrics = {}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0
        all_metrics["DP_CVR_AUC"] = 0
        with torch.no_grad():
            tqdm_dataloader = tqdm(self.test_loader)
            for batch_idx,batch in enumerate(tqdm_dataloader):
                features = batch['features'].to(self.device)
                pay_labels = batch['pay_labels'].to(self.device)
                net_pay_labels = batch['net_pay_labels'].to(self.device)
                delay_pay_labels_afterPay = batch['delay_pay_labels_afterPay'].to(self.device)
                delay_pay_label_afterRefund = batch['delay_pay_label_afterRefund'].to(self.device)
                inw_pay_labels_afterPay = batch['inw_pay_labels_afterPay'].to(self.device)
                outputs = Recmodel.predict(features)

                with torch.no_grad():

                    cvr_auc = auc_score(pay_labels, outputs)
                    all_metrics["CVR_AUC"] += cvr_auc

                    net_cvr_auc = auc_score(net_pay_labels,outputs)
                    all_metrics["NetCVR_AUC"] += net_cvr_auc

                    dp_cvr_auc = auc_score(delay_pay_labels_afterPay,outputs)
                    all_metrics["DP_CVR_AUC"] += dp_cvr_auc

        all_metrics["CVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["NetCVR_AUC"] /= len(tqdm_dataloader)
        all_metrics["DP_CVR_AUC"] /= len(tqdm_dataloader)
        self.logger.info(f"CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.logger.info(f"DP_CVR_AUC: {all_metrics['DP_CVR_AUC']:.5f}")

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
    
class AliDeferStreamTrainer(metaclass=ABCMeta):

    def __init__(self, args,pretrained_model,pretrained_dp_model,train_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.stream_model=copy.deepcopy(pretrained_model)
        self.stream_model.to(self.device)

        self.pretrained_model=pretrained_model
        self.pretrained_model.to(self.device)
        self.pretrained_model.eval()

        self.pretrained_dp_model=pretrained_dp_model
        self.pretrained_dp_model.to(self.device)
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False
        for param in self.pretrained_dp_model.parameters():
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


    def loss_fn(self,outputs, dp_outputs, stream_labels,eps=1e-6):

        logits = outputs.view(-1)
        cvr_probs = torch.sigmoid(logits)
        dp_logits = dp_outputs.view(-1)
        dp_probs = torch.sigmoid(dp_logits)
        z = stream_labels.float().view(-1)

        pos_weight = cvr_probs/torch.abs(cvr_probs - 0.5*dp_probs)
        pos_weight = torch.clamp(pos_weight, min=eps)   
        neg_weight = (1 - cvr_probs) / (1 - cvr_probs + 0.5 * dp_probs)

        pos_weight = pos_weight.detach()
        neg_weight = neg_weight.detach()
        
        pos_loss = stable_log1pex(logits)
        neg_loss = logits + stable_log1pex(logits)
        clf_loss = torch.mean(pos_loss * pos_weight * z + neg_loss * neg_weight * (1 - z))

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
            self.optimizer.zero_grad()
            outputs = self.stream_model(features)
            dp_outputs = self.pretrained_dp_model(features)
            loss = self.loss_fn(outputs, dp_outputs, stream_pay_labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1
            with torch.no_grad():
                outputs = self.stream_model.predict(features)
                cvr_auc = auc_score(pay_labels, outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,outputs)
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
        all_metrics["stream_model_Global_CVR_NLL"] = 0
        all_metrics["stream_model_Global_NetCVR_NLL"] = 0
        all_metrics["stream_model_Global_CVR_PCOC"] = 0
        all_metrics["stream_model_Global_NetCVR_PCOC"] = 0
        all_metrics["stream_model_Global_CVR_PRAUC"] = 0
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = 0

        all_metrics["pretrained_model_Global_CVR_AUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = 0
        all_metrics["pretrained_model_Global_CVR_NLL"] = 0
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = 0
        all_metrics["pretrained_model_Global_CVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = 0
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = 0
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = 0

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
                stream_outputs = stream_model.predict(features)
                pretrained_outputs = self.pretrained_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy())
                stream_model_all_pay_preds.extend(stream_outputs.cpu().numpy())
                stream_model_all_net_pay_preds.extend(stream_outputs.cpu().numpy())
                pretrained_model_all_pay_preds.extend(pretrained_outputs.cpu().numpy())
                pretrained_model_all_net_pay_preds.extend(pretrained_outputs.cpu().numpy())
  
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, stream_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,stream_outputs)
                    all_metrics["stream_model_NetCVR_AUC"] += stream_model_net_cvr_auc

                    pretrained_model_cvr_auc = auc_score(pay_labels, pretrained_outputs)
                    all_metrics["pretrained_model_CVR_AUC"] += pretrained_model_cvr_auc

                    pretrained_model_net_cvr_auc = auc_score(net_pay_labels,pretrained_outputs)
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
        all_metrics["stream_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["stream_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(stream_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_AUC"] = auc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_AUC"] = auc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_NLL"] = nll_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_NLL"] = nll_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_PCOC"] = pcoc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PCOC"] = pcoc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_CVR_PRAUC"] = prauc_score(
            torch.tensor(all_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_pay_preds, dtype=torch.float32)
        )
        all_metrics["pretrained_model_Global_NetCVR_PRAUC"] = prauc_score(
            torch.tensor(all_net_pay_labels, dtype=torch.float32),
            torch.tensor(pretrained_model_all_net_pay_preds, dtype=torch.float32)
        )




        self.logger.info(f"stream_model_CVR_AUC: {all_metrics['stream_model_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_NetCVR_AUC: {all_metrics['stream_model_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_CVR_AUC: {all_metrics['pretrained_model_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_NetCVR_AUC: {all_metrics['pretrained_model_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_AUC: {all_metrics['stream_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_AUC: {all_metrics['stream_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_NLL: {all_metrics['stream_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_NLL: {all_metrics['stream_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PCOC: {all_metrics['stream_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PCOC: {all_metrics['stream_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"stream_model_Global_CVR_PRAUC: {all_metrics['stream_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"stream_model_Global_NetCVR_PRAUC: {all_metrics['stream_model_Global_NetCVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_AUC: {all_metrics['pretrained_model_Global_CVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_AUC: {all_metrics['pretrained_model_Global_NetCVR_AUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_NLL: {all_metrics['pretrained_model_Global_CVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_NLL: {all_metrics['pretrained_model_Global_NetCVR_NLL']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PCOC: {all_metrics['pretrained_model_Global_CVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PCOC: {all_metrics['pretrained_model_Global_NetCVR_PCOC']:.5f}")
        self.logger.info(f"pretrained_model_Global_CVR_PRAUC: {all_metrics['pretrained_model_Global_CVR_PRAUC']:.5f}")
        self.logger.info(f"pretrained_model_Global_NetCVR_PRAUC: {all_metrics['pretrained_model_Global_NetCVR_PRAUC']:.5f}")
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/criteo/processed_data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Criteo', help='Dataset name')
    parser.add_argument('--pay_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--refund_attr_window', type=int, default=7, help='Attribution window size (days)')
    parser.add_argument('--pay_wait_window', type=int, default=1, help='pay wait window size (days)')
    parser.add_argument('--refund_wait_window', type=int, default=1, help='refund wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=15, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=15, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=30, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="defer_train_stream", help='[defer_pretrain,defer_dp_pretrain,defer_train_stream]')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--embed_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    parser.add_argument('--device_idx', type=str, default='1', help='Device index')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='Optimizer')
    parser.add_argument('--model_save_pth', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/pretrain_models/20250623', help='Model save pth')
    parser.add_argument('--pretrain_defer_dp_model_pth', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/pretrain_models/20250623/aux_model_pth1/dp_model.pth', help='if need a pretrain model1 to support')
    parser.add_argument('-reg_loss_decay',type=float,default=1e-4,help='Regularization loss decay coefficient')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()





