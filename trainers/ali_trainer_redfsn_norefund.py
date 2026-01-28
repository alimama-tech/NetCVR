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
from mx_utils.metrics import auc_score,nll_score,prauc_score,pcoc_score



class AliPretrainReDfsnTrainer(metaclass=ABCMeta):
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
        cvr_loss = F.binary_cross_entropy(cvr_outputs, cvr_labels.float())
        net_cvr_loss = F.binary_cross_entropy(net_cvr_outputs, net_cvr_labels.float())
        loss = cvr_loss + net_cvr_loss
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
            cvr_outputs,net_cvr_outputs = self.model(features)
            loss = self.loss_fn(cvr_outputs,pay_labels,net_cvr_outputs,net_pay_labels)
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


    
class AliReDfsnStreamTrainer(metaclass=ABCMeta):


        
    def __init__(self, args,pretrain_model,stream_main_model,stream_feature_model,stream_unbias_model,main_loader,feature_loader,unbias_loader,test_loader):
        
        self.args=args
        self.setup_train(self.args)
        self.pretrain_model=pretrain_model
        self.pretrain_model.to(self.device)
        for param in self.pretrain_model.parameters():
            param.requires_grad = False
        self.stream_main_model=stream_main_model
        self.stream_feature_model=stream_feature_model
        self.stream_unbias_model=stream_unbias_model
        self.stream_main_model.to(self.device)
        self.stream_feature_model.to(self.device)
        self.stream_unbias_model.to(self.device)

        self.main_loader=main_loader
        self.feature_loader=feature_loader
        self.unbias_loader=unbias_loader
        self.test_loader=test_loader

        self.epochs = args.epochs
        self.main_optimizer = self.create_optimizer(self.stream_main_model, args)
        self.feature_optimizer = self.create_optimizer(self.stream_feature_model, args)
        self.unbias_optimizer = self.create_optimizer(self.stream_unbias_model, args)



        self.dataset_name = args.dataset_name
        self.mode = args.mode

        self.main_model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}_main.pth")
        self.feature_model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}_feature.pth")
        self.unbias_model_pth = os.path.join(args.model_save_pth, f"{self.dataset_name}_{self.mode}_unbias.pth")


        self.main_scheduler = lr_scheduler.CosineAnnealingLR(self.main_optimizer, T_max=self.epochs, eta_min=args.lr)
        self.feature_scheduler = lr_scheduler.CosineAnnealingLR(self.feature_optimizer, T_max=self.epochs, eta_min=args.lr)
        self.unbias_scheduler = lr_scheduler.CosineAnnealingLR(self.unbias_optimizer, T_max=self.epochs, eta_min=args.lr)


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


    def feature_loss_fn(self,outputs, stream_labels,stream_pay_mask,eps=1e-6):
        logits = torch.sigmoid(outputs).view(-1).clamp(min=eps, max=1-eps)
        z = stream_labels.float().view(-1)

        pos_loss = -torch.log(logits)
        neg_loss = -torch.log(1-logits)
        clf_loss = torch.mean((pos_loss  * z + neg_loss *  (1 - z))*stream_pay_mask)

        return clf_loss
    
    def unbias_loss_fn(self,outputs, pay_labels,stream_pay_mask,eps=1e-6):
        logits = torch.sigmoid(outputs).view(-1).clamp(min=eps,max=1-eps)
        z = pay_labels.view(-1)
        p_no_grad = logits.clone().detach()
        pos_loss = (1 + p_no_grad) * torch.log(logits)
        neg_loss = (1 - p_no_grad) * (1 + p_no_grad) * (torch.log(1 - logits))
        loss = torch.mean((-pos_loss * z - neg_loss * (1 - z))*stream_pay_mask)
        return loss
    
    def main_loss_fn(self,outputs,unbias_outputs, pay_labels,stream_pay_mask,eps=1e-6):
        logits = torch.sigmoid(outputs+unbias_outputs).view(-1).clamp(min=eps,max=1-eps)

        z = pay_labels.view(-1)
        pos_loss = -torch.log(logits)
        neg_loss = -torch.log(1-logits)
        clf_loss = torch.mean((pos_loss  * z + neg_loss *  (1 - z))*stream_pay_mask)
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
        for day in tqdm(range(len(self.main_loader)), desc="Days"):
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
        self.stream_main_model.train()
        self.stream_feature_model.train()
        self.stream_unbias_model.train()

        main_total_loss = 0
        feature_total_loss = 0
        unbias_total_loss = 0
        
        main_total_batches = 0
        feature_total_batches = 0
        unbias_total_batches = 0

        all_metrics ={}
        all_metrics["CVR_AUC"] = 0
        all_metrics["NetCVR_AUC"] = 0


        main_day_loader = self.main_loader.get_day_dataloader(day_idx)
        feature_day_loader = self.feature_loader.get_day_dataloader(day_idx)
        unbias_day_loader = self.unbias_loader.get_day_dataloader(day_idx)

        tqdm_main_day_dataloader = tqdm(main_day_loader, desc=f"Train Day {day_idx}", leave=False)
        tqdm_feature_day_dataloader = tqdm(feature_day_loader, desc=f"Train Day {day_idx}", leave=False)
        tqdm_unbias_day_dataloader = tqdm(unbias_day_loader, desc=f"Train Day {day_idx}", leave=False)


        for i, batch in enumerate(tqdm_feature_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            self.feature_optimizer.zero_grad()
            cvr_outputs,net_cvr_outputs = self.stream_feature_model(features)
            loss = self.feature_loss_fn(cvr_outputs, stream_pay_labels,stream_pay_mask)
            loss.backward()
            self.feature_optimizer.step()
            feature_total_loss += loss.item()
            feature_total_batches += 1

            tqdm_feature_day_dataloader.set_description('Epoch {}, feature_loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))


        for i, batch in enumerate(tqdm_unbias_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            self.unbias_optimizer.zero_grad()
            cvr_outputs,net_cvr_outputs = self.stream_unbias_model(features)
            loss = self.unbias_loss_fn(cvr_outputs, stream_pay_labels,stream_pay_mask)
            loss.backward()
            self.unbias_optimizer.step()
            unbias_total_loss += loss.item()
            unbias_total_batches += 1
            tqdm_unbias_day_dataloader.set_description('Epoch {}, unbias_loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))


        for i, batch in enumerate(tqdm_main_day_dataloader):
            features = batch['features'].to(self.device)
            pay_labels = batch['pay_labels'].to(self.device)
            net_pay_labels = batch['net_pay_labels'].to(self.device)
            stream_pay_labels = batch['stream_pay_labels'].to(self.device)
            stream_net_pay_labels = batch['stream_net_pay_labels'].to(self.device)
            stream_pay_mask = batch['stream_pay_mask'].to(self.device)
            with torch.no_grad():
                feature_x_embed = self.stream_feature_model.get_x_embed(features)
                feature_cvr_outputs,feature_net_cvr_outputs = self.stream_feature_model(features)
                unbias_x_embed = self.stream_unbias_model.get_x_embed(features)
                unbias_cvr_outputs,unbias_net_cvr_outputs = self.stream_unbias_model(features)

            self.main_optimizer.zero_grad()
            cvr_outputs,net_cvr_outputs = self.stream_main_model(features,feature_x_embed.detach(),unbias_x_embed.detach())
            loss = self.main_loss_fn(cvr_outputs,unbias_cvr_outputs,stream_pay_labels,stream_pay_mask)
            loss.backward()
            self.main_optimizer.step()
            main_total_loss += loss.item()
            main_total_batches += 1

            with torch.no_grad():
                fused_outputs = self.stream_main_model.predict(cvr_outputs,feature_cvr_outputs,unbias_cvr_outputs)
                cvr_auc = auc_score(pay_labels, fused_outputs)
                all_metrics["CVR_AUC"] += cvr_auc
                net_cvr_auc = auc_score(net_pay_labels,fused_outputs)
                all_metrics["NetCVR_AUC"] += net_cvr_auc

            tqdm_main_day_dataloader.set_description('Epoch {}, main_loss {:.3f} '.format(epoch_idx+1, loss.detach().item()))
        main_mean_loss = main_total_loss / main_total_batches
        feature_mean_loss = feature_total_loss / feature_total_batches
        unbias_mean_loss = unbias_total_loss / unbias_total_batches
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Main Mean Loss: {main_mean_loss:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Feature Mean Loss: {feature_mean_loss:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} Day {day_idx}- Unbias Mean Loss: {unbias_mean_loss:.5f}")
        all_metrics["CVR_AUC"] /= main_total_batches
        all_metrics["NetCVR_AUC"] /= main_total_batches
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: CVR_AUC: {all_metrics['CVR_AUC']:.5f}")
        self.logger.info(f"Epoch {epoch_idx+1} train day {day_idx}: NetCVR_AUC: {all_metrics['NetCVR_AUC']:.5f}")
        self.main_scheduler.step()
        self.feature_scheduler.step()
        self.unbias_scheduler.step()

        self.logger.info(f"Epoch {epoch_idx+1} - Learning Rate: {self.main_optimizer.param_groups[0]['lr']:.8f}")

        parent_dir = os.path.dirname(self.main_model_pth)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        torch.save(self.stream_main_model.state_dict(), self.main_model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.main_model_pth}")
        torch.save(self.stream_feature_model.state_dict(), self.feature_model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.feature_model_pth}")
        torch.save(self.stream_unbias_model.state_dict(), self.unbias_model_pth)
        self.logger.info(f"Model saved at Epoch {epoch_idx+1} , Day {day_idx} , Path : {self.unbias_model_pth}")
        return all_metrics


    def test(self,day_idx):
        self.logger.info('Testing best model with test set!')
        self.stream_main_model.eval()
        self.stream_feature_model.eval()
        self.stream_unbias_model.eval()
        self.pretrain_model.eval()
    

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
                
                feature_x_embed = self.stream_feature_model.get_x_embed(features)
                feature_outputs = self.stream_feature_model(features)
                unbias_x_embed = self.stream_unbias_model.get_x_embed(features)
                unbias_outputs = self.stream_unbias_model(features)
                main_outputs = self.stream_main_model(features,feature_x_embed,unbias_x_embed)


                feature_x_embed = self.stream_feature_model.get_x_embed(features)
                feature_cvr_outputs,feature_net_cvr_outputs = self.stream_feature_model(features)
                unbias_x_embed = self.stream_unbias_model.get_x_embed(features)
                unbias_cvr_outputs,unbias_net_cvr_outputs = self.stream_unbias_model(features)
                main_cvr_outputs,main_net_cvr_outputs = self.stream_main_model(features,feature_x_embed.detach(),unbias_x_embed.detach())
                cvr_fused_outputs = self.stream_main_model.predict(main_cvr_outputs,feature_cvr_outputs,unbias_cvr_outputs)


                pretrained_cvr_outputs,pretrained_net_cvr_outputs = self.pretrain_model.predict(features)

                all_pay_labels.extend(pay_labels.cpu().numpy())
                all_net_pay_labels.extend(net_pay_labels.cpu().numpy())
                stream_model_all_pay_preds.extend(cvr_fused_outputs.cpu().numpy())
                stream_model_all_net_pay_preds.extend(cvr_fused_outputs.cpu().numpy())
                pretrained_model_all_pay_preds.extend(pretrained_cvr_outputs.cpu().numpy())
                pretrained_model_all_net_pay_preds.extend(pretrained_net_cvr_outputs.cpu().numpy())
  
                with torch.no_grad():

                    stream_model_cvr_auc = auc_score(pay_labels, cvr_fused_outputs)
                    all_metrics["stream_model_CVR_AUC"] += stream_model_cvr_auc

                    stream_model_net_cvr_auc = auc_score(net_pay_labels,cvr_fused_outputs)
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





