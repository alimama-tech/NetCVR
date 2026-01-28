import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets_esdfmRF import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FusedAttention
import math
class PretrainEsdfmRF_PLE_DealyTimeAware(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_features = len(num_bin_size)
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(num_bin_size) + list(cate_bin_size)

        self.user_idx = 1
        self.item_idx = 5

        self.user_embeddings_for_cvr = nn.Embedding(num_embeddings=cate_bin_size[self.user_idx], embedding_dim=args.embed_dim)
        self.item_embeddings_for_cvr = nn.Embedding(num_embeddings=cate_bin_size[self.item_idx], embedding_dim=args.embed_dim)

        self.user_embeddings_for_rfr = nn.Embedding(num_embeddings=cate_bin_size[self.user_idx], embedding_dim=args.embed_dim)
        self.item_embeddings_for_rfr = nn.Embedding(num_embeddings=cate_bin_size[self.item_idx], embedding_dim=args.embed_dim)

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        self.net_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        self.shared_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        input_dim = (self.num_features + self.cate_features) * args.embed_dim

        self.fc1 = nn.Linear(input_dim*2  , 256)
        self.net_fc1 = nn.Linear(input_dim*2 , 256)
        self.ln1 = nn.BatchNorm1d(256)
        self.net_ln1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.net_fc2 = nn.Linear(256, 256)
        self.ln2 = nn.BatchNorm1d(256)
        self.net_ln2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.net_fc3 = nn.Linear(256, 128)
        self.ln3 = nn.BatchNorm1d(128)
        self.net_ln3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 1)
        self.fc4_net = nn.Linear(128, 1)


        self.cvr_time_net = nn.Sequential(
            nn.Linear(2, 8),                  
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(8, 1)             
        )

        self.rfr_time_net = nn.Sequential(
            nn.Linear(2, 8),                  
            nn.LeakyReLU(negative_slope=0.1),
            nn.Linear(8, 1)             
        )

        self.delay_time_aware_for_cvr_fc1 = nn.Linear(args.embed_dim*2, 128)
        self.delay_time_aware_for_cvr_ln1 = nn.BatchNorm1d(128)
        self.delay_time_aware_for_cvr_fc2 = nn.Linear(128, 128)
        self.delay_time_aware_for_cvr_ln2 = nn.BatchNorm1d(128)
        self.delay_time_aware_for_cvr_fc3 = nn.Linear(128, 1)

        self.delay_time_aware_for_rfr_fc1 = nn.Linear(args.embed_dim*2, 128)
        self.delay_time_aware_for_rfr_ln1 = nn.BatchNorm1d(128)
        self.delay_time_aware_for_rfr_fc2 = nn.Linear(128, 128)
        self.delay_time_aware_for_rfr_ln2 = nn.BatchNorm1d(128)
        self.delay_time_aware_for_rfr_fc3 = nn.Linear(128, 1)




    def get_cvr_hidden_state(self, x , click_hour):
        x = x.long()
        user_features = x[:, self.user_idx]
        item_features = x[:, self.item_idx]
        user_embed_for_delay_time = self.user_embeddings_for_cvr(user_features)
        item_embed_for_delay_time = self.item_embeddings_for_cvr(item_features)
        x_embed_for_delay_time = torch.cat([user_embed_for_delay_time, item_embed_for_delay_time], dim=-1)
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln1(self.delay_time_aware_for_cvr_fc1(x_embed_for_delay_time)))
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln2(self.delay_time_aware_for_cvr_fc2(x_for_delay_time)))
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)


        hours = torch.arange(0,24).float().to(click_hour.device).unsqueeze(1)
        hours_emb = self.encode_hour_to_sin_cos(hours)

        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)
        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        x = x + x_for_delay_time
        return x

    def cvr_delay_time_aware(self, x):
        x = x.long()
        user_features = x[:, self.user_idx]
        item_features = x[:, self.item_idx]
        user_embed_for_delay_time = self.user_embeddings_for_cvr(user_features)
        item_embed_for_delay_time = self.item_embeddings_for_cvr(item_features)
        x_embed_for_delay_time = torch.cat([user_embed_for_delay_time, item_embed_for_delay_time], dim=-1)
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln1(self.delay_time_aware_for_cvr_fc1(x_embed_for_delay_time)))
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln2(self.delay_time_aware_for_cvr_fc2(x_for_delay_time)))
        x_for_delay_time = self.delay_time_aware_for_cvr_fc3(x_for_delay_time)
        return x_for_delay_time.squeeze(1)
    
    def rfr_delay_time_aware(self, x):
        x = x.long()
        user_features = x[:, self.user_idx]
        item_features = x[:, self.item_idx]
        user_embed_for_delay_time = self.user_embeddings_for_rfr(user_features)
        item_embed_for_delay_time = self.item_embeddings_for_rfr(item_features)
        x_embed_for_delay_time = torch.cat([user_embed_for_delay_time, item_embed_for_delay_time], dim=-1)
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_rfr_ln1(self.delay_time_aware_for_rfr_fc1(x_embed_for_delay_time)))
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_rfr_ln2(self.delay_time_aware_for_rfr_fc2(x_for_delay_time)))
        x_for_delay_time = self.delay_time_aware_for_rfr_fc3(x_for_delay_time)
        return x_for_delay_time.squeeze(1)
    

    def cvr_forward(self,x,click_hour):
        x = x.long()
        user_features = x[:, self.user_idx]
        item_features = x[:, self.item_idx]
        user_embed_for_delay_time = self.user_embeddings_for_cvr(user_features)
        item_embed_for_delay_time = self.item_embeddings_for_cvr(item_features)
        x_embed_for_delay_time = torch.cat([user_embed_for_delay_time, item_embed_for_delay_time], dim=-1)
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln1(self.delay_time_aware_for_cvr_fc1(x_embed_for_delay_time)))
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_cvr_ln2(self.delay_time_aware_for_cvr_fc2(x_for_delay_time)))
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)


        hours = torch.arange(0,24).float().to(click_hour.device).unsqueeze(1)
        hours_emb = self.encode_hour_to_sin_cos(hours)

        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)
        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        x = x + x_for_delay_time.detach()
        cvr_logits = self.fc4(x)

        return cvr_logits.squeeze(1)
    

    def net_cvr_forward(self,x,click_hour):
        x = x.long()
        user_features = x[:, self.user_idx]
        item_features = x[:, self.item_idx]
        user_embed_for_delay_time = self.user_embeddings_for_rfr(user_features)
        item_embed_for_delay_time = self.item_embeddings_for_rfr(item_features)
        x_embed_for_delay_time = torch.cat([user_embed_for_delay_time, item_embed_for_delay_time], dim=-1)
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_rfr_ln1(self.delay_time_aware_for_rfr_fc1(x_embed_for_delay_time)))
        x_for_delay_time = F.leaky_relu(self.delay_time_aware_for_rfr_ln2(self.delay_time_aware_for_rfr_fc2(x_for_delay_time)))
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)




        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)
        x = F.leaky_relu(self.net_ln1(self.net_fc1(x_embed)))
        x = F.leaky_relu(self.net_ln2(self.net_fc2(x)))
        x = F.leaky_relu(self.net_ln3(self.net_fc3(x)))
        x = x + x_for_delay_time.detach()
        net_cvr_logits = self.fc4_net(x)


        return net_cvr_logits.squeeze(1)


    def cvr_pos_agument_features(self, x, noise_std=0.05, dropout_rate=0.5):

        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        device = x_embed.device
        batch_size, dim = x_embed.shape

        noise = torch.randn_like(x_embed) * noise_std
        x_emb_aug1 = x_embed
        x1 = F.leaky_relu(self.ln1(self.fc1(x_emb_aug1)))
        x1 = F.leaky_relu(self.ln2(self.fc2(x1)))
        x1 = F.leaky_relu(self.ln3(self.fc3(x1)))
        cvr_logits1 = self.fc4(x1)



        refund_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        refund_x_embed = torch.cat(refund_embed_list, dim=-1)
        refund_x_embed = refund_x_embed.detach()
        refund_x_embed = self.refund_adapter(refund_x_embed)
        x_emb_aug2 = x_embed + refund_x_embed
        x2 = F.leaky_relu(self.ln1(self.fc1(x_emb_aug2)))
        x2 = F.leaky_relu(self.ln2(self.fc2(x2)))
        x2 = F.leaky_relu(self.ln3(self.fc3(x2)))
        cvr_logits2 = self.fc4(x2)



        return cvr_logits1.squeeze(1), cvr_logits2.squeeze(1)



    def predict(self, x , click_hour):
        with torch.no_grad():
            cvr_logits = self.cvr_forward(x, click_hour)
            refund_logits = self.net_cvr_forward(x , click_hour)
            cvr_prob = torch.sigmoid(cvr_logits)
            refund_prob = torch.sigmoid(refund_logits)
            net_cvr_prob = cvr_prob*(1-refund_prob)
            return cvr_prob, net_cvr_prob
        


    def encode_hour_to_sin_cos(self,hour, invalid_value=-1):
        """
        将 hour (0-23 or -1) 转换为 sin/cos 编码
        对于无效值（如 -1），返回 (0, 0) 向量，表示“无信息”
        
        Returns:
            sin_cos: shape [N, 2], each row is [sin(x), cos(x)]
        """
        is_valid = (hour != invalid_value)
        hour_safe = torch.where(is_valid, hour, torch.tensor(0, device=hour.device))
        radians = hour_safe * (2 * math.pi / 24)
        sin_enc = torch.sin(radians)
        cos_enc = torch.cos(radians)
        sin_enc = torch.where(is_valid, sin_enc, torch.tensor(0.0, device=sin_enc.device))
        cos_enc = torch.where(is_valid, cos_enc, torch.tensor(0.0, device=cos_enc.device))
        sin_cos = torch.stack([sin_enc, cos_enc], dim=-1)
        return sin_cos





