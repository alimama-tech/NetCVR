import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets_esdfmRF import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FusedAttention

class PretrainEsdfmRF(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_features = len(num_bin_size)
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(num_bin_size) + list(cate_bin_size)

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        self.net_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        input_dim = (self.num_features + self.cate_features) * args.embed_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.net_fc1 = nn.Linear(input_dim, 256)
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

        self.extra_cvr_network = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )


        self.refund_adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Dropout(0.1)
        )



    def extra_cvr_forward(self,features):
        features = features.long()
        with torch.no_grad():
            embed_list = [emb(features[:, i]) for i, emb in enumerate(self.embeddings)]
            x_embed = torch.cat(embed_list, dim=-1)
            x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
            x = F.leaky_relu(self.ln2(self.fc2(x)))
            x = F.leaky_relu(self.ln3(self.fc3(x)))

            net_embed_list = [emb(features[:, i]) for i, emb in enumerate(self.net_embeddings)]
            net_x_embed = torch.cat(net_embed_list, dim=-1)
            net_x = F.leaky_relu(self.net_ln1(self.net_fc1(net_x_embed)))
            net_x = F.leaky_relu(self.net_ln2(self.net_fc2(net_x)))
            net_x = F.leaky_relu(self.net_ln3(self.net_fc3(net_x)))
        fused_features = torch.cat([x.detach(), net_x.detach()], dim=-1)
        extra_cvr_outputs = self.extra_cvr_network(fused_features)
        return extra_cvr_outputs.squeeze(1)


    def cvr_forward(self,x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))

        cvr_logits = self.fc4(x)
        return cvr_logits.squeeze(1)
    

    def net_cvr_forward(self,x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        x = F.leaky_relu(self.net_ln1(self.net_fc1(x_embed.detach())))
        x = F.leaky_relu(self.net_ln2(self.net_fc2(x)))
        x = F.leaky_relu(self.net_ln3(self.net_fc3(x)))
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



    def predict(self, x):
        with torch.no_grad():
            cvr_logits = self.cvr_forward(x)
            refund_logits = self.net_cvr_forward(x)
            cvr_prob = torch.sigmoid(cvr_logits)
            refund_prob = torch.sigmoid(refund_logits)
            net_cvr_prob = cvr_prob*(1-refund_prob)
            return cvr_prob, net_cvr_prob
        

    def extra_predict(self, x):
        with torch.no_grad():
            cvr_logits = self.extra_cvr_forward(x)
            cvr_prob = torch.sigmoid(cvr_logits)
            return cvr_prob,cvr_prob




