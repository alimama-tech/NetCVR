import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FusedAttention

class PretrainBase(nn.Module):
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
        net_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        net_x_embed = torch.cat(net_embed_list, dim=-1)
        net_h1 = F.leaky_relu(self.net_ln1(self.net_fc1(net_x_embed)))
        net_h2 = F.leaky_relu(self.net_ln2(self.net_fc2(net_h1)))
        net_h3 = F.leaky_relu(self.net_ln3(self.net_fc3(net_h2)))
        net_cvr_logits = self.fc4_net(net_h3)
        return net_cvr_logits.squeeze(1)



    def predict(self, x):
        with torch.no_grad():
            cvr_prob = torch.sigmoid(self.cvr_forward(x))
            net_cvr_prob = torch.sigmoid(self.net_cvr_forward(x))
            return cvr_prob, net_cvr_prob


