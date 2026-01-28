import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainDpDrReDefer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.num_features = len(num_bin_size)
        self.cate_features = len(cate_bin_size)
        self.all_bin_sizes = list(num_bin_size) + list(cate_bin_size)

        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        input_dim = (self.num_features + self.cate_features) * args.embed_dim


        self.fc1 = nn.Linear(input_dim, 256)
        self.ln1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.ln3 = nn.BatchNorm1d(128)

        self.head_main = nn.Linear(128, 1)
        self.head_aux = nn.Linear(128, 1)

    def forward(self, x):

        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        dp_logits = self.head_main(x)
        dr_logits = self.head_aux(x)
        return dp_logits.squeeze(1), dr_logits.squeeze(1)



    def predict(self, x):
        with torch.no_grad():
            dp_logits, dr_logits = self.forward(x)
            dp_probs = torch.sigmoid(dp_logits)
            dr_probs = torch.sigmoid(dr_logits)
            return dp_probs, dr_probs


