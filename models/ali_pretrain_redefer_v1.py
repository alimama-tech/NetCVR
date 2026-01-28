import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F

class PretrainReDefer(nn.Module):
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
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.head_main = nn.Linear(128, 1)
        self.head_aux = nn.Linear(128, 1)

    def extract_features(self, x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        return self.shared_layers(x_embed)

    def forward(self, x):
        feats = self.extract_features(x)
        cvr_logits = self.head_main(feats)
        net_cvr_logits = self.head_aux(feats)
        return torch.sigmoid(cvr_logits).squeeze(1), torch.sigmoid(net_cvr_logits).squeeze(1)



    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


