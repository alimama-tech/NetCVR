import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets import num_bin_size,cate_bin_size


class BiDefuse(nn.Module):
    def __init__(self,args):
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
        self.fc4 = nn.Linear(128, 2)

    def forward(self, x):
    
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))
        logit = self.fc4(x)
        return torch.sigmoid(logit)



    def predict(self, x):
        outs = self.forward(x)
        inw_pred = outs[:,0].view(-1)
        outw_pred = outs[:,1].view(-1)
        pred = inw_pred + outw_pred
        return pred



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/criteo/data.txt', help='Data path')
    parser.add_argument('--data_cache_path', type=str, default='/data1/mxluo/mxluo/CVR_model/AirBench/data/', help='Data path')
    parser.add_argument('--dataset_name', type=str, default='Criteo', help='Dataset name')
    parser.add_argument('--attr_window', type=int, default=30, help='Attribution window size (days)')
    parser.add_argument('--wait_window', type=int, default=1, help='wait window size (days)')
    parser.add_argument('--train_split_days_start', type=int, default=0, help='start day of train (days)')
    parser.add_argument('--train_split_days_end', type=int, default=15, help='end day of train (days)')
    parser.add_argument('--test_split_days_start', type=int, default=15, help='start day of test (days)')
    parser.add_argument('--test_split_days_end', type=int, default=30, help='end day of test (days)')
    parser.add_argument('--mode', type=str, default="defuse_train_stream", help='[defuse_pretrain,defuse_train_stream]')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--embed_dim', type=int, default=8, help='Embedding dimension')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')


    return parser.parse_args()


