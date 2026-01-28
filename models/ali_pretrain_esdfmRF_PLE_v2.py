import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from datasets.ali_odl_datasets_esdfmRF import num_bin_size,cate_bin_size


import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers import FusedAttention

class PretrainEsdfmRF_PLE_v2(nn.Module):
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


        self.shared_embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=bucket_size, embedding_dim=args.embed_dim)
            for bucket_size in self.all_bin_sizes
        ])

        input_dim = (self.num_features + self.cate_features) * args.embed_dim

        self.fc1 = nn.Linear(input_dim*2, 256)
        self.net_fc1 = nn.Linear(input_dim*2, 256)
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


        self.linear_x_hidden = nn.Linear(128, 64)       
        self.linear_net_x_hidden = nn.Linear(128, 64)    
        self.gate_layer = nn.Linear(64, 64)

        self.correction_fc1 = nn.Linear(64, 32)
        self.correction_fc2 = nn.Linear(32, 1)

        
    def cvr_correction_forward(self, features):
        features = features.long()
        with torch.no_grad():
            embed_list = [emb(features[:, i]) for i, emb in enumerate(self.embeddings)]
            shared_embed_list = [emb(features[:, i]) for i, emb in enumerate(self.shared_embeddings)]
            net_embed_list = [emb(features[:, i]) for i, emb in enumerate(self.net_embeddings)]
            x_embed = torch.cat(embed_list, dim=-1)
            net_x_embed = torch.cat(net_embed_list, dim=-1)
            shared_x_embed = torch.cat(shared_embed_list, dim=-1)

            x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)
            net_x_embed = torch.cat([net_x_embed, shared_x_embed], dim=-1)

            x_hidden = F.leaky_relu(self.ln1(self.fc1(x_embed)))
            x_hidden = F.leaky_relu(self.ln2(self.fc2(x_hidden)))
            x_hidden = F.leaky_relu(self.ln3(self.fc3(x_hidden)))
            cvr_logits = self.fc4(x_hidden)

            net_x_hidden = F.leaky_relu(self.net_ln1(self.net_fc1(net_x_embed)))
            net_x_hidden = F.leaky_relu(self.net_ln2(self.net_fc2(net_x_hidden)))
            net_x_hidden = F.leaky_relu(self.net_ln3(self.net_fc3(net_x_hidden)))
            net_cvr_logits = self.fc4_net(net_x_hidden)
        


        x_proj = self.linear_x_hidden(x_hidden.detach())          
        net_proj = self.linear_net_x_hidden(net_x_hidden.detach()) 
        fused_features = torch.stack([x_proj, net_proj], dim=1)

        gate_weights = torch.sigmoid(self.gate_layer(fused_features))

        gated_fused = gate_weights * fused_features
        fused_embedding = torch.sum(gated_fused, dim=1)

        fused_hidden = F.tanh((self.correction_fc1(fused_embedding)))
        fused_correction_x_logits = self.correction_fc2(fused_hidden)
        return cvr_logits.squeeze(-1), net_cvr_logits.squeeze(-1), fused_correction_x_logits.squeeze(-1)

    def correction_predict(self, x):
        with torch.no_grad():
            cvr_logits,refund_logits,correction_logits = self.cvr_correction_forward(x)
            cvr_prob = torch.sigmoid(cvr_logits)
            refund_prob = torch.sigmoid(refund_logits)
            net_cvr_prob = cvr_prob*(1-refund_prob)

            corrected_cvr_logits = cvr_logits + correction_logits
            
            corrected_cvr_prob = torch.sigmoid(corrected_cvr_logits)
            corrected_net_cvr_prob = corrected_cvr_prob * (1-refund_prob)

            return corrected_cvr_prob, corrected_net_cvr_prob
        

    def cvr_forward(self,x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)

        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))

        cvr_logits = self.fc4(x)
        return cvr_logits.squeeze(1)
    

    def net_cvr_forward(self,x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)

        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)
        x = F.leaky_relu(self.net_ln1(self.net_fc1(x_embed)))
        x = F.leaky_relu(self.net_ln2(self.net_fc2(x)))
        x = F.leaky_relu(self.net_ln3(self.net_fc3(x)))
        net_cvr_logits = self.fc4_net(x)
        return net_cvr_logits.squeeze(1)
    

    def get_emb(self, x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)

        net_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.net_embeddings)]
        net_x_embed = torch.cat(net_embed_list, dim=-1)

        return x_embed, net_x_embed

    def get_cvr_hidden_state(self,x):
        x = x.long()
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        shared_embed_list = [emb(x[:, i]) for i, emb in enumerate(self.shared_embeddings)]
        x_embed = torch.cat(embed_list, dim=-1)
        shared_x_embed = torch.cat(shared_embed_list, dim=-1)

        x_embed = torch.cat([x_embed, shared_x_embed], dim=-1)

        x = F.leaky_relu(self.ln1(self.fc1(x_embed)))
        x = F.leaky_relu(self.ln2(self.fc2(x)))
        x = F.leaky_relu(self.ln3(self.fc3(x)))

        return x



    def predict(self, x):
        with torch.no_grad():
            cvr_logits = self.cvr_forward(x)
            refund_logits = self.net_cvr_forward(x)
            cvr_prob = torch.sigmoid(cvr_logits)
            refund_prob = torch.sigmoid(refund_logits)
            net_cvr_prob = cvr_prob*(1-refund_prob)
            return cvr_prob, net_cvr_prob
        






