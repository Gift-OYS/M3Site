import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

from utils.util_classes import CenterLoss, InterLoss
from model.egnn.network import EGNN


class M3Site(torch.nn.Module):

    def __init__(self, config, hidden_size=256):
        super(M3Site, self).__init__()
        self.config = config
        if '3' in config.dataset.tag:
            self.embedding_dim = 1536
        elif 't5' in config.dataset.tag:
            self.embedding_dim = 1024
        else:
            self.embedding_dim = 1280

        self.egnn_model = EGNN(config)
        self.egnn_out_dim = self.config.egnn.output_dim
        self.num_classes = 7
        self.fc1 = nn.Linear(self.embedding_dim+self.egnn_out_dim, hidden_size)
        self.bn1 = BatchNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, self.num_classes)
        self.funicross1 = FunICross(self.egnn_out_dim, self.embedding_dim, condition_dim=768)
        self.funicross2 = FunICross(self.embedding_dim, self.egnn_out_dim, condition_dim=768)
        self.weight_fc = nn.Linear((self.embedding_dim+self.egnn_out_dim) * 2, 1)
        self.center_loss = CenterLoss(num_classes=self.num_classes, feat_dim=7)
        self.inter_loss = InterLoss(margin=0.1)
        self.ab_egnn = nn.Linear(self.egnn_out_dim, self.embedding_dim+self.egnn_out_dim)
        self.ab_esm = nn.Linear(self.embedding_dim, self.embedding_dim+self.egnn_out_dim)

    def forward(self, data, n_select=5):
        esm_rep, batch, ptr, y, func = data.esm_rep, data.batch, data.ptr, data.y, data.func
        graphs = len(ptr) - 1
        egnn_output = self.egnn_model(data)

        esm_data = torch.zeros(graphs, 1024, self.embedding_dim).to(esm_rep.device)
        egnn_data = torch.zeros(graphs, 1024, self.egnn_out_dim).to(egnn_output.device)
        func_data = func.reshape(graphs, 768)
        for graph_idx in range(graphs):
            mask = (batch == graph_idx)
            esm_data[graph_idx][:esm_rep[mask].shape[0]] = esm_rep[mask]
            egnn_data[graph_idx][:egnn_output[mask].shape[0]] = egnn_output[mask]

        total = torch.cat([esm_data, egnn_data], dim=-1)
        stru_seq_seq, attn_weights_1 = self.funicross1(egnn_data, esm_data, esm_data, func_data, return_attn=True)
        seq_stru_stru, attn_weights_2 = self.funicross2(esm_data, egnn_data, egnn_data, func_data, return_attn=True)
        fusion_out = torch.cat([stru_seq_seq, seq_stru_stru], dim=-1)

        combined = torch.cat([fusion_out, total], dim=-1)
        weight = torch.sigmoid(self.weight_fc(combined))
        out = weight * fusion_out + (1 - weight) * total
        out = self.fc1(out).permute(0, 2, 1)
        out = self.bn1(out).permute(0, 2, 1)
        out = torch.relu(out)
        out = self.fc2(out)

        recon_out = torch.tensor([]).to(total.device)
        for graph_idx in range(graphs):
            mask = (batch == graph_idx)
            rep = out[graph_idx][:sum(mask)]
            recon_out = torch.cat((recon_out, rep), dim=0)
        assert recon_out.size(0) == esm_rep.size(0)

        center_loss = self.center_loss(recon_out, y)
        inter_loss = self.inter_loss(self.center_loss.centers)

        recon_out = torch.softmax(recon_out, dim=-1)

        return {
            'token_logits': recon_out,
            'center_loss': center_loss,
            'inter_loss': inter_loss,
        }


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim1 = dim1
        self.key = nn.Linear(dim2, dim1)
        self.value = nn.Linear(dim2, dim1)
        self.out = nn.Linear(dim1, dim1)

    def forward(self, Q, K, V, return_attn=False):
        Q_proj = Q
        K_proj = self.key(K)
        V_proj = self.value(V)
        attention_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1))
        attention_scores = attention_scores / (self.dim1 ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, V_proj)
        output = self.out(context)
        if return_attn:
            return output, attention_probs
        else:
            return output


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim=128, dropout=0.1, condition_dim=None):
        super(FeedForward, self).__init__()
        input_dim = dim + condition_dim if condition_dim is not None else dim
        self.fc1 = nn.Linear(input_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, condition=None):
        if condition is not None:
            condition = condition.unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, condition], dim=-1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class FunICross(nn.Module):
    def __init__(self, dim1, dim2, ff_dim=128, dropout=0.1, condition_dim=None):
        super(FunICross, self).__init__()
        self.attn = CrossAttention(dim1, dim2, dropout)
        self.attn_layer_norm = nn.LayerNorm(dim1)
        self.ff = FeedForward(dim1, ff_dim, dropout, condition_dim)
        self.ff_layer_norm = nn.LayerNorm(dim1)

    def forward(self, Q, K, V, condition=None, return_attn=False):
        if return_attn:
            attn_output, attn_weights = self.attn(Q, K, V, return_attn=True)
        else:
            attn_output = self.attn(Q, K, V)

        Q = self.attn_layer_norm(Q + attn_output)
        ff_output = self.ff(Q, condition)
        Q = self.ff_layer_norm(Q + ff_output)

        if return_attn:
            return Q, attn_weights
        else:
            return Q
