import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm
from utils.util_classes import CenterLoss, InterClassLoss
from model.egnn.network import EGNN


class AP_align_fuse_graph(torch.nn.Module):

    def __init__(self, config, hidden_size=256):
        super(AP_align_fuse_graph, self).__init__()
        self.config = config
        self.seq_max_length = config.dataset.seq_max_length
        if '3' in config.dataset.lm:
            self.embedding_dim = 1536
        elif 't5' in config.dataset.lm:
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
        self.inter_loss = InterClassLoss(margin=0.1)
        self.ab_egnn = nn.Linear(self.egnn_out_dim, self.embedding_dim+self.egnn_out_dim)
        self.ab_esm = nn.Linear(self.embedding_dim, self.embedding_dim+self.egnn_out_dim)

    def forward(self, data):
        esm_rep, batch, func = data.esm_rep, data.batch, data.func
        graphs = 1
        egnn_output = self.egnn_model(data) # [nodes, 16]

        # esm_data = torch.zeros(graphs, 1024, 1280).to(esm_rep.device)   # [1, 1024, 1280]
        # egnn_data = torch.zeros(graphs, 1024, self.egnn_out_dim).to(egnn_output.device) # [1, 1024, 16]
        func_data = func.reshape(graphs, 768)   # [1, 768]
        # for graph_idx in range(graphs):
        #     mask = (batch == graph_idx)
        #     esm_data[graph_idx][:esm_rep[mask].shape[0]] = esm_rep[mask]
        #     egnn_data[graph_idx][:egnn_output[mask].shape[0]] = egnn_output[mask]
        esm_data = F.pad(esm_rep, (0, 0, 0, 1024-esm_rep.shape[0]), value=0).unsqueeze(0)
        egnn_data = F.pad(egnn_output, (0, 0, 0, 1024-egnn_output.shape[0]), value=0).unsqueeze(0)

        total = torch.cat([esm_data, egnn_data], dim=-1)   # [graphs, 1024, 1280+16]
        stru_seq_seq = self.funicross1(egnn_data, esm_data, esm_data, func_data)   # [graphs, 1024, 16]
        seq_stru_stru = self.funicross2(esm_data, egnn_data, egnn_data, func_data)  # [graphs, 1024, 1280]
        fusion_out = torch.cat([stru_seq_seq, seq_stru_stru], dim=-1) # [graphs, 1024, 1280+16]

        combined = torch.cat([fusion_out, total], dim=-1)
        weight = torch.sigmoid(self.weight_fc(combined))
        out = weight * fusion_out + (1 - weight) * total

        out = self.fc1(out).permute(0, 2, 1)
        out = self.bn1(out).permute(0, 2, 1)
        out = torch.relu(out)
        out = self.fc2(out)
        
        recon_out = out[0][:esm_rep.shape[0]]
        recon_out = torch.softmax(recon_out, dim=-1)

        return recon_out


class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim1 = dim1
        self.key = nn.Linear(dim2, dim1)
        self.value = nn.Linear(dim2, dim1)
        self.out = nn.Linear(dim1, dim1)

    def forward(self, Q, K, V):
        Q_proj = Q
        K_proj = self.key(K)       # [len, dim1]
        V_proj = self.value(V)     # [len, dim1]
        attention_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1))  # [len, len] # 由于是分块矩阵，所以可以直接相乘
        attention_scores = attention_scores / (self.dim1 ** 0.5)  # Scale by the square root of dim1
        attention_probs = F.softmax(attention_scores, dim=-1)  # Softmax over the last dimension (keys)
        context = torch.matmul(attention_probs, V_proj)  # [len, dim1]
        output = self.out(context)  # [len, dim1]
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
            condition = condition.unsqueeze(1).expand(-1, x.size(1), -1)  # [len, condition_dim]
            x = torch.cat([x, condition], dim=-1)  # [len, dim + condition_dim]
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

    def forward(self, Q, K, V, condition=None):
        attn_output = self.attn(Q, K, V)
        Q = self.attn_layer_norm(Q + attn_output)
        ff_output = self.ff(Q, condition)   # 把condition加到了feedforward的输入中
        Q = self.ff_layer_norm(Q + ff_output)
        return Q
