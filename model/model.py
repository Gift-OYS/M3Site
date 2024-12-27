import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm

from utils.util_functions import cos_sim, kl_loss
from utils.util_classes import CenterLoss, InterClassLoss
from model.egnn.network import EGNN


class M3Site(torch.nn.Module):

    def __init__(self, config, hidden_size=256):
        super(M3Site, self).__init__()
        self.config = config
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
        stru_seq_seq = self.funicross1(egnn_data, esm_data, esm_data, func_data)
        seq_stru_stru = self.funicross2(esm_data, egnn_data, egnn_data, func_data)
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


    def _softlabel_loss_3d(self, seq_features, text_features, tau):
        seq_features = seq_features.mean(dim=1)
        text_features = text_features.mean(dim=1)
        seq_sim, text_sim = cos_sim(seq_features, seq_features), cos_sim(text_features, text_features)
        logits_per_seq, logits_per_text = self._get_similarity(seq_features, text_features)
        cross_modal_loss = (kl_loss(logits_per_seq, seq_sim, tau=tau) + 
                            kl_loss(logits_per_text, text_sim, tau=tau)) / 2.0
        return cross_modal_loss

    def _get_similarity(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        logits_per_image = image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text

    def _get_model_output(self, input_ids, attention_mask, modal):
        if modal == 'text':
            model = self.text_model
        else:
            model = self.seq_model
        return model(input_ids=input_ids, attention_mask=attention_mask)

    def _get_text_branch_output(self, text_outputs):
        texts_output = [self.texts_encoder[i](text_outputs[i][0]) for i in range(self.num_attr)]
        texts_output_cls = [texts_output[idx][:, 0, :].unsqueeze(1) for idx in range(len(texts_output)) if idx != 3]
        texts_output_cls = torch.cat(texts_output_cls, dim=1)
        texts_output_cls = self.text_suffix_transformer(texts_output_cls)
        text_func = texts_output[3]
        x = texts_output_cls
        for i in range(4):
            _x = x
            x = self.text_crosses[i](x, text_func, text_func)
            x = self.norms[i](x[0] + _x)
        text_branch_output = x
        return text_branch_output
    

class CrossAttention(nn.Module):
    def __init__(self, dim1, dim2, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.dim1 = dim1
        self.key = nn.Linear(dim2, dim1)
        self.value = nn.Linear(dim2, dim1)
        self.out = nn.Linear(dim1, dim1)

    def forward(self, Q, K, V):
        Q_proj = Q
        K_proj = self.key(K)
        V_proj = self.value(V)
        attention_scores = torch.matmul(Q_proj, K_proj.transpose(-2, -1))
        attention_scores = attention_scores / (self.dim1 ** 0.5)
        attention_probs = F.softmax(attention_scores, dim=-1)
        context = torch.matmul(attention_probs, V_proj)
        output = self.out(context)
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

    def forward(self, Q, K, V, condition=None):
        attn_output = self.attn(Q, K, V)
        Q = self.attn_layer_norm(Q + attn_output)
        ff_output = self.ff(Q, condition)
        Q = self.ff_layer_norm(Q + ff_output)
        return Q
