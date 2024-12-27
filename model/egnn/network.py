# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from model.egnn import EGNN_Sparse
from model.egnn.utils import get_edge_feature_dims, get_node_feature_dims


class nodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(nodeEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()
        self.node_feature_dim = get_node_feature_dims()
        for i, dim in enumerate(self.node_feature_dim):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.node_feature_dim)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.node_feature_dim[i]])
            feature_dim_count += self.node_feature_dim[i]
        return x_embedding


class edgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super(edgeEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        self.edge_feature_dims = get_edge_feature_dims()
        for i, dim in enumerate(self.edge_feature_dims):
            emb = torch.nn.Linear(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        feature_dim_count = 0
        for i in range(len(self.edge_feature_dims)):
            x_embedding += self.atom_embedding_list[i](
                x[:, feature_dim_count:feature_dim_count + self.edge_feature_dims[i]])
            feature_dim_count += self.edge_feature_dims[i]
        return x_embedding


class GNNClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features, batch):
        features = features.reshape(max(batch)+1, -1, features.shape[-1])
        x = torch.mean(features, dim=1)  # average pool over the tokens
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
    

class ActiveSiteHead(nn.Module):
    
    def __init__(self, input_dim, output_dim, dropout):
        super(ActiveSiteHead, self).__init__()
        dims = [4**i for i in range(1, 7)]
        lin_dims = [output_dim] + [x for x in dims if output_dim < x < input_dim][1:-1] + [input_dim]
        layers = []
        for in_dim in lin_dims[::-1][:-1]:
            layers.append(nn.Linear(in_dim, lin_dims[lin_dims.index(in_dim) - 1]))
            layers.append(nn.Dropout(dropout))
            layers.append(nn.SiLU())
        layers.pop(); layers.pop()
        self.dense = nn.Sequential(*layers)
        
    def forward(self, x):
        x = self.dense(x)
        return x


class EGNN(nn.Module):
    def __init__(self, config):
        super(EGNN, self).__init__()
        self.config = config
        self.gnn_config = config.egnn
        self.input_dim = config.dataset.property_dim
        self.mpnn_layes = nn.ModuleList([
            EGNN_Sparse(
                self.input_dim, 
                m_dim=int(self.gnn_config["hidden_channels"]), 
                edge_attr_dim=int(self.gnn_config["edge_attr_dim"]), 
                dropout=int(self.gnn_config["dropout"]), 
                mlp_num=int(self.gnn_config["mlp_num"]))
            for _ in range(int(self.gnn_config["n_layers"]))])

        if self.gnn_config["embedding"]:
            self.node_embedding = nodeEncoder(self.input_dim)
            self.edge_embedding = edgeEncoder(self.input_dim)

        self.pred_head = ActiveSiteHead(self.input_dim, self.gnn_config['output_dim'], self.gnn_config['dropout'])
        # self.lin = nn.Linear(input_dim, self.gnn_config['output_dim'])
        # self.droplayer = nn.Dropout(int(self.gnn_config["dropout"]))


    def forward(self, data):
        x, pos, edge_index, edge_attr, batch, esm_rep, prop = (
            data.x, data.pos, 
            data.edge_index,
            data.edge_attr, data.batch,
            data.esm_rep, data.prop
        )

        # 把prop中的第35列和第56列（表示氨基酸类型的one-hot向量）去掉
        if self.config.dataset.property_dim == 41:
            prop = torch.cat([prop[:,:35], prop[:,56:]], dim=1)
        input_x = torch.cat([pos, prop], dim=1)
        # input_x = torch.cat([pos, esm_rep, prop], dim=1)
        # input_x = torch.cat([pos, input_x], dim=1)

        if self.gnn_config['embedding']:
            input_x = self.node_embedding(input_x)
            edge_attr = self.edge_embedding(edge_attr)

        for i, layer in enumerate(self.mpnn_layes):
            h = layer(input_x, edge_index, edge_attr, batch)
            if self.gnn_config['residual']:
                input_x = input_x + h
            else:
                input_x = h

        x = input_x[:, 3:]
        x = self.pred_head(x)
        # x = self.droplayer(x)
        # x = self.lin(x)
        # return x, input_x[:, 3:]
        return x
