import torch
import torch.nn as nn
import torch.nn.functional as F
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.dynamic_rnn import DynamicLSTM
from layers.gnn_layer import GraphAttentionLayer,GraphAttentionLayer_weight


class GraphNN(nn.Module):
    def __init__(self, opt):
        super(GraphNN, self).__init__()
        in_dim = opt.bert_dim
        self.gnn_dims = [in_dim] + [int(dim) for dim in opt.gnn_dims.strip().split(',')]

        self.gnn_layers = len(self.gnn_dims) - 1
        self.att_heads = [int(att_head) for att_head in opt.att_heads.strip().split(',')]
        self.gnn_layer_stack = nn.ModuleList()
        for i in range(self.gnn_layers):
            in_dim = self.gnn_dims[i] * self.att_heads[i - 1] if i != 0 else self.gnn_dims[i]
            self.gnn_layer_stack.append(
                GraphAttentionLayer_weight(opt.device, self.att_heads[i], in_dim, self.gnn_dims[i + 1], opt.dp)
            )

    def forward(self, node_feature, adj):
        # batch, max_doc_len, _ = doc_sents_h.size()
        # assert max(doc_len) == max_doc_len

        for i, gnn_layer in enumerate(self.gnn_layer_stack):
            node_feature,weight = gnn_layer(node_feature, adj)

        return node_feature,weight

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, text, adj):
        hidden = torch.matmul(text, self.weight)
        denom = torch.sum(adj, dim=2, keepdim=True) + 1
        output = torch.matmul(adj, hidden) / denom
        if self.bias is not None:
            return output + self.bias
        else:
            return output
class BERT_SCL_Proto_Graph(nn.Module):
    def __init__(self, opt, bert):
        super(BERT_SCL_Proto_Graph, self).__init__()
        self.bert = bert
        self.bert_dim = opt.bert_dim

        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim*2, opt.num_labels)
        self.gnn = GraphNN(opt)

    def forward(self, inputs):
        concat_bert_indices, concat_segments_indices, centroids = inputs
        batch_size = concat_bert_indices.shape[0]
        centroids = centroids[0]
        _, pooled_output = self.bert(concat_bert_indices, token_type_ids=concat_segments_indices)
        pooled_output = self.dropout(pooled_output)

        # 构建adj矩阵
        matrix = torch.zeros([batch_size, centroids.shape[0]+1, centroids.shape[0]+1]).cuda()
        matrix[:,-1:] = 1
        matrix[:,:,-1] = 1

        # 得到每个图的节点表示
        feature = torch.zeros([batch_size, centroids.shape[0]+1, self.bert_dim]).cuda()

        for i in range(batch_size):
            feature[i][:-1] = centroids
            feature[i][-1] = pooled_output[i]

        node_feature ,weight= self.gnn(feature, matrix)


        weight = weight[:,-1:]


        # state_node_contrastive = F.normalize(state_node_contrastive, dim=2)




        last_node_feature = torch.zeros([batch_size, self.bert_dim]).cuda()

        for i in range(batch_size):
            last_node_feature[i] = node_feature[i][-1]




        #
        # node_for_con = last_node_feature.unsqueeze(1)
        node_for_con = F.normalize(weight, dim=2)



        pooled_output = torch.cat([pooled_output,last_node_feature],dim=1)




        logits = self.dense(pooled_output)

        return logits,node_for_con

    def prototype_encode(self, inputs):

        concat_bert_indices, concat_segments_indices = inputs
        _, pooled_output = self.bert(concat_bert_indices, token_type_ids=concat_segments_indices)
        pooled_output = self.dropout(pooled_output)

        feature = pooled_output.unsqueeze(1)
        feature = F.normalize(feature, dim=2)

        return feature


