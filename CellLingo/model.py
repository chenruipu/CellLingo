# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 10:48:51 2023

@author: chenruipu
"""

import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, edge_names,node_name):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.embedding_gene= dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, in_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.embedding_gotem= dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, in_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.dense = dglnn.HeteroLinear({rel: out_feats for rel in node_name},out_feats)
    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h0_gene_cell= self.embedding_gene(graph,inputs)
        inputs['gene'],inputs['cell'] = F.relu(h0_gene_cell['gene']),F.relu(h0_gene_cell['cell'])
        inputs['gotem']= self.embedding_gotem(graph,inputs)['gotem']
        #inputs['gotem']= F.relu(inputs['gotem'])
        h = self.conv1(graph, inputs)
        h = self.conv2(graph, h)
        h = self.conv3(graph,h)
        h = self.dense(h)
        #h = {k: F.softmax(v,dim=1) for k, v in h.items()}
        return h



