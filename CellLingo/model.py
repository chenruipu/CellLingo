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
            for rel in edge_names}, aggregate='sum') #embedding层，第一次图卷积将细胞节点信息通过图结构传递给基因节点
        self.embedding_gotem= dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, in_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum') #embedding层，第一次图卷积将基因节点信息通过图结构传递给GO节点
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats,activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.dense = dglnn.HeteroLinear({rel: out_feats for rel in node_name},out_feats) #全连接层改变细胞节点维度，便于后接残差神经网络
    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h0_gene_cell= self.embedding_gene(graph,inputs)
        inputs['gene'],inputs['cell'] = F.relu(h0_gene_cell['gene']),F.relu(h0_gene_cell['cell'])
        inputs['gotem']= self.embedding_gotem(graph,inputs)['gotem']
        inputs['gotem']= F.relu(inputs['gotem']) #初始化图结构上的每个节点
        h = self.conv1(graph, inputs)
        h = self.conv2(graph, h)
        h = self.conv3(graph, h)
        h = self.dense(h)
        return h['cell'] #取图中细胞节点作为残差神经网络的输入

#残差模块，三个全连接层和一个dropout层组成
class Baisblock(nn.Module):
    def __init__(self, hid_feats):
        super().__init__()
        self.dense1 = nn.Linear(hid_feats, hid_feats)
        self.dense2 = nn.Linear(hid_feats, hid_feats)
        self.dense3 = nn.Linear(hid_feats, hid_feats)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        h = self.dense1(inputs)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.dense3(h)
        # h = self.relu(h)
        h = h+inputs
        h = self.relu(h)
        return h

#参差神经网络由三个残差模块组成，每两个残差模块之间接一个全理解层
class ResNet (nn.Module):
    def __init__(self, hid_feats,out_feats,num_layer):
        super().__init__()
        self.dense1 = nn.Linear(hid_feats,hid_feats)
        self.Res_layer1 = self.build_resblock(hid_feats,num_layer)
        self.dense2 = nn.Linear(hid_feats, hid_feats//2)
        self.Res_layer2 = self.build_resblock(hid_feats//2,num_layer)
        self.dense3 = nn.Linear(hid_feats//2, hid_feats//4)
        self.Res_layer3 = self.build_resblock(hid_feats//4,num_layer)
        self.out = nn.Linear(hid_feats//4,out_feats)
        self.relu = nn.ReLU()
    def forward(self, inputs):
        h = self.dense1(inputs)
        h = self.relu(h)
        h = self.Res_layer1(h)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.relu(h)
        h = self.Res_layer2(h)
        h = self.relu(h)
        h = self.dense3(h)
        h = self.relu(h)
        h = self.Res_layer3(h)
        h = self.relu(h)
        h = self.out(h)
        return h

    def build_resblock(self,hid_feats, num_layer):
        res_block = nn.Sequential()
        res_block.add_module("dense", nn.Linear(hid_feats,hid_feats))
        res_block.add_module('relu',nn.ReLU())
        for i in range(num_layer):
            res_block.add_module("Res",Baisblock(hid_feats))
        return res_block

# 总体的神经网络由图卷积神经网络和残差神经网络组合而成
class RGCN_ResNet(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layer,edge_names, node_name):
        super().__init__()
        self.RGCN = RGCN(in_feats, hid_feats, hid_feats, edge_names,node_name)
        self.ResNet = ResNet(hid_feats,out_feats,num_layer)
    def forward(self,graph, inputs):
        h = self.RGCN(graph, inputs)
        h = self.ResNet(h)
        return h

