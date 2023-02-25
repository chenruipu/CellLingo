# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:03:34 2023

@author: chenruipu
"""

from .model import RGCN
import dgl 
import torch
def annotation (hetero_graph,epoch,hiddn_units):
    input_features = hetero_graph.ndata['feature']['cell'].size()[1]
    n_class = torch.unique(hetero_graph.ndata['label']['cell'])
    model = RGCN(input_features, hiddn_units, n_class, hetero_graph.etypes,hetero_graph.ntypes)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model=model.to(device)
    hetero_graph= hetero_graph.to(device)
    # hetero_graph.ndata['feature']=hetero_graph.ndata['feature'].to(device)
    h_dict = model(hetero_graph, hetero_graph.ndata['feature'])
    labels = hetero_graph.nodes['cell'].data['label']
    opt = torch.optim.Adam(model.parameters())
    reference_index = torch.nonzero(torch.unique(hetero_graph.ndata['train_mask']['cell'],return_counts=True)[0]==True).squeeze()
    n_reference_cell = torch.unique(hetero_graph.ndata['train_mask']['cell'],return_counts=True)[1][reference_index]#计算参考数据集的细胞数，为返回损

    #%%
    for epoch in range(epoch):
        model.train()
        # 使用所有节点的特征进行前向传播计算，并提取输出的user节点嵌入
        logits = model(hetero_graph, hetero_graph.ndata['feature'])['cell']
        # 计算损失值
        loss = F.cross_entropy(logits[hetero_graph.nodes['cell'].data['train_mask']], labels[hetero_graph.nodes['cell'].data['train_mask']])
        # 计算验证集的准确度。在本例中省略。
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        predict_result=model(hetero_graph, hetero_graph.ndata['feature'])['cell']
        scores, idx = torch.max(predict_result, dim = 1) # [bsz], [bsz]
        correct = torch.eq(idx[n_reference_cell:],labels[n_reference_cell:]).float().mean()
        print('epoch :',epoch,' loss :',loss.item(),' | correct :',correct)

    return predict_result