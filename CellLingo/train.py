# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:03:34 2023

@author: chenruipu
"""
import os

from .model import RGCN_ResNet, RGCN
import dgl
import torch
import torch.nn.functional as F


def build_model(input_features_dim: int = None,
                output_dim: int = None,
                hiddn_units: int = 256,
                num_res_layer: int = 15,
                homolog:bool =False
                ):
    if homolog:
        model = RGCN_ResNet(input_features_dim,
                            hiddn_units,
                            output_dim,
                            num_res_layer,
                            edge_names=['cell_loop', 'express', 'similar_to', 'express_by', 'gene_loop', 'homolog', 'in',
                                        'gotem_loop', 'include'],
                            node_name=['cell', 'gene', 'gotem'])
    else:
        model = RGCN_ResNet(input_features_dim,
                            hiddn_units,
                            output_dim,
                            num_res_layer,
                            edge_names=['cell_loop', 'express', 'similar_to', 'express_by', 'gene_loop', 'in',
                                        'gotem_loop', 'include'],
                            node_name=['cell', 'gene', 'gotem'])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    return model, opt


def train(hetero_graph: dgl.DGLHeteroGraph,
          model: RGCN_ResNet = None,
          opt: torch.optim = None,
          epoch: int = 100
          ):

    if model is None:
        return 'model should not be empty'
    if opt is None:
        return 'opt should not be empty'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hetero_graph = hetero_graph.to(device)
    model = model.to(device)
    labels = hetero_graph.nodes['cell'].data['label']
    for i in range(epoch):
        model.train()
        # 使用所有节点的特征进行前向传播计算，并提取输出的user节点嵌入
        logits = model(hetero_graph, hetero_graph.ndata['feature'])
        # 计算损失值
        loss = F.cross_entropy(logits[hetero_graph.nodes['cell'].data['train_mask']],
                               labels[hetero_graph.nodes['cell'].data['train_mask']])
        # 进行反向传播计算
        opt.zero_grad()
        loss.backward()
        opt.step()
        predict_result = model(hetero_graph, hetero_graph.ndata['feature'])
        scores, idx = torch.max(predict_result, dim=1)  # [bsz], [bsz]
        reference_index = torch.nonzero(
            torch.unique(hetero_graph.ndata['train_mask']['cell'], return_counts=True)[0] == True).squeeze()
        n_reference_cell = torch.unique(hetero_graph.ndata['train_mask']['cell'], return_counts=True)[1][
            reference_index]  # 计算参考数据集的细胞数，为返回损
        correct = torch.eq(idx[n_reference_cell:], labels[n_reference_cell:]).float().mean()
        print('epoch :', i, ' loss :', loss.item(), ' | correct :', correct)


def whole_graph_train(path: str = None,
                      model: RGCN_ResNet = None,
                      opt: torch.optim = None,
                      epoch: int = 100
                      ):
    if path is None:
        return 'path should not be empty'
    if model is None:
        return 'model should not be empty'
    if opt is None:
        return 'opt should not be empty'
    hetero_graph = dgl.load_graphs(path + '/_temp/whole_hetero_graph.bin')[0][0]
    for i in range(epoch):
        print('epoch :', i)
        train(hetero_graph, model, opt, 1)


def batch_train(path: str = None,
                model: RGCN_ResNet = None,
                opt: torch.optim = None,
                epoch: int = 100,
                batch_epoch: int = 5
                ):
    if path is None:
        return 'path should not be empty'
    if model is None:
        return 'model should not be empty'
    if opt is None:
        return 'opt should not be empty'
    graph_dir = os.listdir(path + '/_temp')
    for i in range(epoch):
        print(i)
        for graph_file in graph_dir:
            print(graph_file)
            if graph_file == 'whole_hetero_graph.bin':
                continue
            hetero_graph = dgl.load_graphs(path + '/_temp/' + graph_file)[0][0]
            train(hetero_graph, model, opt, batch_epoch)

    # return model


def get_key(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
    return "There is no such Key"


def predict(model: RGCN_ResNet,
            path: str,
            # hetero_graph:dgl.DGLHeteroGraph,
            all_label_dict: dict
            ):
    hetero_graph = dgl.load_graphs(path + '/_temp/whole_hetero_graph.bin')[0][0]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    hetero_graph = hetero_graph.to(device)
    predict_result = model(hetero_graph, hetero_graph.ndata['feature'])
    _, idx = torch.max(predict_result, dim=1)  # [bsz], [bsz]
    reference_index = torch.nonzero(
        torch.unique(hetero_graph.ndata['train_mask']['cell'], return_counts=True)[0] == True).squeeze()
    n_reference_cell = torch.unique(hetero_graph.ndata['train_mask']['cell'], return_counts=True)[1][
        reference_index]  # 计算参考数据集的细胞数，为返回损
    labels = hetero_graph.nodes['cell'].data['label']
    labels = labels.cpu()
    query_labels = labels[n_reference_cell:].tolist()  # 获取查询数据集的标
    label_dict_reverse = {v:k for k,v in all_label_dict.items()}
    query_labels = [label_dict_reverse[i] for i in query_labels]
    #correct = torch.eq(idx[n_reference_cell:], labels[n_reference_cell:]).float().mean()
    #print(correct, ': correct')

    return query_labels
