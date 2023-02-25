# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:52:19 2023

@author: chenruipu
"""

import pandas as pd
import numpy as np
import scanpy as sc
import torch
import dgl
from typing import Sequence, Union, Mapping, List, Optional, Dict, Callable
import logging
from scipy import sparse
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F


# %%
reference = sc.read_h5ad("E:/Anaconda/envs/env_came/Lib/site-packages/came/sample_data/raw-Baron_human.h5ad")
query = sc.read_h5ad("E:/Anaconda/envs/env_came/Lib/site-packages/came/sample_data/raw-Baron_mouse.h5ad")
gene_homolog = pd.read_csv("E:/Anaconda/envs/env_came/Lib/site-packages/came/sample_data/gene_matches_human2mouse.csv")
reference_go = pd.read_csv("D:/automatic_annotation/ascbgg/human_go.txt", sep='\t', )
query_go = pd.read_csv("D:/automatic_annotation/ascbgg/mouse_go.txt", sep='\t')
gene_homolog = gene_homolog[['human.gene.name', 'gene.name']]
gene_homolog.columns = ['rgene', 'qgene']
ref_label = 'cell_ontology_class'
qry_label = 'cell_ontology_class'

# reference = sc.read_h5ad("D:/automatic_annotation/Dataset/referenceDataset/train_intestine_human_mouse.h5ad")
# query = sc.read_h5ad("D:/automatic_annotation/Dataset/testDataset/Pig/test_pig_intestine_withmeta.h5ad")
# gene_homolog = pd.read_csv("D:/automatic_annotation/CellLingo-main/sampledata/human2pig.txt",sep='\t' )
# reference_go = pd.read_csv("D:/automatic_annotation/CellLingo-main/sampledata/human_go.txt", sep='\t' )
# query_go = pd.read_csv("D:/automatic_annotation/CellLingo-main/sampledata/pig_go.txt", sep='\t')
# gene_homolog.columns = ['rgene', 'qgene']
# ref_label = 'category'
# qry_label = 'cell_type'


# %%
sc.pp.filter_genes(reference, min_cells=3)
sc.pp.filter_genes(query, min_cells=3)
sc.pp.log1p(reference)
sc.pp.highly_variable_genes(reference, flavor='seurat', n_top_genes=2000)
sc.pp.log1p(query)
sc.pp.highly_variable_genes(query, flavor='seurat', n_top_genes=2000)

# %%
sc.tl.pca(reference, svd_solver='arpack')
sc.pp.neighbors(reference)
sc.tl.rank_genes_groups(reference, groupby=ref_label, method='t-test')
ref_DEG = list(set([x for p in [list(x) for x in reference.uns['rank_genes_groups']['names'][0:50]] for x in p]))
ref_HVG = list(reference.var.index[reference.var['highly_variable']])
ref_D_H_gene = list(set(ref_DEG + ref_HVG))

sc.tl.pca(query, svd_solver='arpack')
sc.pp.neighbors(query)
sc.tl.leiden(query)
sc.tl.rank_genes_groups(query, 'leiden', method='t-test')
qry_DEG = list(set([x for p in [list(x) for x in query.uns['rank_genes_groups']['names'][0:50]] for x in p]))
qry_HVG = list(query.var.index[query.var['highly_variable']])
qry_D_H_gene = list(set(qry_DEG + ref_HVG))

# %%
# 在同源关系表中将所有和之前选取基因相关的全部行取出
gene_homolg_nodes = pd.concat(
    [gene_homolog[gene_homolog['rgene'].isin(ref_D_H_gene)], gene_homolog[gene_homolog['qgene'].isin(qry_D_H_gene)]],
    ignore_index=True)
ref_gene = set(gene_homolg_nodes['rgene'])
qry_gene = set(gene_homolg_nodes['qgene'])

# 与在数据文件中存在的gene取交集
ref_gene = list(set(reference.var.index).intersection(ref_gene))
qry_gene = list(set(query.var.index).intersection(qry_gene))
gene_homolg_nodes = gene_homolog[(gene_homolog['rgene'].isin(ref_gene)) & (gene_homolog['qgene'].isin(qry_gene))]

reference = reference[:, ref_gene]
query = query[:, qry_gene]
# gene_homolg_featuers = pd.concat([gene_homolog[(gene_homolog['rgene'].isin(ref_DEG))],gene_homolog[(gene_homolog['qgene'].isin(qry_DEG))]], ignore_index=True)


# %%


all_gene = pd.DataFrame({'gene_name': ref_gene + qry_gene})
# all_gene_dict = {}
# for index,row in all_gene.iterrows():
#     all_gene_dict[row['gene_name']]=index
ref_gene_dict = {}
for index in range(0, len(ref_gene)):
    ref_gene_dict[ref_gene[index]] = index
qry_gene_dict = {}
for index in range(0, len(qry_gene)):
    qry_gene_dict[qry_gene[index]] = index + len(ref_gene_dict)
# 创建gene对应ID的同源对应表
gene_homolg_nodes['rgene'] = gene_homolg_nodes['rgene'].replace(ref_gene_dict)
gene_homolg_nodes['qgene'] = gene_homolg_nodes['qgene'].replace(qry_gene_dict)
r_gene_homolg_nodes = gene_homolg_nodes.rename(columns={'rgene': 'qgene', "qgene": "rgene"})
gene_homolg_gene = pd.concat([gene_homolg_nodes, r_gene_homolg_nodes], ignore_index=True)
# gene_homolg_gene_src = list()
# %%
gene_homolg_gene_src = np.array(gene_homolg_gene['rgene'])
gene_homolg_gene_dst = np.array(gene_homolg_gene['qgene'])

# %%


c2g_cell_id = np.concatenate([reference.X.nonzero()[0], query.X.nonzero()[0] + reference.shape[0]])
c2g_gene_id = np.concatenate([reference.X.nonzero()[1], query.X.nonzero()[1] + len(ref_gene)])
all_cell_id = np.arange(0, reference.shape[0] + query.shape[0])
all_gene_id = np.arange(0, len(all_gene))


# %%

def get_ccnet(adata: sc.AnnData):
    """ Extract the pre-computed single-cell KNN network

    If the adata has not been preprocessed, please run 
    `adata_processed = quick_preprocess(adata, **kwds)` first.

    Parameters
    ----------
    adata
        the data object
    """
    key0 = 'neighbors'
    key = 'connectivities'
    if key in adata.obsp.keys():
        adj = adata.obsp[key]
    else:
        adj = adata.uns[key0][key]
    return adj


# %%
ref_cell_net, qry_cell_net = get_ccnet(reference), get_ccnet(query)
c2c_cell_1 = np.concatenate([ref_cell_net.nonzero()[0], qry_cell_net.nonzero()[0] + reference.shape[0]])
c2c_cell_2 = np.concatenate([ref_cell_net.nonzero()[1], qry_cell_net.nonzero()[1] + reference.shape[0]])


# %%
def change_names(names: Sequence,
                 foo_change: Callable,
                 **kwargs):
    """
    Parameters
    ----------
    names
        a list of names to be modified
    foo_change: function to map a name-string to a new one
    **kwargs: other kwargs for foo_change
    """
    return list(map(foo_change, names, **kwargs))


def pivot_df_to_sparse(df: pd.DataFrame, row=0, col=1, key_data=None, **kwds):
    """
    row, col:
        str or int, int for column index, and str for column name
    """

    def _get_df_vals(key):
        if isinstance(key, str):
            return df[key].values
        else:
            return df.iloc[:, key].values

    rows, cols = list(map(_get_df_vals, [row, col]))

    if key_data is None:
        data = np.ones(df.shape[0], dtype=int)
    else:
        data = _get_df_vals(key_data)
    return pivot_to_sparse(rows, cols, data, **kwds)


def pivot_to_sparse(rows: Sequence, cols: Sequence,
                    data: Optional[Sequence] = None,
                    rownames: Sequence = None,
                    colnames: Sequence = None):
    """
    Parameters
    ----------
    rownames:
        If provided, the resulting matrix rows are restricted to `rownames`,
        Names will be removed if they are not in `rownames`, and names that
        not occur in `rows` but in `rownames` will take ALL-zeros in that row
        of the resulting matrix.
        if not provided, will be set as `rows.unique()`
    colnames: 
        if not provided, will be set as `cols.unique()`

    Notes
    -----
        * `rows` and `cols` should be of the same length!

    """

    def _make_ids_from_name(args):  # vals, names=None
        vals, names = args
        if names is None:
            names = np.unique(vals)
        name2id = pd.Series(np.arange(len(names)), index=names)
        ii = change_names(vals, lambda x: name2id[x])
        return names, ii

    if data is None:
        data = np.ones_like(rows, dtype=float)
    # make sure that all of the row or column names are in the provided names
    if rownames is not None or colnames is not None:
        # r_kept, c_kept = np.ones_like(rows).astype(bool), np.ones_like(rows).astype(bool)
        if rownames is not None:
            tmp = set(rownames)
            r_kept = list(map(lambda x: x in tmp, rows))
            logging.debug(sum(r_kept))
        else:
            r_kept = np.ones_like(rows).astype(bool)
        if colnames is not None:
            tmp = set(colnames)
            c_kept = list(map(lambda x: x in tmp, cols))
        else:
            c_kept = np.ones_like(rows).astype(bool)
        kept = np.minimum(r_kept, c_kept)
        logging.debug(len(kept), sum(kept))
        rows, cols, data = rows[kept], cols[kept], data[kept]

    (rownames, ii), (colnames, jj) = list(map(
        _make_ids_from_name, [(rows, rownames), (cols, colnames)]))

    sparse_mat = sparse.coo_matrix(
        (data, (ii, jj)), shape=(len(rownames), len(colnames)))
    return sparse_mat, rownames, colnames


# %%
# 细胞节点中特征选择

gene_homolg_featuers = pd.concat(
    [gene_homolog[(gene_homolog['rgene'].isin(ref_DEG))], gene_homolog[(gene_homolog['qgene'].isin(qry_DEG))]],
    ignore_index=True)
ref_node_gene = set(gene_homolg_featuers['rgene'])
qry_node_gene = set(gene_homolg_featuers['qgene'])
ref_node_gene = list(set(reference.var.index).intersection(ref_node_gene))
qry_node_gene = list(set(query.var.index).intersection(qry_node_gene))
ref_node_gene.sort(key=list(gene_homolg_featuers['rgene']).index)
qry_node_gene.sort(key=list(gene_homolg_featuers['qgene']).index)
gene_homolg_featuers = pd.merge(gene_homolog[(gene_homolog['rgene'].isin(ref_node_gene))],
                                gene_homolog[(gene_homolog['qgene'].isin(qry_node_gene))], how='inner')

trans_adj, ref_node_gene, qry_node_gene = pivot_df_to_sparse(gene_homolg_featuers)

# 生成细胞节点特征
ref_node_feats = reference[:, ref_node_gene].X.A
qry_node_feats = query[:, qry_node_gene].X.A
trans_adj = trans_adj.A
qry_node_feats = trans_adj.dot(qry_node_feats.T) / trans_adj.sum(1).reshape(trans_adj.shape[0], 1)
qry_node_feats = qry_node_feats.T
# ref_node_feats = torch.Tensor(ref_node_feats)
# qry_node_feats = torch.Tensor(qry_node_feats)
cell_node_feats = np.concatenate((ref_node_feats, qry_node_feats))
cell_node_feats = torch.Tensor(cell_node_feats)
# %%
# go

reference_go = reference_go.dropna()
query_go = query_go.dropna()
reference_go = reference_go[reference_go['Gene name'].isin(ref_gene)]
query_go = query_go[query_go['Gene name'].isin(qry_gene)]
reference_go['Gene name'] = reference_go['Gene name'].replace(ref_gene_dict)
query_go['Gene name'] = query_go['Gene name'].replace(qry_gene_dict)
all_gene_go = pd.concat([reference_go, query_go])
all_go = pd.DataFrame({'GOID': list(set(all_gene_go['GO term accession']))})
all_go_dict = {}
for index, row in all_go.iterrows():
    all_go_dict[row['GOID']] = index
all_gene_go = all_gene_go.replace(all_go_dict)
gotem_id = np.arange(0, len(all_go))
gene_in_gotem = all_gene_go['Gene name'].values
gotem_include_gene = all_gene_go['GO term accession'].values
# %%
cell_label = pd.DataFrame(
    {'label': list(reference.obs[ref_label]) + list(query.obs[qry_label])})
all_label = list(set(cell_label['label']))
all_label_dict = {}
for i in range(0, len(all_label)):
    all_label_dict[str(all_label[i])] = i
cell_label = cell_label.replace(all_label_dict)
cell_label = torch.Tensor(cell_label['label']).long()
# %%
train_mask = torch.cat(
    (torch.ones(reference.shape[0], dtype=torch.bool), torch.zeros(query.shape[0], dtype=torch.bool)))

# %%


hetero_graph = dgl.heterograph({
    ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
    ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
    ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
    ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
    ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
    ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
    ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
    ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
})

hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
hetero_graph.nodes['cell'].data['label'] = cell_label
hetero_graph.nodes['cell'].data['train_mask'] = train_mask


#%%
sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2) #TODO
dataloader = dgl.dataloading.DistNodeDataLoader(
    hetero_graph, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=4) #TODO
input_nodes, output_nodes, blocks = next(iter(dataloader)) #TODO
print(blocks)
# %%

# class RGCN(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, edge_names,node_name):
#         super().__init__()
#         # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
#         self.embedding_gene= dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(in_feats, in_feats,activation=F.relu)
#             for rel in edge_names}, aggregate='sum')
#         self.embedding_gotem= dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(in_feats, in_feats,activation=F.relu)
#             for rel in edge_names}, aggregate='sum')
#         self.conv1 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(in_feats, hid_feats,activation=F.relu)
#             for rel in edge_names}, aggregate='sum')
#         self.conv2 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, hid_feats,activation=F.relu)
#             for rel in edge_names}, aggregate='sum')
#         self.conv3 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, out_feats,activation=F.relu)
#             for rel in edge_names}, aggregate='sum')
#         self.dense = dglnn.HeteroLinear({rel: out_feats for rel in node_name},out_feats)
#     def forward(self, graph, inputs):
#         # 输入是节点的特征字典
#         h0_gene_cell= self.embedding_gene(graph,inputs)
#         inputs['gene'],inputs['cell'] = F.relu(h0_gene_cell['gene']),F.relu(h0_gene_cell['cell'])
#         inputs['gotem']= self.embedding_gotem(graph,inputs)['gotem']
#         #inputs['gotem']= F.relu(inputs['gotem'])
#         h = self.conv1(graph, inputs)
#         h = self.conv2(graph, h)
#         h = self.conv3(graph,h)
#         h = self.dense(h)

#         #h = {k: F.softmax(v,dim=1) for k, v in h.items()}
#         return h
# %%

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, edge_names, node_name):
        super().__init__()
        # 实例化HeteroGraphConv，in_feats是输入特征的维度，out_feats是输出特征的维度，aggregate是聚合函数的类型
        self.embedding_gene = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, in_feats, activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.embedding_gotem = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, in_feats, activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats, activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats, activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats, activation=F.relu)
            for rel in edge_names}, aggregate='sum')
        self.dense = dglnn.HeteroLinear({rel: hid_feats for rel in node_name}, hid_feats)

    def forward(self, graph, inputs):
        # 输入是节点的特征字典
        h0_gene_cell = self.embedding_gene(graph, inputs)
        inputs['gene'], inputs['cell'] = F.relu(h0_gene_cell['gene']), F.relu(h0_gene_cell['cell'])
        inputs['gotem'] = self.embedding_gotem(graph, inputs)['gotem']
        # inputs['gotem']= F.relu(inputs['gotem'])
        h = self.conv1(graph, inputs)
        h = self.conv2(graph, h)
        h = self.conv3(graph, h)
        h = self.dense(h)

        # h = {k: F.softmax(v,dim=1) for k, v in h.items()}
        return h['cell']

class Baisblock(nn.Module):
    def __init__(self, hid_feats):
        super().__init__()
        self.dense1 = nn.Linear(hid_feats, hid_feats)
        self.dense2 = nn.Linear(hid_feats, hid_feats)
        self.dense3 = nn.Linear(hid_feats, hid_feats)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        h = self.dense1(inputs)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.dense2(h)
        h = self.dropout(h)
        h = self.relu(h)
        h = self.dense3(h)
        h = self.relu(h)
        h = h+inputs
        h = self.relu(h)
        return h


class ResNet (nn.Module):
    def __init__(self, hid_feats,out_feats,num_layer):
        super().__init__()
        self.dense1 = nn.Linear(hid_feats,hid_feats//2)
        self.Res_layer1 = self.build_resblock(hid_feats//2,num_layer)
        self.dense2 = nn.Linear(hid_feats//2, hid_feats//4)
        self.Res_layer2 = self.build_resblock(hid_feats//4,num_layer)
        self.dense3 = nn.Linear(hid_feats//4, hid_feats//8)
        self.Res_layer3 = self.build_resblock(hid_feats//8,num_layer)
        self.out = nn.Linear(hid_feats//8,out_feats)
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

class RGCN_ResNet(nn.Module):
    def __init__(self, in_feats, hid_feats,out_feats,num_layer,edge_names, node_name):
        super().__init__()
        self.RGCN = RGCN(in_feats, hid_feats, edge_names, node_name)
        self.ResNet = ResNet(hid_feats,out_feats,num_layer)
    def forward(self,graph, inputs):
        h = self.RGCN(graph, inputs)
        h = self.ResNet(h)
        return h



#%%
model = RGCN_ResNet(in_feats=cell_node_feats.size()[1],
                    hid_feats=512,
                    out_feats =len(all_label_dict),
                    num_layer=14,
                    edge_names= hetero_graph.etypes,
                    node_name=hetero_graph.ntypes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
hetero_graph = hetero_graph.to(device)
# hetero_graph.ndata['feature']=hetero_graph.ndata['feature'].to(device)
h_dict = model(hetero_graph, hetero_graph.ndata['feature'])
labels = hetero_graph.nodes['cell'].data['label']
opt = torch.optim.Adam(model.parameters())

# %%
for epoch in range(200):
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
    correct = torch.eq(idx[ref_node_feats.shape[0]:], labels[ref_node_feats.shape[0]:]).float().mean()
    print('epoch :', epoch, ' loss :', loss.item(), ' | correct :', correct)

# %%
predict_result = model(hetero_graph, hetero_graph.ndata['feature'])
scores, idx = torch.max(predict_result, dim=1)  # [bsz], [bsz]
correct = torch.eq(idx[ref_node_feats.shape[0]:], labels[ref_node_feats.shape[0]:]).float().mean()
