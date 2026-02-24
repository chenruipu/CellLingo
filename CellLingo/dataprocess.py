# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 08:52:19 2023

@author: chenruipu
"""
import pandas as pd
import numpy as np
import scanpy as sc
from typing import Sequence, Union, Mapping, List, Optional, Dict, Callable, AnyStr
import logging
from scipy import sparse
import torch
import dgl
import os


def get_ccnet(adata: sc.AnnData):
    key0 = 'neighbors'
    key = 'connectivities'
    if key in adata.obsp.keys():
        adj = adata.obsp[key]
    else:
        adj = adata.uns[key0][key]
    return adj


def change_names(names: Sequence,
                 foo_change: Callable,
                 **kwargs):

    return list(map(foo_change, names, **kwargs))


def pivot_df_to_sparse(df: pd.DataFrame, row=0, col=1, key_data=None, **kwds):
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
def dataprocess(reference: sc.AnnData,
                query: sc.AnnData,
                gene_homolog: pd.DataFrame,
                reference_go: pd.DataFrame,
                query_go: pd.DataFrame,
                reference_label: AnyStr,
                query_label: AnyStr = None,
                ref_batch: str = None,
                qry_batch: str = None,
                batch: int = None,
                path: str = None,
                homolog: bool=True,
                ):
    homolog = homolog
    # 不容许单独输入qry_batch或ref_batch
    if batch is None:
        if (qry_batch is not None) & (ref_batch is not None):
            pass
        elif (qry_batch is None) & (ref_batch is None):
            pass
        else:
            return 'error with batch,qry_batch,ref_batch setting'

    # 指定工作路径
    if path is None:
        path = os.getcwd()
        temp_graph_path = path
    else:
        temp_graph_path = path + '/_temp'
        if os.path.exists(temp_graph_path):
            pass
        else:
            os.makedirs(temp_graph_path)

    # 数据预处理
    reference.obs[reference_label] = reference.obs[reference_label].astype('category')
    if ref_batch is not None:
        reference.obs[ref_batch] = reference.obs[ref_batch].astype('category')
    if query_label is not None:
        query.obs[query_label] = query.obs[query_label].astype('category')
    if qry_batch is not None:
        query.obs[qry_batch] = query.obs[qry_batch].astype('category')
    #过滤基因，每个基因至少在三个细胞中表达
    sc.pp.filter_genes(reference, min_cells=3)
    sc.pp.filter_genes(query, min_cells=3)
    #归一化参考数据集的基因表达
    sc.pp.normalize_total(reference, target_sum=1e4)
    sc.pp.log1p(reference)
    #计算高变基因取前2000
    sc.pp.highly_variable_genes(reference, flavor='seurat', n_top_genes=2000)
    reference_leiden = reference[:, reference.var.highly_variable]
    sc.pp.scale(reference_leiden, max_value=10)
    sc.tl.pca(reference_leiden, svd_solver='arpack')
    sc.pp.neighbors(reference_leiden)
    reference.obsp = reference_leiden.obsp
    reference.uns = reference_leiden.uns
    sc.tl.rank_genes_groups(reference, groupby=reference_label, method='t-test') #计算特异表达基因
    ref_DEG = list(set([x for p in [list(x) for x in reference.uns['rank_genes_groups']['names'][0:50]] for x in p])) #特异表达基因列表
    ref_HVG = list(reference.var.index[reference.var['highly_variable']]) #高变基因列表
    ref_D_H_gene = list(set(ref_DEG + ref_HVG)) #高变基因和特异表达基因取交集
    #归一化查询数据集的基因表达
    sc.pp.normalize_total(query, target_sum=1e4)
    sc.pp.log1p(query)
    sc.pp.highly_variable_genes(query, flavor='seurat', n_top_genes=2000)
    query_leiden = query[:, query.var.highly_variable]
    sc.pp.scale(query_leiden, max_value=10)
    sc.tl.pca(query_leiden, svd_solver='arpack')
    sc.pp.neighbors(query_leiden)
    #无监督聚类
    sc.tl.leiden(query_leiden)
    query.obsp = query_leiden.obsp
    query.uns = query_leiden.uns
    query.obs['leiden'] = query_leiden.obs['leiden']
    sc.tl.rank_genes_groups(query, 'leiden', method='t-test')
    qry_DEG = list(set([x for p in [list(x) for x in query.uns['rank_genes_groups']['names'][0:50]] for x in p]))
    qry_HVG = list(query.var.index[query.var['highly_variable']])
    qry_D_H_gene = list(set(qry_DEG + qry_HVG))
    #同源基因表处理
    gene_homolog.columns = ['rgene', 'qgene'] #修改列明
    gene_homolog = gene_homolog.dropna() #去掉空行
    gene_homolg_nodes = pd.concat([gene_homolog[gene_homolog['rgene'].isin(ref_D_H_gene)],
                                   gene_homolog[gene_homolog['qgene'].isin(qry_D_H_gene)]], ignore_index=True) #保留同源基因
    ref_gene = set(gene_homolg_nodes['rgene'])
    qry_gene = set(gene_homolg_nodes['qgene'])

    #与在数据中存在的基因取交集
    ref_gene = list(set(reference.var.index).intersection(ref_gene))
    qry_gene = list(set(query.var.index).intersection(qry_gene))
    gene_homolg_nodes = gene_homolog[(gene_homolog['rgene'].isin(ref_gene)) & (gene_homolog['qgene'].isin(qry_gene))]
    #取相应的基因
    reference = reference[:, ref_gene]
    query = query[:, qry_gene]
    all_gene = pd.DataFrame({'gene_name': ref_gene + qry_gene})
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
    #细胞表达基因与基因被细胞表达关系
    gene_homolg_gene_src = np.array(gene_homolg_gene['rgene'])
    gene_homolg_gene_dst = np.array(gene_homolg_gene['qgene'])
    c2g_cell_id = np.concatenate([reference.X.nonzero()[0], query.X.nonzero()[0] + reference.shape[0]])
    c2g_gene_id = np.concatenate([reference.X.nonzero()[1], query.X.nonzero()[1] + len(ref_gene)])
    all_cell_id = np.arange(0, reference.shape[0] + query.shape[0])
    all_gene_id = np.arange(0, len(all_gene))
    #生成细胞节点连接关系
    ref_cell_net, qry_cell_net = get_ccnet(reference), get_ccnet(query)
    c2c_cell_1 = np.concatenate([ref_cell_net.nonzero()[0], qry_cell_net.nonzero()[0] + reference.shape[0]])
    c2c_cell_2 = np.concatenate([ref_cell_net.nonzero()[1], qry_cell_net.nonzero()[1] + reference.shape[0]])

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
    cell_node_feats = np.concatenate((ref_node_feats, qry_node_feats))
    cell_node_feats = torch.Tensor(cell_node_feats)
    # go
    reference_go.columns = ['Gene name', 'GO term accession']
    query_go.columns = ['Gene name', 'GO term accession']
    reference_go = reference_go.dropna()
    query_go = query_go.dropna()
    reference_go = reference_go[reference_go['Gene name'].isin(ref_gene)]
    query_go = query_go[query_go['Gene name'].isin(qry_gene)]
    reference_go['Gene name'] = reference_go['Gene name'].replace(ref_gene_dict)
    query_go['Gene name'] = query_go['Gene name'].replace(qry_gene_dict)
    all_gene_go = pd.concat([reference_go, query_go])
    all_go = pd.DataFrame({'GOID': list(set(all_gene_go['GO term accession']))})
    #生成GOID字典
    all_go_dict = {}
    for index, row in all_go.iterrows():
        all_go_dict[row['GOID']] = index
    all_gene_go = all_gene_go.replace(all_go_dict)
    gotem_id = np.arange(0, len(all_go))
    gene_in_gotem = all_gene_go['Gene name'].values
    gotem_include_gene = all_gene_go['GO term accession'].values

    if query_label is None: #如果查询数据集为提供预分类结果
        n_cell_label = len(reference.obs[reference_label].cat.categories)
        cell_label = pd.DataFrame(
            {'label': list(reference.obs[reference_label]) + list(np.random.randint(0, n_cell_label, len(query)))})
        all_label = list(reference.obs[reference_label].cat.categories)
        all_label_dict = {}
        for i in range(0, len(all_label)):
            all_label_dict[str(all_label[i])] = i
        cell_label = cell_label.replace(all_label_dict)
        cell_label = torch.Tensor(cell_label['label']).long()
    else:
        cell_label = pd.DataFrame({'label': list(reference.obs[reference_label]) + list(query.obs[query_label])})
        all_label = list(
            set(list(reference.obs[reference_label].cat.categories) + list(query.obs[query_label].cat.categories)))
        all_label_dict = {}
        for i in range(0, len(all_label)):
            all_label_dict[str(all_label[i])] = i
        cell_label = cell_label.replace(all_label_dict)
        cell_label = torch.Tensor(cell_label['label']).long()
    #只计算参考数据集的loss
    train_mask = torch.cat((torch.ones(reference.shape[0], dtype=torch.bool), torch.zeros(query.shape[0], dtype=torch.bool)))
    #全图训练
    if batch is None:
        if (ref_batch is None) & (qry_batch is None):
            #同源基因连接
            if homolog:
                hetero_graph = dgl.heterograph({
                    ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                    ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                    ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                    ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
                    ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                    ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                })
            else: #不使用同源基因连接
                hetero_graph = dgl.heterograph({
                    ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                    ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                    ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                    ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                    ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                })
            hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
            hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
            hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
            hetero_graph.nodes['cell'].data['label'] = cell_label
            hetero_graph.nodes['cell'].data['train_mask'] = train_mask
            dgl.save_graphs(temp_graph_path + '/whole_hetero_graph.bin', hetero_graph)
            parameter = {'path': path, 'input_dim': len(ref_node_gene), 'output_dim': len(all_label_dict),
                         'label_dict': all_label_dict,'homolog': homolog}
            np.save(path + '/parameter.npy', parameter)
            return path, len(ref_node_gene), len(all_label_dict), all_label_dict,homolog
        else:
            if homolog:
                whole_hetero_graph = dgl.heterograph({
                    ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                    ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                    ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                    ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
                    ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                    ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                })
            else:
                whole_hetero_graph = dgl.heterograph({
                    ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                    ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                    ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                    ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                    ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                    ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                    ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                })
            whole_hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
            whole_hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
            whole_hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
            whole_hetero_graph.nodes['cell'].data['label'] = cell_label
            whole_hetero_graph.nodes['cell'].data['train_mask'] = train_mask
            dgl.save_graphs(temp_graph_path + '/whole_hetero_graph.bin', whole_hetero_graph)
            reference_group = reference.obs.groupby(ref_batch).indices
            query_group = query.obs.groupby(qry_batch).indices
            reference_group = [reference[inds] for inds in reference_group.values()]
            query_group = [query[inds] for inds in query_group.values()]
            i = 0
            #分别将参考数据集和测试数据集分割，两两组合形成小图，便于训练
            for reference in reference_group:
                for query in query_group:
                    c2g_cell_id = np.concatenate([reference.X.nonzero()[0], query.X.nonzero()[0] + reference.shape[0]])
                    c2g_gene_id = np.concatenate([reference.X.nonzero()[1], query.X.nonzero()[1] + len(ref_gene)])
                    all_cell_id = np.arange(0, reference.shape[0] + query.shape[0])
                    all_gene_id = np.arange(0, len(all_gene))
                    # 生成细胞节点特征
                    ref_node_feats = reference[:, ref_node_gene].X.A
                    qry_node_feats = query[:, qry_node_gene].X.A
                    # ref_node_feats = reference[:, ref_node_gene].X
                    # qry_node_feats = query[:, qry_node_gene].X
                    # trans_adj = trans_adj.A
                    qry_node_feats = trans_adj.dot(qry_node_feats.T) / trans_adj.sum(1).reshape(trans_adj.shape[0], 1)
                    qry_node_feats = qry_node_feats.T
                    # ref_node_feats = torch.Tensor(ref_node_feats)
                    # qry_node_feats = torch.Tensor(qry_node_feats)
                    cell_node_feats = np.concatenate((ref_node_feats, qry_node_feats))
                    cell_node_feats = torch.Tensor(cell_node_feats)
                    ref_cell_net, qry_cell_net = get_ccnet(reference), get_ccnet(query)
                    c2c_cell_1 = np.concatenate(
                        [ref_cell_net.nonzero()[0], qry_cell_net.nonzero()[0] + reference.shape[0]])
                    c2c_cell_2 = np.concatenate(
                        [ref_cell_net.nonzero()[1], qry_cell_net.nonzero()[1] + reference.shape[0]])
                    if query_label is None:
                        n_cell_label = len(reference.obs[reference_label].cat.categories)
                        cell_label = pd.DataFrame({'label': list(reference.obs[reference_label]) + list(
                            np.random.randint(0, n_cell_label, len(query)))})
                    else:
                        cell_label = pd.DataFrame(
                            {'label': list(reference.obs[reference_label]) + list(query.obs[query_label])})
                    # all_label = list(set(cell_label['label']))
                    cell_label = cell_label.replace(all_label_dict)
                    cell_label = torch.Tensor(cell_label['label']).long()
                    train_mask = torch.cat((torch.ones(reference.shape[0], dtype=torch.bool),
                                            torch.zeros(query.shape[0], dtype=torch.bool)))
                    if homolog:
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
                    else:
                        hetero_graph = dgl.heterograph({
                            ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                            ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                            ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                            ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                            ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                            ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                            ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                            ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                        })

                    hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
                    hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
                    hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
                    hetero_graph.nodes['cell'].data['label'] = cell_label
                    hetero_graph.nodes['cell'].data['train_mask'] = train_mask
                    # print(hetero_graph)
                    file_name = '/hetero_graph_' + str(i) + '.bin'
                    i = i + 1
                    print(file_name)
                    dgl.save_graphs(temp_graph_path + file_name, hetero_graph)
                    # result_hetero_graph.append(hetero_graph)
            parameter = {'path': path, 'input_dim': len(ref_node_gene), 'output_dim': len(all_label_dict),
                         'label_dict': all_label_dict,'homolog': homolog}
            np.save(path + '/parameter.npy', parameter)
            return path, len(ref_node_gene), len(all_label_dict), all_label_dict,homolog
    else:
        if homolog:
            whole_hetero_graph = dgl.heterograph({
                ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
                ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
            })
        else:
            whole_hetero_graph = dgl.heterograph({
                ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                # ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
                ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
            })
        whole_hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
        whole_hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
        whole_hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
        whole_hetero_graph.nodes['cell'].data['label'] = cell_label
        whole_hetero_graph.nodes['cell'].data['train_mask'] = train_mask
        dgl.save_graphs(temp_graph_path + '/whole_hetero_graph.bin', whole_hetero_graph)
        ref_batch = np.random.randint(0, batch, reference.shape[0])
        qry_batch = np.random.randint(0, batch, query.shape[0])
        reference.obs['batch'] = list(ref_batch)
        query.obs['batch'] = list(qry_batch)
        reference_group = reference.obs.groupby('batch').indices
        query_group = query.obs.groupby('batch').indices
        reference_group = [reference[inds] for inds in reference_group.values()]
        query_group = [query[inds] for inds in query_group.values()]
        i = 0
        for reference in reference_group:
            for query in query_group:
                c2g_cell_id = np.concatenate([reference.X.nonzero()[0], query.X.nonzero()[0] + reference.shape[0]])
                c2g_gene_id = np.concatenate([reference.X.nonzero()[1], query.X.nonzero()[1] + len(ref_gene)])
                all_cell_id = np.arange(0, reference.shape[0] + query.shape[0])
                all_gene_id = np.arange(0, len(all_gene))
                # 生成细胞节点特征
                ref_node_feats = reference[:, ref_node_gene].X.A
                qry_node_feats = query[:, qry_node_gene].X.A
                qry_node_feats = trans_adj.dot(qry_node_feats.T) / trans_adj.sum(1).reshape(trans_adj.shape[0], 1)
                qry_node_feats = qry_node_feats.T
                cell_node_feats = np.concatenate((ref_node_feats, qry_node_feats))
                cell_node_feats = torch.Tensor(cell_node_feats)
                ref_cell_net, qry_cell_net = get_ccnet(reference), get_ccnet(query)
                c2c_cell_1 = np.concatenate([ref_cell_net.nonzero()[0], qry_cell_net.nonzero()[0] + reference.shape[0]])
                c2c_cell_2 = np.concatenate([ref_cell_net.nonzero()[1], qry_cell_net.nonzero()[1] + reference.shape[0]])
                if query_label is None:
                    n_cell_label = len(reference.obs[reference_label].cat.categories)
                    cell_label = pd.DataFrame({'label': list(reference.obs[reference_label]) + list(
                        np.random.randint(0, n_cell_label, len(query)))})
                else:
                    cell_label = pd.DataFrame(
                        {'label': list(reference.obs[reference_label]) + list(query.obs[query_label])})
                cell_label = cell_label.replace(all_label_dict)
                cell_label = torch.Tensor(cell_label['label']).long()
                train_mask = torch.cat(
                    (torch.ones(reference.shape[0], dtype=torch.bool), torch.zeros(query.shape[0], dtype=torch.bool)))
                if homolog:
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
                else:
                    hetero_graph = dgl.heterograph({
                        ('cell', 'cell_loop', 'cell'): (all_cell_id, all_cell_id),
                        ('gene', 'gene_loop', 'gene'): (all_gene_id, all_gene_id),
                        ('gotem', 'gotem_loop', 'gotem'): (gotem_id, gotem_id),
                        ('cell', 'similar_to', 'cell'): (c2c_cell_1, c2c_cell_2),
                        ('cell', 'express', 'gene'): (c2g_cell_id, c2g_gene_id),
                        ('gene', 'express_by', 'cell'): (c2g_gene_id, c2g_cell_id),
                        # ('gene', 'homolog', 'gene'): (gene_homolg_gene_src, gene_homolg_gene_dst),
                        ('gotem', 'include', 'gene'): (gotem_include_gene, gene_in_gotem),
                        ('gene', 'in', 'gotem'): (gene_in_gotem, gotem_include_gene)
                    })

                hetero_graph.nodes['cell'].data['feature'] = cell_node_feats
                hetero_graph.nodes['gene'].data['feature'] = torch.zeros(len(all_gene), cell_node_feats.size()[1])
                hetero_graph.nodes['gotem'].data['feature'] = torch.zeros(len(all_go), cell_node_feats.size()[1])
                hetero_graph.nodes['cell'].data['label'] = cell_label
                hetero_graph.nodes['cell'].data['train_mask'] = train_mask
                # print(hetero_graph)
                file_name = '/hetero_graph_' + str(i) + '.bin'
                i = i + 1
                print(file_name)
                dgl.save_graphs(temp_graph_path + file_name, hetero_graph)
        parameter = {'path': path, 'input_dim': len(ref_node_gene), 'output_dim': len(all_label_dict),
                     'label_dict': all_label_dict, 'homolog': homolog}
        np.save(path + '/parameter.npy', parameter)
        return path, len(ref_node_gene), len(all_label_dict), all_label_dict, homolog


def load_parameter(path: str):
    parameter = np.load(path + '/parameter.npy',allow_pickle=True).item()
    path = parameter['path']
    input_dim = parameter['input_dim']
    output_dim = parameter['output_dim']
    all_label_dic = parameter['label_dict']
    homolog = parameter['homolog']
    return path, input_dim, output_dim, all_label_dic,homolog
