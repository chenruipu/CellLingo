## CellLingo

CellLingo 是一个用于**跨物种单细胞细胞类型注释**的 Python 包。它将细胞（cell）、基因（gene）与 GO term（项目代码中节点名为 `gotem`）组织为异构图，并使用基于 DGL 的图神经网络进行训练与预测。

## 特性

- **跨物种注释**：通过同源基因表（homolog）对齐不同物种的基因空间  
- **引入 Gene Ontology**：将 GO term 作为图节点参与信息传播  
- **端到端训练与预测**：提供 `dataprocess` → `build_model` → `whole_graph_train` → `predict` 的完整流程

## 安装

在仓库根目录（包含 `setup.py` 的目录）运行：

```bash
pip install .
```

### 依赖

`setup.py` 中声明的依赖如下：

- `scanpy`
- `dgl >= 1`
- `torch >= 1.13, < 2`

## 快速开始（最小示例）

下面示例与仓库中的 `CellLingo/test.py` 一致（请把路径替换成你自己的数据路径）。

```python
import CellLingo
import pandas as pd
import numpy as np
import scanpy as sc

reference = sc.read_h5ad("raw-Baron_human.h5ad")
query = sc.read_h5ad("raw-Baron_mouse.h5ad")

# 同源基因表：两列分别为参考物种基因与查询物种基因
gene_homolog = pd.read_csv("gene_matches_human2mouse.csv")[["human.gene.name", "gene.name"]]

# GO 注释表：两列分别为 Gene name 与 GO term accession（文本/tsv 均可读入）
reference_go = pd.read_csv("human_go.txt", sep="\t")
query_go = pd.read_csv("mouse_go.txt", sep="\t")

ref_label = "cell_ontology_class"
qry_label = "cell_ontology_class"

CellLingo.dataprocess(
    reference=reference,
    query=query,
    gene_homolog=gene_homolog,
    reference_go=reference_go,
    query_go=query_go,
    reference_label=ref_label,
    query_label=qry_label,
    path="human2mouse_test",
)

# 这里的 input_features_dim / output_dim 与你的数据处理结果有关
model, opt = CellLingo.build_model(input_features_dim=674, output_dim=15, homolog=True)
CellLingo.whole_graph_train(path="human2mouse_test", model=model, opt=opt)

parameter = np.load("human2mouse_test/parameter.npy", allow_pickle=True).item()
pred = CellLingo.predict(model=model, path="human2mouse_test", all_label_dict=parameter["label_dict"])
query.obs["predicted_cell_type"] = pred
```

## API 概览

### `CellLingo.dataprocess(...)`

负责：

- 预处理参考/查询 `AnnData`
- 构建异构图（`cell/gene/gotem`）
- 保存图到 `path/_temp/whole_hetero_graph.bin`
- 保存参数到 `path/parameter.npy`

常用参数（与源码保持一致）：

- `reference`, `query`：`scanpy.AnnData`
- `gene_homolog`：同源基因表（两列）
- `reference_go`, `query_go`：GO 注释表
- `reference_label`, `query_label`：细胞类型标签列名
- `homolog`：是否启用同源基因连接（默认 `True`）
- `go`：是否启用 GO term（默认 `True`）

### `CellLingo.build_model(...)`

构建模型与优化器：

- `input_features_dim`：细胞节点输入特征维度
- `output_dim`：类别数
- `homolog`：是否包含同源基因边（影响 `edge_names`）

### `CellLingo.whole_graph_train(...)` / `CellLingo.predict(...)`

- `whole_graph_train`：从 `path/_temp/whole_hetero_graph.bin` 读取整图训练  
- `predict`：返回 query 细胞的预测标签（字符串列表）

---

## CellLingo (English)

CellLingo is a Python package for **cross-species single-cell cell-type annotation**. It builds a heterogeneous graph with cell, gene, and GO term nodes (the GO node type is named `gotem` in the codebase), and trains a DGL-based graph neural network for prediction.

## Features

- **Cross-species annotation**: aligns gene spaces across species using a homolog table  
- **Gene Ontology support**: introduces GO terms as graph nodes for message passing  
- **End-to-end workflow**: `dataprocess` → `build_model` → `whole_graph_train` → `predict`

## Installation

From the repository root (the directory containing `setup.py`):

```bash
pip install .
```

### Dependencies

As declared in `setup.py`:

- `scanpy`
- `dgl >= 1`
- `torch >= 1.13, < 2`

## Quick start (minimal example)

This example mirrors `CellLingo/test.py` in this repository. Replace file paths with your own.

```python
import CellLingo
import pandas as pd
import numpy as np
import scanpy as sc

reference = sc.read_h5ad("raw-Baron_human.h5ad")
query = sc.read_h5ad("raw-Baron_mouse.h5ad")

# Homolog table: two columns for reference/query species genes
gene_homolog = pd.read_csv("gene_matches_human2mouse.csv")[["human.gene.name", "gene.name"]]

# GO annotations: two columns for Gene name and GO term accession
reference_go = pd.read_csv("human_go.txt", sep="\t")
query_go = pd.read_csv("mouse_go.txt", sep="\t")

ref_label = "cell_ontology_class"
qry_label = "cell_ontology_class"

CellLingo.dataprocess(
    reference=reference,
    query=query,
    gene_homolog=gene_homolog,
    reference_go=reference_go,
    query_go=query_go,
    reference_label=ref_label,
    query_label=qry_label,
    path="human2mouse_test",
)

# input_features_dim / output_dim depend on your processed data
model, opt = CellLingo.build_model(input_features_dim=674, output_dim=15, homolog=True)
CellLingo.whole_graph_train(path="human2mouse_test", model=model, opt=opt)

parameter = np.load("human2mouse_test/parameter.npy", allow_pickle=True).item()
pred = CellLingo.predict(model=model, path="human2mouse_test", all_label_dict=parameter["label_dict"])
query.obs["predicted_cell_type"] = pred
```

## API overview

### `CellLingo.dataprocess(...)`

Responsibilities:

- preprocess reference/query `AnnData`
- build a heterogeneous graph (`cell/gene/gotem`)
- save the graph to `path/_temp/whole_hetero_graph.bin`
- save parameters to `path/parameter.npy`

Common arguments (as in the source code):

- `reference`, `query`: `scanpy.AnnData`
- `gene_homolog`: homolog table (two columns)
- `reference_go`, `query_go`: GO annotation tables
- `reference_label`, `query_label`: column names of cell-type labels
- `homolog`: whether to enable homolog gene edges (default `True`)
- `go`: whether to enable GO terms (default `True`)

### `CellLingo.build_model(...)`

Builds the model and optimizer:

- `input_features_dim`: input feature dimension of cell nodes
- `output_dim`: number of classes
- `homolog`: whether homolog edges are included (affects `edge_names`)

### `CellLingo.whole_graph_train(...)` / `CellLingo.predict(...)`

- `whole_graph_train`: trains on the full graph loaded from `path/_temp/whole_hetero_graph.bin`  
- `predict`: returns predicted labels for query cells (a list of strings)

