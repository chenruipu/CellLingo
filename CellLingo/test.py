import CellLingo
import pandas as pd
import numpy as np
import scanpy as sc


# %%
reference = sc.read_h5ad("/data2/chenruipu/data/cellLingo/sample_data/raw-Baron_human.h5ad")
query = sc.read_h5ad("/data2/chenruipu/data/cellLingo/sample_data/raw-Baron_mouse.h5ad")
gene_homolog = pd.read_csv("/data2/chenruipu/data/cellLingo/sample_data/gene_matches_human2mouse.csv")
reference_go = pd.read_csv("/data2/chenruipu/data/cellLingo/CellLingo-main/sampledata/human_go.txt", sep='\t', )
query_go = pd.read_csv("/data2/chenruipu/data/cellLingo/CellLingo-main/sampledata/mouse_go.txt", sep='\t')
gene_homolog = gene_homolog[['human.gene.name', 'gene.name']]
gene_homolog.columns = ['rgene', 'qgene']
ref_label = 'cell_ontology_class'
qry_label = 'cell_ontology_class'


#%%
human2mouse = CellLingo.dataprocess(reference, query, gene_homolog, reference_go, query_go, ref_label,qry_label,path = "human2mouse_test")
model,opt = CellLingo.build_model(input_features_dim=674,output_dim=15,homolog=True)
CellLingo.whole_graph_train(path = '/data2/chenruipu/data/cellLingo/human2mouse_test',model=model,opt=opt)
parameter = np.load('/data2/chenruipu/data/cellLingo/human2mouse_test/parameter.npy',allow_pickle=True).item()
predict_label = CellLingo.predict(model=model, path='/data2/chenruipu/data/cellLingo/human2mouse_test', all_label_dict=parameter['label_dict'])
query.obs['predicted_cell_type'] = predict_label
