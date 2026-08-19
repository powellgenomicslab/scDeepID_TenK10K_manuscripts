# scDeepID_TenK10K_manuscripts

This repository contains the scDeepID code used in the TenK10K manuscripts for cell state modelling.

We computed cell function scores as cell states using scDeepID a multi-task transformer, where a biologically-informed masked-encoder is used to generate cell function embeddings along the supplied database, followed by combining a <cls> token for representing each cell’s identity. A set of transformer blocks are then used to learn the high-level data representation of the data, followed by two task heads for cell type identification and cell simulation. The attention-based latent space is extracted from the model by attention rollout132 1150 , and cell function scores are calculated by attribution methods for deep learning models. scDeepID can be flexibly used with different supplied databases such as Reactome, gene ontology (GO), and more.

## Quickstart

### 1) Install dependencies

```bash
pip install torch scanpy anndata numpy pandas scipy scikit-learn tqdm einops matplotlib seaborn
```

### 2) Run a training workflow

```python
import scanpy as sc
import scDeepID_TenK10K_manuscripts as scDeepID

adata = sc.read_h5ad("your_data.h5ad")
adata_sub = scDeepID.preprocess(adata, gmt_path="human_reactome")

scDeepID.train(
    adata_sub,
    gmt_path="human_reactome",
    label_name="celltype",
    epochs=10
)
```
### 3) Run prediction

```python
pred_real, pred_recon, pred_attn = scDeepID.pred(
    adata_sub,
    model_weight_path="demo_project/model-9.pth",
    label_name="celltype"
)
```
