# scDeepID_TenK10K_manuscripts

This repository contains the scDeepID code used in the TenK10K manuscripts for cell state modelling.

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
