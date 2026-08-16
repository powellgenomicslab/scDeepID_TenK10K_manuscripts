# scDeepID_TenK10K_manuscripts

This repository contains the scDeepID code used in the TenK10K manuscripts for single-cell cell-type modeling with pathway-aware deep learning.

## Quickstart

### 1) Install dependencies

```bash
pip install torch scanpy anndata numpy pandas scipy scikit-learn tqdm einops matplotlib seaborn
```

### 2) Run a minimal training workflow

```python
import scanpy as sc
import scDeepID_TenK10K_manuscripts as scDeepID

# Load your AnnData object (cells x genes)
adata = sc.read_h5ad("your_data.h5ad")

# Optional preprocessing against a pathway database
adata_sub = scDeepID.preprocess(adata, gmt_path="human_reactome")

# Train and save outputs to ./demo_project
scDeepID.train(
    adata_sub,
    gmt_path="human_reactome",
    project="demo_project",
    label_name="celltype",
    epochs=10
)
```

Training writes artifacts (for example `model-*.pth`, `mask.npy`, and `pathway.csv`) into the project directory.

### 3) Run prediction

```python
pred_real, pred_recon, pred_attn = scDeepID.pred(
    adata_sub,
    model_weight_path="demo_project/model-9.pth",
    project="demo_project",
    label_name="celltype"
)
```
