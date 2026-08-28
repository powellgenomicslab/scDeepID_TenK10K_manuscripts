# scDeepID_TenK10K_manuscripts

This repository contains the scDeepID code used in the TenK10K manuscripts for cell state modelling.

We computed cell function scores as cell states using scDeepID, a multi-task transformer. A biologically-informed
masked encoder generates cell function embeddings against a supplied gene-set database, and a `<cls>` token is
prepended to represent each cell's identity. A stack of transformer blocks then learns a high-level representation
of the data, followed by two task heads for cell type identification and cell simulation. The attention-based
latent space is extracted from the model by attention rollout, and cell function scores are calculated by
attribution methods for deep learning models. scDeepID can be flexibly used with different supplied databases such
as Reactome, gene ontology (GO), and more.

## Repository structure

```
scDeepID_TenK10K_manuscripts/
├── __init__.py             Public API: preprocess(), train(), pred(), attribution()
├── pre.py                  Inference: cell type prediction, reconstruction, attention rollout output
├── train.py                GMT parsing, pathway tokenizer, dataset/dataloader, training & evaluation loops
├── scDeepID_model.py       Model definition: FeatureEmbed, Attention, Block, Transformer, attention rollout
├── customized_linear.py    Mask-constrained linear layer backing the biologically-informed encoder
├── util.py                 Cell function inference via attribution (Captum)
├── LICENSE                 GPL-3.0
└── README.md
```

## Installation

### 1) Create and activate a virtual environment

```bash
git clone https://github.com/powellgenomicslab/scDeepID_TenK10K_manuscripts.git
cd scDeepID_TenK10K_manuscripts

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install dependencies

```bash
pip install torch scanpy anndata numpy pandas scipy scikit-learn tqdm einops matplotlib captum
```

### 3) Make the package importable

```python
import sys
sys.path.insert(0, "/path/to/parent/of/clone")
import scDeepID_TenK10K_manuscripts as scDeepID
```

## Input data requirements

Before running the tutorial, make sure your `AnnData` satisfies the following:

- **`adata.X` must be normalised and log-transformed.** Raw count matrix can be processed as below:
  ```python
  sc.pp.normalize_total(adata, target_sum=1e4)
  sc.pp.log1p(adata)
  ```
- **Cell type** labels live in `adata.obs`, default as `celltype`.
- `epochs` must be **2 or more**.

## Tutorial

Throughout, replace the two placeholders with your own paths:

| Placeholder | Meaning |
|---|---|
| `path/to/your/gmt` | Your gene-set database in GMT format. |
| `path/to/your/project` | Output directory for all artefacts. **Resolved relative to your current working directory**, and created if absent. |

```python
import sys
import scanpy as sc
sys.path.insert(0, "/path/to/parent/of/clone")
import scDeepID_TenK10K_manuscripts as scDeepID

GMT     = "path/to/your/gmt"
PROJECT = "path/to/your/project"
EPOCHS  = 10
MODEL   = f"{PROJECT}/model-{EPOCHS - 1}.pth"   # training saves the last epoch only
```

### 1) Preprocess

Restricts the data to genes present in the supplied database, then selects highly variable genes.

```python
adata = sc.read_h5ad("your_data.h5ad")     # normalised + log1p (see above)

adata_sub = scDeepID.preprocess(
    adata,
    gmt_path=GMT,
    project=PROJECT,
    hvg_num=4000,                # number of HVGs to keep
)
```

### 2) Train

```python
scDeepID.train(
    adata_sub,
    gmt_path=GMT,
    project=PROJECT,
    label_name="celltype",
    epochs=EPOCHS,
    max_gs=300,                 # cap on the number of pathway tokens
)
```

Training saves a single checkpoint — the last epoch — as `<PROJECT>/model-<EPOCHS-1>.pth`, since epochs are
zero-indexed. With `EPOCHS = 10` that is `model-9.pth`, which is what `MODEL` above resolves to.

### 3) Prediction

```python
res= scDeepID.pred(
    adata_sub,
    model_weight_path=MODEL,
    project=PROJECT,
    label_name="celltype"
)
print(res[2].obs['pred'])
```

### 4) Cell function computation

Cell function scores are computed by attribution methods via `scDeepID.attribution`. Returns an `AnnData` of per-cell × per-pathway function scores, and a `DataFrame` of
per-cell × per-gene attributions.

```python
pathway_scores, gene_attr = scDeepID.attribution(
    adata_sub,
    project=PROJECT,
    model_weight_path=MODEL,
    batch_size=32,
    method="IntegratedGradients",
)
```

## Integration with single-cell polygenic enrichment

For the polygenic analyses that integrate scDeepID cell function scores with single-cell eQTL and GWAS data, see:

https://github.com/powellgenomicslab/tenk10k-effector-genes/tree/main/scripts/4-polygenic

## Citation

If you use this code in your research, please cite:

Cuomo, A., Spenceley, E., Tanudisastro, H., et al. Impact of rare and common genetic variation on cell type-specific
gene expression in human blood. *medRxiv* 2025.03.20.25324352 (2025). https://doi.org/10.1101/2025.03.20.25324352

Tanudisastro, H., Cuomo, A., Weisburd, B., et al. Tandem repeat variation shapes immune cell type-specific gene
expression. *bioRxiv* 2024.11.02.621562 (2024). https://doi.org/10.1101/2024.11.02.621562

Henry, A., Senabouth, A., Tyebally, R., et al. Single-cell genetics identifies cell-type-specific effector genes
across complex traits and diseases. *medRxiv* 2025.08.28.25334614 (2025). https://doi.org/10.1101/2025.08.28.25334614
