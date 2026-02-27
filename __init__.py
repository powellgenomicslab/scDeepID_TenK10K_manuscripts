from . import scDeepID_model, pre
from .train import fit_model, get_dbhvg_simple
from .pre import predict
from .util import calculate_attributions

def preprocess(adata, gmt_path, project=None, min_g=12,max_g=5000,hvg_num=None):
    r"""
    Preprocess the data to only keep the genes within the data base, and select highly variable genes (HVGs) if specified.
    """
    adata_sub_list = get_dbhvg_simple(adata, gmt_path, project=project, hvg_num=hvg_num,min_g=min_g,max_g=max_g)
    return adata_sub_list

def train(adata, gmt_path,project=None,hvg_list_path=None,autobalance=False, pre_weights='', label_name='celltype',
          min_g=12, max_g=300, max_gs=300, min_hvg_genes=12, max_hvg_genes=500,
          mask_ratio=0.015, n_fc_tokens=0, batch_size=8, embed_dim=48,z_dim=48,depth=4,num_heads=4,
          lr=0.001, epochs=10, lrf=0.01, lambda1=0.9, lambda2=0.1,beta=1.0,
          drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., attn_temperature=1.0,
          anneal=True, jaccard_threshold=0.5, retention_ratio=0.5):
    r"""
    Main training function.
    """
    fit_model(adata, gmt_path,project=project,hvg_list_path=hvg_list_path,autobalance=autobalance, pre_weights=pre_weights,
               label_name=label_name, min_g=min_g, max_g=max_g, min_hvg_genes=min_hvg_genes, max_hvg_genes=max_hvg_genes, 
               max_gs=max_gs, mask_ratio = mask_ratio,n_fc_tokens = n_fc_tokens,batch_size=batch_size, embed_dim=embed_dim, 
               z_dim=z_dim,depth=depth,num_heads=num_heads,lr=lr, epochs= epochs, lrf=lrf, lambda1=lambda1, 
               lambda2=lambda2, beta=beta, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=drop_path_ratio, 
               attn_temperature=attn_temperature, anneal=anneal, jaccard_threshold=jaccard_threshold, retention_ratio=retention_ratio)

def pred(adata,model_weight_path,project,
        label_name=None,cutoff=0.1,
        n_fc_tokens = 0,batch_size=50,embed_dim=48,z_dim=48,depth=4,
        num_heads=4,attn_temperature=1.0):
    r"""
    Main prediction function.
    """
    adata = predict(adata,model_weight_path,project=project,label_name=label_name,cutoff=cutoff,
    n_fc_tokens = n_fc_tokens,batch_size=batch_size,embed_dim=embed_dim,z_dim=z_dim,depth=depth,num_heads=num_heads,
    attn_temperature=attn_temperature)
    return(adata)

def attribution(adata, project, model_weight_path=None, batch_size=8, n_steps=25, method='IntegratedGradients', adata_umap=None):
    r"""
    >>> # Calculate attributions with IntegratedGradients
    >>> pathway_attr, gene_attr = scDeepID.attribution(adata, project='my_project')
    """
    return calculate_attributions(
        adata=adata,
        project=project,
        model_weight_path=model_weight_path,
        batch_size=batch_size,
        n_steps=n_steps,
        method=method,
        adata_umap=adata_umap
    )

