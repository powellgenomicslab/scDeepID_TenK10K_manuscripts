import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scanpy as sc
from tqdm import tqdm
from captum.attr import (
    IntegratedGradients,  
    DeepLift,            
    GradientShap,        
    FeatureAblation,     
    Occlusion,           
    Lime,                
    LRP,                 
    KernelShap,          
    DeepLiftShap         
)

from .scDeepID_model import scDeepID_model as create_model

def calculate_attributions(adata, project, model_weight_path=None, batch_size=32, n_steps=50, method='IntegratedGradients', adata_umap=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    project_path = os.path.join(os.getcwd(), project)
    mask = np.load(os.path.join(project_path, 'mask.npy'))
    pathways = pd.read_csv(os.path.join(project_path, 'pathway.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(project_path, 'label_dictionary.csv'), index_col=0)
    model = create_model(
        num_classes=len(labels),        
        num_genes=adata.shape[1],      
        mask=mask,                      
        embed_dim=48,                  
        z_dim=48,                       
        depth=4,                        
        num_heads=4,                    
        drop_ratio=0.,                  
        attn_drop_ratio=0.,             
        drop_path_ratio=0.,
    ).to(device)
    
    if model_weight_path is None:
        import glob
        model_files = sorted(glob.glob(os.path.join(project_path, 'model-*.pth')))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {project_path}")
        model_weight_path = model_files[-1]  
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()  
    parm = {}
    for name, parameters in model.named_parameters():
        parm[name] = parameters.detach().cpu().numpy()
    gene_pathway_weights = parm['feature_embed.fe.weight']
    gene_pathway_weights = gene_pathway_weights.reshape((int(gene_pathway_weights.shape[0] / 48), 48, adata.shape[1]))
    gene_pathway_weights = abs(gene_pathway_weights)
    gene_pathway_weights = np.mean(gene_pathway_weights, axis=1)
    gene_pathway_weights = pd.DataFrame(gene_pathway_weights)
    gene_pathway_weights.columns = adata.var_names
    gene_pathway_weights.index = pathways['0']
    gene_pathway_weights.to_csv(project_path + '/gene_pathway_weights_meanweight.csv')
    def forward_func(inputs):
        outputs = model(inputs)
        return outputs[0]  

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, inputs):
            outputs = self.model(inputs)
            return outputs[0]
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    print(f"\nUsing method: {method}")
    if method == 'IntegratedGradients':
        attr_method = IntegratedGradients(forward_func)
    elif method == 'GradientShap':
        attr_method = GradientShap(forward_func)
    elif method == 'DeepLift':
        attr_method = DeepLift(wrapped_model)
    elif method == 'FeatureAblation':
        attr_method = FeatureAblation(wrapped_model)
    elif method == 'Occlusion':
        attr_method = Occlusion(wrapped_model)
    elif method == 'Lime':
        attr_method = Lime(wrapped_model)
    elif method == 'LRP':
        attr_method = LRP(wrapped_model)
    elif method == 'KernelShap':
        attr_method = KernelShap(wrapped_model)
    elif method == 'DeepLiftShap':
        attr_method = DeepLiftShap(wrapped_model)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"Calculating attributions for {adata.shape[0]} cells...")
    gene_attributions = []  
    cell_ids = []          
    
    for i in tqdm(range(0, adata.shape[0], batch_size)):
        batch_end = min(i + batch_size, adata.shape[0])
        if hasattr(adata.X, 'toarray'):
            batch_data = adata.X[i:batch_end].toarray()
        else:
            batch_data = adata.X[i:batch_end]
        batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(device)
        baseline = torch.zeros_like(batch_tensor) 
        targets = []
        for j in range(i, batch_end):
            celltype = adata.obs.iloc[j].get('celltype', None)
            if celltype is None:
                raise ValueError(f"Cell at index {j} has no 'celltype' in adata.obs")
            matches = np.where(labels.values == celltype)[0]
            if len(matches) == 0:
                raise ValueError(f"Cell type '{celltype}' not found in label dictionary. "
                               f"Available types: {list(labels.values.flatten())}")
            target = int(matches[0])
            targets.append(target)
        target_tensor = torch.tensor(targets).to(device)

        if method == 'IntegratedGradients':
            attr = attr_method.attribute(
                batch_tensor,
                baselines=baseline,
                target=target_tensor,
                n_steps=n_steps,
                internal_batch_size=None
            )
        elif method == 'GradientShap':
            attr = attr_method.attribute(
                batch_tensor,
                baselines=baseline,
                target=target_tensor,
                n_samples=n_steps
            )
        elif method == 'FeatureAblation':
            attr = attr_method.attribute(
                batch_tensor,
                target=target_tensor,
            )
        elif method == 'Occlusion':
            attr = attr_method.attribute(
                batch_tensor,
                target=target_tensor,
                sliding_window_shapes=(3,)
            )
        elif method == 'Lime':
            attr = attr_method.attribute(
                batch_tensor,
                target=target_tensor,
                n_samples=min(n_steps, 500)
            )
        elif method == 'KernelShap':
            attr = attr_method.attribute(
                batch_tensor,
                target=target_tensor,
                n_samples=min(n_steps, 200)
            )
        elif method == 'DeepLiftShap':
            attr = attr_method.attribute(
                batch_tensor,
                baselines=baseline,
                target=target_tensor
            )
        else:
            attr = attr_method.attribute(
                batch_tensor,
                target=target_tensor
            )
        gene_attributions.extend(attr.cpu().detach().numpy())
        cell_ids.extend(adata.obs_names[i:batch_end])

        del attr, batch_tensor, baseline, target_tensor
        torch.cuda.empty_cache()

    gene_attr_df = pd.DataFrame(
        gene_attributions,
        index=cell_ids,
        columns=adata.var_names
    )
    
    print("Aggregating to pathway level...")
    gene_attr_abs = gene_attr_df.abs()
    pathway_scores = gene_attr_abs @ gene_pathway_weights[gene_attr_df.columns].abs().T
    pathway_adata = sc.AnnData(
        X=pathway_scores.values,                  
        obs=adata.obs.loc[cell_ids].copy()       
    )
    pathway_adata.var_names = gene_pathway_weights.index
    
    if adata_umap is not None and 'X_umap' in adata_umap.obsm:
        common_cells = [cid for cid in cell_ids if cid in adata_umap.obs_names]
        if len(common_cells) == len(cell_ids):
            pathway_adata.obsm['X_umap'] = adata_umap[cell_ids].obsm['X_umap']
        else:
            print(f"Warning: Only {len(common_cells)}/{len(cell_ids)} cells found in adata_umap. Skipping UMAP transfer.")
    elif 'X_umap' in adata.obsm:
        pathway_adata.obsm['X_umap'] = adata[cell_ids].obsm['X_umap']
    
    return pathway_adata, gene_attr_df