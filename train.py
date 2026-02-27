import os
import sys
import csv
import time
import math
import random
import platform
from collections import OrderedDict


import numpy as np
import pandas as pd
import scipy.sparse
import scanpy as sc
import anndata as ad
import torch.utils
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .scDeepID_model import scDeepID_model as create_model

def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)

class MyDataSet(Dataset):  
    def __init__(self, exp, label=None):  
        self.exp = exp
        self.label = label if label is not None else torch.full((exp.shape[0],), -1, dtype=torch.long)
        self.input_size = self.exp.shape[1]
        self.sample_num = self.exp.shape[0]

    def __getitem__(self, index):
        sample = np.array(self.exp[index].todense())
        in_data = sample.astype(np.float32)
        in_data = np.squeeze(in_data)
        in_label = self.label[index]

        return in_data, in_label

    def __len__(self):
        return self.sample_num
    
def prepareDataSet(adata, label_name="Celltype", batch_size=8, autobalance=False, project_path=""):
    if label_name is None:
        exp = scipy.sparse.csr_matrix(adata.X)
        label = None
        test_exp = exp
        test_label = None   
    else:
        if autobalance:
            print("autobalancing data...")
            label = adata.obs[label_name].astype("str")
            unique_labels, counts = np.unique(label, return_counts=True)
            mean_count = int(np.mean(counts))
            median_count = int(np.median(counts))
            print(f"mean count: {mean_count}, median count: {median_count}")
            anchor_count = int(max(mean_count, median_count))
            adata_list = []
            for cls,num in zip(unique_labels,counts):
                if num < anchor_count:
                    fold_num = anchor_count // num
                    remaining = anchor_count % num
                    print(f"doing augementation for celltype {cls} with fold_num {fold_num}, remaining {remaining} ...")
                    adata_tmp = adata[adata.obs[label_name]==cls].copy()
                    if fold_num >=2:
                        for _ in range(int(fold_num)-1): 
                            adata_list.append(adata_tmp.copy())
                    if remaining > 0:
                        sampled_indices = np.random.choice(adata_tmp.obs.index, remaining, replace=False)
                        adata_list.append(adata_tmp[sampled_indices].copy())
                else:
                    print(f"skip augementation for celltype {cls} with original num {num}")
            adata_copy = ad.concat(adata_list, index_unique="_")
            del adata_list
            adata_copy = ad.concat([adata, adata_copy], index_unique="_")
            print("adata shape after augementation: {}".format(adata_copy.shape))
            print(adata_copy.obs[label_name].value_counts())
            exp = scipy.sparse.csr_matrix(adata_copy.X)
            label = adata_copy.obs[label_name].astype("str")
            test_exp = scipy.sparse.csr_matrix(adata.X)
            test_label = adata.obs[label_name].astype("str")
            print("finish balancing data!")
        else:
            exp = scipy.sparse.csr_matrix(adata.X)
            label = adata.obs[label_name].astype("str")
            test_exp = exp
            test_label = label

    label_encoder = LabelEncoder()
    if label is not None:
        label_encoder.fit(label)
        label_encoded = label_encoder.transform(label)
        label_max = np.max(label_encoded)
        test_label_encoded = label_encoder.transform(test_label)
    else:
        label_encoded = None
        label_max = None
        test_label_encoded = None

    trainset = MyDataSet(exp, label_encoded)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    testset = MyDataSet(test_exp, test_label_encoded) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    inverse = label_encoder.inverse_transform(range(0, label_max + 1)) if label is not None else None
    genes = adata.var_names.to_list()

    return trainloader, testloader, int(math.ceil(len(trainset) / batch_size)), inverse, genes, label_max

def get_gmt(gmt):
    import pathlib
    root = pathlib.Path(__file__).parent
    gmt_files = {
        "human_gobp": [root / "resources/GO_bp.gmt"],
        "human_immune": [root / "resources/immune.gmt"],
        "human_reactome": [root / "resources/reactome.gmt"], 
        "human_tf": [root / "resources/TF.gmt"],
        "mouse_gobp": [root / "resources/m_GO_bp.gmt"], 
        "mouse_reactome": [root / "resources/m_reactome.gmt"], 
        "mouse_tf": [root / "resources/m_TF.gmt"],
        "mouse_gobp_2024": [root / "resources/m_GO_bp_2024.gmt"], 
        "mouse_reactome_2024": [root / "resources/m_reactome_2024.gmt"], 
        "human_gobp_2024": [root / "resources/GO_bp_2024.gmt"],
        "human_reactome_2024": [root / "resources/reactome_2024.gmt"], 
        "human_reactome_2023": [root / "resources/reactome_2023.gmt"], 
    }
    return gmt_files[gmt][0]

def read_gmt(fname, sep='\t', min_g=0, max_g=5000):
    """
    Read GMT file into dictionary of gene_module:genes.\n
    min_g and max_g are optional gene set size filters.

    Args:
        fname (str): Path to gmt file
        sep (str): Separator used to read gmt file.
        min_g (int): Minimum of gene members in gene module.
        max_g (int): Maximum of gene members in gene module.
    Returns:
        OrderedDict: Dictionary of gene_module:genes.
    """
    dict_pathway = OrderedDict()
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            val = line.split(sep)
            if min_g <= len(val[2:]) <= max_g:
                genes_upper = [gene.upper() for gene in val[2:]]
                dict_pathway[val[0]] = genes_upper
    return dict_pathway


def get_dbhvg_simple(adata, gmt_path, project=None, min_g=12, max_g=5000, hvg_num=None):
    today = time.strftime('%Y%m%d', time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project

    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    if '.gmt' in gmt_path:
        gmt_path = gmt_path
    else:
        gmt_path = get_gmt(gmt_path)
    
    dict_pathway = read_gmt(gmt_path, min_g=min_g, max_g=max_g)
    all_genes = set()
    for genes in dict_pathway.values():
        all_genes.update(genes)
    
    adata = adata[:, adata.var_names.isin(all_genes)]
    print("check db adata shape: ", adata.shape)
    
    if hvg_num is None:
        sc.pp.highly_variable_genes(adata, subset=True)
    else:
        sc.pp.highly_variable_genes(adata, n_top_genes=hvg_num, subset=True)
    
    print("check preprocessed adata shape: ", adata.shape)
    return adata

def pathway_tokenizer(feature_list, dict_pathway, hvg_list=None, add_missing=0, fully_connected=True, to_tensor=False,
                     min_genes=12, max_genes=500, jaccard_threshold=0.5, plot_qc=True, save_path=None, retention_ratio=0.5):
    """
    Generate tokens of pathways/cell functions from expression.
    """
    assert type(dict_pathway) == OrderedDict
    
    p_mask = np.zeros((len(feature_list), len(dict_pathway)))
    pathway = list()
    for j, k in enumerate(dict_pathway.keys()):
        pathway.append(k)
        for i in range(p_mask.shape[0]):
            if hvg_list is None:
                if feature_list[i] in dict_pathway[k]:
                    p_mask[i,j] = 1.
            else:
                if (feature_list[i] in dict_pathway[k]) and (feature_list[i] in hvg_list):
                    p_mask[i,j] = 1.
    pathway = np.array(pathway)
    print(f"Initial tokenizer: {p_mask.shape[1]} pathways/tokens")
    
    pathway_sizes = np.sum(p_mask, axis=0)
    keep_mask = (max_genes >= pathway_sizes) & (pathway_sizes >= min_genes)
    p_mask = p_mask[:, keep_mask]
    print("check mask shape aftr min max hvg filtering: ",p_mask.shape)
    pathway = pathway[keep_mask]
    pathway_sizes = pathway_sizes[keep_mask]
    print(f"Filtered {np.sum(~keep_mask)} pathways outside size range base on presenting genes: [{min_genes}, {max_genes}]")

    n_pathways = p_mask.shape[1]
    jaccard_matrix = np.zeros((n_pathways, n_pathways))
    for i in range(n_pathways):
        genes_i = set(np.where(p_mask[:, i] == 1)[0])
        for j in range(i + 1, n_pathways):
            genes_j = set(np.where(p_mask[:, j] == 1)[0])
            intersection = len(genes_i & genes_j)
            union = len(genes_i | genes_j)
            jaccard = intersection / union if union > 0 else 0 
            jaccard_matrix[i, j] = jaccard
            jaccard_matrix[j, i] = jaccard
    np.fill_diagonal(jaccard_matrix, 1.0)
    distance_matrix = 1 - jaccard_matrix
    np.fill_diagonal(distance_matrix, 0)
    condensed_dist = squareform(distance_matrix)
    Z = linkage(condensed_dist, method='average')
    clusters = fcluster(Z, t=jaccard_threshold, criterion='distance')
    cluster_info = {}
    is_representative = np.zeros(len(pathway), dtype=bool)
    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_pathways = pathway[cluster_indices]  
        if len(cluster_indices) == 1:
            is_representative[cluster_indices[0]] = True
            representative_idx = cluster_indices[0]
            avg_similarity = 1.0  
        else:
            cluster_jaccard = jaccard_matrix[np.ix_(cluster_indices, cluster_indices)]
            avg_similarity = np.mean(cluster_jaccard, axis=1)
            best_idx = np.argmax(avg_similarity)
            representative_idx = cluster_indices[best_idx]
            is_representative[representative_idx] = True
            
        cluster_info[f'Cluster_{cluster_id}'] = {
            'pathways': cluster_pathways.tolist() if isinstance(cluster_pathways, np.ndarray) else cluster_pathways,
            'size': len(cluster_indices),
            'avg_similarity': float(np.mean(avg_similarity)) if len(cluster_indices) > 1 else 1.0,
            'representative': pathway[representative_idx],
            'representative_n_genes': int(pathway_sizes[representative_idx]),
            'avg_pathway_size': float(np.mean(pathway_sizes[cluster_indices]))
        }
    pre_pathway_summary = pd.DataFrame(cluster_info).T
    p_mask_sel = p_mask[:, is_representative]
    pathway_sel = pathway[is_representative]
    pathway_sizes_sel = pathway_sizes[is_representative]
    jaccard_matrix_sel = jaccard_matrix[np.ix_(is_representative, is_representative)]
    print(f"Selected {np.sum(is_representative)} representatives from {len(np.unique(clusters))} clusters")
    
    if plot_qc:
        import matplotlib.pyplot as plt
        original_sizes = [len(dict_pathway.get(pw, [])) for pw in pathway_sel]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

        ax1.hist(pathway_sizes_sel, bins=30, alpha=0.7)
        ax1.set_xlabel('Number of HVG genes per pathway')
        ax1.set_ylabel('Count')
        ax1.set_title(f'Selected Pathway Sizes (n={len(pathway_sizes_sel)})')
        ax1.axvline(np.mean(pathway_sizes_sel), color='red', linestyle='--', label=f'Mean={np.mean(pathway_sizes_sel):.1f}')
        ax1.legend()
        
        retention_ratios = np.array(pathway_sizes_sel) / np.array(original_sizes)
        colors = ['grey' if r < 0.25 else 'lightblue' if r < 0.5 else 'orange' if r < 0.75 else 'red' for r in retention_ratios]
        ax2.scatter(original_sizes, pathway_sizes_sel, c=colors, alpha=0.6)
        ax2.plot([0, max(original_sizes)], [0, max(original_sizes)], 'k--', alpha=0.3)
        ax2.set_xlabel('Original pathway size')
        ax2.set_ylabel('HVG-filtered size')
        ax2.set_title('HVG Retention in Pathways')
        
        from matplotlib.patches import Patch
        n1 = sum(r < 0.25 for r in retention_ratios)
        n2 = sum(0.25 <= r < 0.5 for r in retention_ratios)
        n3 = sum(0.5 <= r < 0.75 for r in retention_ratios)
        n4 = sum(r >= 0.75 for r in retention_ratios)
        
        legend_elements = [Patch(facecolor='grey', label=f'0-25% (n={n1})'),
                         Patch(facecolor='lightblue', label=f'25-50% (n={n2})'),
                         Patch(facecolor='orange', label=f'50-75% (n={n3})'),
                         Patch(facecolor='red', label=f'75-100% (n={n4})')]
        ax2.legend(handles=legend_elements, title='Retention', loc='upper left')
        
        filtered_sizes = np.array(pathway_sizes_sel)[retention_ratios >= retention_ratio]
        ax3.hist(filtered_sizes, bins=30, alpha=0.7, color='green')
        ax3.set_xlabel('Number of HVG genes per pathway')
        ax3.set_ylabel('Count')
        ax3.set_title(f'After {retention_ratio*100:.0f}% Retention Filter (n={len(filtered_sizes)})')
        ax3.axvline(np.mean(filtered_sizes), color='red', linestyle='--', label=f'Mean={np.mean(filtered_sizes):.1f}')
        ax3.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(os.path.join(save_path, 'selected_pathway_sizes.png'), dpi=150)
        plt.close()
        print(f"Mean pathway size: {np.mean(pathway_sizes_sel):.1f} ± {np.std(pathway_sizes_sel):.1f}")
        print(f"Retention categories: 0-25%: {n1}, 25-50%: {n2}, 50-75%: {n3}, 75-100%: {n4}")
        
        good_retention = retention_ratios >= retention_ratio
        p_mask_sel = p_mask_sel[:, good_retention]
        pathway_sel = pathway_sel[good_retention]
        n_removed = sum(retention_ratios < retention_ratio)
        print(f"Removed {n_removed} pathways with <{retention_ratio*100:.0f}% retention")

    if add_missing:
        n = 1 if type(add_missing)==bool else add_missing
        if not fully_connected:
            idx_0 = np.where(np.sum(p_mask_sel, axis=1)==0)
            vec = np.zeros((p_mask_sel.shape[0],n))
            vec[idx_0,:] = 1.
        else:
            vec = np.ones((p_mask_sel.shape[0], n))
        p_mask_sel = np.hstack((p_mask_sel, vec))
        new_pathways = np.array(['node %d' % i for i in range(n)])
        pathway_sel = np.concatenate((pathway_sel, new_pathways))
    
    print(f"Final tokenizer: {p_mask_sel.shape[1]} pathways/tokens, with pathway mask shape: {p_mask_sel.shape}")
    
    if to_tensor:
        p_mask_sel = torch.Tensor(p_mask_sel)
    
    return p_mask_sel, pathway_sel, jaccard_matrix, jaccard_matrix_sel, pre_pathway_summary


def KL_loss(mu, logvar, beta=1):
        total_kld = -0.5 * torch.mean(1 + logvar - mu**2 -  logvar.exp())
        return  beta*total_kld

def train_one_epoch(model, optimizer, data_loader, device, epoch, total_epochs, lambda1=1.0, lambda2=1.0, beta=1.0, anneal=True):
    """
    Train the model and update weights.
    """
    model.train()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1).to(device)
    accu_loss = torch.zeros(1).to(device)
    accu_loss_vae1 = torch.zeros(1).to(device)
    accu_loss_vae2 = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()
    sample_num = 0

    iter_rna_loader = tqdm(data_loader)
    for batch_idx, data in enumerate(iter_rna_loader):
        exp, label = data
        sample_num += exp.shape[0]
        pred, recon, mu, var, cls_latent, rec_latent, latent, attn_weights, bottleneck_weight = model(exp.to(device))
        assert torch.all(label >= 0) and torch.all(label < pred.size(1)), \
            f"Labels should be between 0 and {pred.size(1)-1}. Found labels outside this range."
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, label.to(device)).sum()

        loss_cls = loss_function(pred, label.to(device))
        loss_mse = (exp.to(device) - recon) ** 2  
        loss_mse = loss_mse.mean()
        kl_annealing_factor = min(1, epoch / (total_epochs * 0.5))  
        if anneal:
            loss_sam = kl_annealing_factor * (1 / exp.shape[1] * KL_loss(mu, var, beta=beta))
        else:
            loss_sam = 1 / exp.shape[1] * KL_loss(mu, var, beta=beta)
        loss_total = lambda1 * (loss_mse + loss_sam) + lambda2 * loss_cls
        loss_total.backward(retain_graph=True)

        accu_loss += loss_cls.detach()
        accu_loss_vae1 += loss_mse.detach()
        accu_loss_vae2 += (1 / exp.shape[1] * KL_loss(mu, var, beta=beta)).detach()
        iter_rna_loader.desc = "[train epoch {}] mse loss: {:.3f}, sam loss: {:.5f}, cls loss: {:.3f}, acc: {:.3f}, lr: {:.8f}".format(epoch,
                                                                                                                                        accu_loss_vae1.item() / (batch_idx + 1),
                                                                                                                                        accu_loss_vae2.item() / (batch_idx + 1),
                                                                                                                                        accu_loss.item() / (batch_idx + 1),
                                                                                                                                        accu_num.item() / sample_num, optimizer.param_groups[0]['lr'])
        if not torch.isfinite(loss_total):
            print('WARNING: non-finite loss, ending training ', loss_total)
            sys.exit(1)
        optimizer.step()
        optimizer.zero_grad()
    return accu_loss_vae1.item() / (batch_idx + 1), accu_loss_vae2.item() / (batch_idx + 1), accu_loss.item() / (batch_idx + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate_celltype(model, data_loader, num_classes, device, lambda1=1, lambda2=1):
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accu_loss = torch.zeros(1).to(device)
    accu_loss_vae1 = torch.zeros(1).to(device)
    accu_loss_vae2 = torch.zeros(1).to(device)
    accu_num = torch.zeros(num_classes).to(device)
    sample_num = torch.zeros(num_classes).to(device)

    iter_rna_loader = tqdm(data_loader)
    for batch_idx, data in enumerate(iter_rna_loader):
        exp, label = data
        pred, recon, mu, var, cls_latent, rec_latent, latent, attn_weights, attn_weights_sim = model(exp.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        for j in range(num_classes):
            label_mask = torch.eq(label.to(device), j)
            accu_mask = torch.eq(pred_classes, label.to(device))
            accu_num[j] += (label_mask & accu_mask).sum().item()
            sample_num[j] += (label.to(device) == j).sum().item()

        loss_cls = loss_function(pred, label.to(device))
        loss_mse = (exp.to(device) - recon) ** 2
        loss_mse = loss_mse.mean()
        loss_sam = 1 / exp.shape[1] * (KL_loss(mu, var))

        accu_loss += loss_cls.detach()
        accu_loss_vae1 += loss_mse.detach()
        accu_loss_vae2 += loss_sam.detach()
        iter_rna_loader.desc = "[Evaluation celltype-wise]  mse loss: {:.3f}, sam loss: {:.5f}, cls loss: {:.3f}, acc: {:.3f}".format(
            accu_loss_vae1.item() / (batch_idx + 1),
            accu_loss_vae2.item() / (batch_idx + 1),
            accu_loss.item() / (batch_idx + 1),
            accu_num.sum().item() / sample_num.sum().item())
    for j in range(num_classes):
        print("Evaluation celltype-wise: cell type {}, \t prec: {}, \t num: {}".format(j, accu_num[j] / sample_num[j], sample_num[j]))

def fit_model(adata, gmt_path, project=None, hvg_list_path=None, autobalance=False, pre_weights='', label_name='Celltype', 
              min_g=12, max_g=300, min_hvg_genes=12, max_hvg_genes=500, max_gs=300, mask_ratio=0.015, n_fc_tokens=0, batch_size=8, 
              embed_dim=48, z_dim=48, depth=4, num_heads=4, lr=0.001, epochs=10, lrf=0.1, lambda1=0.9, lambda2=0.1, 
              beta=1.0, drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0., attn_temperature=1.0, anneal=True, jaccard_threshold=0.5, retention_ratio=0.5):
    device = 'cuda'
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(device)
    today = time.strftime('%Y%m%d',time.localtime(time.time()))
    project = project or gmt_path.replace('.gmt','')+'_%s'%today
    project_path = os.getcwd()+'/%s'%project
    if os.path.exists(project_path) is False:
        os.makedirs(project_path)
    if hvg_list_path is None:
        hvg_list = None
        print("no database hvg is loaded for pathway mask, use all genes in the adata...")
    else:
        hvg_list = pd.read_csv(hvg_list_path)
        hvg_list = hvg_list.iloc[:, 0].tolist()
        print("{} hvgs loaded!".format(len(hvg_list)))
    train_loader, test_loader, training_iters, inverse, genes, label_max = prepareDataSet(adata=adata, label_name=label_name, batch_size=batch_size, autobalance=autobalance,  project_path=project_path)
    if gmt_path is None:
        mask = np.random.binomial(1,mask_ratio,size=(len(genes), max_gs))
        pathway = list()
        for i in range(max_gs):
            x = 'node %d' % i
            pathway.append(x)
        print('Full connection!')
    else:
        if '.gmt' in gmt_path:
            gmt_path = gmt_path
        else:
            gmt_path = get_gmt(gmt_path)
        reactome_dict = read_gmt(gmt_path, min_g=min_g, max_g=max_g)
        print("number of loaded pathway is: ", len(reactome_dict))
        mask,pathway,jaccard_matrix,jaccard_matrix_sel,pre_pathway_summary = pathway_tokenizer(feature_list=genes,dict_pathway=reactome_dict, hvg_list = hvg_list,add_missing=n_fc_tokens, 
        fully_connected=True,min_genes=min_hvg_genes, max_genes=max_hvg_genes, jaccard_threshold=jaccard_threshold,plot_qc=True, save_path=project_path, retention_ratio=retention_ratio)
        pre_pathway_summary.to_csv(project_path+'/pre_pathway_summary.csv')
        pathway = pathway[np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):][::-1]]
        mask = mask[:,np.argsort(np.sum(mask,axis=0))[-min(max_gs,mask.shape[1]):][::-1]]
        print('Mask loaded with shape: ', mask.shape)
    np.save(project_path+'/mask.npy',mask)
    pd.DataFrame(pathway).to_csv(project_path+'/pathway.csv') 
    pd.DataFrame(inverse,columns=[label_name]).to_csv(project_path+'/label_dictionary.csv', quoting=None)
    num_classes = np.int64(label_max+1)
    print("the number of classes in the dataset: ",num_classes)
    model = create_model(num_classes=num_classes, num_genes=len(genes), mask=mask, embed_dim=embed_dim, z_dim=z_dim, 
                        depth=depth, num_heads=num_heads, drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, 
                        drop_path_ratio=drop_path_ratio, attn_temperature=attn_temperature).to(device)
    torch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert mask.shape[0] == len(genes), "mask shape does not match genes shape, please check!"
    print(model)
    print("model created with {} trainable parameters!".format(torch_total_params))
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)
    if pre_weights != "":
        assert os.path.exists(pre_weights), "pre_weights file: '{}' not exist.".format(pre_weights)
        preweights_dict = torch.load(pre_weights, map_location=device)
        print(model.load_state_dict(preweights_dict, strict=False))
    print('Model builded!')
    pg = [p for p in model.parameters() if p.requires_grad]  
    optimizer = optim.SGD(pg, lr=lr, momentum=0.9, weight_decay=5E-5) 
    lf = lambda x: ((1 + math.cos(x * math.pi / (epochs-1))) / 2) * (1 - lrf) + lrf  
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    log_file = open(project_path+'/train_log.txt', 'w', newline='\n')
    writer = csv.writer(log_file)
    writer.writerow(['epoch', 'mse_loss', 'sampling_loss', 'classification_loss', 'accuracy', 'lr'])
    log_file.close()
    for epoch in range(epochs):
        mse_loss, sam_loss, cls_loss, accu = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,epoch=epoch,total_epochs=epochs, lambda1=lambda1, lambda2=lambda2,beta=beta,anneal=anneal)
        log_file = open(project_path+'/train_log.txt', 'a', newline='\n')
        writer = csv.writer(log_file)
        writer.writerow([epoch, mse_loss, sam_loss, cls_loss, accu, optimizer.param_groups[0]['lr']])
        log_file.close()
        scheduler.step()
    log_file.close()
    if platform.system().lower() == 'windows':
            torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), 
                    project_path + f"/model-{epoch}.pth")
    else:
        torch.save(model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(), 
                os.path.join(project_path, f"model-{epoch}.pth"))
    evaluate_celltype(model=model, data_loader=test_loader, num_classes=num_classes, device=device,lambda1=lambda1, lambda2=lambda2)
    print('Stage1 training finished!')


