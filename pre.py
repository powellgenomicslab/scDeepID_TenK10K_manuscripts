import os
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
import scanpy as sc
import anndata as ad
from .scDeepID_model import scDeepID_model as create_model
from .train import *

def predict(adata, model_weight_path, project, label_name=None,
              cutoff=0.1, n_fc_tokens=0, batch_size=50, embed_dim=48, z_dim=48, depth=4, num_heads=4,
              attn_temperature=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    project_path = os.getcwd() + '/%s' % project
    mask_path = project_path + '/mask.npy'
    mask = np.load(mask_path)
    pathway = pd.read_csv(project_path + '/pathway.csv', index_col=0)
    dictionary = pd.read_table(project_path + '/label_dictionary.csv', sep=',', header=0, index_col=0)
    n_c = len(dictionary)
    label_name_dic = dictionary.columns[0]
    dictionary.loc[(dictionary.shape[0])] = 'Unknown'
    dic = {}
    for i in range(len(dictionary)):
        dic[i] = dictionary[label_name_dic][i]
    train_loader, test_loader, training_iters, inverse, genes, label_max = prepareDataSet(adata=adata, label_name=label_name, batch_size=batch_size, autobalance=False, project_path=project_path)
    model = create_model(num_classes=n_c, num_genes=len(genes), mask=mask, embed_dim=embed_dim, z_dim=z_dim, 
                        depth=depth, num_heads=num_heads,
                        drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                        attn_temperature=attn_temperature).to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    parm = {}
    for name, parameters in model.named_parameters():
        parm[name] = parameters.detach().cpu().numpy()
    gene_pathway_weights = parm['feature_embed.fe.weight']
    gene_pathway_weights = gene_pathway_weights.reshape((int(gene_pathway_weights.shape[0] / embed_dim), embed_dim, adata.shape[1]))
    gene_pathway_weights = abs(gene_pathway_weights)
    gene_pathway_weights = np.mean(gene_pathway_weights, axis=1)
    gene_pathway_weights = pd.DataFrame(gene_pathway_weights)
    gene_pathway_weights.columns = adata.var_names
    gene_pathway_weights.index = pathway['0']
    gene_pathway_weights.to_csv(project_path + '/gene_pathway_weights.csv')

    test_loader = tqdm(test_loader)

    with torch.no_grad():
        adata_list = []
        adata_list_real = []
        adata_list_attn = []
        for batch_idx, data in enumerate(test_loader):
            exp, label = data
            pre, recon, mu, var, cls_latent, rec_latent, latent, attn, bottle_weight = model(exp.to(device))
            pre = pre.cpu()
            pre = F.softmax(pre, 1)
            predict_class = np.empty(shape=0)
            pre_class = np.empty(shape=0)
            for i in range(len(pre)):
                if torch.max(pre, dim=1)[0][i] >= cutoff:
                    predict_class = np.r_[predict_class, torch.max(pre, dim=1)[1][i].numpy()]
                else:
                    predict_class = np.r_[predict_class, n_c]
                pre_class = np.r_[pre_class, torch.max(pre, dim=1)[0][i]]
            meta = np.c_[predict_class, pre_class]
            meta = pd.DataFrame(meta)
            meta.columns = ['pred', 'prob']
            meta.index = meta.index.astype('str')
            att = attn.cpu().numpy()
            if n_fc_tokens > 0:
                att = att[:, 0:(att.shape[1] - n_fc_tokens)]
            att = att.astype('float32')

            if n_fc_tokens > 0:
                varinfo = pd.DataFrame(pathway.iloc[0:len(pathway) - n_fc_tokens, 0].values, index=pathway.iloc[0:len(pathway) - n_fc_tokens, 0], columns=['pathway_index'])
            else:
                varinfo = pd.DataFrame(pathway.iloc[:, 0].values, index=pathway.iloc[:, 0], columns=['pathway_index'])
            adata_list_attn.append(sc.AnnData(att, obs=meta, var=varinfo))
            adata_list.append(sc.AnnData(recon.cpu().numpy(), obs=meta, var=adata.var))
            adata_list_real.append(sc.AnnData(exp.cpu().numpy(), obs=meta, var=adata.var))

    new = ad.concat(adata_list, index_unique="_")
    new.obs.index = adata.obs.index
    new.obs['pred'] = new.obs['pred'].map(dic)
    new.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values

    attn = ad.concat(adata_list_attn,index_unique="_")
    attn.obs.index = adata.obs.index
    attn.obs['pred'] = attn.obs['pred'].map(dic)
    attn.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    real = ad.concat(adata_list_real, index_unique="_")
    real.obs.index = adata.obs.index
    real.obs[adata.obs.columns] = adata.obs[adata.obs.columns].values
    new.X[new.X < real.X.min()] = real.X.min()
    new.X[new.X > real.X.max()] = real.X.max()  
    res = [real, new, attn]
    return res