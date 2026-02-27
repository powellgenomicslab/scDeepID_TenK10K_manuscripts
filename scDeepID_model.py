from functools import partial
import torch
import torch.nn as nn
import copy
from .customized_linear import CustomizedLinear
from einops import rearrange
import numpy as np

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class FeatureEmbed(nn.Module):
    def __init__(self, num_genes, mask, embed_dim=192, fe_bias=True, norm_layer=None): 
        super().__init__()
        self.num_genes = num_genes
        self.num_patches = mask.shape[1]
        self.embed_dim = embed_dim
        mask = np.repeat(mask,embed_dim,axis=1)
        self.mask = mask
        self.fe = CustomizedLinear(self.mask)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        x = rearrange(self.fe(x), 'h (c w) -> h c w ', c=self.num_patches)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.,
                 attn_temperature=1.0):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.attn_temperature = attn_temperature
        self.attn_gradients = None
        
    def save_attn_gradients(self, grad):
        """Hook function to save attention gradients"""
        self.attn_gradients = grad
        
    def get_attn_gradients(self):
        """Get saved attention gradients"""
        return self.attn_gradients

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn / self.attn_temperature
        attn = attn.softmax(dim=-1)
        
        if x.requires_grad:
            attn.register_hook(self.save_attn_gradients)
        
        weights = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features 
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self,
                 dim, 
                 num_heads,
                 mlp_ratio=4., 
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0., 
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 attn_temperature=1.0):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio,
                              attn_temperature=attn_temperature)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)
    def forward(self, x):
        raw, weights = self.attn(self.norm1(x))
        x = x + self.drop_path(raw)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights

def attention_rollout(att_mat):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    att_mat = torch.stack(att_mat).squeeze(1)

    if att_mat.dim() == 4:
        att_mat = att_mat.unsqueeze(1)

    att_mat = torch.mean(att_mat, dim=2)
    residual_att = torch.eye(att_mat.size(3)).to(device)
    aug_att_mat = att_mat.to(device) + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

    v = joint_attentions[-1]
    v = v[:, 0, 1:] 
    return v

class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, mask, fe_bias=True,
                 embed_dim=64, z_dim=64, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=FeatureEmbed, norm_layer=None,
                 act_layer=None, attn_temperature=1.0):

        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_genes = num_genes
        self.z_dim = z_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.feature_embed = embed_layer(num_genes, mask = mask, embed_dim=embed_dim, fe_bias=fe_bias) 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer, attn_temperature=attn_temperature)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        self.fc_mean = nn.Sequential(
            nn.Linear(self.embed_dim, self.z_dim)
        )
        self.fc_var = nn.Sequential(
            nn.Linear(self.embed_dim, self.z_dim)
        )
        self.decoder_pred = nn.Linear(self.z_dim, self.num_genes)

        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.xavier_uniform_(self.cls_token)
        self.apply(_init_weights)

    def reparameterize(self, mu, logvar): 
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward_features(self, x):
        x = self.feature_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        attn_weights = []
        tem = x
        for layer_block in self.blocks:
            tem, weights = layer_block(tem)
            attn_weights.append(weights)
        x = self.norm(tem)

        attn_weights_all = attn_weights.copy()
        attn_weights = attention_rollout(attn_weights)
        if self.dist_token is None:
            return x[:, 0, :], x[:,1:,:], attn_weights, attn_weights_all
        else:
            return [x[:, 0, :], x[:, 1, :]], x[:,1:,:], attn_weights, attn_weights_all
        
    def forward(self, x):
        cls_latent, latent, attn_weights, attn_weights_all = self.forward_features(x)
        mu = self.fc_mean(cls_latent)
        var = self.fc_var(cls_latent)
        rec_latent = self.reparameterize(mu, var) 
        rec = self.decoder_pred(rec_latent)
        rec = torch.nn.functional.relu(rec)

        if self.head_dist is not None:
            cls_latent, cls_latent_dist = self.head(cls_latent[0]), self.head_dist(cls_latent[1])
            if self.training and not torch.jit.is_scripting():
                return cls_latent, cls_latent_dist
            else:
                return (cls_latent+cls_latent_dist) / 2
        else:
            pre = self.head(cls_latent) 

        if pre.dim() == 1:
            pre = pre.unsqueeze(0)

        return pre, rec, mu, var, cls_latent, rec_latent, latent, attn_weights, attn_weights_all

def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

def scDeepID_model(num_classes, num_genes, mask, embed_dim=48, z_dim=48, depth=4, num_heads=4,
                   drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
                   attn_temperature=1.0):
    model = Transformer(num_classes=num_classes,
                        num_genes=num_genes,
                        mask=mask,
                        embed_dim=embed_dim,
                        z_dim=z_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=drop_ratio,
                        attn_drop_ratio=attn_drop_ratio,
                        drop_path_ratio=drop_path_ratio,
                        attn_temperature=attn_temperature)
    return model

