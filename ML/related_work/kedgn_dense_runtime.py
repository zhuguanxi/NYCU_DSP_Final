"""Dense, masked equivalent of the author's irregular graph cell for GPU."""

from types import MethodType

import torch
import torch.nn.functional as F


def dense_forward(self, obs_emb, observed_mask, lengths, avg_interval, var_plm_rep_tensor):
    batch,steps,nodes,features=obs_emb.shape
    device=obs_emb.device
    h=self.init_hidden_states(obs_emb)
    output=torch.zeros_like(h)
    eye=torch.eye(nodes,device=device)[None]
    # The same variable text embeddings are shared across every encounter.
    query=self.projection_f(var_plm_rep_tensor)
    node_emb=F.normalize(self.projection_g(var_plm_rep_tensor),p=2,dim=-1)
    adj=torch.softmax(node_emb@node_emb.T,dim=-1)[None]
    weights={}
    for key in ('reset_gate','update_gate','candidate_gate'):
        gate=getattr(self.gated_update,key)
        weights[key]=(torch.einsum('vq,qio->vio',query,gate.W_1),query@gate.b_1)
    def project(key,value):
        weight,bias=weights[key]
        return torch.einsum('bvi,vio->bvo',value,weight)+bias[None]
    counts=observed_mask.sum(dim=1)
    maximum=int(lengths.max().item())
    for step in range(maximum):
        mask=observed_mask[:,step]
        observed=mask.bool()[...,None]
        rarity=self.rarity_alpha*torch.tanh(avg_interval[:,step]/(counts+1))
        distance=(rarity[:,:,None]-rarity[:,None,:]).abs()
        score=-self.rarity_W[None]*distance
        both=mask[:,:,None]*mask[:,None,:]
        current=adj*(1+score)*both*(1-eye)+eye
        x=torch.cat([obs_emb[:,step],rarity[...,None]],dim=-1)
        combined=current@torch.cat([x,h],dim=-1)
        reset=torch.sigmoid(project('reset_gate',combined))
        update=torch.sigmoid(project('update_gate',combined))
        reset_h=torch.where(observed,reset*h,h)
        candidate=torch.tanh(project('candidate_gate',torch.cat([x,reset_h],dim=-1)))
        next_h=(1-update)*reset_h+update*candidate
        h=torch.where(observed,next_h,reset_h)
        finished=(step==(lengths[:,0]-1))[:,None,None]
        output=torch.where(finished,h,output)
    return output


def install(model):
    model.GCRNN.forward=MethodType(dense_forward,model.GCRNN)
    return model
