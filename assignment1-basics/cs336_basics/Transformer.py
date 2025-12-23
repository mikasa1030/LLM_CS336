import math
import numbers
from turtle import forward
import torch
import torch.nn as nn


from einops import einsum,rearrange
from typing import Optional

class Linear(nn.Module):
    def __init__(self,in_features:int,out_features:int,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None)->None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.W = nn.Parameter(torch.empty(out_features,in_features,dtype=dtype,device=device))
    
        # Xavier初始化标准差
        std = (2/(in_features+out_features))**0.5 # 1. Xavier Normal
        # std = (6/(in_features+out_features))**0.5 2. Xavier Uniform
        a,b = -30/std,30/std
        nn.init.trunc_normal_(self.W,mean=0.0,std=std,a=a,b=b)

    def forward(self,x:torch.tensor)->torch.tensor:
        """
            x.shape:(... in_features)
        """
        return einsum(x,self.W,"... i,o i -> ... o")

class Embedding(nn.Module):
    def __init__(self,num_embeddings:int,embedding_dim:int,device:Optional[torch.device]=None,dtype:Optional[torch.dtype]=None)->None:
        super().__init__()
        self.embedding = nn.Parameter(torch.empty(num_embeddings,embedding_dim,device=device,dtype=dtype))
        nn.init.trunc_normal_(self.embedding,mean=0.0,std=1.0,a=-3.0,b=3.0)

    def forward(self,token_ids):
        return self.embedding[token_ids]

class SwiGLU(nn.Module):
    def __init__(self,d_model:int,d_ff:int,device: Optional[torch.device]=None,dtype: Optional[torch.dtype]=None)->None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.W1 = nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))
        self.W3 = nn.Parameter(torch.empty(d_ff,d_model,device=device,dtype=dtype))
        self.W2 = nn.Parameter(torch.empty(d_model,d_ff,device=device,dtype=dtype))

    def Switch(self,x: torch.tensor):
        return x*(1/(1+torch.exp(-x)))

    def forward(self,x: torch.tensor):
        """
            x.shape = (... d_model)
        """
        out1 = einsum(x,self.W1,"... i,o i -> ... o")
        out3 = einsum(x,self.W3,"... i,o i -> ... o")
        out1 = self.Switch(out1)
        out = einsum(out1*out3,self.W2,"... o,i o -> ... i")
        return out

def softmax(x: torch.Tensor):
    maxval = torch.max(x,dim=-1,keepdim=True).values
    exp_x = torch.exp(x-maxval)
    return exp_x/(torch.sum(exp_x,dim=-1,keepdim=True))


def scaled_dot_product_attention(Q:torch.tensor,K:torch.Tensor,V:torch.Tensor,mask:torch.Tensor)->torch.Tensor:
    d_k = Q.size(-1)
    atten:torch.Tensor = einsum(Q,K,"... q d_k,... k d_k -> ... q k")/math.sqrt(d_k)
    if mask is not None:
        atten = atten.masked_fill_(mask==False,float("-inf"))
    atten = softmax(atten)
    output = einsum(atten,V,"... q k,... k d_k -> ... q d_k")
    return output

class RoPE(nn.Module):
    def __init__(self,d_model:int,max_seqlen:int,theta:int=10000,device: Optional[torch.device]=None,dtype:Optional[torch.dtype]=None) -> None:
        super().__init__()
        inverse_freqs = 1/(theta**(torch.arange(0,d_model,2,device=device,dtype=dtype)/d_model))
        t = torch.arange(0,max_seqlen,device=device,dtype=dtype)

        freqs = einsum(t,inverse_freqs,"i,j -> i j")
        self.register_buffer("freqs_cos",freqs.cos())
        self.register_buffer("freqs_sin",freqs.sin())

    def forward(self,x: torch.Tensor,token_poistions:Optional[torch.Tensor]=None)->torch.Tensor:
        max_seqlen = x.size(-2)
        if token_poistions is None:
            cos = self.freqs_cos[:max_seqlen]
            sin = self.freqs_sin[:max_seqlen]
        else:
            cos = self.freqs_cos[token_poistions]
            sin = self.freqs_sin[token_poistions]
            if cos.ndim == x.ndim - 1:
                cos = cos.unsqueeze(-3)
                sin = sin.unsqueeze(-3)
        x1 = x[...,0::2]
        x2 = x[...,1::2]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos                                                                                                                                                                                                                                                                                                                                                                                                                                             

        rotated_x = torch.stack((rotated_x1,rotated_x2),dim=-1)

        return rearrange(rotated_x,"... d two -> ... (d two)")


class multihead_self_attention(nn.Module):
    def __init__(self,d_model:int,numheads:int,rope:Optional[RoPE]=None,device: Optional[torch.device]=None,dtype: Optional[torch.dtype]=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.numheads = numheads
        assert d_model % numheads == 0
        self.d_k = d_model//numheads
        self.rope = rope
        self.device = device
        self.dtype = dtype

        self.qkv_proj = Linear(d_model,3*d_model,device=device,dtype=dtype)
        self.o_proj = Linear(d_model,d_model,device=device,dtype=dtype)

     
    def forward(self,x:torch.Tensor,mask:Optional[torch.Tensor]=None,token_poisitions:Optional[torch.Tensor]=None)->torch.Tensor:
        qkv = self.qkv_proj(x)
        q,k,v = rearrange(qkv,"... s (three numheads dk) -> three ... numheads s dk",three=3,numheads=self.numheads)
        if self.rope:
            q = self.rope(q,token_poisitions)
            k = self.rope(k,token_poisitions)
        if mask is None:
            seqlen = x.size(-2)
            mask = torch.tril(torch.ones((seqlen,seqlen),device=x.device,dtype=torch.bool))
        
        atten_out = scaled_dot_product_attention(q,k,v,mask)
        atten_out = rearrange(atten_out,"... numheads s d -> ... s (numheads d)")
        return self.o_proj(atten_out)

class RMSNorm(nn.Module):
    def __init__(self,d_model:int,eps: float=1e-8,device: Optional[torch.device]=None,dtype: Optional[torch.dtype]=None) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.ones(d_model,device=device,dtype=dtype))
        self.eps = eps

    def forward(self,x:torch.Tensor):
        original_dtype = x.dtype
        x_float = x.to(torch.float32)

        inverse_x = torch.rsqrt(x_float.pow(2).mean(dim=-1,keepdim=True)+self.eps)

        norm_x = x*inverse_x*self.W
        return norm_x
        


    
        
        
        
        
        
        

