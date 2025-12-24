import math
from numpy.random import beta
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from einops import einsum,rearrange

def cross_entroy(inputs:torch.Tensor,targets:torch.Tensor)->torch.Tensor:
    """
        inputs.shape = [... vocab_size]
        targets.shape = [vocab_size]
    """

    max_logits = torch.max(inputs,dim=-1,keepdim=True).values
    logits = inputs-max_logits

    log_sum_exp = torch.log(torch.sum(torch.exp(logits),dim=-1))+max_logits.squeeze(-1)

    logits_t = torch.gather(inputs,dim=-1,index=targets.unsqueeze(-1)).squeeze(-1)

    return torch.mean(log_sum_exp-logits_t)
    
class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2) -> None:
        default = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay)
        super().__init__(params, default)
    
    @torch.no_grad()
    def step(self,closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1,beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]
                if len(state)==0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                
                exp_avg,exp_avg_sq = state['exp_avg'],state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad,alpha=1-beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad,grad,value=1-beta2)

                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t

                alpha_t = lr * (math.sqrt(bias_correction2)/bias_correction1)

                deomn = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg,deomn,value=-alpha_t)

                if weight_decay != 0:
                    p.add_(p,alpha=-lr*weight_decay)
        return loss

def get_lr_consine_schedule(it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int)->float:

    if it<warmup_iters:
        return max_learning_rate * it/warmup_iters
    
    if it > cosine_cycle_iters:
        return min_learning_rate
    
    progress = (it - warmup_iters)/(cosine_cycle_iters - warmup_iters)
    decay_ratio = 0.5 * (1 + math.cos(math.pi * progress))

    return min_learning_rate + (max_learning_rate - min_learning_rate) * decay_ratio

def clip_grad_norm(parameters, max_norm: float, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    if not isinstance(parameters,torch.Tensor):
        parameters = [parameters]
    grads = [p.grad for p in parameters if p.grad is not None]
    if len(grads)==0:
        return torch.tensor(0.)

    eps = 1e-8
    total_norm = torch.linalg.norm(torch.stack([torch.linalg.norm(g.detach(),ord=norm_type) for g in grads]),ord=norm_type)
    clip_norm = max_norm/(eps+total_norm)

    if clip_norm<1:
        for g in grads:
            g.detach().mul_(clip_norm.to(g.device))

def save_checkpoint( model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out):

    model_weight = model.state_dict()
    optimizer_weight = optimizer.state_dict()

    checkpoint = {
        "model_weight":model_weight,
        "optimizer_states":optimizer_weight,
        "iteration":iteration
    }
    torch.save(obj=checkpoint,f=out)

def load_checkpoint(src,model:torch.nn.Module,optimizer:torch.optim.Optimizer)->int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_weight"])
    optimizer.load_state_dict(checkpoint["optimizer_states"])
    return checkpoint["iteration"]

def get_batch(dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    dtype: torch.dtype = torch.long,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    total_len = dataset.shape[0]
    max_start_index = total_len - context_length

    starts = torch.randint(low=0,high=max_start_index,size=(batch_size,))
    label_batch = torch.empty((batch_size,context_length),dtype=dtype)
    data_batch = torch.empty((batch_size,context_length),dtype=dtype)

    for i,start in enumerate(starts):
        data_batch[i] = torch.tensor(dataset[start:start+context_length])
        label_batch[i] = torch.tensor(dataset[start+1,start+context_length+1])

    return data_batch.to(device),label_batch.to(device)

