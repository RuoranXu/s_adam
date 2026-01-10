import torch
from torch.optim.optimizer import Optimizer
import math
from torch.func import functional_call, jvp, vmap

class SAdam(Optimizer):
    """
    S-Adam: Singularity-Aware Adam
    Implements the algorithm described in 'Singularity-Aware Optimization'.
    
    Args:
        params: Parameters to optimize
        lr: Base learning rate
        betas: (beta1, beta2) for momentum
        eps: Epsilon for numerical stability
        k_dir: Number of random directions to sample (default: 4)
        lambda_lgi: Damping coefficient for LGI (default: 0.5)
        zeta: Small constant for LGI stability (default: 1e-6)
    """
    def __init__(self, params, model_ref=None, lr=1e-3, betas=(0.9, 0.999), 
                 eps=1e-8, weight_decay=0, k_dir=4, lambda_lgi=0.5, zeta=1e-6):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        
        # model_ref is required for functional_call to compute directional derivatives
        self.model_ref = model_ref
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        k_dir=k_dir, lambda_lgi=lambda_lgi, zeta=zeta)
        super(SAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, inputs=None, targets=None, criterion=None):
        """
        Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            inputs, targets, criterion: Required if using functional approach for LGI calculation.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_steps = []
            
            beta1, beta2 = group['betas']
            k_dir = group['k_dir']
            lambda_lgi = group['lambda_lgi']
            zeta = group['zeta']

            # 1. Standard Adam Preparation
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('SAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    state_steps.append(state['step'])
                    
                    state['step'] += 1

            # 2. Singularity Detection (LGI Calculation)
            # Only perform if we have model reference and data to compute functional JVP
            lgi_score = 0.0
            if self.model_ref is not None and inputs is not None and criterion is not None:
                lgi_score = self._compute_lgi(group['params'], inputs, targets, criterion, k_dir, zeta)

            # 3. Adaptive Damping Factor (Theorem 4.1)
            # alpha(t) = exp(-lambda * LGI)
            damping = math.exp(-lambda_lgi * lgi_score)
            
            # 4. Parameter Update
            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                step = state_steps[i]

                if group['weight_decay'] != 0:
                    grad = grad.add(param, alpha=group['weight_decay'])

                # Momentum Update
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                
                # Apply Damping to Step Size
                step_size = group['lr'] * damping * math.sqrt(bias_correction2) / bias_correction1
                
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    def _compute_lgi(self, params, inputs, targets, criterion, k, zeta):
        """
        Computes Local Geometric Instability (LGI) using torch.func.jvp and vmap.
        """
        # Prepare functional call
        params_dict = {k: v for k, v in self.model_ref.named_parameters() if v.requires_grad}
        buffers_dict = {k: v for k, v in self.model_ref.named_buffers()}
        
        def func_model(params_d):
            out = functional_call(self.model_ref, (params_d, buffers_dict), (inputs,))
            return criterion(out, targets)

        # Generate k random tangent vectors (normalized)
        # We flatten params to treat them as a single vector for direction generation
        flat_vs_list = []
        for _ in range(k):
            vs = {name: torch.randn_like(p) for name, p in params_dict.items()}
            # Normalize efficiently? For now, component-wise random is fine for "random direction"
            flat_vs_list.append(vs)
            
        # Parallel JVP computation
        # Input: params, Tangents: (k, params)
        # We use a wrapper to handle the dict structure in vmap
        
        # Trick: vmap over the 'tangents' argument
        def compute_jvp_single_dir(tangent):
            _, jvp_val = jvp(func_model, (params_dict,), (tangent,))
            return jvp_val

        # vmap allows computing JVP for k directions in ONE forward pass (theoretically)
        # Note: In practice, this batches the operations.
        # tangets_batch needs to be a dict of stacked tensors
        tangents_batch = {}
        for key in params_dict.keys():
            # Stack k tensors: [k, ...]
            tangents_batch[key] = torch.stack([v[key] for v in flat_vs_list])
            
        jvps = vmap(compute_jvp_single_dir)(tangents_batch) # Shape: [k, 1] or [k]
        
        # Calculate Variance (LGI)
        # LGI = Var(JVP) / (E[JVP^2] + zeta)
        var_jvp = torch.var(jvps)
        mean_sq_jvp = torch.mean(jvps**2)
        
        lgi = var_jvp / (mean_sq_jvp + zeta)
        return lgi.item()
