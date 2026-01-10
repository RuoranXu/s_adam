import torch
import numpy as np
import matplotlib.pyplot as plt
from s_adam import SAdam

# 1. Define Nonsmooth Landscapes
def nonsmooth_rosenbrock(x, y):
    # The valley is at y = x^2, but nonsmooth due to abs
    return 10 * torch.abs(y - x**2) + torch.abs(1 - x)

def train_trajectory(optimizer_cls, start_pos, steps=200, lr=0.05, **kwargs):
    pos = torch.tensor(start_pos, requires_grad=True, dtype=torch.float32)
    
    # Mock model for S-Adam (since it needs a structure for LGI)
    # For synthetic functions, we bypass the functional part in SAdam for simplicity
    # or implement a minimal wrapper.
    # HERE: We use standard optimization for simple vars. 
    # NOTE: SAdam in this script will use lgi=0 (standard Adam behavior) 
    # unless we manually compute LGI for this simple function.
    
    # Simulating S-Adam logic manually for this simple 2D function 
    # because the SAdam class above relies on nn.Module structure.
    
    path = []
    
    # Initialize momentum buffers
    m = torch.zeros_like(pos)
    v = torch.zeros_like(pos)
    beta1, beta2 = 0.9, 0.999
    
    for t in range(1, steps + 1):
        path.append(pos.detach().numpy().copy())
        
        # 1. Calculate Gradients
        loss = nonsmooth_rosenbrock(pos[0], pos[1])
        loss.backward()
        grad = pos.grad
        
        # 2. Simulate LGI Calculation (S-Adam Logic)
        lgi = 0.0
        if optimizer_cls == 'S-Adam':
            # Sample directions
            k = 5
            jvps = []
            for _ in range(k):
                d = torch.randn_like(pos)
                d = d / torch.norm(d)
                # Finite difference approx for direction derivative
                eps = 1e-4
                with torch.no_grad():
                    l_plus = nonsmooth_rosenbrock(pos[0] + eps*d[0], pos[1] + eps*d[1])
                    l_minus = nonsmooth_rosenbrock(pos[0], pos[1])
                    jvp_val = (l_plus - l_minus) / eps
                    jvps.append(jvp_val.item())
            jvps = np.array(jvps)
            lgi = np.var(jvps) / (np.mean(jvps**2) + 1e-6)
            
        # 3. Update (Manual Implementation for visibility)
        damping = np.exp(-1.0 * lgi) if optimizer_cls == 'S-Adam' else 1.0
        
        with torch.no_grad():
            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * grad**2
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            step_size = lr * damping / (torch.sqrt(v_hat) + 1e-8)
            pos -= step_size * m_hat
            
        pos.grad.zero_()
        
    return np.array(path)

# Run Experiment
start = [-1.5, 2.0]
path_adam = train_trajectory('Adam', start)
path_sadam = train_trajectory('S-Adam', start)

# Plotting
x = np.linspace(-2, 2, 200)
y = np.linspace(-1, 3, 200)
X, Y = np.meshgrid(x, y)
Z = 10 * np.abs(Y - X**2) + np.abs(1 - X)

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, np.log(Z + 1), levels=30, cmap='gray_r', alpha=0.6)
plt.plot(path_adam[:, 0], path_adam[:, 1], 'r.-', label='Standard Adam', linewidth=1)
plt.plot(path_sadam[:, 0], path_sadam[:, 1], 'g.-', label='S-Adam (Ours)', linewidth=2)
plt.plot(1, 1, 'b*', markersize=15, label='Global Opt')
plt.title('Traversal on Nonsmooth Rosenbrock Ridge')
plt.legend()
plt.savefig('synthetic_trajectory.png')
plt.show()
