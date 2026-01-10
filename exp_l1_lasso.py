import torch
import torch.nn as nn
from s_adam import SAdam
import matplotlib.pyplot as plt
import numpy as np

# Setup
torch.manual_seed(42)
D_in = 1000   # Dimensions
N = 500       # Samples
Sparsity = 0.95 # 95% weights are zero

# Synthetic Data
true_w = torch.zeros(D_in, 1)
idx = torch.randperm(D_in)[:int(D_in * (1-Sparsity))]
true_w[idx] = torch.randn(len(idx), 1)

X = torch.randn(N, D_in)
y = X @ true_w + 0.1 * torch.randn(N, 1)

# Model
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(D_in, 1, bias=False)
    def forward(self, x):
        return self.linear(x)

def train(opt_name, epochs=300):
    model = LinearRegression()
    criterion = nn.MSELoss()
    lambda_l1 = 0.5
    
    if opt_name == 'S-Adam':
        # Pass model_ref to SAdam for LGI computation
        opt = SAdam(model.parameters(), model_ref=model, lr=0.01, lambda_lgi=0.5)
    elif opt_name == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    losses = []
    sparsity_checks = []
    
    for epoch in range(epochs):
        def closure():
            opt.zero_grad()
            out = model(X)
            mse = criterion(out, y)
            l1 = lambda_l1 * torch.norm(model.linear.weight, 1)
            loss = mse + l1
            loss.backward()
            return loss

        # For S-Adam, we pass inputs to compute LGI
        if opt_name == 'S-Adam':
            loss = opt.step(closure, inputs=X, targets=y, criterion=criterion)
        else:
            loss = opt.step(closure)
            
        losses.append(loss.item())
        
        # Check pseudo-sparsity (weights < 1e-3)
        with torch.no_grad():
            sparse_count = torch.sum(torch.abs(model.linear.weight) < 1e-2).item()
            sparsity_checks.append(sparse_count)
            
    return losses, sparsity_checks

loss_adam, sparse_adam = train('Adam')
loss_sadam, sparse_sadam = train('S-Adam')

# Visualization
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss (MSE + L1)', color='tab:red')
ax1.plot(loss_adam, color='tab:red', alpha=0.5, label='Adam Loss')
ax1.plot(loss_sadam, color='tab:red', linestyle='--', label='S-Adam Loss')
ax1.tick_params(axis='y', labelcolor='tab:red')

ax2 = ax1.twinx()  
ax2.set_ylabel('Zero Parameters Count (Approx)', color='tab:blue')  
ax2.plot(sparse_adam, color='tab:blue', alpha=0.5, label='Adam Sparsity')
ax2.plot(sparse_sadam, color='tab:blue', linestyle='--', label='S-Adam Sparsity')
ax2.axhline(int(D_in * Sparsity), color='grey', linestyle=':', label='True Sparsity')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('L1 Regularized Regression: Stability & Sparsity')
fig.tight_layout()
plt.legend(loc='center right')
plt.savefig('l1_experiment.png')
plt.show()
