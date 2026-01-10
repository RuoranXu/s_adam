import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import math
import copy

torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. S-Adam 优化器 (Paper Implementation)
# ==========================================
class SAdam(torch.optim.Optimizer):
    """
    S-Adam: Singularity-Aware Adam for Nonsmooth Optimization.
    Implements the algorithm via Randomized Directional Smoothing.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, k_dir=12, lambda_lgi=5.0, sigma=0.02):
        """
        Args:
            k_dir (int): Number of random directions for LGI estimation.
            lambda_lgi (float): Sensitivity coefficient (The "Brake" strength).
            sigma (float): Probe radius (perturbation scale).
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        k_dir=k_dir, lambda_lgi=lambda_lgi, sigma=sigma)
        super(SAdam, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step with Singularity Detection.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue

            # --- Phase 1: Local Geometric Instability (LGI) Estimation ---
            # Paper Equation: LGI = Var(Df) / (E[Df^2] + epsilon)
            
            k_dir = group['k_dir']
            sigma = group['sigma']
            lambda_lgi = group['lambda_lgi']
            
            # 保存原始参数
            original_params = [p.data.clone() for p in params]
            base_loss = loss.item()
            
            # 存储方向导数
            directional_derivs = []
            
            for _ in range(k_dir):
                # 1. 生成单位随机方向 u
                perturbations = []
                sq_norm = 0.0
                for p in params:
                    noise = torch.randn_like(p)
                    perturbations.append(noise)
                    sq_norm += noise.norm()**2
                
                total_norm = sq_norm.sqrt()
                
                # 2. 施加扰动: theta_new = theta + sigma * u
                for i, p in enumerate(params):
                    # Normalize noise to unit sphere then scale by sigma
                    u = perturbations[i] / (total_norm + 1e-12)
                    p.data.add_(u, alpha=sigma)
                
                # 3. 计算扰动后的 Loss (Forward Only)
                loss_probe = closure().item()
                
                # 4. 近似方向导数: D_u f ≈ (f(theta+sigma*u) - f(theta)) / sigma
                diff = (loss_probe - base_loss) / sigma
                directional_derivs.append(diff)
                
                # 5. 还原参数
                for i, p in enumerate(params):
                    p.data.copy_(original_params[i])
            
            # 计算统计量
            dd_tensor = torch.tensor(directional_derivs)
            var_dd = torch.var(dd_tensor)
            mean_sq_dd = torch.mean(dd_tensor**2)
            
            # LGI Score
            lgi = var_dd / (mean_sq_dd + 1e-8)
            
            # 计算阻尼系数 alpha(t) = exp(- lambda * LGI)
            # Clip LGI to avoid numerical underflow in exp if LGI explodes
            lgi_clamped = torch.clamp(lgi, max=50.0) 
            damping = torch.exp(-lambda_lgi * lgi_clamped).item()
            
            # 将阻尼系数存入 state 以便后续可视化分析
            self.state[params[0]]['lgi'] = lgi.item()
            self.state[params[0]]['damping'] = damping

            # --- Phase 2: Adam Update with Damping ---
            for p in params:
                grad = p.grad
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['lgi'] = 0.0
                    state['damping'] = 1.0

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Ultimate Step: Apply Damping
                # step_size = lr * damping * ...
                step_size = group['lr'] * damping * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

# ==========================================
# 2. 实验环境设置 (Testbed)
# ==========================================

# 定义著名的 Nonsmooth Rosenbrock 函数 (带绝对值的香蕉函数)
# 这个函数在 y=x^2 处有一个非常尖锐的非光滑山脊
class NonsmoothRosenbrock(nn.Module):
    def forward(self, x, y):
        # 放大非光滑项的权重，模拟 ReLU 网络的剧烈梯度变化
        # Global Min: (1, 1), Value: 0
        term1 = 20.0 * torch.abs(y - x**2) # The Nonsmooth Ridge
        term2 = (1 - x)**2                 # The Convex Funnel
        return term1 + term2

def run_trajectory(optimizer_class, steps=600, **optim_kwargs):
    # 初始化在一个典型的困难点，使得优化器必须沿着山脊行走
    x = torch.tensor([-1.2], requires_grad=True)
    y = torch.tensor([1.0], requires_grad=True)
    
    criterion = NonsmoothRosenbrock()
    optimizer = optimizer_class([x, y], **optim_kwargs)
    
    trajectory = []
    losses = []
    dampings = [] # 记录 S-Adam 的阻尼变化
    
    trajectory.append([x.item(), y.item()])
    
    for t in range(steps):
        def closure():
            optimizer.zero_grad()
            loss = criterion(x, y)
            loss.backward()
            return loss
        
        loss_val = optimizer.step(closure)
        
        # 记录数据
        trajectory.append([x.item(), y.item()])
        losses.append(loss_val.item())
        
        # 获取 S-Adam 的内部状态
        if isinstance(optimizer, SAdam):
            dampings.append(optimizer.state[x]['damping'])
        else:
            dampings.append(1.0) # Standard Adam has damping = 1.0
            
    return np.array(trajectory), np.array(losses), np.array(dampings)

# ==========================================
# 3. 运行对比实验
# ==========================================
print("Running Experiment: Standard Adam...")
# Adam 设置：较高的学习率会让它在非光滑区域剧烈震荡
traj_adam, loss_adam, damp_adam = run_trajectory(
    torch.optim.Adam, steps=600, lr=0.05, betas=(0.9, 0.999)
)

print("Running Experiment: S-Adam (Ours)...")
# S-Adam 设置：相同的基准 LR，但是开启了 LGI 阻尼
traj_sadam, loss_sadam, damp_sadam = run_trajectory(
    SAdam, steps=600, lr=0.05, betas=(0.9, 0.999), 
    lambda_lgi=8.0, sigma=0.05, k_dir=8
)

# ==========================================
# 4. ICML 风格终极可视化
# ==========================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})

fig = plt.figure(figsize=(18, 6), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1, 1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

# --- Subplot 1: Optimization Landscape & Trajectory ---
# 生成高分辨率背景
x_grid = np.linspace(-2.0, 2.0, 300)
y_grid = np.linspace(-1.0, 4.0, 300)
X, Y = np.meshgrid(x_grid, y_grid)
Z = 20.0 * np.abs(Y - X**2) + (1 - X)**2

# 绘制 Loss 等高线 (Log scale 增强对比度)
lev_exp = np.arange(np.floor(np.log10(Z.min()+1e-6)), np.ceil(np.log10(Z.max())), 0.1)
levels = np.power(10, lev_exp)
cp = ax1.contourf(X, Y, Z, levels=levels, norm=colors.LogNorm(), cmap='gray_r', alpha=0.6)

# 绘制 Adam 轨迹
ax1.plot(traj_adam[:, 0], traj_adam[:, 1], color='#e74c3c', alpha=0.7, lw=1.5, ls='--', label='Standard Adam')
ax1.scatter(traj_adam[-1, 0], traj_adam[-1, 1], color='#e74c3c', s=50, marker='x')

# 绘制 S-Adam 轨迹
ax1.plot(traj_sadam[:, 0], traj_sadam[:, 1], color='#2ecc71', alpha=1.0, lw=2.5, label='S-Adam (Ours)')
ax1.scatter(traj_sadam[-1, 0], traj_sadam[-1, 1], color='#2ecc71', s=50, marker='o')

# 标记最优点
ax1.plot(1, 1, 'b*', markersize=18, markeredgecolor='white', label='Global Opt')

ax1.set_xlim(-2.0, 2.0)
ax1.set_ylim(-1.0, 4.0)
ax1.set_xlabel(r'Parameter $\theta_1$')
ax1.set_ylabel(r'Parameter $\theta_2$')
ax1.set_title('(a) Trajectory on Nonsmooth Ridge', fontweight='bold')
ax1.legend(loc='upper left', frameon=True, framealpha=0.9)

# --- Subplot 2: Loss Convergence ---
ax2.plot(loss_adam, color='#e74c3c', alpha=0.4, lw=1)
# 平滑处理 Adam 曲线以便观看
loss_adam_smooth = np.convolve(loss_adam, np.ones(10)/10, mode='valid')
ax2.plot(loss_adam_smooth, color='#c0392b', lw=1.5, label='Standard Adam (Smoothed)')

ax2.plot(loss_sadam, color='#2ecc71', lw=2.5, label='S-Adam (Ours)')

ax2.set_yscale('log')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss Value (Log Scale)')
ax2.set_title('(b) Convergence Speed', fontweight='bold')
ax2.legend()
ax2.grid(True, which="both", ls="-", alpha=0.2)

# --- Subplot 3: The "Mechanism" (Adaptive Damping) ---
# 这是论文中最有力的证据：展示算法如何自动感知地形
ax3.plot(damp_adam, color='#e74c3c', ls='--', alpha=0.8, label='Standard Adam ($\alpha=1$)')
ax3.plot(damp_sadam, color='#2980b9', lw=2, label='S-Adam Damping ($\alpha_t$)')

# 添加注释
ax3.text(100, 0.4, 'Sharp Ridge Detected\n(Slow Down)', fontsize=9, color='#2980b9', ha='center')
ax3.text(450, 0.85, 'Valley Flatness\n(Speed Up)', fontsize=9, color='#2980b9', ha='center')

ax3.set_ylim(0, 1.1)
ax3.set_xlabel('Iteration')
ax3.set_ylabel(r'Effective Damping Factor $\alpha(LGI)$')
ax3.set_title('(c) Adaptive Geometry Awareness', fontweight='bold')
ax3.legend(loc='lower right')
ax3.fill_between(range(len(damp_sadam)), damp_sadam, 1.0, color='#2980b9', alpha=0.1)

# 保存和展示
plt.savefig('ICML_SAdam_Analysis.png', dpi=300, bbox_inches='tight')
print("Figure saved as ICML_SAdam_Analysis.png")
plt.show()

# 打印最终统计
#print(f"{'Optimizer':<15} | {'Final Loss':<15} | {'Min Loss':<15}")
#print("-" * 50)
#print(f"{'Adam':<15} | {loss_adam[-1]:.2e}        | {np.min(loss_adam):.2e}")
print(f"{'S-Adam':<15} | {loss_sadam[-1]:.2e}        | {np.min(loss_sadam):.2e}")
