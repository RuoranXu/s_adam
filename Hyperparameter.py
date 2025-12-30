import torch
import numpy as np
import matplotlib.pyplot as plt
from module2 import SimpleFCN, NoisySGD, train, test  # 复用模块2的代码

# ===================== 1. 超参数网格设置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 步长η（对数间隔）
etas = np.logspace(-5, -1, 5)  # [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# 噪声系数k（σ = k·η^(2/3)）
k_list = [0.1, 0.5, 1, 2, 5]
n_trials = 3  # 简化实验次数（可改30）

# 存储结果：acc_grid[η_idx][k_idx] = 平均准确率
acc_grid = np.zeros((len(etas), len(k_list)))
std_grid = np.zeros((len(etas), len(k_list)))

# ===================== 2. 网格搜索实验 =====================
for eta_idx, eta in enumerate(etas):
    print(f"\nTesting eta = {eta:.5f}")
    sigma_base = eta ** (2/3)
    for k_idx, k in enumerate(k_list):
        sigma = k * sigma_base
        accs = []
        for trial in range(n_trials):
            model = SimpleFCN().to(device)
            optimizer = NoisySGD(model.parameters(), lr=eta, sigma=sigma)
            # 训练5轮（简化）
            train(model, optimizer, train_loader, device, epochs=5)
            acc = test(model, test_loader, device)
            accs.append(acc)
        # 记录均值和标准差
        acc_grid[eta_idx, k_idx] = np.mean(accs)
        std_grid[eta_idx, k_idx] = np.std(accs)

# ===================== 3. 绘制热力图 =====================
plt.figure(figsize=(10, 8))
# 热力图（平均准确率）
im = plt.imshow(acc_grid, cmap="viridis")
# 设置坐标轴标签
plt.xticks(range(len(k_list)), [f"k={k}" for k in k_list])
plt.yticks(range(len(etas)), [f"η={eta:.5f}" for eta in etas])
plt.xlabel("Noise Coefficient k (σ = k·η^(2/3))")
plt.ylabel("Learning Rate η")
plt.title("Test Accuracy Heatmap (Mean)")

# 添加数值标注
for i in range(len(etas)):
    for j in range(len(k_list)):
        text = plt.text(j, i, f"{acc_grid[i,j]:.1f}",
                       ha="center", va="center", color="white" if acc_grid[i,j] > 90 else "black")

# 颜色条
cbar = plt.colorbar(im)
cbar.set_label("Test Accuracy (%)")
plt.savefig("hyperparam_heatmap.png")
plt.show()

# ===================== 4. 分析最优参数区间 =====================
# 找到准确率最高的参数组合
max_acc_idx = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
best_eta = etas[max_acc_idx[0]]
best_k = k_list[max_acc_idx[1]]
best_acc = acc_grid[max_acc_idx]

print(f"\n最优参数组合：η={best_eta:.5f}, k={best_k}, 准确率={best_acc:.2f}%")
print(f"对应理论约束：σ = {best_k} · {best_eta:.5f}^(2/3) = {best_k * (best_eta**(2/3)):.6f}")

# 计算最优区间的鲁棒性（波动幅度）
# 找到准确率≥95%的区域
robust_region = (acc_grid >= 95)
if np.any(robust_region):
    robust_std = std_grid[robust_region]
    print(f"\n高准确率区域（≥95%）的准确率波动：均值±标准差 = {np.mean(robust_std):.2f}±{np.std(robust_std):.2f}")
else:
    print("\n无准确率≥95%的区域")
