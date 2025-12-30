import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ===================== 1. 定义非凸函数及梯度 =====================
def double_well(x):
    """双井函数: f(x) = x^4 - 2x^2，临界点：x=-1（极小）、x=0（鞍点）、x=1（极小）"""
    return x**4 - 2 * x**2

def double_well_gradient(x):
    """梯度：f’(x) = 4x^3 - 4x"""
    return 4 * x**3 - 4 * x

# ===================== 2. 实现带噪声的一阶优化器 =====================
def gd_sgd(x0, lr, sigma, max_iter, is_sgd=True):
    """
    梯度下降/随机梯度下降
    :param x0: 初始值
    :param lr: 步长η
    :param sigma: 噪声强度σ
    :param max_iter: 迭代次数
    :param is_sgd: 是否加噪声（SGD）
    :return: 轨迹列表、最终值
    """
    x = x0
    trajectory = [x]
    for _ in range(max_iter):
        grad = double_well_gradient(x)
        # 加性高斯噪声（SGD）/无噪声（GD）
        noise = np.random.normal(0, sigma) if is_sgd else 0
        x = x - lr * (grad + noise)
        trajectory.append(x)
    return np.array(trajectory), x

# ===================== 3. 实验配置与运行 =====================
# 理论约束：σ ∝ η^(2/3)
eta = 0.05  # 步长
sigma_opt = eta ** (2/3)  # 理论最优噪声强度
sigma_under = sigma_opt * 0.5  # 噪声不足
sigma_over = sigma_opt * 2  # 噪声过度

x0 = 0.5  # 初始值
max_iter = 1000
n_trials = 100  # 实验次数

# 记录各实验组结果
results = {
    "GD": {"trajectories": [], "final_vals": []},
    "SGD_under": {"trajectories": [], "final_vals": []},
    "SGD_opt": {"trajectories": [], "final_vals": []},
    "SGD_over": {"trajectories": [], "final_vals": []}
}

# 批量运行实验
for _ in range(n_trials):
    # GD（无噪声）
    traj, final = gd_sgd(x0, eta, 0, max_iter, is_sgd=False)
    results["GD"]["trajectories"].append(traj)
    results["GD"]["final_vals"].append(final)
    
    # SGD-噪声不足
    traj, final = gd_sgd(x0, eta, sigma_under, max_iter, is_sgd=True)
    results["SGD_under"]["trajectories"].append(traj)
    results["SGD_under"]["final_vals"].append(final)
    
    # SGD-理论最优噪声
    traj, final = gd_sgd(x0, eta, sigma_opt, max_iter, is_sgd=True)
    results["SGD_opt"]["trajectories"].append(traj)
    results["SGD_opt"]["final_vals"].append(final)
    
    # SGD-噪声过度
    traj, final = gd_sgd(x0, eta, sigma_over, max_iter, is_sgd=True)
    results["SGD_over"]["trajectories"].append(traj)
    results["SGD_over"]["final_vals"].append(final)

# ===================== 4. 结果分析与可视化 =====================
# 定义临界点邻域（判断是否遍历）
critical_points = [-1.0, 0.0, 1.0]
neighborhood = 1e-3

# 计算遍历覆盖率
def calculate_coverage(trajectories):
    coverage = []
    for traj in trajectories:
        covered = []
        for cp in critical_points:
            if np.any(np.abs(traj - cp) < neighborhood):
                covered.append(True)
            else:
                covered.append(False)
        coverage.append(all(covered))  # 是否覆盖所有临界点
    return np.mean(coverage)

# 计算各实验组覆盖率
coverage_gd = calculate_coverage(results["GD"]["trajectories"])
coverage_under = calculate_coverage(results["SGD_under"]["trajectories"])
coverage_opt = calculate_coverage(results["SGD_opt"]["trajectories"])
coverage_over = calculate_coverage(results["SGD_over"]["trajectories"])

print(f"遍历覆盖率：")
print(f"GD: {coverage_gd:.2f}, SGD-噪声不足: {coverage_under:.2f}, SGD-最优: {coverage_opt:.2f}, SGD-过度: {coverage_over:.2f}")

# 计算收敛到最优极小的概率（x≈±1，损失=-1）
def calculate_opt_prob(final_vals):
    # 最优极小的损失值为-1，允许1e-2误差
    opt_vals = [1 if np.abs(double_well(x) - (-1)) < 1e-2 else 0 for x in final_vals]
    return np.mean(opt_vals)

prob_gd = calculate_opt_prob(results["GD"]["final_vals"])
prob_under = calculate_opt_prob(results["SGD_under"]["final_vals"])
prob_opt = calculate_opt_prob(results["SGD_opt"]["final_vals"])
prob_over = calculate_opt_prob(results["SGD_over"]["final_vals"])

print(f"收敛到最优极小概率：")
print(f"GD: {prob_gd:.2f}, SGD-噪声不足: {prob_under:.2f}, SGD-最优: {prob_opt:.2f}, SGD-过度: {prob_over:.2f}")

# 可视化轨迹（选单次实验）
plt.figure(figsize=(12, 6))
x_range = np.linspace(-1.5, 1.5, 1000)
plt.plot(x_range, double_well(x_range), label="Double-Well Function", color="black")
# 标记临界点
for cp in critical_points:
    plt.scatter(cp, double_well(cp), color="red", s=50, label="Critical Point" if cp == -1 else "")

# 绘制各实验组轨迹
plt.plot(results["GD"]["trajectories"][0], double_well(results["GD"]["trajectories"][0]), label="GD", alpha=0.7)
plt.plot(results["SGD_under"]["trajectories"][0], double_well(results["SGD_under"]["trajectories"][0]), label="SGD-Under", alpha=0.7)
plt.plot(results["SGD_opt"]["trajectories"][0], double_well(results["SGD_opt"]["trajectories"][0]), label="SGD-Opt", alpha=0.7)
plt.plot(results["SGD_over"]["trajectories"][0], double_well(results["SGD_over"]["trajectories"][0]), label="SGD-Over", alpha=0.7)

plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Trajectory of GD/SGD on Double-Well Function")
plt.legend()
plt.grid(True)
plt.savefig("double_well_trajectory.png")
plt.show()
