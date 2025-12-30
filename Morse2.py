import torch
from pyhessian import hessian

# ===================== 补充：神经网络损失函数的Morse指标 =====================
def calculate_morse_index_nn(model, data, target, device):
    """计算神经网络损失函数在当前参数处的Morse指标（海森矩阵负特征值数量）"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    
    # 计算海森矩阵（PyHessian支持计算前N个特征值）
    hessian_comp = hessian(model, loss_fn, data=(data.to(device), target.to(device)))
    # 计算前50个特征值（平衡计算量和准确性）
    eigenvalues, _ = hessian_comp.eigenvalues(top_n=50)
    
    # 统计负特征值数量（Morse指标）
    morse_index = sum(1 for eig in eigenvalues if eig < 0)
    return morse_index, eigenvalues

# ===================== 整合到原有神经网络实验中 =====================
# 训练后分析每个模型的Morse指标
for trial in range(n_trials):
    # 以SGD-最优噪声组为例
    model_opt = SimpleFCN().to(device)
    optimizer_opt = NoisySGD(model_opt.parameters(), lr=eta, sigma=sigma_opt)
    train(model_opt, optimizer_opt, train_loader, device)
    
    # 取一批数据计算Morse指标
    data, target = next(iter(train_loader))
    morse_idx, eigvals = calculate_morse_index_nn(model_opt, data, target, device)
    
    print(f"\nTrial {trial+1} - SGD-Opt: Morse指标={morse_idx}")
    print(f"海森矩阵特征值前10个：{eigvals[:10]}")

# 验证Morse理论核心结论：噪声驱动下Morse指标从高→低（鞍点→局部极小）
# 统计不同实验组的平均Morse指标
morse_indices_under = []
morse_indices_opt = []
morse_indices_over = []

for trial in range(n_trials):
    # 噪声不足组
    model_under = SimpleFCN().to(device)
    optimizer_under = NoisySGD(model_under.parameters(), lr=eta, sigma=sigma_under)
    train(model_under, optimizer_under, train_loader, device)
    morse_idx, _ = calculate_morse_index_nn(model_under, data, target, device)
    morse_indices_under.append(morse_idx)
    
    # 最优噪声组
    model_opt = SimpleFCN().to(device)
    optimizer_opt = NoisySGD(model_opt.parameters(), lr=eta, sigma=sigma_opt)
    train(model_opt, optimizer_opt, train_loader, device)
    morse_idx, _ = calculate_morse_index_nn(model_opt, data, target, device)
    morse_indices_opt.append(morse_idx)
    
    # 噪声过度组
    model_over = SimpleFCN().to(device)
    optimizer_over = NoisySGD(model_over.parameters(), lr=eta, sigma=sigma_over)
    train(model_over, optimizer_over, train_loader, device)
    morse_idx, _ = calculate_morse_index_nn(model_over, data, target, device)
    morse_indices_over.append(morse_idx)

# 打印平均Morse指标
print(f"\n平均Morse指标：")
print(f"SGD-噪声不足: {np.mean(morse_indices_under):.2f}")
print(f"SGD-最优噪声: {np.mean(morse_indices_opt):.2f}")  # 应显著更低（更接近局部极小）
print(f"SGD-噪声过度: {np.mean(morse_indices_over):.2f}")
