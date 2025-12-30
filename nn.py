import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pyhessian import hessian  # 海森矩阵分析工具
import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 数据加载与预处理 =====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# ===================== 2. 定义小规模全连接网络 =====================
class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ===================== 3. 自定义带可控噪声的SGD =====================
class NoisySGD(optim.Optimizer):
    def __init__(self, params, lr=0.01, sigma=0.0):
        defaults = dict(lr=lr, sigma=sigma)
        super(NoisySGD, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            sigma = group['sigma']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                # 加性高斯噪声
                noise = torch.normal(0, sigma, size=grad.size()).to(grad.device)
                p.data.add_( -lr * (grad + noise) )
        return loss

# ===================== 4. 训练与评估函数 =====================
def train(model, optimizer, train_loader, device, epochs=10):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}, Train Loss: {avg_loss:.4f}")
    return train_losses

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    acc = 100 * correct / total
    return acc

# ===================== 5. 统计临界点类型（鞍点/局部极小） =====================
def count_critical_points(model, data, target, device):
    """用海森矩阵特征值判断临界点类型：负特征值数=0→局部极小，>0→鞍点"""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    # 计算海森矩阵
    hessian_comp = hessian(model, loss_fn, data=(data.to(device), target.to(device)))
    # 获取顶层10个特征值（近似判断）
    top_eigenvalues, _ = hessian_comp.eigenvalues(top_n=10)
    # 统计负特征值数量
    neg_eig = sum(1 for eig in top_eigenvalues if eig < 0)
    if neg_eig == 0:
        return "local_min"
    else:
        return "saddle"

# ===================== 6. 实验主流程 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eta = 0.01  # 步长
sigma_opt = eta ** (2/3)  # 理论最优噪声
sigma_under = sigma_opt * 0.5
sigma_over = sigma_opt * 2

n_trials = 5  # 简化实验次数（可改50）
results = {
    "SGD_under": {"train_losses": [], "test_accs": [], "critical_types": []},
    "SGD_opt": {"train_losses": [], "test_accs": [], "critical_types": []},
    "SGD_over": {"train_losses": [], "test_accs": [], "critical_types": []}
}

# 批量训练
for trial in range(n_trials):
    print(f"\nTrial {trial+1}/{n_trials}")
    
    # 实验组1：噪声不足
    model_under = SimpleFCN().to(device)
    optimizer_under = NoisySGD(model_under.parameters(), lr=eta, sigma=sigma_under)
    losses_under = train(model_under, optimizer_under, train_loader, device)
    acc_under = test(model_under, test_loader, device)
    # 取第一批数据统计临界点
    data, target = next(iter(train_loader))
    crit_type_under = count_critical_points(model_under, data, target, device)
    results["SGD_under"]["train_losses"].append(losses_under)
    results["SGD_under"]["test_accs"].append(acc_under)
    results["SGD_under"]["critical_types"].append(crit_type_under)
    
    # 实验组2：理论最优噪声
    model_opt = SimpleFCN().to(device)
    optimizer_opt = NoisySGD(model_opt.parameters(), lr=eta, sigma=sigma_opt)
    losses_opt = train(model_opt, optimizer_opt, train_loader, device)
    acc_opt = test(model_opt, test_loader, device)
    crit_type_opt = count_critical_points(model_opt, data, target, device)
    results["SGD_opt"]["train_losses"].append(losses_opt)
    results["SGD_opt"]["test_accs"].append(acc_opt)
    results["SGD_opt"]["critical_types"].append(crit_type_opt)
    
    # 实验组3：噪声过度
    model_over = SimpleFCN().to(device)
    optimizer_over = NoisySGD(model_over.parameters(), lr=eta, sigma=sigma_over)
    losses_over = train(model_over, optimizer_over, train_loader, device)
    acc_over = test(model_over, test_loader, device)
    crit_type_over = count_critical_points(model_over, data, target, device)
    results["SGD_over"]["train_losses"].append(losses_over)
    results["SGD_over"]["test_accs"].append(acc_over)
    results["SGD_over"]["critical_types"].append(crit_type_over)

# ===================== 7. 结果分析 =====================
# 计算平均准确率和方差（收敛稳定性）
acc_under_mean = np.mean(results["SGD_under"]["test_accs"])
acc_under_std = np.std(results["SGD_under"]["test_accs"])
acc_opt_mean = np.mean(results["SGD_opt"]["test_accs"])
acc_opt_std = np.std(results["SGD_opt"]["test_accs"])
acc_over_mean = np.mean(results["SGD_over"]["test_accs"])
acc_over_std = np.std(results["SGD_over"]["test_accs"])

print("\n测试准确率（均值±标准差）：")
print(f"SGD-噪声不足: {acc_under_mean:.2f}±{acc_under_std:.2f}")
print(f"SGD-最优噪声: {acc_opt_mean:.2f}±{acc_opt_std:.2f}")
print(f"SGD-噪声过度: {acc_over_mean:.2f}±{acc_over_std:.2f}")

# 统计临界点类型
crit_under = results["SGD_under"]["critical_types"]
crit_opt = results["SGD_opt"]["critical_types"]
crit_over = results["SGD_over"]["critical_types"]

print("\n临界点类型统计：")
print(f"SGD-噪声不足: 局部极小={crit_under.count('local_min')}, 鞍点={crit_under.count('saddle')}")
print(f"SGD-最优噪声: 局部极小={crit_opt.count('local_min')}, 鞍点={crit_opt.count('saddle')}")
print(f"SGD-噪声过度: 局部极小={crit_over.count('local_min')}, 鞍点={crit_over.count('saddle')}")

# 可视化训练损失
plt.figure(figsize=(10, 6))
epochs = range(1, 11)
# 绘制各实验组平均损失
loss_under_mean = np.mean(results["SGD_under"]["train_losses"], axis=0)
loss_opt_mean = np.mean(results["SGD_opt"]["train_losses"], axis=0)
loss_over_mean = np.mean(results["SGD_over"]["train_losses"], axis=0)

plt.plot(epochs, loss_under_mean, label="SGD-Under", marker='o')
plt.plot(epochs, loss_opt_mean, label="SGD-Opt", marker='s')
plt.plot(epochs, loss_over_mean, label="SGD-Over", marker='^')
plt.xlabel("Epoch")
plt.ylabel("Average Train Loss")
plt.title("Train Loss of Noisy SGD on FCN-MNIST")
plt.legend()
plt.grid(True)
plt.savefig("fcn_train_loss.png")
plt.show()
