import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigvalsh  # 特征值计算

# ===================== 补充：Morse指标计算 =====================
def double_well_hessian(x):
    """双井函数的海森矩阵（一维函数的二阶导数）：f''(x) = 12x² - 4"""
    return 12 * x**2 - 4

def calculate_morse_index(hessian):
    """计算Morse指标：海森矩阵负特征值的数量（一维时直接判断符号）"""
    if isinstance(hessian, np.ndarray):
        eigvals = eigvalsh(hessian)  # 高维海森矩阵用特征值分解
    else:
        eigvals = [hessian]  # 一维时直接取二阶导数
    
    morse_index = sum(1 for eig in eigvals if eig < 0)
    return morse_index

# ===================== 整合到原有低维实验中 =====================
def analyze_critical_points_morse(trajectory, func_hessian):
    """分析轨迹中经过的临界点及其Morse指标"""
    critical_info = []
    # 遍历轨迹中的点，筛选临界点（梯度≈0）
    for x in trajectory:
        grad = double_well_gradient(x)
        if np.abs(grad) < 1e-3:  # 梯度接近0，判定为临界点
            hess = func_hessian(x)
            morse_idx = calculate_morse_index(hess)
            critical_info.append({
                "x": x,
                "hessian": hess,
                "morse_index": morse_idx,
                "type": "local_min" if morse_idx == 0 else "saddle" if morse_idx > 0 else "local_max"
            })
    # 去重（同一临界点可能被多次检测）
    unique_critical = []
    seen_x = []
    for info in critical_info:
        if not any(np.abs(info["x"] - sx) < 1e-3 for sx in seen_x):
            seen_x.append(info["x"])
            unique_critical.append(info)
    return unique_critical

# 复用原有低维实验的轨迹结果，分析Morse指标
# 以SGD-最优噪声组为例
traj_opt = results["SGD_opt"]["trajectories"][0]
critical_morse = analyze_critical_points_morse(traj_opt, double_well_hessian)

# 打印Morse指标分析结果
print("\n=== Morse指标分析（SGD-最优噪声）===")
for cp in critical_morse:
    print(f"临界点x={cp['x']:.4f}, 海森矩阵={cp['hessian']:.4f}, Morse指标={cp['morse_idx']}, 类型={cp['type']}")

# 验证Morse不等式（一维双井函数的拓扑不变量：欧拉示性数=1）
# Morse不等式：局部极小数量 - 鞍点数量 + 局部极大数量 = 欧拉示性数
local_min_count = sum(1 for cp in critical_morse if cp["type"] == "local_min")
saddle_count = sum(1 for cp in critical_morse if cp["type"] == "saddle")
euler_char = local_min_count - saddle_count  # 一维无局部极大
print(f"\nMorse不等式验证：局部极小({local_min_count}) - 鞍点({saddle_count}) = {euler_char}（理论欧拉示性数=1）")
