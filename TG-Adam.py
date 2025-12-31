import torch
import torch.nn as nn
import torch.optim as optim
import gudhi as gd
from sklearn.decomposition import PCA

class TopologyGuidedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 weight_decay=0, amsgrad=False, 
                 betti_threshold=5, morse_threshold=2):
        super().__init__(params, lr, betas, eps, weight_decay, amsgrad)
        # 拓扑引导的超参数
        self.betti_threshold = betti_threshold  # 高Betti数的阈值
        self.morse_threshold = morse_threshold  # 高Morse指标的阈值
        self.pca = PCA(n_components=2)  # 降维工具
    
    def compute_topology_features(self, model):
        """计算当前模型参数的Betti数和Morse指标"""
        # 1. 展平参数
        flat_params = []
        for p in model.parameters():
            flat_params.append(p.data.cpu().numpy().flatten())
        flat_params = np.concatenate(flat_params).reshape(1, -1)  # 单样本
        
        # 2. PCA降维（注：实际需采样多个参数点，这里简化为单样本+历史缓存）
        # （正式实验中需缓存最近N轮的参数点，再降维计算Betti数）
        params_lowdim = self.pca.fit_transform(flat_params)
        
        # 3. 计算Betti数（简化版，正式需用持久同调）
        # 这里用参数的梯度范数近似：梯度范数越小，Betti数可能越低
        grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in model.parameters()])).item()
        b1 = 10 if grad_norm < 1e-3 else 3  # 示例：梯度小→Betti数高（简化）
        
        # 4. 计算Morse指标（简化版，正式需用海森矩阵）
        # 这里用损失的二阶导数近似：损失变化越小，Morse指标可能越低
        morse_ind = 0 if grad_norm < 1e-4 else 3  # 示例：梯度极小→Morse指标=0
        
        return b1, morse_ind
    
    def step(self, closure=None, model=None):
        # 1. 计算当前拓扑特征（必须传入model）
        if model is not None:
            b1, morse_ind = self.compute_topology_features(model)
            
            # 2. 动态调整动量β₁
            if b1 > self.betti_threshold:
                self.param_groups[0]['betas'] = (0.7, self.param_groups[0]['betas'][1])  # 降低动量
            else:
                self.param_groups[0]['betas'] = (0.95, self.param_groups[0]['betas'][1])  # 增强动量
            
            # 3. 动态调整ε
            if morse_ind > self.morse_threshold:
                self.param_groups[0]['eps'] = 1e-10  # 减小ε，增强缩放
            else:
                self.param_groups[0]['eps'] = 1e-6  # 增大ε，减弱缩放
        
        # 4. 执行标准Adam的更新
        return super().step(closure)
