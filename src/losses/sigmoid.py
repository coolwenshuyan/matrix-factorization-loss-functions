# src/losses/sigmoid.py
import numpy as np
from .base import BaseLoss


class SigmoidLikeLoss(BaseLoss):
    """
    类Sigmoid损失函数
    
    L(e) = L_max * (1 - exp(-α * e²))
    ∂L/∂e = 2 * α * L_max * e * exp(-α * e²)
    """
    
    def __init__(self, alpha: float = 1.0, l_max: float = 3.0):
        """
        初始化Sigmoid-like损失函数
        
        Args:
            alpha: 增长速度参数
            l_max: 损失上界
        """
        super().__init__("SigmoidLike")
        
        if alpha <= 0:
            raise ValueError("alpha必须大于0")
        if l_max <= 0:
            raise ValueError("l_max必须大于0")
        
        self.alpha = alpha
        self.l_max = l_max
        self._config = {
            'alpha': alpha,
            'l_max': l_max
        }
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算Sigmoid-like损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        errors_squared = errors ** 2
        
        # 防止指数溢出
        exp_arg = -self.alpha * errors_squared
        exp_arg = np.clip(exp_arg, -50, 0)  # exp(-50) ≈ 0
        
        loss = self.l_max * (1 - np.exp(exp_arg))
        
        return float(np.mean(loss))
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Sigmoid-like损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        errors_squared = errors ** 2
        
        # 防止指数溢出
        exp_arg = -self.alpha * errors_squared
        exp_arg = np.clip(exp_arg, -50, 0)
        
        exp_term = np.exp(exp_arg)
        grad = 2 * self.alpha * self.l_max * errors * exp_term
        
        return grad
    
    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Sigmoid-like损失的二阶导数"""
        errors = predictions - targets
        errors_squared = errors ** 2
        
        exp_arg = -self.alpha * errors_squared
        exp_arg = np.clip(exp_arg, -50, 0)
        exp_term = np.exp(exp_arg)
        
        # H = 2αL_max * exp(-αe²) * (1 - 2αe²)
        hess = 2 * self.alpha * self.l_max * exp_term * (1 - 2 * self.alpha * errors_squared)
        
        return hess