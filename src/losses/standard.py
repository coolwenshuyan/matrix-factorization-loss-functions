# src/losses/standard.py
import numpy as np
from .base import BaseLoss


class L2Loss(BaseLoss):
    """
    L2损失函数 (Mean Squared Error)
    
    L(e) = 0.5 * e²
    ∂L/∂e = e
    """
    
    def __init__(self):
        super().__init__("L2")
        self._config = {}
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算L2损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        loss = 0.5 * np.mean(errors ** 2)
        
        return float(loss)
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算L2损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        return errors
    
    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """L2损失的二阶导数是常数1"""
        return np.ones_like(predictions)


class L1Loss(BaseLoss):
    """
    L1损失函数 (Mean Absolute Error)
    
    L(e) = |e|
    ∂L/∂e = sign(e)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """
        初始化L1损失
        
        Args:
            epsilon: 用于数值稳定性的小常数
        """
        super().__init__("L1")
        self.epsilon = epsilon
        self._config = {'epsilon': epsilon}
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算L1损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        loss = np.mean(np.abs(errors))
        
        return float(loss)
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算L1损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        
        # 处理e=0的情况
        # 使用次梯度：在0处返回0
        grad = np.sign(errors)
        
        # 可选：平滑处理（类似于Smooth L1）
        # mask = np.abs(errors) < self.epsilon
        # grad[mask] = errors[mask] / self.epsilon
        
        return grad
    
    def is_differentiable_at(self, x: float) -> bool:
        """L1在0处不可导"""
        return abs(x) > self.epsilon