# src/losses/robust.py
import numpy as np
from .base import BaseLoss


class HuberLoss(BaseLoss):
    """
    Huber损失函数
    
    L(e) = 0.5 * e²                    if |e| ≤ δ
         = δ * |e| - 0.5 * δ²          if |e| > δ
    
    ∂L/∂e = e                          if |e| ≤ δ
          = δ * sign(e)                if |e| > δ
    """
    
    def __init__(self, delta: float = 1.0):
        """
        初始化Huber损失
        
        Args:
            delta: 阈值参数
        """
        super().__init__("Huber")
        
        if delta <= 0:
            raise ValueError("delta必须大于0")
        
        self.delta = delta
        self._config = {'delta': delta}
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算Huber损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 分段计算
        quadratic_mask = abs_errors <= self.delta
        linear_mask = ~quadratic_mask
        
        loss = np.zeros_like(errors)
        
        # 二次损失部分
        loss[quadratic_mask] = 0.5 * errors[quadratic_mask] ** 2
        
        # 线性损失部分
        loss[linear_mask] = self.delta * abs_errors[linear_mask] - 0.5 * self.delta ** 2
        
        return float(np.mean(loss))
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Huber损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 分段计算梯度
        grad = np.zeros_like(errors)
        
        # 二次部分的梯度
        quadratic_mask = abs_errors <= self.delta
        grad[quadratic_mask] = errors[quadratic_mask]
        
        # 线性部分的梯度
        linear_mask = ~quadratic_mask
        grad[linear_mask] = self.delta * np.sign(errors[linear_mask])
        
        return grad
    
    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Huber损失的二阶导数"""
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        hess = np.zeros_like(errors)
        quadratic_mask = abs_errors <= self.delta
        hess[quadratic_mask] = 1.0
        
        return hess


class LogcoshLoss(BaseLoss):
    """
    Log-cosh损失函数
    
    L(e) = log(cosh(e))
    ∂L/∂e = tanh(e)
    """
    
    def __init__(self, epsilon: float = 1e-12):
        """
        初始化Logcosh损失
        
        Args:
            epsilon: 数值稳定性常数
        """
        super().__init__("Logcosh")
        self.epsilon = epsilon
        self._config = {'epsilon': epsilon}
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算Logcosh损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        
        # 数值稳定的实现
        # log(cosh(x)) = |x| + log(2) - log(1 + exp(-2|x|))
        abs_errors = np.abs(errors)
        
        # 对于大的|x|，使用近似：log(cosh(x)) ≈ |x| - log(2)
        # 对于小的|x|，使用标准公式
        
        threshold = 20.0  # 阈值，超过此值使用近似
        
        loss = np.zeros_like(errors)
        
        # 小值：使用精确公式
        small_mask = abs_errors < threshold
        if np.any(small_mask):
            loss[small_mask] = np.log(np.cosh(errors[small_mask]) + self.epsilon)
        
        # 大值：使用近似公式
        large_mask = ~small_mask
        if np.any(large_mask):
            loss[large_mask] = abs_errors[large_mask] - np.log(2.0)
        
        return float(np.mean(loss))
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Logcosh损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        
        # 梯度是tanh(e)
        # 对于大的|e|，tanh(e) ≈ sign(e)
        threshold = 20.0
        
        grad = np.zeros_like(errors)
        
        # 小值：使用精确公式
        small_mask = np.abs(errors) < threshold
        if np.any(small_mask):
            grad[small_mask] = np.tanh(errors[small_mask])
        
        # 大值：使用近似
        large_mask = ~small_mask
        if np.any(large_mask):
            grad[large_mask] = np.sign(errors[large_mask])
        
        return grad
    
    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算Logcosh损失的二阶导数"""
        errors = predictions - targets
        
        # 二阶导数是 sech²(e) = 1 - tanh²(e)
        tanh_e = np.tanh(errors)
        return 1.0 - tanh_e ** 2