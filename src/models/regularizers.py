# src/models/regularizers.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseRegularizer(ABC):
    """正则化器的基类"""
    
    def __init__(self, lambda_reg: float = 0.01):
        """
        初始化
        
        Args:
            lambda_reg: 正则化强度
        """
        self.lambda_reg = lambda_reg
    
    @abstractmethod
    def compute_penalty(self, parameters: Dict[str, np.ndarray]) -> float:
        """
        计算正则化惩罚项
        
        Args:
            parameters: 参数字典
            
        Returns:
            正则化损失
        """
        pass
    
    @abstractmethod
    def compute_gradient(self, parameter_name: str, parameter: np.ndarray) -> np.ndarray:
        """
        计算正则化梯度
        
        Args:
            parameter_name: 参数名称
            parameter: 参数值
            
        Returns:
            正则化梯度
        """
        pass


class L2Regularizer(BaseRegularizer):
    """L2正则化器"""
    
    def __init__(self, lambda_reg: float = 0.01,
                 lambda_user: Optional[float] = None,
                 lambda_item: Optional[float] = None,
                 lambda_bias: Optional[float] = None):
        """
        初始化L2正则化器
        
        Args:
            lambda_reg: 默认正则化强度
            lambda_user: 用户因子的正则化强度
            lambda_item: 物品因子的正则化强度
            lambda_bias: 偏差项的正则化强度
        """
        super().__init__(lambda_reg)
        self.lambda_user = lambda_user if lambda_user is not None else lambda_reg
        self.lambda_item = lambda_item if lambda_item is not None else lambda_reg
        self.lambda_bias = lambda_bias if lambda_bias is not None else lambda_reg * 0.1
    
    def compute_penalty(self, parameters: Dict[str, np.ndarray]) -> float:
        """计算L2正则化惩罚"""
        penalty = 0.0
        
        for name, param in parameters.items():
            lambda_val = self._get_lambda(name)
            penalty += lambda_val * np.sum(param ** 2)
        
        return penalty
    
    def compute_gradient(self, parameter_name: str, parameter: np.ndarray) -> np.ndarray:
        """计算L2正则化梯度"""
        lambda_val = self._get_lambda(parameter_name)
        return 2 * lambda_val * parameter
    
    def _get_lambda(self, parameter_name: str) -> float:
        """获取特定参数的正则化强度"""
        if 'user' in parameter_name.lower() or parameter_name == 'P':
            return self.lambda_user
        elif 'item' in parameter_name.lower() or parameter_name == 'Q':
            return self.lambda_item
        elif 'bias' in parameter_name.lower():
            return self.lambda_bias
        else:
            return self.lambda_reg


class L1Regularizer(BaseRegularizer):
    """L1正则化器"""
    
    def __init__(self, lambda_reg: float = 0.01, epsilon: float = 1e-8):
        """
        初始化L1正则化器
        
        Args:
            lambda_reg: 正则化强度
            epsilon: 平滑参数
        """
        super().__init__(lambda_reg)
        self.epsilon = epsilon
    
    def compute_penalty(self, parameters: Dict[str, np.ndarray]) -> float:
        """计算L1正则化惩罚"""
        penalty = 0.0
        
        for param in parameters.values():
            penalty += self.lambda_reg * np.sum(np.abs(param))
        
        return penalty
    
    def compute_gradient(self, parameter_name: str, parameter: np.ndarray) -> np.ndarray:
        """计算L1正则化梯度"""
        # 使用平滑的L1梯度避免不可导点
        return self.lambda_reg * parameter / (np.abs(parameter) + self.epsilon)


class ElasticNetRegularizer(BaseRegularizer):
    """弹性网络正则化器（L1 + L2）"""
    
    def __init__(self, lambda_reg: float = 0.01, l1_ratio: float = 0.5,
                 epsilon: float = 1e-8):
        """
        初始化弹性网络正则化器
        
        Args:
            lambda_reg: 总正则化强度
            l1_ratio: L1正则化的比例
            epsilon: L1平滑参数
        """
        super().__init__(lambda_reg)
        self.l1_ratio = l1_ratio
        self.l2_ratio = 1 - l1_ratio
        self.epsilon = epsilon
    
    def compute_penalty(self, parameters: Dict[str, np.ndarray]) -> float:
        """计算弹性网络正则化惩罚"""
        l1_penalty = 0.0
        l2_penalty = 0.0
        
        for param in parameters.values():
            l1_penalty += np.sum(np.abs(param))
            l2_penalty += np.sum(param ** 2)
        
        return self.lambda_reg * (self.l1_ratio * l1_penalty + self.l2_ratio * l2_penalty)
    
    def compute_gradient(self, parameter_name: str, parameter: np.ndarray) -> np.ndarray:
        """计算弹性网络正则化梯度"""
        l1_grad = parameter / (np.abs(parameter) + self.epsilon)
        l2_grad = 2 * parameter
        
        return self.lambda_reg * (self.l1_ratio * l1_grad + self.l2_ratio * l2_grad)