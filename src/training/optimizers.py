# src/training/optimizers.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any


class Optimizer(ABC):
    """优化器基类"""
    
    def __init__(self, learning_rate: float = 0.01):
        """
        初始化优化器
        
        Args:
            learning_rate: 学习率
        """
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.state = {}
        
    @abstractmethod
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        更新参数
        
        Args:
            params: 参数字典
            grads: 梯度字典
            
        Returns:
            更新后的参数
        """
        pass
    
    def get_learning_rate(self) -> float:
        """获取当前学习率"""
        return self.learning_rate
    
    def set_learning_rate(self, learning_rate: float):
        """设置学习率"""
        self.learning_rate = learning_rate
    
    def get_state(self) -> Dict:
        """获取优化器状态"""
        return {
            'learning_rate': self.learning_rate,
            'state': self.state
        }
    
    def set_state(self, state: Dict):
        """设置优化器状态"""
        self.learning_rate = state['learning_rate']
        self.state = state['state']
    
    def get_config(self) -> Dict:
        """获取优化器配置"""
        return {
            'class': self.__class__.__name__,
            'learning_rate': self.initial_learning_rate
        }


class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def __init__(self, learning_rate: float = 0.01, weight_decay: float = 0.0):
        """
        初始化SGD优化器
        
        Args:
            learning_rate: 学习率
            weight_decay: 权重衰减（L2正则化）
        """
        super().__init__(learning_rate)
        self.weight_decay = weight_decay
        
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """SGD参数更新"""
        updated_params = {}
        
        for name, param in params.items():
            if name in grads:
                grad = grads[name]
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # 更新参数
                updated_params[name] = param - self.learning_rate * grad
            else:
                updated_params[name] = param
        
        return updated_params


class MomentumSGD(Optimizer):
    """带动量的SGD优化器"""
    
    def __init__(self, learning_rate: float = 0.01, 
                 momentum: float = 0.9,
                 weight_decay: float = 0.0,
                 nesterov: bool = False):
        """
        初始化动量SGD
        
        Args:
            learning_rate: 学习率
            momentum: 动量系数
            weight_decay: 权重衰减
            nesterov: 是否使用Nesterov加速
        """
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov
        
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """动量SGD参数更新"""
        updated_params = {}
        
        for name, param in params.items():
            if name in grads:
                grad = grads[name]
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # 初始化动量
                if name not in self.state:
                    self.state[name] = {'velocity': np.zeros_like(param)}
                
                velocity = self.state[name]['velocity']
                
                # 更新速度
                velocity = self.momentum * velocity - self.learning_rate * grad
                self.state[name]['velocity'] = velocity
                
                # 更新参数
                if self.nesterov:
                    updated_params[name] = param + self.momentum * velocity - self.learning_rate * grad
                else:
                    updated_params[name] = param + velocity
            else:
                updated_params[name] = param
        
        return updated_params


class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, learning_rate: float = 0.001,
                 beta1: float = 0.9,
                 beta2: float = 0.999,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        初始化Adam优化器
        
        Args:
            learning_rate: 学习率
            beta1: 一阶矩估计的指数衰减率
            beta2: 二阶矩估计的指数衰减率
            eps: 数值稳定性常数
            weight_decay: 权重衰减
        """
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0  # 时间步
        
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Adam参数更新"""
        self.t += 1
        updated_params = {}
        
        for name, param in params.items():
            if name in grads:
                grad = grads[name]
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # 初始化状态
                if name not in self.state:
                    self.state[name] = {
                        'm': np.zeros_like(param),  # 一阶矩
                        'v': np.zeros_like(param)   # 二阶矩
                    }
                
                m = self.state[name]['m']
                v = self.state[name]['v']
                
                # 更新矩估计
                m = self.beta1 * m + (1 - self.beta1) * grad
                v = self.beta2 * v + (1 - self.beta2) * grad**2
                
                # 保存更新后的矩
                self.state[name]['m'] = m
                self.state[name]['v'] = v
                
                # 偏差修正
                m_hat = m / (1 - self.beta1**self.t)
                v_hat = v / (1 - self.beta2**self.t)
                
                # 更新参数
                updated_params[name] = param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)
            else:
                updated_params[name] = param
        
        return updated_params


class AdaGrad(Optimizer):
    """AdaGrad优化器"""
    
    def __init__(self, learning_rate: float = 0.01,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        初始化AdaGrad优化器
        
        Args:
            learning_rate: 学习率
            eps: 数值稳定性常数
            weight_decay: 权重衰减
        """
        super().__init__(learning_rate)
        self.eps = eps
        self.weight_decay = weight_decay
        
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """AdaGrad参数更新"""
        updated_params = {}
        
        for name, param in params.items():
            if name in grads:
                grad = grads[name]
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # 初始化累积梯度
                if name not in self.state:
                    self.state[name] = {'sum_squared_grad': np.zeros_like(param)}
                
                sum_squared_grad = self.state[name]['sum_squared_grad']
                
                # 累积梯度平方
                sum_squared_grad += grad**2
                self.state[name]['sum_squared_grad'] = sum_squared_grad
                
                # 更新参数
                updated_params[name] = param - self.learning_rate * grad / (np.sqrt(sum_squared_grad) + self.eps)
            else:
                updated_params[name] = param
        
        return updated_params


class RMSprop(Optimizer):
    """RMSprop优化器"""
    
    def __init__(self, learning_rate: float = 0.001,
                 alpha: float = 0.99,
                 eps: float = 1e-8,
                 weight_decay: float = 0.0):
        """
        初始化RMSprop优化器
        
        Args:
            learning_rate: 学习率
            alpha: 平滑常数
            eps: 数值稳定性常数
            weight_decay: 权重衰减
        """
        super().__init__(learning_rate)
        self.alpha = alpha
        self.eps = eps
        self.weight_decay = weight_decay
        
    def update(self, params: Dict[str, np.ndarray], 
               grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """RMSprop参数更新"""
        updated_params = {}
        
        for name, param in params.items():
            if name in grads:
                grad = grads[name]
                
                # 添加权重衰减
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * param
                
                # 初始化均方根
                if name not in self.state:
                    self.state[name] = {'square_avg': np.zeros_like(param)}
                
                square_avg = self.state[name]['square_avg']
                
                # 更新均方根
                square_avg = self.alpha * square_avg + (1 - self.alpha) * grad**2
                self.state[name]['square_avg'] = square_avg
                
                # 更新参数
                updated_params[name] = param - self.learning_rate * grad / (np.sqrt(square_avg) + self.eps)
            else:
                updated_params[name] = param
        
        return updated_params