# src/losses/base.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional, Tuple
import json


class BaseLoss(ABC):
    """
    损失函数的抽象基类
    
    所有损失函数必须继承此类并实现相应的方法
    """
    
    def __init__(self, name: str):
        """
        初始化损失函数
        
        Args:
            name: 损失函数名称
        """
        self.name = name
        self._config = {}
        
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> Union[float, np.ndarray]:
        """
        计算损失值
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            损失值（标量或数组）
        """
        pass
    
    @abstractmethod
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        计算损失对预测值的梯度
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            梯度数组
        """
        pass
    
    def __call__(self, predictions: np.ndarray, targets: np.ndarray, 
                 return_gradient: bool = False) -> Union[float, Tuple[float, np.ndarray]]:
        """
        使损失函数可调用
        
        Args:
            predictions: 预测值
            targets: 真实值
            return_gradient: 是否同时返回梯度
            
        Returns:
            损失值，或 (损失值, 梯度) 元组
        """
        loss = self.forward(predictions, targets)
        
        if return_gradient:
            grad = self.gradient(predictions, targets)
            return loss, grad
        
        return loss
    
    def hessian(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        计算二阶导数（Hessian）
        
        默认实现返回None，子类可以覆盖此方法
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            Hessian矩阵或None
        """
        return None
    
    def is_differentiable_at(self, x: float) -> bool:
        """
        检查函数在某点是否可导
        
        Args:
            x: 检查点
            
        Returns:
            是否可导
        """
        # 默认实现：通过数值方法检查
        eps = 1e-8
        try:
            # 创建测试点
            test_pred = np.array([x - eps, x, x + eps])
            test_target = np.zeros(3)
            
            # 计算梯度
            grads = self.gradient(test_pred, test_target)
            
            # 检查梯度是否有限
            return np.all(np.isfinite(grads))
        except:
            return False
    
    def get_config(self) -> Dict:
        """
        获取损失函数的配置参数
        
        Returns:
            配置字典
        """
        config = {
            'name': self.name,
            'class': self.__class__.__name__
        }
        config.update(self._config)
        return config
    
    def set_config(self, config: Dict):
        """
        设置损失函数的配置参数
        
        Args:
            config: 配置字典
        """
        self._config.update(config)
    
    def save_config(self, filepath: str):
        """
        保存配置到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w') as f:
            json.dump(self.get_config(), f, indent=2)
    
    def __repr__(self) -> str:
        """
        返回损失函数的字符串表示
        
        Returns:
            字符串表示
        """
        params = []
        for key, value in self._config.items():
            if key not in ['name', 'class']:
                params.append(f"{key}={value}")
        
        param_str = ", ".join(params) if params else ""
        return f"{self.__class__.__name__}({param_str})"
    
    def validate_inputs(self, predictions: np.ndarray, targets: np.ndarray):
        """
        验证输入的有效性
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Raises:
            ValueError: 如果输入无效
        """
        if predictions.shape != targets.shape:
            raise ValueError(f"预测值和目标值形状不匹配: {predictions.shape} vs {targets.shape}")
        
        if np.any(np.isnan(predictions)) or np.any(np.isnan(targets)):
            raise ValueError("输入包含NaN值")
        
        if np.any(np.isinf(predictions)) or np.any(np.isinf(targets)):
            raise ValueError("输入包含无穷大值")