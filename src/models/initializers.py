# src/models/initializers.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional


class BaseInitializer(ABC):
    """参数初始化器的基类"""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        初始化
        
        Args:
            random_seed: 随机种子
        """
        self.random_seed = random_seed
        if random_seed is not None:
            self.rng = np.random.RandomState(random_seed)
        else:
            self.rng = np.random.RandomState()
    
    @abstractmethod
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """
        初始化参数
        
        Args:
            shape: 参数形状
            
        Returns:
            初始化的参数数组
        """
        pass
    
    def __call__(self, shape: Tuple[int, ...]) -> np.ndarray:
        """使初始化器可调用"""
        return self.initialize(shape)


class NormalInitializer(BaseInitializer):
    """正态分布初始化器"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01, 
                 random_seed: Optional[int] = None):
        """
        初始化
        
        Args:
            mean: 均值
            std: 标准差
            random_seed: 随机种子
        """
        super().__init__(random_seed)
        self.mean = mean
        self.std = std
    
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """使用正态分布初始化"""
        return self.rng.normal(self.mean, self.std, shape).astype(np.float32)


class UniformInitializer(BaseInitializer):
    """均匀分布初始化器"""
    
    def __init__(self, low: float = -0.01, high: float = 0.01,
                 random_seed: Optional[int] = None):
        """
        初始化
        
        Args:
            low: 下界
            high: 上界
            random_seed: 随机种子
        """
        super().__init__(random_seed)
        self.low = low
        self.high = high
    
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """使用均匀分布初始化"""
        return self.rng.uniform(self.low, self.high, shape).astype(np.float32)


class XavierInitializer(BaseInitializer):
    """Xavier/Glorot初始化器"""
    
    def __init__(self, mode: str = 'fan_avg', random_seed: Optional[int] = None):
        """
        初始化
        
        Args:
            mode: 'fan_in', 'fan_out', 或 'fan_avg'
            random_seed: 随机种子
        """
        super().__init__(random_seed)
        self.mode = mode
    
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """使用Xavier初始化"""
        if len(shape) != 2:
            raise ValueError("Xavier初始化仅支持2D数组")
        
        fan_in, fan_out = shape
        
        if self.mode == 'fan_in':
            scale = np.sqrt(1.0 / fan_in)
        elif self.mode == 'fan_out':
            scale = np.sqrt(1.0 / fan_out)
        else:  # fan_avg
            scale = np.sqrt(2.0 / (fan_in + fan_out))
        
        return self.rng.normal(0, scale, shape).astype(np.float32)


class TruncatedNormalInitializer(BaseInitializer):
    """截断正态分布初始化器"""
    
    def __init__(self, mean: float = 0.0, std: float = 0.01,
                 num_std: float = 2.0, random_seed: Optional[int] = None):
        """
        初始化
        
        Args:
            mean: 均值
            std: 标准差
            num_std: 截断的标准差数量
            random_seed: 随机种子
        """
        super().__init__(random_seed)
        self.mean = mean
        self.std = std
        self.num_std = num_std
    
    def initialize(self, shape: Tuple[int, ...]) -> np.ndarray:
        """使用截断正态分布初始化"""
        lower = self.mean - self.num_std * self.std
        upper = self.mean + self.num_std * self.std
        
        # 生成正态分布样本并截断
        samples = self.rng.normal(self.mean, self.std, shape)
        samples = np.clip(samples, lower, upper)
        
        return samples.astype(np.float32)