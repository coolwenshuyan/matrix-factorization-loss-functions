"""
参数空间定义模块

提供超参数空间的定义和管理功能
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union, Callable
from abc import ABC, abstractmethod


class Parameter(ABC):
    """参数基类"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def sample(self, random_state: np.random.RandomState) -> Any:
        """采样一个值"""
        pass
    
    @abstractmethod
    def normalize(self, value: Any) -> float:
        """将值归一化到[0,1]"""
        pass
    
    @abstractmethod
    def denormalize(self, normalized_value: float) -> Any:
        """将归一化值转换回原始值"""
        pass


class ContinuousParameter(Parameter):
    """连续参数"""
    
    def __init__(self, name: str, low: float, high: float, 
                 scale: str = 'linear'):
        """
        Args:
            name: 参数名
            low: 最小值
            high: 最大值
            scale: 缩放方式 ('linear' 或 'log')
        """
        super().__init__(name)
        self.low = low
        self.high = high
        self.scale = scale
        
        if scale == 'log':
            if low <= 0 or high <= 0:
                raise ValueError("对数缩放需要正数范围")
            self.log_low = np.log(low)
            self.log_high = np.log(high)
    
    def sample(self, random_state: np.random.RandomState) -> float:
        """采样一个值"""
        if self.scale == 'log':
            log_value = random_state.uniform(self.log_low, self.log_high)
            return np.exp(log_value)
        else:
            return random_state.uniform(self.low, self.high)
    
    def normalize(self, value: float) -> float:
        """归一化到[0,1]"""
        if self.scale == 'log':
            log_value = np.log(max(value, 1e-10))  # 防止log(0)
            return (log_value - self.log_low) / (self.log_high - self.log_low)
        else:
            return (value - self.low) / (self.high - self.low)
    
    def denormalize(self, normalized_value: float) -> float:
        """反归一化"""
        normalized_value = np.clip(normalized_value, 0, 1)
        
        if self.scale == 'log':
            log_value = self.log_low + normalized_value * (self.log_high - self.log_low)
            return np.exp(log_value)
        else:
            return self.low + normalized_value * (self.high - self.low)


class DiscreteParameter(Parameter):
    """离散参数"""
    
    def __init__(self, name: str, low: int, high: int, step: int = 1):
        """
        Args:
            name: 参数名
            low: 最小值
            high: 最大值
            step: 步长
        """
        super().__init__(name)
        self.low = low
        self.high = high
        self.step = step
        self.values = list(range(low, high + 1, step))
    
    def sample(self, random_state: np.random.RandomState) -> int:
        """采样一个值"""
        return random_state.choice(self.values)
    
    def normalize(self, value: int) -> float:
        """归一化到[0,1]"""
        try:
            index = self.values.index(value)
            return index / (len(self.values) - 1) if len(self.values) > 1 else 0
        except ValueError:
            # 如果值不在列表中，找最近的
            closest_idx = np.argmin([abs(v - value) for v in self.values])
            return closest_idx / (len(self.values) - 1) if len(self.values) > 1 else 0
    
    def denormalize(self, normalized_value: float) -> int:
        """反归一化"""
        normalized_value = np.clip(normalized_value, 0, 1)
        index = int(round(normalized_value * (len(self.values) - 1)))
        return self.values[index]


class CategoricalParameter(Parameter):
    """分类参数"""
    
    def __init__(self, name: str, choices: List[Any]):
        """
        Args:
            name: 参数名
            choices: 选项列表
        """
        super().__init__(name)
        self.choices = choices
    
    def sample(self, random_state: np.random.RandomState) -> Any:
        """采样一个值"""
        return random_state.choice(self.choices)
    
    def normalize(self, value: Any) -> float:
        """归一化到[0,1]"""
        try:
            index = self.choices.index(value)
            return index / (len(self.choices) - 1) if len(self.choices) > 1 else 0
        except ValueError:
            return 0  # 如果值不在选项中，返回0
    
    def denormalize(self, normalized_value: float) -> Any:
        """反归一化"""
        normalized_value = np.clip(normalized_value, 0, 1)
        index = int(round(normalized_value * (len(self.choices) - 1)))
        return self.choices[index]


class ConditionalParameter(Parameter):
    """条件参数"""
    
    def __init__(self, name: str, base_param: Parameter, 
                 condition: Callable[[Dict], bool]):
        """
        Args:
            name: 参数名
            base_param: 基础参数
            condition: 激活条件函数
        """
        super().__init__(name)
        self.base_param = base_param
        self.condition = condition
    
    def is_active(self, config: Dict) -> bool:
        """检查参数是否激活"""
        return self.condition(config)
    
    def sample(self, random_state: np.random.RandomState) -> Any:
        """采样一个值"""
        return self.base_param.sample(random_state)
    
    def normalize(self, value: Any) -> float:
        """归一化到[0,1]"""
        return self.base_param.normalize(value)
    
    def denormalize(self, normalized_value: float) -> Any:
        """反归一化"""
        return self.base_param.denormalize(normalized_value)


class ParameterSpace:
    """参数空间"""
    
    def __init__(self):
        self.parameters: Dict[str, Parameter] = {}
        self.param_order: List[str] = []
    
    def add_continuous(self, name: str, low: float, high: float, 
                      scale: str = 'linear'):
        """添加连续参数"""
        param = ContinuousParameter(name, low, high, scale)
        self.parameters[name] = param
        self.param_order.append(name)
    
    def add_discrete(self, name: str, low: int, high: int, step: int = 1):
        """添加离散参数"""
        param = DiscreteParameter(name, low, high, step)
        self.parameters[name] = param
        self.param_order.append(name)
    
    def add_categorical(self, name: str, choices: List[Any]):
        """添加分类参数"""
        param = CategoricalParameter(name, choices)
        self.parameters[name] = param
        self.param_order.append(name)
    
    def add_conditional(self, name: str, base_param: Parameter,
                       condition: Callable[[Dict], bool]):
        """添加条件参数"""
        param = ConditionalParameter(name, base_param, condition)
        self.parameters[name] = param
        self.param_order.append(name)
    
    def sample(self, random_state: np.random.RandomState) -> Dict:
        """采样一个配置"""
        config = {}
        
        for name in self.param_order:
            param = self.parameters[name]
            
            # 检查条件参数
            if isinstance(param, ConditionalParameter):
                if param.is_active(config):
                    config[name] = param.sample(random_state)
            else:
                config[name] = param.sample(random_state)
        
        return config
    
    def normalize(self, config: Dict) -> np.ndarray:
        """将配置归一化为向量"""
        vector = []
        
        for name in self.param_order:
            if name in config:
                param = self.parameters[name]
                normalized_value = param.normalize(config[name])
                vector.append(normalized_value)
            else:
                # 条件参数可能不存在
                vector.append(0.0)
        
        return np.array(vector)
    
    def denormalize(self, vector: np.ndarray) -> Dict:
        """将归一化向量转换为配置"""
        config = {}
        
        for i, name in enumerate(self.param_order):
            if i < len(vector):
                param = self.parameters[name]
                
                # 检查条件参数
                if isinstance(param, ConditionalParameter):
                    if param.is_active(config):
                        config[name] = param.denormalize(vector[i])
                else:
                    config[name] = param.denormalize(vector[i])
        
        return config
    
    def get_bounds(self) -> List[tuple]:
        """获取参数边界（用于优化算法）"""
        bounds = []
        
        for name in self.param_order:
            param = self.parameters[name]
            if isinstance(param, (ContinuousParameter, DiscreteParameter)):
                bounds.append((0.0, 1.0))  # 归一化后的边界
            elif isinstance(param, CategoricalParameter):
                bounds.append((0.0, 1.0))
            else:
                bounds.append((0.0, 1.0))
        
        return bounds
    
    def get_dimension(self) -> int:
        """获取参数空间维度"""
        return len(self.param_order)
    
    def validate_config(self, config: Dict) -> bool:
        """验证配置是否有效"""
        for name, value in config.items():
            if name in self.parameters:
                param = self.parameters[name]
                
                # 检查类型和范围
                if isinstance(param, ContinuousParameter):
                    if not isinstance(value, (int, float)):
                        return False
                    if not (param.low <= value <= param.high):
                        return False
                
                elif isinstance(param, DiscreteParameter):
                    if not isinstance(value, int):
                        return False
                    if value not in param.values:
                        return False
                
                elif isinstance(param, CategoricalParameter):
                    if value not in param.choices:
                        return False
        
        return True
    
    def get_parameter_info(self) -> Dict:
        """获取参数信息"""
        info = {}
        
        for name, param in self.parameters.items():
            if isinstance(param, ContinuousParameter):
                info[name] = {
                    'type': 'continuous',
                    'low': param.low,
                    'high': param.high,
                    'scale': param.scale
                }
            elif isinstance(param, DiscreteParameter):
                info[name] = {
                    'type': 'discrete',
                    'low': param.low,
                    'high': param.high,
                    'step': param.step,
                    'values': param.values
                }
            elif isinstance(param, CategoricalParameter):
                info[name] = {
                    'type': 'categorical',
                    'choices': param.choices
                }
            elif isinstance(param, ConditionalParameter):
                base_info = {}
                if isinstance(param.base_param, ContinuousParameter):
                    base_info = {
                        'type': 'continuous',
                        'low': param.base_param.low,
                        'high': param.base_param.high,
                        'scale': param.base_param.scale
                    }
                # 可以添加其他基础参数类型的处理
                
                info[name] = {
                    'type': 'conditional',
                    'base_param': base_info
                }
        
        return info
