"""
超参数优化模块

提供完整的超参数优化功能，包括：
- 参数空间定义
- 多种采样策略
- 约束管理
- 并行执行
- 实验追踪
"""

from .space import (
    ParameterSpace,
    ContinuousParameter,
    DiscreteParameter,
    CategoricalParameter,
    ConditionalParameter
)
from .sampler import (
    RandomSampler,
    LatinHypercubeSampler,
    SobolSampler,
    AdaptiveSampler
)
from .optimizer import HyperOptimizer
from .constraints import (
    Constraint,
    RangeConstraint,
    RelationalConstraint,
    ConditionalConstraint,
    ConstraintManager
)
from .parallel import ParallelExecutor
from .tracker import ExperimentTracker
from .utils import (
    normalize_config,
    denormalize_config,
    config_to_vector,
    vector_to_config
)

__all__ = [
    # 参数空间
    'ParameterSpace',
    'ContinuousParameter',
    'DiscreteParameter',
    'CategoricalParameter',
    'ConditionalParameter',
    
    # 采样器
    'RandomSampler',
    'LatinHypercubeSampler',
    'SobolSampler',
    'AdaptiveSampler',
    
    # 优化器
    'HyperOptimizer',
    
    # 约束
    'Constraint',
    'RangeConstraint',
    'RelationalConstraint',
    'ConditionalConstraint',
    'ConstraintManager',
    
    # 并行执行
    'ParallelExecutor',
    
    # 实验追踪
    'ExperimentTracker',
    
    # 工具函数
    'normalize_config',
    'denormalize_config',
    'config_to_vector',
    'vector_to_config'
]

__version__ = '1.0.0'