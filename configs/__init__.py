"""
矩阵分解初始化器配置包

提供各种初始化策略的预定义配置和便利函数。

主要功能:
- get_initializer_configs(): 获取所有预定义的初始化器配置
- create_config_with_initializer(): 创建带有特定初始化器的完整配置
- get_recommended_configs_by_scenario(): 获取按场景分类的推荐配置
- generate_experiment_configs(): 生成用于对比实验的配置列表

使用示例:
    from configs.initializer_configs import get_initializer_configs, create_config_with_initializer
    
    # 获取所有配置
    configs = get_initializer_configs()
    
    # 创建使用Xavier初始化的配置
    base_config = create_base_config()
    config = create_config_with_initializer(base_config, 'xavier_fan_avg')
"""

from .initializer_configs import (
    get_initializer_configs,
    create_config_with_initializer,
    create_base_config,
    get_recommended_configs_by_scenario,
    generate_experiment_configs,
    get_initializer_description,
    print_all_configurations
)

__all__ = [
    'get_initializer_configs',
    'create_config_with_initializer',
    'create_base_config',
    'get_recommended_configs_by_scenario',
    'generate_experiment_configs',
    'get_initializer_description',
    'print_all_configurations'
]

# 版本信息
__version__ = '1.0.0'
__author__ = 'coolwen'
__description__ = 'Initializer configurations for matrix factorization models'
