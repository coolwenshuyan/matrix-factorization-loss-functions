#!/usr/bin/env python3
"""
矩阵分解模型的各种初始化策略配置示例
包含Normal、Xavier、Uniform、TruncatedNormal等初始化方法
"""

def get_initializer_configs():
    """
    获取各种初始化策略的配置字典

    Returns:
        dict: 包含多种初始化配置的字典
    """

    initializer_configs = {

        # 1. Normal初始化 (正态分布初始化)
        'normal_small': {
            'type': 'Normal',
            'mean': 0.0,
            'std': 0.01         # 小标准差，适合大部分情况
        },

        'normal_medium': {
            'type': 'Normal',
            'mean': 0.0,
            'std': 0.1          # 中等标准差
        },

        'normal_large': {
            'type': 'Normal',
            'mean': 0.0,
            'std': 0.5          # 大标准差，可能导致训练不稳定
        },

        'normal_custom': {
            'type': 'Normal',
            'mean': 0.05,       # 非零均值
            'std': 0.02
        },

        # 2. Xavier/Glorot初始化
        'xavier_fan_avg': {
            'type': 'Xavier',
            'mode': 'fan_avg'   # 默认模式，基于输入输出维度平均值
        },

        'xavier_fan_in': {
            'type': 'Xavier',
            'mode': 'fan_in'    # 基于输入维度
        },

        'xavier_fan_out': {
            'type': 'Xavier',
            'mode': 'fan_out'   # 基于输出维度
        },

        # 3. Uniform初始化 (均匀分布初始化)
        'uniform_small': {
            'type': 'Uniform',
            'low': -0.01,
            'high': 0.01        # 小范围均匀分布
        },

        'uniform_medium': {
            'type': 'Uniform',
            'low': -0.1,
            'high': 0.1         # 中等范围均匀分布
        },

        'uniform_large': {
            'type': 'Uniform',
            'low': -0.5,
            'high': 0.5         # 大范围均匀分布
        },

        'uniform_asymmetric': {
            'type': 'Uniform',
            'low': 0.0,
            'high': 0.1         # 非对称范围（仅正值）
        },

        # 4. TruncatedNormal初始化 (截断正态分布初始化)
        'truncated_normal_2std': {
            'type': 'TruncatedNormal',
            'mean': 0.0,
            'std': 0.01,
            'num_std': 2.0      # 截断在±2σ
        },

        'truncated_normal_3std': {
            'type': 'TruncatedNormal',
            'mean': 0.0,
            'std': 0.05,
            'num_std': 3.0      # 截断在±3σ
        },

        'truncated_normal_custom': {
            'type': 'TruncatedNormal',
            'mean': 0.02,
            'std': 0.03,
            'num_std': 2.5      # 自定义截断
        }
    }

    return initializer_configs


def create_config_with_initializer(base_config, initializer_name):
    """
    创建带有特定初始化器的完整配置

    Args:
        base_config (dict): 基础配置
        initializer_name (str): 初始化器名称

    Returns:
        dict: 完整的配置字典
    """
    initializer_configs = get_initializer_configs()

    if initializer_name not in initializer_configs:
        raise ValueError(f"未知的初始化器: {initializer_name}")

    config = base_config.copy()
    config['initializer'] = initializer_configs[initializer_name]

    return config


def create_base_config():
    """创建基础配置模板"""
    return {
        # 数据配置
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'center_data': True,

        # 模型配置
        'n_factors': 50,
        'use_bias': True,

        # 训练配置
        'n_epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.01,
        'clip_gradient': 5.0,
        'momentum': 0.9,
        'lr_schedule': 'exponential',
        'early_stopping_patience': 10,
        'verbose': 1,

        # 损失函数配置
        'loss_function': {
            'type': 'L2'
        },

        # 正则化配置
        'regularizer': {
            'type': 'L2',
            'lambda_reg': 0.01,
            'lambda_bias': 0.001
        },

        # 初始化配置（将被替换）
        'initializer': {
            'type': 'Normal',
            'mean': 0.0,
            'std': 0.01
        }
    }


def get_recommended_configs_by_scenario():
    """
    根据不同场景推荐初始化配置

    Returns:
        dict: 按场景分类的推荐配置
    """

    scenarios = {

        # 场景1: 标准推荐系统
        'standard_recommendation': {
            'description': '适用于大多数推荐系统场景',
            'configs': [
                'normal_small',      # 最常用
                'xavier_fan_avg',    # 自适应调整
                'uniform_small'      # 稳定选择
            ]
        },

        # 场景2: 大规模数据集
        'large_scale': {
            'description': '适用于大规模用户/物品数据集',
            'configs': [
                'xavier_fan_avg',    # 自动缩放
                'normal_small',      # 保守选择
                'truncated_normal_2std'  # 避免极值
            ]
        },

        # 场景3: 稀疏数据
        'sparse_data': {
            'description': '适用于非常稀疏的评分矩阵',
            'configs': [
                'normal_small',      # 小幅度初始化
                'uniform_small',     # 控制范围
                'truncated_normal_2std'  # 避免过大初值
            ]
        },

        # 场景4: 高维潜在因子
        'high_dimensional': {
            'description': '潜在因子数量较大(>100)的情况',
            'configs': [
                'xavier_fan_avg',    # 根据维度调整
                'xavier_fan_in',     # 基于输入维度
                'normal_small'       # 保守策略
            ]
        },

        # 场景5: 快速原型
        'prototyping': {
            'description': '快速实验和原型开发',
            'configs': [
                'normal_small',      # 默认选择
                'uniform_medium',    # 快速收敛
                'xavier_fan_avg'     # 平衡选择
            ]
        },

        # 场景6: 精细调优
        'fine_tuning': {
            'description': '需要精细调参的生产环境',
            'configs': [
                'truncated_normal_2std',    # 稳定性好
                'xavier_fan_avg',           # 理论基础强
                'normal_custom'             # 可定制
            ]
        }
    }

    return scenarios


def generate_experiment_configs():
    """
    生成用于对比实验的配置列表

    Returns:
        list: 实验配置列表
    """
    base_config = create_base_config()
    initializer_configs = get_initializer_configs()

    experiment_configs = []

    # 为每种初始化器创建实验配置
    for name, init_config in initializer_configs.items():
        config = base_config.copy()
        config['initializer'] = init_config
        config['experiment_name'] = f'init_{name}'

        experiment_configs.append({
            'name': name,
            'config': config,
            'description': get_initializer_description(name)
        })

    return experiment_configs


def get_initializer_description(initializer_name):
    """
    获取初始化器的描述信息

    Args:
        initializer_name (str): 初始化器名称

    Returns:
        str: 描述信息
    """
    descriptions = {
        'normal_small': 'Normal(μ=0, σ=0.01) - 小标准差正态分布，最常用',
        'normal_medium': 'Normal(μ=0, σ=0.1) - 中等标准差正态分布',
        'normal_large': 'Normal(μ=0, σ=0.5) - 大标准差正态分布，可能不稳定',
        'normal_custom': 'Normal(μ=0.05, σ=0.02) - 自定义均值和标准差',

        'xavier_fan_avg': 'Xavier初始化 (fan_avg模式) - 自适应缩放',
        'xavier_fan_in': 'Xavier初始化 (fan_in模式) - 基于输入维度',
        'xavier_fan_out': 'Xavier初始化 (fan_out模式) - 基于输出维度',

        'uniform_small': 'Uniform[-0.01, 0.01] - 小范围均匀分布',
        'uniform_medium': 'Uniform[-0.1, 0.1] - 中等范围均匀分布',
        'uniform_large': 'Uniform[-0.5, 0.5] - 大范围均匀分布',
        'uniform_asymmetric': 'Uniform[0, 0.1] - 非对称均匀分布',

        'truncated_normal_2std': 'TruncatedNormal(μ=0, σ=0.01, ±2σ) - 截断正态分布',
        'truncated_normal_3std': 'TruncatedNormal(μ=0, σ=0.05, ±3σ) - 宽截断正态分布',
        'truncated_normal_custom': 'TruncatedNormal(μ=0.02, σ=0.03, ±2.5σ) - 自定义截断'
    }

    return descriptions.get(initializer_name, '未知初始化器')


def print_all_configurations():
    """打印所有可用的初始化配置"""

    print("=" * 80)
    print("矩阵分解初始化策略配置大全")
    print("=" * 80)

    initializer_configs = get_initializer_configs()

    # 按类型分组显示
    config_groups = {
        'Normal初始化': [k for k in initializer_configs if k.startswith('normal_')],
        'Xavier初始化': [k for k in initializer_configs if k.startswith('xavier_')],
        'Uniform初始化': [k for k in initializer_configs if k.startswith('uniform_')],
        'TruncatedNormal初始化': [k for k in initializer_configs if k.startswith('truncated_')]
    }

    for group_name, config_names in config_groups.items():
        print(f"\n{group_name}:")
        print("-" * 40)

        for name in config_names:
            config = initializer_configs[name]
            description = get_initializer_description(name)

            print(f"\n'{name}': {{")
            for key, value in config.items():
                print(f"    '{key}': {repr(value)},")
            print("}")
            print(f"# {description}")

    # 显示推荐场景
    print("\n" + "=" * 80)
    print("推荐使用场景")
    print("=" * 80)

    scenarios = get_recommended_configs_by_scenario()
    for scenario_name, scenario_info in scenarios.items():
        print(f"\n{scenario_name.replace('_', ' ').title()}:")
        print(f"描述: {scenario_info['description']}")
        print("推荐配置:", ", ".join(scenario_info['configs']))


# 使用示例
if __name__ == '__main__':

    # 示例1: 获取特定初始化器配置
    configs = get_initializer_configs()
    xavier_config = configs['xavier_fan_avg']
    print("Xavier配置:", xavier_config)

    # 示例2: 创建完整的训练配置
    base_config = create_base_config()
    full_config = create_config_with_initializer(base_config, 'xavier_fan_avg')
    print("\n完整配置示例:")
    print(f"初始化器: {full_config['initializer']}")

    # 示例3: 生成实验配置
    experiment_configs = generate_experiment_configs()
    print(f"\n总共生成 {len(experiment_configs)} 个实验配置")

    # 示例4: 打印所有配置
    print_all_configurations()

    # 示例5: 特定场景推荐
    scenarios = get_recommended_configs_by_scenario()
    large_scale_configs = scenarios['large_scale']['configs']
    print(f"\n大规模数据推荐配置: {large_scale_configs}")
