#!/usr/bin/env python3
"""
使用初始化器配置进行矩阵分解训练的示例
展示如何使用configs/initializer_configs.py中的配置
"""

import sys
import os
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入配置和训练模块
from configs.initializer_configs import (
    get_initializer_configs, 
    create_config_with_initializer,
    create_base_config,
    get_recommended_configs_by_scenario,
    print_all_configurations
)
from example.complete_training_example import MatrixFactorizationTrainer

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_single_initializer():
    """演示使用单个初始化器训练"""
    logger.info("演示单个初始化器训练...")
    
    # 创建基础配置
    base_config = create_base_config()
    base_config['n_epochs'] = 20  # 减少训练轮数用于演示
    
    # 使用Xavier初始化器
    config = create_config_with_initializer(base_config, 'xavier_fan_avg')
    
    # 创建训练器
    trainer = MatrixFactorizationTrainer(config)
    
    try:
        # 准备数据
        trainer.prepare_data(
            dataset_name='movielens100k',
            data_path='dataset/20201202M100K_data_all_random.txt'
        )
        
        # 创建和训练模型
        trainer.create_model()
        trainer.train()
        trainer.evaluate()
        
        logger.info("单个初始化器训练完成!")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")


def demo_scenario_based_selection():
    """演示基于场景选择初始化器"""
    logger.info("演示基于场景选择初始化器...")
    
    # 获取场景推荐
    scenarios = get_recommended_configs_by_scenario()
    
    # 选择大规模数据场景
    large_scale_configs = scenarios['large_scale']['configs']
    logger.info(f"大规模数据推荐的初始化器: {large_scale_configs}")
    
    # 使用推荐的第一个配置
    base_config = create_base_config()
    base_config['n_epochs'] = 10  # 更少轮数用于演示
    
    recommended_init = large_scale_configs[0]  # xavier_fan_avg
    config = create_config_with_initializer(base_config, recommended_init)
    
    logger.info(f"使用推荐的初始化器: {recommended_init}")
    logger.info(f"初始化器配置: {config['initializer']}")


def demo_batch_experiment():
    """演示批量初始化器实验"""
    logger.info("演示批量初始化器实验...")
    
    # 获取所有配置
    all_configs = get_initializer_configs()
    
    # 选择几个代表性配置进行快速测试
    test_configs = [
        'normal_small',
        'xavier_fan_avg', 
        'uniform_small',
        'truncated_normal_2std'
    ]
    
    base_config = create_base_config()
    base_config['n_epochs'] = 5  # 很少轮数用于快速演示
    
    results = {}
    
    for init_name in test_configs:
        logger.info(f"测试初始化器: {init_name}")
        
        # 创建配置
        config = create_config_with_initializer(base_config, init_name)
        
        # 记录配置信息
        results[init_name] = {
            'config': config['initializer'],
            'description': all_configs[init_name]
        }
        
        logger.info(f"  配置: {config['initializer']}")
    
    # 打印结果汇总
    print("\n" + "="*60)
    print("批量初始化器测试配置汇总")
    print("="*60)
    
    for init_name, result in results.items():
        print(f"\n{init_name}:")
        print(f"  配置: {result['config']}")


def demo_custom_initializer():
    """演示自定义初始化器配置"""
    logger.info("演示自定义初始化器配置...")
    
    # 创建自定义配置
    custom_config = create_base_config()
    custom_config['n_epochs'] = 15
    
    # 手动设置初始化器配置
    custom_config['initializer'] = {
        'type': 'TruncatedNormal',
        'mean': 0.01,          # 小的正偏移
        'std': 0.05,           # 中等标准差
        'num_std': 1.5         # 较紧的截断
    }
    
    logger.info("自定义初始化器配置:")
    logger.info(f"  {custom_config['initializer']}")
    
    # 也可以测试是否有效
    try:
        trainer = MatrixFactorizationTrainer(custom_config)
        logger.info("自定义配置验证成功!")
    except Exception as e:
        logger.error(f"自定义配置验证失败: {e}")


def demo_initializer_analysis():
    """演示初始化器特性分析"""
    logger.info("演示初始化器特性分析...")
    
    # 模拟不同矩阵大小下的初始化
    from src.models.initializers import (
        NormalInitializer, XavierInitializer, 
        UniformInitializer, TruncatedNormalInitializer
    )
    
    # 测试不同的矩阵形状
    shapes = [(100, 50), (1000, 100), (5000, 200)]
    
    print("\n" + "="*80)
    print("初始化器在不同矩阵形状下的统计特性")
    print("="*80)
    
    for shape in shapes:
        print(f"\n矩阵形状: {shape}")
        print("-" * 40)
        
        # Normal初始化
        normal_init = NormalInitializer(mean=0.0, std=0.01, random_seed=42)
        normal_data = normal_init.initialize(shape)
        
        # Xavier初始化
        xavier_init = XavierInitializer(mode='fan_avg', random_seed=42)
        xavier_data = xavier_init.initialize(shape)
        
        # Uniform初始化
        uniform_init = UniformInitializer(low=-0.01, high=0.01, random_seed=42)
        uniform_data = uniform_init.initialize(shape)
        
        # TruncatedNormal初始化
        truncated_init = TruncatedNormalInitializer(mean=0.0, std=0.01, num_std=2.0, random_seed=42)
        truncated_data = truncated_init.initialize(shape)
        
        # 打印统计信息
        initializers = {
            'Normal': normal_data,
            'Xavier': xavier_data,
            'Uniform': uniform_data,
            'TruncatedNormal': truncated_data
        }
        
        for name, data in initializers.items():
            print(f"{name:<15} 均值: {np.mean(data):.6f}, "
                  f"标准差: {np.std(data):.6f}, "
                  f"范围: [{np.min(data):.6f}, {np.max(data):.6f}]")


def main():
    """主函数 - 运行所有演示"""
    print("初始化器配置使用演示")
    print("="*60)
    
    # 首先显示所有可用配置
    print_all_configurations()
    
    # 运行各种演示
    demos = [
        demo_single_initializer,
        demo_scenario_based_selection,
        demo_batch_experiment,
        demo_custom_initializer,
        demo_initializer_analysis
    ]
    
    for i, demo_func in enumerate(demos, 1):
        print(f"\n{'='*20} 演示 {i} {'='*20}")
        try:
            demo_func()
        except Exception as e:
            logger.error(f"演示 {i} 失败: {e}")
        print("="*50)


if __name__ == '__main__':
    main()
