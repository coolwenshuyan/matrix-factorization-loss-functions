#!/usr/bin/env python3
"""
初始化器配置功能测试脚本
验证所有初始化器配置是否正常工作
"""

import sys
import os
import numpy as np
import logging

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_initializer_configs():
    """测试初始化器配置功能"""
    logger.info("测试初始化器配置...")
    
    try:
        from configs.initializer_configs import (
            get_initializer_configs,
            create_config_with_initializer,
            create_base_config,
            get_recommended_configs_by_scenario,
            generate_experiment_configs
        )
        
        # 测试1: 获取所有配置
        configs = get_initializer_configs()
        logger.info(f"✓ 成功获取 {len(configs)} 个初始化器配置")
        
        # 测试2: 创建配置
        base_config = create_base_config()
        test_config = create_config_with_initializer(base_config, 'xavier_fan_avg')
        logger.info("✓ 成功创建带初始化器的配置")
        
        # 测试3: 场景推荐
        scenarios = get_recommended_configs_by_scenario()
        logger.info(f"✓ 成功获取 {len(scenarios)} 个场景推荐")
        
        # 测试4: 实验配置生成
        exp_configs = generate_experiment_configs()
        logger.info(f"✓ 成功生成 {len(exp_configs)} 个实验配置")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 配置功能测试失败: {e}")
        return False

def test_initializer_classes():
    """测试初始化器类"""
    logger.info("测试初始化器类...")
    
    try:
        from src.models.initializers import (
            NormalInitializer,
            XavierInitializer,
            UniformInitializer,
            TruncatedNormalInitializer
        )
        
        # 测试形状
        test_shape = (100, 50)
        
        # 测试每个初始化器
        initializers = {
            'Normal': NormalInitializer(mean=0.0, std=0.01, random_seed=42),
            'Xavier': XavierInitializer(mode='fan_avg', random_seed=42),
            'Uniform': UniformInitializer(low=-0.01, high=0.01, random_seed=42),
            'TruncatedNormal': TruncatedNormalInitializer(mean=0.0, std=0.01, num_std=2.0, random_seed=42)
        }
        
        for name, initializer in initializers.items():
            data = initializer.initialize(test_shape)
            
            # 检查形状
            assert data.shape == test_shape, f"{name}: 形状不匹配"
            
            # 检查数据类型
            assert data.dtype == np.float32, f"{name}: 数据类型不正确"
            
            # 检查是否有NaN
            assert not np.any(np.isnan(data)), f"{name}: 包含NaN值"
            
            logger.info(f"✓ {name}初始化器测试通过 - 均值: {np.mean(data):.6f}, 标准差: {np.std(data):.6f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 初始化器类测试失败: {e}")
        return False

def test_training_integration():
    """测试训练集成"""
    logger.info("测试训练集成...")
    
    try:
        from example.complete_training_example import MatrixFactorizationTrainer
        from configs.initializer_configs import create_config_with_initializer, create_base_config
        
        # 创建最小配置用于测试
        base_config = create_base_config()
        base_config['n_epochs'] = 1  # 只训练1轮用于测试
        
        # 测试不同初始化器
        test_initializers = ['normal_small', 'xavier_fan_avg', 'uniform_small']
        
        for init_name in test_initializers:
            config = create_config_with_initializer(base_config, init_name)
            trainer = MatrixFactorizationTrainer(config)
            
            # 验证训练器可以创建
            assert trainer.config['initializer']['type'] in ['Normal', 'Xavier', 'Uniform'], \
                f"初始化器类型不正确: {trainer.config['initializer']['type']}"
            
            logger.info(f"✓ {init_name}训练器创建成功")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 训练集成测试失败: {e}")
        return False

def test_configuration_completeness():
    """测试配置完整性"""
    logger.info("测试配置完整性...")
    
    try:
        from configs.initializer_configs import get_initializer_configs
        
        configs = get_initializer_configs()
        required_types = ['Normal', 'Xavier', 'Uniform', 'TruncatedNormal']
        
        # 检查每种类型是否都有配置
        found_types = set()
        for config in configs.values():
            found_types.add(config['type'])
        
        for req_type in required_types:
            assert req_type in found_types, f"缺少 {req_type} 类型的配置"
            logger.info(f"✓ 找到 {req_type} 类型配置")
        
        # 检查特定配置的存在
        required_configs = [
            'normal_small', 'xavier_fan_avg', 'uniform_small', 'truncated_normal_2std'
        ]
        
        for req_config in required_configs:
            assert req_config in configs, f"缺少必需配置: {req_config}"
            logger.info(f"✓ 找到必需配置: {req_config}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 配置完整性测试失败: {e}")
        return False

def test_parameter_validation():
    """测试参数验证"""
    logger.info("测试参数验证...")
    
    try:
        from src.models.initializers import NormalInitializer, XavierInitializer
        
        # 测试正常参数
        normal_init = NormalInitializer(mean=0.0, std=0.01)
        data = normal_init.initialize((10, 5))
        assert data.shape == (10, 5)
        logger.info("✓ 正常参数测试通过")
        
        # 测试Xavier初始化的mode参数
        for mode in ['fan_in', 'fan_out', 'fan_avg']:
            xavier_init = XavierInitializer(mode=mode)
            data = xavier_init.initialize((10, 5))
            assert data.shape == (10, 5)
            logger.info(f"✓ Xavier {mode} 模式测试通过")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 参数验证测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("="*60)
    print("初始化器配置功能测试")
    print("="*60)
    
    tests = [
        ("配置功能", test_initializer_configs),
        ("初始化器类", test_initializer_classes),
        ("训练集成", test_training_integration),
        ("配置完整性", test_configuration_completeness),
        ("参数验证", test_parameter_validation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
    
    # 总结
    print(f"\n{'='*20} 测试总结 {'='*20}")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{test_name:<15} {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！初始化器配置功能正常。")
        return True
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return False

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
