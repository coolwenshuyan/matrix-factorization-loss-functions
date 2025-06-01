#!/usr/bin/env python3
"""
快速配置MAE优化示例

展示如何在现有框架中快速配置MAE作为优化目标的简单示例
"""

import sys
import os
import numpy as np

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data.data_manager import DataManager
from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.initializers import NormalInitializer
from src.models.regularizers import L2Regularizer
from src.losses.hpl import HybridPiecewiseLoss
from src.losses.standard import L2Loss, L1Loss
from src.evaluation.metrics import MAE, RMSE


def quick_mae_example():
    """快速MAE配置示例"""
    print("快速MAE配置示例")
    print("="*50)
    
    # 1. 数据准备
    print("1. 准备数据...")
    data_config = {
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'center_data': True
    }
    
    data_manager = DataManager(data_config)
    data_manager.load_dataset('movielens100k', 'dataset/20201202M100K_data_all_random.txt')
    data_manager.preprocess()
    
    train_data, val_data, test_data = data_manager.get_splits()
    stats = data_manager.get_statistics()
    
    print(f"数据统计: {stats['n_users']} 用户, {stats['n_items']} 物品")
    print(f"训练集: {len(train_data)} 条")
    
    # 2. 初始化评估指标
    mae_metric = MAE()
    rmse_metric = RMSE()
    
    # 3. 定义三种损失函数配置
    loss_configs = {
        'HPL': {
            'loss': HybridPiecewiseLoss(delta1=0.5, delta2=1.5, l_max=4.0, c_sigmoid=1.0),
            'name': 'HPL混合分段损失'
        },
        'L2': {
            'loss': L2Loss(),
            'name': 'L2损失(MSE)'
        },
        'L1': {
            'loss': L1Loss(epsilon=1e-8),
            'name': 'L1损失(MAE)'
        }
    }
    
    # 4. 测试每种损失函数在MAE指标上的表现
    results = {}
    
    for loss_name, loss_config in loss_configs.items():
        print(f"\n2. 测试 {loss_config['name']}...")
        
        # 🎯 关键配置：创建模型时指定损失函数
        model = MatrixFactorizationSGD(
            n_users=stats['n_users'],
            n_items=stats['n_items'],
            n_factors=50,                    # 潜在因子数
            learning_rate=0.02,             # 学习率
            regularizer=L2Regularizer(lambda_reg=0.01),  # 正则化
            loss_function=loss_config['loss'],           # 🎯 指定损失函数
            use_bias=True,
            global_mean=data_manager.global_mean or 0.0
        )
        
        # 初始化参数
        initializer = NormalInitializer(mean=0.0, std=0.01)
        model.initialize_parameters(initializer)
        
        # 训练模型
        print(f"   训练 {loss_name} 模型...")
        model.fit(
            train_data=train_data,
            val_data=val_data,
            n_epochs=30,                    # 训练轮数
            verbose=0                       # 不显示详细过程
        )
        
        # 🎯 关键评估：在验证集上计算MAE和RMSE
        val_predictions = model.predict(
            val_data[:, 0].astype(int),
            val_data[:, 1].astype(int)
        )
        
        # 还原数据尺度
        if data_manager.global_mean is not None:
            val_predictions += data_manager.global_mean
            val_targets = val_data[:, 2] + data_manager.global_mean
        else:
            val_targets = val_data[:, 2]
        
        # 🎯 计算MAE和RMSE指标
        mae_score = mae_metric.calculate(val_targets, val_predictions)
        rmse_score = rmse_metric.calculate(val_targets, val_predictions)
        
        results[loss_name] = {
            'mae': mae_score,
            'rmse': rmse_score,
            'name': loss_config['name']
        }
        
        print(f"   {loss_name} 结果: MAE={mae_score:.4f}, RMSE={rmse_score:.4f}")
    
    # 5. 🎯 关键结果：MAE指标对比
    print("\n" + "="*50)
    print("MAE指标对比结果")
    print("="*50)
    
    print(f"{'损失函数':<15} {'MAE':<10} {'RMSE':<10} {'MAE排名':<10}")
    print("-" * 50)
    
    # 按MAE排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
    
    for rank, (loss_name, result) in enumerate(sorted_results, 1):
        print(f"{result['name']:<15} {result['mae']:<10.4f} {result['rmse']:<10.4f} {rank:<10}")
    
    # 6. 分析和建议
    print(f"\n分析:")
    best_loss, best_result = sorted_results[0]
    print(f"✅ {best_result['name']} 在MAE指标上表现最佳: {best_result['mae']:.4f}")
    
    if best_loss == 'L1':
        print("🎯 符合预期：L1损失直接优化MAE，理论上应该最优")
    elif best_loss == 'HPL':
        print("🎉 HPL损失在MAE上超越了L1！分段策略确实有效")
    else:
        print("📊 L2损失意外表现最佳，可能需要调整其他参数")
    
    print(f"\n💡 配置建议:")
    print(f"如果要优化MAE指标，推荐配置:")
    print(f"1. 损失函数: {best_result['name']}")
    print(f"2. 评估指标: MAE")
    print(f"3. 参数调整: 可以进一步优化学习率、正则化等")
    
    return results


def show_mae_configuration_guide():
    """展示MAE配置指南"""
    print("\n" + "="*60)
    print("MAE配置完整指南")
    print("="*60)
    
    guide = """
📋 如何配置MAE作为优化目标：

1️⃣ 导入MAE评估指标：
   from src.evaluation.metrics import MAE
   mae_metric = MAE()

2️⃣ 选择适合MAE的损失函数：
   方案A：直接使用L1损失（理论最优）
   from src.losses.standard import L1Loss
   loss_function = L1Loss(epsilon=1e-8)
   
   方案B：使用HPL损失（可能更优）
   from src.losses.hpl import HybridPiecewiseLoss
   loss_function = HybridPiecewiseLoss(delta1=0.5, delta2=1.5)
   
   方案C：使用L2损失（作为基线）
   from src.losses.standard import L2Loss
   loss_function = L2Loss()

3️⃣ 创建目标函数优化MAE：
   def objective_function(config):
       # 创建模型（使用选定的损失函数）
       model = MatrixFactorizationSGD(
           n_users=n_users,
           n_items=n_items,
           n_factors=config['factors'],
           learning_rate=config['lr'],
           loss_function=loss_function,  # 🎯 关键：指定损失函数
           ...
       )
       
       # 训练模型
       model.fit(train_data, val_data, ...)
       
       # 🎯 关键：评估MAE
       predictions = model.predict(...)
       mae = mae_metric.calculate(targets, predictions)
       return mae  # 返回MAE作为优化目标

4️⃣ 配置超参数优化：
   optimizer = HyperOptimizer(
       objective_fn=objective_function,
       maximize=False,  # 🎯 关键：最小化MAE
       ...
   )

5️⃣ 运行优化并分析结果：
   best_trial = optimizer.optimize(n_trials=50)
   print(f"最佳MAE: {best_trial.score}")

💡 重要提示：
- L1损失理论上最适合MAE优化
- HPL损失可能通过分段策略获得更好效果
- 记得设置maximize=False来最小化MAE
- 可以同时监控RMSE来全面评估
"""
    
    print(guide)


if __name__ == '__main__':
    print("MAE配置示例和指南")
    print("="*60)
    
    # 运行快速示例
    results = quick_mae_example()
    
    # 显示配置指南
    show_mae_configuration_guide()
    
    print("\n✅ 示例运行完成！")
    print("💡 参考上面的配置指南来设置您自己的MAE优化实验")
