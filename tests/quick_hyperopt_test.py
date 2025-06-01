#!/usr/bin/env python3
"""
Hyperopt模块快速演示

这是一个简化的演示，展示hyperopt模块的核心功能
"""

import numpy as np
import time
import json

def run_hyperopt_demo():
    """运行hyperopt演示"""
    
    print("🚀 Hyperopt模块演示开始")
    print("=" * 50)
    
    # 设置随机种子确保结果可重现
    np.random.seed(42)
    
    # 1. 参数空间定义
    print("\n📋 1. 参数空间定义")
    param_info = {
        'learning_rate': 'continuous [0.001, 0.1] log-scale',
        'latent_factors': 'discrete [20, 30, 40, 50, 60, 70, 80, 90, 100]',
        'lambda_reg': 'continuous [0.001, 0.1] log-scale', 
        'delta1': 'continuous [0.1, 1.0] linear-scale',
        'delta2': 'continuous [1.0, 3.0] linear-scale',
        'loss_type': 'categorical [hpl, l2]'
    }
    
    for param, info in param_info.items():
        print(f"   {param}: {info}")
    
    print("\n⚖️  约束条件: delta1 < delta2 (HPL损失函数要求)")
    
    # 2. 采样函数
    def sample_hyperparameters():
        """采样一组超参数"""
        config = {}
        
        # 学习率 (对数尺度)
        log_lr = np.random.uniform(np.log(0.001), np.log(0.1))
        config['learning_rate'] = np.exp(log_lr)
        
        # 潜在因子数 (离散)
        factors_choices = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        config['latent_factors'] = np.random.choice(factors_choices)
        
        # 正则化参数 (对数尺度)
        log_reg = np.random.uniform(np.log(0.001), np.log(0.1))
        config['lambda_reg'] = np.exp(log_reg)
        
        # HPL参数
        config['delta1'] = np.random.uniform(0.1, 1.0)
        config['delta2'] = np.random.uniform(1.0, 3.0)
        
        # 约束检查和修正
        if config['delta1'] >= config['delta2']:
            config['delta1'] = config['delta2'] * 0.8  # 自动修正约束
        
        # 损失函数类型
        config['loss_type'] = np.random.choice(['hpl', 'l2'])
        
        return config
    
    # 3. 目标函数 (模拟矩阵分解训练)
    def evaluate_config(config):
        """
        模拟矩阵分解模型训练和评估
        返回验证集RMSE (越小越好)
        """
        
        # 提取参数
        lr = config['learning_rate']
        nf = config['latent_factors'] 
        reg = config['lambda_reg']
        d1 = config['delta1']
        d2 = config['delta2']
        loss_type = config['loss_type']
        
        # 模拟基准性能
        base_rmse = 0.85
        
        # 各参数对性能的影响 (基于经验知识)
        
        # 学习率影响 (最优在0.01附近)
        lr_penalty = 0.1 * abs(lr - 0.01)
        
        # 因子数影响 (最优在50附近)  
        nf_penalty = 0.05 * abs(nf - 50) / 50
        
        # 正则化影响 (最优在0.01附近)
        reg_penalty = 0.1 * abs(reg - 0.01)
        
        # HPL特定参数影响
        hpl_penalty = 0
        if loss_type == 'hpl':
            hpl_penalty += 0.03 * abs(d1 - 0.5)    # delta1最优在0.5
            hpl_penalty += 0.03 * abs(d2 - 2.0)    # delta2最优在2.0
            hpl_penalty -= 0.02  # HPL整体性能稍好
        
        # 计算最终RMSE
        rmse = base_rmse + lr_penalty + nf_penalty + reg_penalty + hpl_penalty
        
        # 添加随机噪声模拟实验不确定性
        rmse += np.random.normal(0, 0.02)
        
        # 确保RMSE在合理范围内
        return max(0.5, rmse)
    
    # 4. 超参数优化过程
    print("\n🔍 2. 开始超参数优化")
    print("-" * 40)
    
    best_config = None
    best_rmse = float('inf')
    optimization_history = []
    
    start_time = time.time()
    
    # 运行20次试验
    n_trials = 20
    
    for trial_id in range(1, n_trials + 1):
        # 采样新配置
        config = sample_hyperparameters()
        
        # 评估配置
        rmse = evaluate_config(config)
        
        # 记录试验
        trial_info = {
            'trial_id': trial_id,
            'config': config,
            'rmse': rmse,
            'is_best': False
        }
        
        # 检查是否是新的最佳配置
        if rmse < best_rmse:
            best_rmse = rmse
            best_config = config.copy()
            trial_info['is_best'] = True
            
            print(f"Trial {trial_id:2d}: 🎯 新最佳! RMSE = {rmse:.4f} "
                  f"(lr={config['learning_rate']:.4f}, "
                  f"factors={config['latent_factors']}, "
                  f"loss={config['loss_type']})")
        else:
            print(f"Trial {trial_id:2d}: RMSE = {rmse:.4f}")
        
        optimization_history.append(trial_info)
    
    optimization_time = time.time() - start_time
    
    # 5. 结果分析
    print("\n📊 3. 优化结果分析")
    print("=" * 50)
    
    print(f"总试验数: {n_trials}")
    print(f"优化耗时: {optimization_time:.2f}秒")
    print(f"最佳RMSE: {best_rmse:.4f}")
    
    # 计算性能改进
    first_rmse = optimization_history[0]['rmse']
    improvement_pct = (first_rmse - best_rmse) / first_rmse * 100
    print(f"性能改进: {improvement_pct:.1f}% (从 {first_rmse:.4f} 提升到 {best_rmse:.4f})")
    
    # 最佳配置详情
    print(f"\n🏆 最佳配置:")
    for param, value in best_config.items():
        if isinstance(value, float):
            print(f"   {param}: {value:.6f}")
        else:
            print(f"   {param}: {value}")
    
    # 6. 损失函数类型分析
    print(f"\n🔬 4. 损失函数类型分析")
    
    hpl_trials = [t for t in optimization_history if t['config']['loss_type'] == 'hpl']
    l2_trials = [t for t in optimization_history if t['config']['loss_type'] == 'l2']
    
    if hpl_trials and l2_trials:
        hpl_avg = np.mean([t['rmse'] for t in hpl_trials])
        l2_avg = np.mean([t['rmse'] for t in l2_trials])
        
        print(f"HPL损失函数: {len(hpl_trials)} 次试验, 平均RMSE = {hpl_avg:.4f}")
        print(f"L2损失函数:  {len(l2_trials)} 次试验, 平均RMSE = {l2_avg:.4f}")
        
        if hpl_avg < l2_avg:
            print("✅ HPL损失函数表现更好!")
        else:
            print("✅ L2损失函数表现更好!")
    
    # 7. 详细试验表
    print(f"\n📋 5. 详细试验结果")
    print("-" * 85)
    print(f"{'Trial':<6} {'RMSE':<8} {'学习率':<10} {'因子':<6} {'正则化':<10} {'损失':<6} {'δ1':<6} {'δ2':<6}")
    print("-" * 85)
    
    for trial in optimization_history:
        config = trial['config']
        marker = "🎯" if trial['is_best'] else "  "
        
        print(f"{marker}{trial['trial_id']:<4} "
              f"{trial['rmse']:<8.4f} "
              f"{config['learning_rate']:<10.4f} "
              f"{config['latent_factors']:<6} "
              f"{config['lambda_reg']:<10.4f} "
              f"{config['loss_type']:<6} "
              f"{config['delta1']:<6.3f} "
              f"{config['delta2']:<6.3f}")
    
    # 8. 优化建议
    print(f"\n💡 6. 优化建议")
    print("=" * 50)
    
    if best_config['loss_type'] == 'hpl':
        print("🎯 建议使用HPL损失函数")
        print(f"   推荐HPL参数: δ1={best_config['delta1']:.3f}, δ2={best_config['delta2']:.3f}")
    else:
        print("🎯 建议使用L2损失函数")
    
    print(f"\n🔧 关键超参数设置:")
    print(f"   学习率: {best_config['learning_rate']:.5f}")
    print(f"   潜在因子数: {best_config['latent_factors']}")
    print(f"   正则化强度: {best_config['lambda_reg']:.5f}")
    
    print(f"\n📈 下一步建议:")
    print("   1. 在最佳配置周围进行精细搜索")
    print("   2. 使用更多训练轮数验证结果")
    print("   3. 在完整数据集上评估性能")
    print("   4. 考虑使用贝叶斯优化提升效率")
    
    # 9. 保存结果
    results = {
        'best_config': best_config,
        'best_rmse': best_rmse,
        'improvement_percent': improvement_pct,
        'optimization_time': optimization_time,
        'all_trials': optimization_history
    }
    
    try:
        with open('hyperopt_demo_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 结果已保存到: hyperopt_demo_results.json")
    except Exception as e:
        print(f"\n❌ 保存结果失败: {e}")
    
    print(f"\n✅ Hyperopt演示完成!")
    
    return results

def demonstrate_sampling_strategies():
    """演示不同采样策略的差异"""
    
    print(f"\n" + "=" * 60)
    print("🎲 采样策略对比演示")
    print("=" * 60)
    
    np.random.seed(42)
    
    # 1. 随机采样
    print(f"\n1️⃣ 随机采样 (Random Sampling)")
    print("   完全随机选择参数值")
    print("   优点: 简单易实现  缺点: 可能聚集在某些区域")
    
    random_samples = []
    for i in range(5):
        lr = np.exp(np.random.uniform(np.log(0.001), np.log(0.1)))
        factors = np.random.choice([20, 30, 40, 50, 60, 70, 80, 90, 100])
        random_samples.append((lr, factors))
        print(f"   样本{i+1}: 学习率={lr:.4f}, 因子数={factors}")
    
    # 2. 拉丁超立方采样 (简化版)
    print(f"\n2️⃣ 拉丁超立方采样 (Latin Hypercube Sampling)")
    print("   确保每个维度的均匀覆盖")
    print("   优点: 更好的空间覆盖  缺点: 实现复杂")
    
    n_samples = 5
    
    # 为学习率创建LHS样本
    lr_intervals = np.arange(n_samples) / n_samples
    lr_perm = np.random.permutation(n_samples)
    
    # 为因子数创建LHS样本  
    factor_intervals = np.arange(n_samples) / n_samples
    factor_perm = np.random.permutation(n_samples)
    
    lhs_samples = []
    for i in range(n_samples):
        # 在每个区间内随机采样
        lr_norm = (lr_perm[i] + np.random.random()) / n_samples
        factor_norm = (factor_perm[i] + np.random.random()) / n_samples
        
        # 转换到实际范围
        lr = np.exp(np.log(0.001) + lr_norm * (np.log(0.1) - np.log(0.001)))
        
        factor_choices = [20, 30, 40, 50, 60, 70, 80, 90, 100]
        factor_idx = int(factor_norm * len(factor_choices))
        factors = factor_choices[min(factor_idx, len(factor_choices)-1)]
        
        lhs_samples.append((lr, factors))
        print(f"   样本{i+1}: 学习率={lr:.4f}, 因子数={factors}")
    
    # 3. 对比分析
    print(f"\n3️⃣ 采样效果对比")
    
    # 学习率分布分析
    random_lrs = [s[0] for s in random_samples]
    lhs_lrs = [s[0] for s in lhs_samples]
    
    print(f"   学习率覆盖范围:")
    print(f"   随机采样: [{min(random_lrs):.4f}, {max(random_lrs):.4f}]")
    print(f"   LHS采样:  [{min(lhs_lrs):.4f}, {max(lhs_lrs):.4f}]")
    
    # 因子数分布分析
    random_factors = [s[1] for s in random_samples]
    lhs_factors = [s[1] for s in lhs_samples]
    
    print(f"   因子数覆盖范围:")
    print(f"   随机采样: [{min(random_factors)}, {max(random_factors)}]")
    print(f"   LHS采样:  [{min(lhs_factors)}, {max(lhs_factors)}]")
    
    print(f"\n🎯 总结:")
    print(f"   • 随机采样适合快速探索和简单场景")
    print(f"   • LHS采样适合需要均匀覆盖的精细优化")
    print(f"   • 实际使用中可根据试验预算选择策略")

if __name__ == '__main__':
    print("🔬 开始Hyperopt模块完整演示")
    
    # 运行主要演示
    results = run_hyperopt_demo()
    
    # 演示采样策略对比
    demonstrate_sampling_strategies()
    
    print(f"\n" + "=" * 60)
    print("🎉 演示完成!")
    print("=" * 60)
    print("这个演示展示了hyperopt模块的核心功能:")
    print("✓ 多类型参数空间定义 (连续、离散、分类)")
    print("✓ 约束条件自动检查和修正")
    print("✓ 智能超参数采样")
    print("✓ 模拟目标函数优化")
    print("✓ 完整的实验追踪和分析")
    print("✓ 多种采样策略对比")
    print(f"\n💡 在实际项目中，目标函数将是真实的模型训练过程")
    print(f"   可以直接替换 evaluate_config 函数为真实的训练流程")
