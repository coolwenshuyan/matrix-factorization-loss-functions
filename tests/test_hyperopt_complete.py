#!/usr/bin/env python3
"""
超参数优化简单演示
"""

import numpy as np
import json
import time

def main():
    print("=== Hyperopt模块演示 ===")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 1. 定义参数空间
    print("\n1. 参数空间定义:")
    param_space = {
        'learning_rate': {'type': 'continuous', 'bounds': (0.001, 0.1), 'scale': 'log'},
        'latent_factors': {'type': 'discrete', 'bounds': (20, 100), 'step': 10},
        'lambda_reg': {'type': 'continuous', 'bounds': (0.001, 0.1), 'scale': 'log'},
        'delta1': {'type': 'continuous', 'bounds': (0.1, 1.0), 'scale': 'linear'},
        'delta2': {'type': 'continuous', 'bounds': (1.0, 3.0), 'scale': 'linear'},
        'loss_type': {'type': 'categorical', 'choices': ['hpl', 'l2']}
    }
    
    for param, spec in param_space.items():
        print(f"  {param}: {spec}")
    
    # 2. 定义约束
    print("\n2. 约束条件:")
    constraints = ['delta1 < delta2']
    print(f"  {constraints}")
    
    # 3. 采样函数
    def sample_config():
        config = {}
        
        # learning_rate (log scale)
        log_lr = np.random.uniform(np.log(0.001), np.log(0.1))
        config['learning_rate'] = np.exp(log_lr)
        
        # latent_factors (discrete)
        factors = list(range(20, 101, 10))
        config['latent_factors'] = np.random.choice(factors)
        
        # lambda_reg (log scale)
        log_reg = np.random.uniform(np.log(0.001), np.log(0.1))
        config['lambda_reg'] = np.exp(log_reg)
        
        # delta1, delta2 (with constraint)
        config['delta1'] = np.random.uniform(0.1, 1.0)
        config['delta2'] = np.random.uniform(1.0, 3.0)
        
        # 确保约束
        if config['delta1'] >= config['delta2']:
            config['delta1'] = config['delta2'] * 0.8
        
        # loss_type
        config['loss_type'] = np.random.choice(['hpl', 'l2'])
        
        return config
    
    # 4. 目标函数（模拟）
    def objective_function(config):
        """模拟矩阵分解训练和评估"""
        
        # 模拟不同参数对性能的影响
        lr = config['learning_rate']
        nf = config['latent_factors']
        reg = config['lambda_reg']
        d1 = config['delta1']
        d2 = config['delta2']
        loss_type = config['loss_type']
        
        # 基准RMSE
        rmse = 0.85
        
        # 学习率影响（最优约0.01）
        rmse += 0.1 * abs(lr - 0.01)
        
        # 因子数影响（最优约50）
        rmse += 0.05 * abs(nf - 50) / 50
        
        # 正则化影响（最优约0.01）
        rmse += 0.1 * abs(reg - 0.01)
        
        # HPL参数影响
        if loss_type == 'hpl':
            rmse += 0.03 * abs(d1 - 0.5)
            rmse += 0.03 * abs(d2 - 2.0)
            rmse -= 0.02  # HPL稍微好一点
        
        # 添加噪声
        rmse += np.random.normal(0, 0.02)
        
        return max(0.5, rmse)
    
    # 5. 运行优化
    print("\n3. 开始超参数优化:")
    print("-" * 50)
    
    best_config = None
    best_score = float('inf')
    all_trials = []
    
    start_time = time.time()
    
    for trial in range(20):
        # 采样配置
        config = sample_config()
        
        # 评估配置
        score = objective_function(config)
        
        # 记录试验
        trial_info = {
            'trial_id': trial + 1,
            'config': config,
            'score': score,
            'time': time.time() - start_time
        }
        all_trials.append(trial_info)
        
        # 更新最佳配置
        if score < best_score:
            best_score = score
            best_config = config
            print(f"Trial {trial+1:2d}: 🎯 新最佳! RMSE = {score:.4f} "
                  f"(lr={config['learning_rate']:.4f}, factors={config['latent_factors']}, "
                  f"loss={config['loss_type']})")
        else:
            print(f"Trial {trial+1:2d}: RMSE = {score:.4f}")
    
    total_time = time.time() - start_time
    
    # 6. 显示结果
    print("\n" + "="*70)
    print("优化结果")
    print("="*70)
    
    print(f"总试验数: {len(all_trials)}")
    print(f"优化耗时: {total_time:.2f}秒")
    print(f"最佳RMSE: {best_score:.4f}")
    
    print("\n最佳配置:")
    for key, value in best_config.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # 7. 分析结果
    print("\n" + "="*70)
    print("结果分析")
    print("="*70)
    
    # 计算改进幅度
    first_score = all_trials[0]['score']
    improvement = (first_score - best_score) / first_score * 100
    print(f"性能改进: {improvement:.1f}% (从 {first_score:.4f} 到 {best_score:.4f})")
    
    # 分析最佳试验
    best_trial_idx = np.argmin([t['score'] for t in all_trials])
    print(f"最佳试验: Trial {best_trial_idx + 1}")
    
    # 统计损失函数类型
    hpl_trials = [t for t in all_trials if t['config']['loss_type'] == 'hpl']
    l2_trials = [t for t in all_trials if t['config']['loss_type'] == 'l2']
    
    if hpl_trials and l2_trials:
        hpl_avg = np.mean([t['score'] for t in hpl_trials])
        l2_avg = np.mean([t['score'] for t in l2_trials])
        print(f"损失函数对比: HPL平均 {hpl_avg:.4f}, L2平均 {l2_avg:.4f}")
    
    # 8. 显示详细结果表
    print("\n详细试验结果:")
    print("-" * 90)
    print(f"{'Trial':<6} {'RMSE':<8} {'学习率':<10} {'因子数':<8} {'正则化':<10} {'损失':<6} {'δ1':<6} {'δ2':<6}")
    print("-" * 90)
    
    for trial in all_trials:
        config = trial['config']
        print(f"{trial['trial_id']:<6} "
              f"{trial['score']:<8.4f} "
              f"{config['learning_rate']:<10.4f} "
              f"{config['latent_factors']:<8} "
              f"{config['lambda_reg']:<10.4f} "
              f"{config['loss_type']:<6} "
              f"{config['delta1']:<6.3f} "
              f"{config['delta2']:<6.3f}")
    
    # 9. 保存结果到文件
    # 转换所有NumPy类型
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    results = {
        'best_config': convert_numpy_types(best_config),
        'best_score': float(best_score),
        'total_trials': len(all_trials),
        'optimization_time': total_time,
        'improvement_percent': improvement,
        'all_trials': convert_numpy_types(all_trials)
    }
    
    try:
        with open('hyperopt_demo_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n结果已保存到: hyperopt_demo_results.json")
    except Exception as e:
        print(f"\n保存结果时出错: {e}")
    
    # 10. 给出建议
    print("\n" + "="*70)
    print("优化建议")
    print("="*70)
    
    print("基于当前结果，建议的最佳配置为:")
    # 将NumPy类型转换为Python原生类型
    best_config_json = {}
    for key, value in best_config.items():
        if isinstance(value, np.integer):
            best_config_json[key] = int(value)
        elif isinstance(value, np.floating):
            best_config_json[key] = float(value)
        elif isinstance(value, np.ndarray):
            best_config_json[key] = value.tolist()
        else:
            best_config_json[key] = value
    
    print(json.dumps(best_config_json, indent=2))
    
    if best_config['loss_type'] == 'hpl':
        print("\n✅ HPL损失函数表现更好，建议使用")
        print(f"   推荐参数: δ1={best_config['delta1']:.3f}, δ2={best_config['delta2']:.3f}")
    else:
        print("\n✅ L2损失函数表现更好，建议使用简单损失函数")
    
    print(f"\n🔧 其他关键参数:")
    print(f"   学习率: {best_config['learning_rate']:.4f}")
    print(f"   潜在因子数: {best_config['latent_factors']}")
    print(f"   正则化强度: {best_config['lambda_reg']:.4f}")
    
    print("\n📈 下一步优化建议:")
    print("1. 在最佳配置周围进行更精细的搜索")
    print("2. 增加训练轮数进行更准确的评估")
    print("3. 使用交叉验证提高结果可靠性")
    print("4. 考虑添加更多超参数（如动量、学习率衰减等）")
    
    return results

def demonstrate_sampling_strategies():
    """演示不同采样策略"""
    print("\n" + "="*70)
    print("采样策略对比演示")
    print("="*70)
    
    np.random.seed(42)
    
    # 1. 随机采样
    print("\n1. 随机采样 (Random Sampling):")
    random_samples = []
    for i in range(5):
        lr = np.exp(np.random.uniform(np.log(0.001), np.log(0.1)))
        factors = np.random.choice(list(range(20, 101, 10)))
        random_samples.append({'lr': lr, 'factors': factors})
        print(f"  样本{i+1}: lr={lr:.4f}, factors={factors}")
    
    # 2. 拉丁超立方采样模拟
    print("\n2. 拉丁超立方采样 (Latin Hypercube Sampling):")
    n_samples = 5
    # 简化的LHS实现
    lr_intervals = np.linspace(0, 1, n_samples)
    factor_intervals = np.linspace(0, 1, n_samples)
    
    # 随机排列
    lr_perm = np.random.permutation(n_samples)
    factor_perm = np.random.permutation(n_samples)
    
    lhs_samples = []
    for i in range(n_samples):
        # 在每个区间内随机采样
        lr_normalized = (lr_perm[i] + np.random.random()) / n_samples
        factor_normalized = (factor_perm[i] + np.random.random()) / n_samples
        
        # 转换到实际范围
        lr = np.exp(np.log(0.001) + lr_normalized * (np.log(0.1) - np.log(0.001)))
        factors = int(20 + factor_normalized * (100 - 20))
        factors = ((factors - 20) // 10) * 10 + 20  # 对齐到10的倍数
        
        lhs_samples.append({'lr': lr, 'factors': factors})
        print(f"  样本{i+1}: lr={lr:.4f}, factors={factors}")
    
    # 3. 对比分析
    print("\n3. 采样策略对比:")
    print("随机采样的学习率分布:", [s['lr'] for s in random_samples])
    print("LHS采样的学习率分布:", [s['lr'] for s in lhs_samples])
    
    # 计算覆盖度
    print(f"\n覆盖度分析:")
    print(f"随机采样因子数范围: {min(s['factors'] for s in random_samples)} - {max(s['factors'] for s in random_samples)}")
    print(f"LHS采样因子数范围: {min(s['factors'] for s in lhs_samples)} - {max(s['factors'] for s in lhs_samples)}")

if __name__ == '__main__':
    print("开始Hyperopt模块演示...")
    
    # 运行主要演示
    results = main()
    
    # 演示不同采样策略
    demonstrate_sampling_strategies()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("这个演示展示了hyperopt模块的核心功能:")
    print("✓ 参数空间定义 (连续、离散、分类参数)")
    print("✓ 约束条件管理 (δ1 < δ2)")
    print("✓ 随机采样策略")
    print("✓ 目标函数优化 (最小化RMSE)")
    print("✓ 实验追踪和结果分析")
    print("✓ 多种采样策略对比")
    print("\n在实际应用中，目标函数将是真实的模型训练和评估过程。")


