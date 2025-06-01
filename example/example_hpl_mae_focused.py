    print("-" * 50)
    print(f"{'HPL':<12} {hpl_mae:<12.4f} {'对比':<12} {((l2_mae - hpl_mae) / l2_mae * 100 if l2_mae != 0 else 0):<12.2f}%")
    print(f"{'L2':<12} {l2_mae:<12.4f} {'基准':<12} {'0.00':<12}%")
    print(f"{'L1(MAE)':<12} {l1_mae:<12.4f} {'对比':<12} {((l2_mae - l1_mae) / l2_mae * 100 if l2_mae != 0 else 0):<12.2f}%")
    
    # 分析结果
    if best_mae == hpl_mae:
        print(f"\n🎉 HPL损失函数在MAE指标上表现最优!")
        if l2_mae != float('inf'):
            improvement = (l2_mae - hpl_mae) / l2_mae * 100
            print(f"   相比L2改进了: {improvement:.2f}%")
        if l1_mae != float('inf'):
            improvement = (l1_mae - hpl_mae) / l1_mae * 100
            print(f"   相比L1改进了: {improvement:.2f}%")
        winner = 'HPL'
    elif best_mae == l1_mae:
        print(f"\n🎯 L1(MAE)损失函数在MAE指标上表现最优!")
        print(f"   这是符合预期的，因为L1损失直接优化MAE")
        if l2_mae != float('inf'):
            improvement = (l2_mae - l1_mae) / l2_mae * 100
            print(f"   相比L2改进了: {improvement:.2f}%")
        if hpl_mae != float('inf'):
            improvement = (hpl_mae - l1_mae) / hpl_mae * 100
            print(f"   相比HPL改进了: {improvement:.2f}%")
        winner = 'L1'
    elif best_mae == l2_mae:
        print(f"\n📊 L2损失函数在MAE指标上表现最优")
        winner = 'L2'
    else:
        print(f"\n🤝 多种损失函数表现相当")
        winner = 'Tie'
    
    # 5. 详细配置对比
    print(f"\n最佳配置对比:")
    if results['HPL']['best_config']:
        hpl_config = results['HPL']['best_config']
        print(f"\nHPL最佳配置:")
        print(f"  学习率: {hpl_config['learning_rate']:.4f}")
        print(f"  因子数: {hpl_config['latent_factors']}")
        print(f"  正则化: {hpl_config['lambda_reg']:.6f}")
        print(f"  delta1: {hpl_config['delta1']:.3f}")
        print(f"  delta2: {hpl_config['delta2']:.3f}")
        print(f"  l_max: {hpl_config.get('l_max', 4.0):.2f}")
        print(f"  c_sigmoid: {hpl_config.get('c_sigmoid', 1.0):.2f}")
    
    if results['L2']['best_config']:
        l2_config = results['L2']['best_config']
        print(f"\nL2最佳配置:")
        print(f"  学习率: {l2_config['learning_rate']:.4f}")
        print(f"  因子数: {l2_config['latent_factors']}")
        print(f"  正则化: {l2_config['lambda_reg']:.6f}")
    
    if results['L1']['best_config']:
        l1_config = results['L1']['best_config']
        print(f"\nL1最佳配置:")
        print(f"  学习率: {l1_config['learning_rate']:.4f}")
        print(f"  因子数: {l1_config['latent_factors']}")
        print(f"  正则化: {l1_config['lambda_reg']:.6f}")
        print(f"  L1_epsilon: {l1_config.get('l1_epsilon', 1e-8):.2e}")
    
    # 6. 保存对比结果
    comparison_results = {
        'winner': winner,
        'hpl_mae': hpl_mae,
        'l2_mae': l2_mae,
        'l1_mae': l1_mae,
        'best_mae': best_mae,
        'hpl_improvement_over_l2': (l2_mae - hpl_mae) / l2_mae * 100 if l2_mae != 0 else 0,
        'l1_improvement_over_l2': (l2_mae - l1_mae) / l2_mae * 100 if l2_mae != 0 else 0,
        'hpl_config': results['HPL']['best_config'],
        'l2_config': results['L2']['best_config'],
        'l1_config': results['L1']['best_config']
    }
    
    # 保存详细结果
    save_comparison_results(comparison_results, "hpl_vs_l2_vs_l1_mae_comparison.json")
    
    return results


def save_comparison_results(results, filename):
    """保存对比结果"""
    try:
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif hasattr(obj, 'item'):
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        clean_results = convert_numpy_types(results)
        clean_results['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        clean_results['evaluation_metric'] = 'MAE'
        clean_results['description'] = 'HPL vs L2 vs L1 losses optimized for MAE metric'
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n对比结果已保存到: {filename}")
        
    except Exception as e:
        print(f"保存对比结果失败: {e}")


def evaluate_final_performance(data_manager, best_configs):
    """在测试集上评估最终性能"""
    print("\n" + "="*60)
    print("测试集最终性能评估")
    print("="*60)
    
    mae_metric = MAE()
    rmse_metric = RMSE()
    
    test_results = {}
    train_data, val_data, test_data = data_manager.get_splits()
    
    loss_functions = {
        'HPL': lambda config: HybridPiecewiseLoss(
            delta1=config['delta1'],
            delta2=config['delta2'], 
            l_max=config.get('l_max', 4.0),
            c_sigmoid=config.get('c_sigmoid', 1.0)
        ),
        'L2': lambda config: L2Loss(),
        'L1': lambda config: L1Loss(epsilon=config.get('l1_epsilon', 1e-8))
    }
    
    for loss_name, config in best_configs.items():
        if config is None:
            continue
            
        print(f"\n评估 {loss_name} 在测试集上的性能...")
        
        try:
            # 创建模型
            loss_function = loss_functions[loss_name](config)
            regularizer = L2Regularizer(lambda_reg=config['lambda_reg'])
            
            model = MatrixFactorizationSGD(
                n_users=data_manager.get_statistics()['n_users'],
                n_items=data_manager.get_statistics()['n_items'],
                n_factors=config['latent_factors'],
                learning_rate=config['learning_rate'],
                regularizer=regularizer,
                loss_function=loss_function,
                use_bias=True,
                global_mean=data_manager.global_mean or 0.0
            )
            
            # 初始化并训练
            initializer = NormalInitializer(mean=0.0, std=0.01)
            model.initialize_parameters(initializer)
            
            # 在训练+验证集上训练
            combined_data = np.vstack([train_data, val_data])
            model.fit(
                train_data=combined_data,
                val_data=None,  # 不使用验证集（因为要在测试集评估）
                n_epochs=50,
                verbose=0
            )
            
            # 测试集预测
            test_predictions = model.predict(
                test_data[:, 0].astype(int),
                test_data[:, 1].astype(int)
            )
            
            # 还原到原始尺度
            if data_manager.global_mean is not None:
                test_predictions += data_manager.global_mean
                test_targets = test_data[:, 2] + data_manager.global_mean
            else:
                test_targets = test_data[:, 2]
            
            # 计算指标
            test_mae = mae_metric.calculate(test_targets, test_predictions)
            test_rmse = rmse_metric.calculate(test_targets, test_predictions)
            
            test_results[loss_name] = {
                'mae': test_mae,
                'rmse': test_rmse,
                'config': config
            }
            
            print(f"{loss_name} 测试结果:")
            print(f"  MAE: {test_mae:.4f}")
            print(f"  RMSE: {test_rmse:.4f}")
            
        except Exception as e:
            print(f"{loss_name} 测试评估失败: {e}")
            test_results[loss_name] = None
    
    # 汇总测试结果
    print(f"\n{'损失函数':<12} {'测试MAE':<12} {'测试RMSE':<12}")
    print("-" * 40)
    for loss_name, result in test_results.items():
        if result:
            print(f"{loss_name:<12} {result['mae']:<12.4f} {result['rmse']:<12.4f}")
    
    return test_results


def main():
    """HPL专用MAE优化主函数"""
    try:
        print("HPL损失函数MAE指标深度优化实验")
        print("="*60)
        
        # 1. 基础HPL MAE优化实验
        print("\n1. 运行HPL vs L2 vs L1 MAE对比...")
        comparison_results = run_mae_comparison()
        
        # 2. 提取最佳配置
        best_configs = {
            'HPL': comparison_results['HPL']['best_config'],
            'L2': comparison_results['L2']['best_config'], 
            'L1': comparison_results['L1']['best_config']
        }
        
        # 3. 测试集最终评估
        print("\n2. 运行测试集最终评估...")
        data_manager = setup_improved_data()
        test_results = evaluate_final_performance(data_manager, best_configs)
        
        # 4. 生成最终报告
        print("\n" + "="*60)
        print("MAE优化实验最终总结")
        print("="*60)
        
        print("\n验证集最佳MAE结果:")
        if comparison_results['HPL']['best_mae'] != float('inf'):
            print(f"  HPL: {comparison_results['HPL']['best_mae']:.4f}")
        if comparison_results['L2']['best_mae'] != float('inf'):
            print(f"  L2:  {comparison_results['L2']['best_mae']:.4f}")
        if comparison_results['L1']['best_mae'] != float('inf'):
            print(f"  L1:  {comparison_results['L1']['best_mae']:.4f}")
        
        print("\n测试集最终结果:")
        for loss_name, result in test_results.items():
            if result:
                print(f"  {loss_name}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}")
        
        # 5. 关键发现和建议
        print("\n" + "="*60)
        print("关键发现和建议")
        print("="*60)
        
        best_val_mae = min(
            comparison_results['HPL']['best_mae'],
            comparison_results['L2']['best_mae'],
            comparison_results['L1']['best_mae']
        )
        
        if best_val_mae == comparison_results['L1']['best_mae']:
            print("✅ L1损失在MAE优化中表现最佳，这符合理论预期")
            print("   - L1损失直接优化MAE目标，应该是最优选择")
            print("   - 建议在注重MAE指标的应用中使用L1损失")
        elif best_val_mae == comparison_results['HPL']['best_mae']:
            print("🎉 HPL损失在MAE优化中超越了L1损失!")
            print("   - 这表明HPL的分段策略对MAE优化也有帮助")
            print("   - HPL可能在处理不同误差范围时比简单L1更有效")
        else:
            print("📊 L2损失在MAE优化中表现意外地好")
            print("   - 可能需要调整其他损失函数的超参数范围")
        
        # 6. 配置建议
        print("\n配置建议:")
        for loss_name, config in best_configs.items():
            if config:
                print(f"\n{loss_name}最佳配置:")
                print(f"  学习率: {config['learning_rate']:.4f}")
                print(f"  潜在因子: {config['latent_factors']}")
                print(f"  正则化: {config['lambda_reg']:.6f}")
                if loss_name == 'HPL':
                    print(f"  delta1: {config['delta1']:.3f}")
                    print(f"  delta2: {config['delta2']:.3f}")
                elif loss_name == 'L1':
                    print(f"  epsilon: {config.get('l1_epsilon', 1e-8):.2e}")
        
        print("\n实验完成！详细结果已保存到JSON文件中。")
        
        return True
        
    except Exception as e:
        print(f"MAE优化实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
