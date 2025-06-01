#!/usr/bin/env python3
"""
简化版MAE优化实验

如果完整版本遇到导入问题，可以使用这个简化版本
专注于核心功能：三种损失函数在MAE指标上的对比
"""

import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from data.data_manager import DataManager
    from src.models.mf_sgd import MatrixFactorizationSGD
    from src.models.initializers import NormalInitializer
    from src.models.regularizers import L2Regularizer
    from src.losses.hpl import HybridPiecewiseLoss
    from src.losses.standard import L2Loss, L1Loss
    from src.evaluation.metrics import MAE, RMSE
    print("✅ 成功导入所有必需模块")
except ImportError as e:
    print(f"❌ 导入模块失败: {e}")
    print("请确保在正确的项目目录中运行此脚本")
    sys.exit(1)


def setup_data():
    """设置数据"""
    print("📁 准备数据...")
    
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
    
    return data_manager


def test_single_loss(loss_name, loss_function, data_manager, config):
    """测试单个损失函数的MAE性能"""
    print(f"🔬 测试 {loss_name} 损失函数...")
    
    # 获取数据
    train_data, val_data, test_data = data_manager.get_splits()
    stats = data_manager.get_statistics()
    
    # 创建模型
    model = MatrixFactorizationSGD(
        n_users=stats['n_users'],
        n_items=stats['n_items'],
        n_factors=config['n_factors'],
        learning_rate=config['learning_rate'],
        regularizer=L2Regularizer(lambda_reg=config['lambda_reg']),
        loss_function=loss_function,
        use_bias=True,
        global_mean=data_manager.global_mean or 0.0
    )
    
    # 初始化参数
    initializer = NormalInitializer(mean=0.0, std=0.01)
    model.initialize_parameters(initializer)
    
    # 训练模型
    print(f"   训练中...")
    start_time = time.time()
    model.fit(
        train_data=train_data,
        val_data=val_data,
        n_epochs=config['n_epochs'],
        verbose=0
    )
    train_time = time.time() - start_time
    
    # 预测
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
    
    # 计算指标
    mae_metric = MAE()
    rmse_metric = RMSE()
    
    mae_score = mae_metric.calculate(val_targets, val_predictions)
    rmse_score = rmse_metric.calculate(val_targets, val_predictions)
    
    return {
        'mae': mae_score,
        'rmse': rmse_score,
        'train_time': train_time
    }


def run_mae_comparison_simple():
    """简化版MAE对比实验"""
    print("🎯 简化版MAE对比实验")
    print("="*50)
    
    # 1. 准备数据
    data_manager = setup_data()
    
    # 2. 定义测试配置
    base_config = {
        'n_factors': 50,
        'learning_rate': 0.02,
        'lambda_reg': 0.01,
        'n_epochs': 30
    }
    
    # 3. 定义三种损失函数
    loss_configs = {
        'L1(MAE)': L1Loss(epsilon=1e-8),
        'HPL': HybridPiecewiseLoss(delta1=0.5, delta2=1.5, l_max=4.0, c_sigmoid=1.0),
        'L2(MSE)': L2Loss()
    }
    
    # 4. 测试每种损失函数
    results = {}
    for loss_name, loss_function in loss_configs.items():
        try:
            result = test_single_loss(loss_name, loss_function, data_manager, base_config)
            results[loss_name] = result
            print(f"   ✅ {loss_name}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, 时间={result['train_time']:.1f}s")
        except Exception as e:
            print(f"   ❌ {loss_name} 测试失败: {e}")
            results[loss_name] = None
    
    # 5. 分析结果
    print("\n" + "="*50)
    print("📊 MAE对比结果")
    print("="*50)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ 没有成功的测试结果")
        return
    
    # 按MAE排序
    sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['mae'])
    
    print(f"{'排名':<4} {'损失函数':<12} {'MAE':<10} {'RMSE':<10} {'训练时间':<10}")
    print("-" * 50)
    
    for rank, (loss_name, result) in enumerate(sorted_results, 1):
        print(f"{rank:<4} {loss_name:<12} {result['mae']:<10.4f} {result['rmse']:<10.4f} {result['train_time']:<10.1f}s")
    
    # 6. 结论分析
    print(f"\n🏆 结论分析:")
    best_loss, best_result = sorted_results[0]
    print(f"在MAE指标上表现最佳的是: {best_loss}")
    print(f"最佳MAE: {best_result['mae']:.4f}")
    
    if best_loss == 'L1(MAE)':
        print("✅ 符合理论预期：L1损失直接优化MAE，应该表现最佳")
    elif best_loss == 'HPL':
        print("🎉 HPL损失超越了L1！这证明了分段策略的有效性")
    else:
        print("📊 L2损失表现最佳，这可能需要进一步调整超参数")
    
    # 7. 配置建议
    print(f"\n💡 如果要优化MAE指标，建议:")
    print(f"1. 使用 {best_loss} 损失函数")
    print(f"2. 可以进一步调整学习率和正则化参数")
    print(f"3. 考虑增加训练轮数以获得更好的收敛")
    
    return results


def quick_mae_test():
    """超快速MAE测试（仅用于验证配置）"""
    print("⚡ 快速MAE配置验证")
    print("="*30)
    
    try:
        # 测试MAE指标
        mae_metric = MAE()
        test_targets = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        test_preds = np.array([1.1, 2.2, 2.8, 4.3, 4.9])
        mae_score = mae_metric.calculate(test_targets, test_preds)
        print(f"✅ MAE指标测试成功: {mae_score:.4f}")
        
        # 测试L1损失
        l1_loss = L1Loss(epsilon=1e-8)
        print(f"✅ L1损失函数创建成功")
        
        # 测试HPL损失
        hpl_loss = HybridPiecewiseLoss(delta1=0.5, delta2=1.5)
        print(f"✅ HPL损失函数创建成功")
        
        print(f"✅ 所有核心组件配置正确，可以运行完整实验")
        return True
        
    except Exception as e:
        print(f"❌ 配置验证失败: {e}")
        return False


def main():
    """主函数"""
    print("MAE配置和对比实验")
    print("="*60)
    
    # 1. 快速验证
    print("\n1. 快速配置验证...")
    if not quick_mae_test():
        print("❌ 基础配置验证失败，请检查环境")
        return False
    
    # 2. 询问用户选择
    print(f"\n请选择要运行的实验:")
    print(f"1. 快速验证（推荐）")
    print(f"2. 简化版MAE对比实验")
    print(f"3. 完整版MAE优化实验")
    
    try:
        choice = input("请输入选择 (1/2/3，默认1): ").strip()
        if not choice:
            choice = "1"
    except:
        choice = "1"
    
    if choice == "1":
        print(f"\n✅ 快速验证已完成，MAE配置正确")
        print(f"💡 您可以参考 MAE_Configuration_Guide.md 进行详细配置")
        return True
    
    elif choice == "2":
        print(f"\n2. 运行简化版MAE对比实验...")
        try:
            results = run_mae_comparison_simple()
            print(f"\n✅ 简化版实验完成")
            return True
        except Exception as e:
            print(f"❌ 简化版实验失败: {e}")
            return False
    
    elif choice == "3":
        print(f"\n3. 运行完整版MAE优化实验...")
        print(f"💡 请运行: python example_hpl_mae_complete.py")
        return True
    
    else:
        print(f"无效选择，运行快速验证")
        return True


if __name__ == '__main__':
    success = main()
    print(f"\n{'='*60}")
    if success:
        print("🎉 实验完成！")
        print("📖 详细配置说明请参考: MAE_Configuration_Guide.md")
        print("🚀 完整实验请运行: python example_hpl_mae_complete.py")
    else:
        print("❌ 实验失败，请检查环境配置")
    
    sys.exit(0 if success else 1)
