#!/usr/bin/env python3
"""
HPL损失函数专用优化实验

专门针对混合分段损失函数(HPL)的超参数优化
修正了数据分割问题，增加了HPL专用的参数空间和优化策略
"""

import sys
import os
import numpy as np
import time
import json

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入模块
try:
    from src.hyperopt.space import ParameterSpace
    from src.hyperopt.samplers import LatinHypercubeSampler
    from src.hyperopt.constraints import ConstraintManager
    from src.hyperopt.optimizer import HyperOptimizer
    from src.hyperopt.tracker import ExperimentTracker
    print("成功导入hyperopt模块")
except ImportError as e:
    print(f"导入hyperopt模块失败: {e}")

try:
    from data.data_manager import DataManager
    from src.models.mf_sgd import MatrixFactorizationSGD
    from src.models.initializers import NormalInitializer
    from src.models.regularizers import L2Regularizer
    from src.losses.hpl import HybridPiecewiseLoss
    from src.losses.standard import L2Loss
    print("成功导入数据和模型模块")
except ImportError as e:
    print(f"导入数据/模型模块失败: {e}")
    sys.exit(1)

try:
    from example_hyperopt import (
        SimpleParameterSpace, SimpleLatinHypercubeSampler,
        SimpleConstraintManager, SimpleHyperOptimizer, SimpleExperimentTracker
    )
    print("简化版本类已准备就绪")
except ImportError:
    print("警告：无法导入简化版本类")


class HPLFocusedObjectiveFunction:
    """专门针对HPL的目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        print(f"HPL目标函数初始化:")
        print(f"  训练集: {len(self.train_data)} 条")
        print(f"  验证集: {len(self.val_data)} 条")
        print(f"  测试集: {len(self.test_data)} 条")

    def __call__(self, config):
        """HPL专用目标函数：强制使用HPL损失函数"""
        try:
            # 强制使用HPL损失函数
            loss_function = HybridPiecewiseLoss(
                delta1=config['delta1'],
                delta2=config['delta2'],
                l_max=config.get('l_max', 4.0),
                c_sigmoid=config.get('c_sigmoid', 1.0)
            )

            regularizer = L2Regularizer(lambda_reg=config['lambda_reg'])

            model = MatrixFactorizationSGD(
                n_users=self.n_users,
                n_items=self.n_items,
                n_factors=config['latent_factors'],
                learning_rate=config['learning_rate'],
                regularizer=regularizer,
                loss_function=loss_function,
                use_bias=True,
                global_mean=self.data_manager.global_mean or 0.0
            )

            initializer = NormalInitializer(mean=0.0, std=0.01)
            model.initialize_parameters(initializer)

            # 增加训练时间，HPL需要更多时间收敛
            model.fit(
                train_data=self.train_data,
                val_data=self.val_data,
                n_epochs=50,        # 增加到50轮
                verbose=0,          # 不打印详细训练过程
                early_stopping_patience=15  # 增加耐心
            )

            # 验证集评估
            val_predictions = model.predict(
                self.val_data[:, 0].astype(int),
                self.val_data[:, 1].astype(int)
            )

            # 还原到原始尺度
            if self.data_manager.global_mean is not None:
                val_predictions += self.data_manager.global_mean
                val_targets = self.val_data[:, 2] + self.data_manager.global_mean
            else:
                val_targets = self.val_data[:, 2]

            # 计算RMSE
            rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))

            # 修改打印格式，包含所有重要参数
            print(f"HPL配置: 学习率={config['learning_rate']:.4f}, "
                  f"因子数={config['latent_factors']}, "
                  f"正则化={config['lambda_reg']:.6f}, "
                  f"δ1={config['delta1']:.3f}, δ2={config['delta2']:.3f}, "
                  f"l_max={config.get('l_max', 4.0):.2f}, c_sig={config.get('c_sigmoid', 1.0):.2f}, "
                  f"RMSE={rmse:.4f}")

            return rmse

        except Exception as e:
            print(f"HPL配置评估失败 {config}: {e}")
            return 10.0  # 返回较大值表示失败


class L2FocusedObjectiveFunction:
    """专门针对L2的目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

    def __call__(self, config):
        try:
            # 强制使用L2损失函数
            loss_function = L2Loss()
            regularizer = L2Regularizer(lambda_reg=config['lambda_reg'])

            model = MatrixFactorizationSGD(
                n_users=self.n_users,
                n_items=self.n_items,
                n_factors=config['latent_factors'],
                learning_rate=config['learning_rate'],
                regularizer=regularizer,
                loss_function=loss_function,
                use_bias=True,
                global_mean=self.data_manager.global_mean or 0.0
            )

            initializer = NormalInitializer(mean=0.0, std=0.01)
            model.initialize_parameters(initializer)

            # 相同的训练设置
            model.fit(
                train_data=self.train_data,
                val_data=self.val_data,
                n_epochs=50,        # 与HPL相同
                verbose=0,
                early_stopping_patience=15  # 与HPL相同
            )

            # 验证集评估
            val_predictions = model.predict(
                self.val_data[:, 0].astype(int),
                self.val_data[:, 1].astype(int)
            )

            # 还原到原始尺度
            if self.data_manager.global_mean is not None:
                val_predictions += self.data_manager.global_mean
                val_targets = self.val_data[:, 2] + self.data_manager.global_mean
            else:
                val_targets = self.val_data[:, 2]

            # 计算RMSE
            rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))

            # 修改打印格式，包含所有重要参数
            print(f"L2配置: 学习率={config['learning_rate']:.4f}, "
                  f"因子数={config['latent_factors']}, "
                  f"正则化={config['lambda_reg']:.6f}, "
                  f"RMSE={rmse:.4f}")

            return rmse

        except Exception as e:
            print(f"L2配置评估失败 {config}: {e}")
            return 10.0  # 返回较大值表示失败


def setup_improved_data():
    """改进的数据设置，修正分割比例问题"""
    print("准备改进的数据...")

    # 修正数据配置
    data_config = {
        'random_seed': 42,
        'train_ratio': 0.8,          # 确保80%训练数据
        'val_ratio': 0.1,            # 10%验证数据
        'test_ratio': 0.1,           # 10%测试数据
        'batch_size': 128,
        'shuffle': True,
        'center_data': True,         # 数据中心化有助于HPL
        'ensure_user_in_train': True
    }

    data_manager = DataManager(data_config)
    data_manager.load_dataset('movielens100k', 'dataset/20201202M100K_data_all_random.txt')
    data_manager.preprocess()

    # 验证数据分割
    train_data, val_data, test_data = data_manager.get_splits()
    print(f"修正后数据分割:")
    print(f"  训练集: {len(train_data)} 条 ({len(train_data)/100000*100:.1f}%)")
    print(f"  验证集: {len(val_data)} 条 ({len(val_data)/100000*100:.1f}%)")
    print(f"  测试集: {len(test_data)} 条 ({len(test_data)/100000*100:.1f}%)")

    data_manager.print_summary()
    return data_manager


def create_hpl_parameter_space():
    """创建HPL专用参数空间"""
    print("创建HPL专用参数空间...")

    try:
        space = ParameterSpace()
    except:
        space = SimpleParameterSpace()

    # 基础模型参数 - 基于之前实验结果优化范围
    space.add_continuous('learning_rate', 0.01, 0.08, scale='log')  # 缩小到有效范围
    space.add_discrete('latent_factors', 15, 75, step=5)            # 聚焦有效区间
    space.add_continuous('lambda_reg', 0.001, 0.02, scale='log')    # 专注弱正则化

    # HPL专用参数 - 扩展搜索范围
    space.add_continuous('delta1', 0.05, 1.5)      # 扩展下界，允许更小的delta1
    space.add_continuous('delta2', 0.8, 4.0)       # 扩展上界，允许更大的delta2
    space.add_continuous('l_max', 2.5, 6.0)        # 损失函数上限
    space.add_continuous('c_sigmoid', 0.3, 3.0)    # Sigmoid函数陡峭度

    print(f"HPL参数空间维度: {space.get_dimension()}")
    print("HPL参数信息:")
    for name, info in space.get_parameter_info().items():
        print(f"  {name}: {info}")

    return space


def create_l2_parameter_space():
    """创建L2专用参数空间"""
    try:
        space = ParameterSpace()
    except:
        space = SimpleParameterSpace()

    # 只包含基础模型参数，不包含HPL参数
    space.add_continuous('learning_rate', 0.01, 0.08, scale='log')
    space.add_discrete('latent_factors', 15, 75, step=5)
    space.add_continuous('lambda_reg', 0.001, 0.02, scale='log')

    return space


def create_hpl_constraints():
    """创建HPL专用约束条件"""
    print("创建HPL专用约束条件...")

    try:
        constraints = ConstraintManager()
    except:
        constraints = SimpleConstraintManager()

    # 核心约束：delta1 < delta2
    constraints.add_relation('delta1', 'delta2', '<')

    print(f"HPL约束数量: {len(constraints.constraints) if hasattr(constraints, 'constraints') else 1}")

    return constraints


def run_hpl_focused_optimization():
    """
    运行HPL损失函数专用优化实验

    该函数执行完整的HPL优化流程，包括数据准备、参数空间定义、优化器配置、
    参数优化搜索以及结果分析。采用分段式优化策略，优先使用完整版hyperopt组件，
    在失败时自动降级为简化版组件。

    Returns:
        tuple: 包含两个元素的元组
            - optimizer: 使用的优化器实例（HyperOptimizer/SimpleHyperOptimizer）
            - best_trial: 最优试验结果对象，包含最优配置和性能指标
    """
    print("="*60)
    print("HPL损失函数专用优化实验")
    print("="*60)

    # 初始化数据管理器
    # 创建针对HPL损失函数改进的数据处理管道
    data_manager = setup_improved_data()

    # 构建HPL专用目标函数
    # 基于改进数据创建专门的优化目标函数
    objective = HPLFocusedObjectiveFunction(data_manager)

    # 定义HPL参数搜索空间
    # 包含学习率、正则化参数、潜在因子数等核心参数的搜索范围
    space = create_hpl_parameter_space()

    # 配置参数约束条件
    # 设置参数间的逻辑约束关系，确保参数组合的有效性
    constraints = create_hpl_constraints()

    # 组件初始化与兼容性检查
    # 尝试加载完整版优化组件，失败时自动切换简化版
    try:
        sampler = LatinHypercubeSampler(space, seed=42)  # 优先使用LHS
        tracker = ExperimentTracker('hpl_focused_optimization', backend='memory')
        optimizer = HyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=constraints,
            tracker=tracker,
            maximize=False,  # 最小化RMSE
            seed=42
        )
        print("使用完整版hyperopt组件")
    except:
        sampler = SimpleLatinHypercubeSampler(space, seed=42)
        tracker = SimpleExperimentTracker('hpl_focused_optimization', backend='memory')
        optimizer = SimpleHyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=constraints,
            tracker=tracker,
            maximize=False,
            seed=42
        )
        print("使用简化版hyperopt组件")

    print("开始HPL专用优化...")
    start_time = time.time()

    # 执行优化搜索过程
    # 使用增强的试验次数和耐心值进行参数优化
    best_trial = optimizer.optimize(
        n_trials=40,                 # 增加试验数
        no_improvement_rounds=15,    # 增加耐心
        batch_size=1
    )

    end_time = time.time()

    # 优化结果解析与展示
    # 输出最优参数配置、性能指标及参数合理性分析
    print("\n" + "="*60)
    print("HPL专用优化结果")
    print("="*60)

    if best_trial:
        print(f"HPL最佳配置: {best_trial.config}")
        print(f"HPL最佳验证RMSE: {best_trial.score:.4f}")
        print(f"HPL优化耗时: {end_time - start_time:.2f}秒")

        # 参数特征分析
        # 对关键HPL参数进行详细解析和间隔合理性验证
        config = best_trial.config
        print(f"\nHPL参数分析:")
        print(f"  学习率: {config['learning_rate']:.4f}")
        print(f"  潜在因子数: {config['latent_factors']}")
        print(f"  正则化参数: {config['lambda_reg']:.6f}")
        print(f"  HPL delta1: {config['delta1']:.3f}")
        print(f"  HPL delta2: {config['delta2']:.3f}")
        print(f"  HPL l_max: {config.get('l_max', 4.0):.2f}")
        print(f"  HPL c_sigmoid: {config.get('c_sigmoid', 1.0):.2f}")
        print(f"  delta差值: {config['delta2'] - config['delta1']:.3f}")

        # 参数合理性验证
        # 检查delta间隔和l_max设置是否符合HPL理论要求
        delta_gap = config['delta2'] - config['delta1']
        l_max_val = config.get('l_max', 4.0)

        print(f"\nHPL参数合理性分析:")
        if delta_gap < 0.3:
            print(f"  ⚠️  delta间隔过小 ({delta_gap:.3f})，可能影响分段效果")
        elif delta_gap > 2.0:
            print(f"  ⚠️  delta间隔过大 ({delta_gap:.3f})，可能失去分段优势")
        else:
            print(f"  ✅ delta间隔合理 ({delta_gap:.3f})")

        if l_max_val <= config['delta2']:
            print(f"  ⚠️  l_max过小，应该 > delta2")
        else:
            print(f"  ✅ l_max设置合理 ({l_max_val:.2f} > {config['delta2']:.3f})")
    else:
        print("HPL优化失败，未找到有效配置")

    # 优化过程统计汇总
    # 展示试验总数、成功数和失败数等统计信息
    results = optimizer.get_results()
    print(f"\nHPL优化统计:")
    print(f"总试验数: {results['n_trials']}")
    print(f"成功试验数: {results['n_completed']}")
    print(f"失败试验数: {results['n_failed']}")

    return optimizer, best_trial


def run_l2_focused_optimization():
    """L2损失函数专用优化（用于公平对比）"""
    print("开始L2专用优化...")

    # 1. 准备相同的数据
    data_manager = setup_improved_data()

    # 2. 创建L2专用目标函数
    objective = L2FocusedObjectiveFunction(data_manager)

    # 3. 创建L2参数空间（不包含HPL参数）
    space = create_l2_parameter_space()

    # 4. 创建采样器和优化器
    try:
        sampler = LatinHypercubeSampler(space, seed=42)
        tracker = ExperimentTracker('l2_focused_optimization', backend='memory')
        optimizer = HyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=None,
            tracker=tracker,
            maximize=False,
            seed=42
        )
    except:
        sampler = SimpleLatinHypercubeSampler(space, seed=42)
        tracker = SimpleExperimentTracker('l2_focused_optimization', backend='memory')
        optimizer = SimpleHyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=None,
            tracker=tracker,
            maximize=False,
            seed=42
        )

    # 5. 运行优化（相同的试验预算）
    best_trial = optimizer.optimize(
        n_trials=40,                 # 与HPL相同的试验数
        no_improvement_rounds=15,    # 相同的耐心
        batch_size=1
    )

    if best_trial:
        print(f"L2最佳RMSE: {best_trial.score:.4f}")
        print(f"L2最佳配置: {best_trial.config}")

    return optimizer, best_trial


def run_hpl_vs_l2_comparison():
    """HPL与L2损失函数的公平对比"""
    print("\n" + "="*60)
    print("HPL vs L2 公平对比实验")
    print("="*60)

    results = {}

    # 1. HPL专用优化
    print("\n1. 运行HPL专用优化...")
    hpl_optimizer, hpl_best = run_hpl_focused_optimization()
    results['HPL'] = {
        'best_rmse': hpl_best.score if hpl_best else float('inf'),
        'best_config': hpl_best.config if hpl_best else None,
        'optimizer': hpl_optimizer
    }

    # 2. L2专用优化（使用相同的试验预算和训练设置）
    print("\n2. 运行L2专用优化...")
    l2_optimizer, l2_best = run_l2_focused_optimization()
    results['L2'] = {
        'best_rmse': l2_best.score if l2_best else float('inf'),
        'best_config': l2_best.config if l2_best else None,
        'optimizer': l2_optimizer
    }

    # 3. 对比分析
    print("\n" + "="*60)
    print("HPL vs L2 对比结果")
    print("="*60)

    hpl_rmse = results['HPL']['best_rmse']
    l2_rmse = results['L2']['best_rmse']

    print(f"{'损失函数':<12} {'最佳RMSE':<12} {'相对表现':<12}")
    print("-" * 40)
    print(f"{'HPL':<12} {hpl_rmse:<12.4f} {'基准':<12}")
    print(f"{'L2':<12} {l2_rmse:<12.4f} {'对比':<12}")

    if hpl_rmse < l2_rmse:
        improvement = (l2_rmse - hpl_rmse) / l2_rmse * 100
        print(f"\n🎉 HPL损失函数表现更优!")
        print(f"   相比L2改进了: {improvement:.2f}%")
        print(f"   绝对改进: {l2_rmse - hpl_rmse:.4f}")
        winner = 'HPL'
    elif l2_rmse < hpl_rmse:
        degradation = (hpl_rmse - l2_rmse) / l2_rmse * 100
        print(f"\n📊 L2损失函数表现更优")
        print(f"   HPL相比L2差了: {degradation:.2f}%")
        print(f"   绝对差距: {hpl_rmse - l2_rmse:.4f}")
        winner = 'L2'
    else:
        print(f"\n🤝 两种损失函数表现相当")
        winner = 'Tie'

    # 4. 详细配置对比
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

    # 5. 保存对比结果
    comparison_results = {
        'winner': winner,
        'hpl_rmse': hpl_rmse,
        'l2_rmse': l2_rmse,
        'improvement_percentage': (l2_rmse - hpl_rmse) / l2_rmse * 100 if l2_rmse != 0 else 0,
        'hpl_config': results['HPL']['best_config'],
        'l2_config': results['L2']['best_config']
    }

    # 保存详细结果
    save_comparison_results(comparison_results, "hpl_vs_l2_comparison.json")

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

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False)

        print(f"\n对比结果已保存到: {filename}")

    except Exception as e:
        print(f"保存对比结果失败: {e}")


def main():
    """HPL专用优化主函数"""
    try:
        print("HPL损失函数深度优化实验")
        print("="*60)

        # 1. 基础HPL优化实验
        print("\n1. 运行HPL专用优化实验...")
        hpl_optimizer, hpl_best = run_hpl_focused_optimization()

        # 2. HPL vs L2 公平对比
        print("\n2. 运行HPL vs L2公平对比...")
        comparison_results = run_hpl_vs_l2_comparison()

        # 3. 生成最终报告
        print("\n" + "="*60)
        print("最终实验总结")
        print("="*60)

        if hpl_best:
            print(f"\nHPL专用优化最佳RMSE: {hpl_best.score:.4f}")
            print(f"HPL最佳配置:")
            config = hpl_best.config
            print(f"  学习率: {config['learning_rate']:.4f}")
            print(f"  潜在因子数: {config['latent_factors']}")
            print(f"  正则化: {config['lambda_reg']:.6f}")
            print(f"  δ1: {config['delta1']:.3f}")
            print(f"  δ2: {config['delta2']:.3f}")
            print(f"  l_max: {config.get('l_max', 4.0):.2f}")
            print(f"  c_sigmoid: {config.get('c_sigmoid', 1.0):.2f}")

        if comparison_results:
            hpl_rmse = comparison_results['HPL']['best_rmse']
            l2_rmse = comparison_results['L2']['best_rmse']
            print(f"\n公平对比结果:")
            print(f"  HPL最佳RMSE: {hpl_rmse:.4f}")
            print(f"  L2最佳RMSE: {l2_rmse:.4f}")
            if hpl_rmse < l2_rmse:
                improvement = (l2_rmse - hpl_rmse) / l2_rmse * 100
                print(f"  🎉 HPL优于L2，改进: {improvement:.2f}%")
            else:
                degradation = (hpl_rmse - l2_rmse) / l2_rmse * 100
                print(f"  📊 L2优于HPL，差距: {degradation:.2f}%")

        print("\n实验完成！详细结果已保存到相应的JSON文件中。")

        # 提供改进建议
        print("\n改进建议:")
        print("1. 如果HPL性能不佳，请考虑增加训练轮数和试验数")
        print("2. 可以调整delta1和delta2的搜索范围")
        print("3. 考虑添加更多的HPL高级参数优化")
        print("4. 可以尝试不同的数据集和初始化策略")

        return True

    except Exception as e:
        print(f"HPL优化实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


