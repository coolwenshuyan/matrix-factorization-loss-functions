#!/usr/bin/env python3
"""
HPL损失函数专用优化实验 - MAE指标完整版本

专门针对混合分段损失函数(HPL)的超参数优化，使用MAE作为优化目标
完整的HPL vs L2 vs L1在MAE指标上的对比实验
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
    from src.losses.standard import L2Loss, L1Loss  # 添加L1Loss用于MAE训练
    from src.evaluation.metrics import MAE, RMSE  # 导入MAE和RMSE评估指标
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


class HPLMAEObjectiveFunction:
    """专门针对HPL优化MAE的目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        # 初始化MAE评估器
        self.mae_metric = MAE()

        print(f"HPL-MAE目标函数初始化:")
        print(f"  训练集: {len(self.train_data)} 条")
        print(f"  验证集: {len(self.val_data)} 条")
        print(f"  测试集: {len(self.test_data)} 条")
        print(f"  优化目标: MAE (平均绝对误差)")

    def __call__(self, config):
        """HPL专用目标函数：强制使用HPL损失函数，优化MAE指标"""
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

            # 🎯 关键修改：计算MAE而不是RMSE
            mae = self.mae_metric.calculate(val_targets, val_predictions)

            # 记录HPL特有信息
            print(f"HPL配置: δ1={config['delta1']:.3f}, δ2={config['delta2']:.3f}, "
                  f"l_max={config.get('l_max', 4.0):.2f}, c_sig={config.get('c_sigmoid', 1.0):.2f}, "
                  f"MAE={mae:.4f}")

            return mae

        except Exception as e:
            print(f"HPL配置评估失败 {config}: {e}")
            return 10.0  # 返回较大值表示失败


class L2MAEObjectiveFunction:
    """专门针对L2损失优化MAE的目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        # 初始化MAE评估器
        self.mae_metric = MAE()

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

            if self.data_manager.global_mean is not None:
                val_predictions += self.data_manager.global_mean
                val_targets = self.val_data[:, 2] + self.data_manager.global_mean
            else:
                val_targets = self.val_data[:, 2]

            # 🎯 计算MAE
            mae = self.mae_metric.calculate(val_targets, val_predictions)

            print(f"L2配置: lr={config['learning_rate']:.4f}, factors={config['latent_factors']}, "
                  f"reg={config['lambda_reg']:.6f}, MAE={mae:.4f}")

            return mae

        except Exception as e:
            print(f"L2配置评估失败 {config}: {e}")
            return 10.0


class L1MAEObjectiveFunction:
    """专门针对L1损失（MAE损失）优化MAE的目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        # 初始化MAE评估器
        self.mae_metric = MAE()

    def __call__(self, config):
        try:
            # 🎯 使用L1损失函数（MAE损失）进行训练
            loss_function = L1Loss(epsilon=config.get('l1_epsilon', 1e-8))
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
                n_epochs=50,
                verbose=0,
                early_stopping_patience=15
            )

            # 验证集评估
            val_predictions = model.predict(
                self.val_data[:, 0].astype(int),
                self.val_data[:, 1].astype(int)
            )

            if self.data_manager.global_mean is not None:
                val_predictions += self.data_manager.global_mean
                val_targets = self.val_data[:, 2] + self.data_manager.global_mean
            else:
                val_targets = self.val_data[:, 2]

            # 计算MAE
            mae = self.mae_metric.calculate(val_targets, val_predictions)

            print(f"L1配置: lr={config['learning_rate']:.4f}, factors={config['latent_factors']}, "
                  f"reg={config['lambda_reg']:.6f}, eps={config.get('l1_epsilon', 1e-8):.2e}, MAE={mae:.4f}")

            return mae

        except Exception as e:
            print(f"L1配置评估失败 {config}: {e}")
            return 10.0


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


def create_l1_parameter_space():
    """创建L1专用参数空间"""
    try:
        space = ParameterSpace()
    except:
        space = SimpleParameterSpace()

    # L1损失专用参数
    space.add_continuous('learning_rate', 0.01, 0.08, scale='log')
    space.add_discrete('latent_factors', 15, 75, step=5)
    space.add_continuous('lambda_reg', 0.001, 0.02, scale='log')
    space.add_continuous('l1_epsilon', 1e-10, 1e-6, scale='log')  # L1平滑参数

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


def run_hpl_mae_optimization():
    """运行HPL针对MAE的专用优化实验"""
    print("="*60)
    print("HPL损失函数专用优化实验 - MAE目标")
    print("="*60)

    # 1. 准备改进的数据
    data_manager = setup_improved_data()

    # 2. 创建HPL专用目标函数（MAE版本）
    objective = HPLMAEObjectiveFunction(data_manager)

    # 3. 创建HPL专用参数空间
    space = create_hpl_parameter_space()

    # 4. 创建HPL专用约束
    constraints = create_hpl_constraints()

    # 5. 创建采样器和优化器
    try:
        sampler = LatinHypercubeSampler(space, seed=42)  # 优先使用LHS
        tracker = ExperimentTracker('hpl_mae_optimization', backend='memory')
        optimizer = HyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=constraints,
            tracker=tracker,
            maximize=False,  # 最小化MAE
            seed=42
        )
        print("使用完整版hyperopt组件")
    except:
        sampler = SimpleLatinHypercubeSampler(space, seed=42)
        tracker = SimpleExperimentTracker('hpl_mae_optimization', backend='memory')
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

    print("开始HPL专用MAE优化...")
    start_time = time.time()

    # 6. 运行优化 - 增加试验数和耐心
    best_trial = optimizer.optimize(
        n_trials=40,                 # 增加试验数
        no_improvement_rounds=15,    # 增加耐心
        batch_size=1
    )

    end_time = time.time()

    # 7. 打印HPL专用结果
    print("\n" + "="*60)
    print("HPL专用MAE优化结果")
    print("="*60)

    if best_trial:
        print(f"HPL最佳配置: {best_trial.config}")
        print(f"HPL最佳验证MAE: {best_trial.score:.4f}")
        print(f"HPL优化耗时: {end_time - start_time:.2f}秒")

        # HPL参数分析
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

        # 分析HPL参数的合理性
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

    # 8. 获取优化历史
    results = optimizer.get_results()
    print(f"\nHPL优化统计:")
    print(f"总试验数: {results['n_trials']}")
    print(f"成功试验数: {results['n_completed']}")
    print(f"失败试验数: {results['n_failed']}")

    return optimizer, best_trial


def run_l2_mae_optimization():
    """L2损失函数专用MAE优化（用于公平对比）"""
    print("开始L2专用MAE优化...")

    # 1. 准备相同的数据
    data_manager = setup_improved_data()

    # 2. 创建L2专用目标函数（MAE版本）
    objective = L2MAEObjectiveFunction(data_manager)

    # 3. 创建L2参数空间（不包含HPL参数）
    space = create_l2_parameter_space()

    # 4. 创建采样器和优化器
    try:
        sampler = LatinHypercubeSampler(space, seed=42)
        tracker = ExperimentTracker('l2_mae_optimization', backend='memory')
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
        tracker = SimpleExperimentTracker('l2_mae_optimization', backend='memory')
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
        print(f"L2最佳MAE: {best_trial.score:.4f}")
        print(f"L2最佳配置: {best_trial.config}")

    return optimizer, best_trial


def run_l1_mae_optimization():
    """L1损失函数专用MAE优化（直接优化MAE损失）"""
    print("开始L1专用MAE优化...")

    # 1. 准备相同的数据
    data_manager = setup_improved_data()

    # 2. 创建L1专用目标函数（MAE版本）
    objective = L1MAEObjectiveFunction(data_manager)

    # 3. 创建L1参数空间
    space = create_l1_parameter_space()

    # 4. 创建采样器和优化器
    try:
        sampler = LatinHypercubeSampler(space, seed=42)
        tracker = ExperimentTracker('l1_mae_optimization', backend='memory')
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
        tracker = SimpleExperimentTracker('l1_mae_optimization', backend='memory')
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
        print(f"L1最佳MAE: {best_trial.score:.4f}")
        print(f"L1最佳配置: {best_trial.config}")

    return optimizer, best_trial


def run_mae_comparison():
    """HPL vs L2 vs L1 在MAE指标上的公平对比"""
    print("\n" + "="*60)
    print("HPL vs L2 vs L1 MAE指标公平对比实验")
    print("="*60)

    results = {}

    # 1. HPL专用MAE优化
    print("\n1. 运行HPL专用MAE优化...")
    hpl_optimizer, hpl_best = run_hpl_mae_optimization()
    results['HPL'] = {
        'best_mae': hpl_best.score if hpl_best else float('inf'),
        'best_config': hpl_best.config if hpl_best else None,
        'optimizer': hpl_optimizer
    }

    # 2. L2专用MAE优化
    print("\n2. 运行L2专用MAE优化...")
    l2_optimizer, l2_best = run_l2_mae_optimization()
    results['L2'] = {
        'best_mae': l2_best.score if l2_best else float('inf'),
        'best_config': l2_best.config if l2_best else None,
        'optimizer': l2_optimizer
    }

    # 3. L1专用MAE优化
    print("\n3. 运行L1专用MAE优化...")
    l1_optimizer, l1_best = run_l1_mae_optimization()
    results['L1'] = {
        'best_mae': l1_best.score if l1_best else float('inf'),
        'best_config': l1_best.config if l1_best else None,
        'optimizer': l1_optimizer
    }

    # 4. 对比分析
    print("\n" + "="*60)
    print("HPL vs L2 vs L1 MAE对比结果")
    print("="*60)

    hpl_mae = results['HPL']['best_mae']
    l2_mae = results['L2']['best_mae']
    l1_mae = results['L1']['best_mae']

    # 找出最佳表现
    best_mae = min(hpl_mae, l2_mae, l1_mae)

    print(f"{'损失函数':<12} {'最佳MAE':<12} {'相对表现':<12} {'改进幅度':<12}")
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
