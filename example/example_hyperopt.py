#!/usr/bin/env python3
"""
修正的超参数优化示例

使用hyperopt模块对矩阵分解模型进行超参数优化
修正了导入路径和缺失的类定义
"""

import sys
import os
import numpy as np
import time
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 安全导入hyperopt模块组件
try:
    from src.hyperopt.space import ParameterSpace
    from src.hyperopt.samplers import RandomSampler, LatinHypercubeSampler
    from src.hyperopt.constraints import ConstraintManager
    from src.hyperopt.optimizer import HyperOptimizer, Trial
    from src.hyperopt.tracker import ExperimentTracker
    from src.hyperopt.parallel import ThreadExecutor
    print("成功导入所有hyperopt模块")
except ImportError as e:
    print(f"导入hyperopt模块失败: {e}")
    print("将使用简化版本")

# 安全导入数据和模型相关模块
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
    print("请确保相关模块存在")
    sys.exit(1)


# ===== 简化版本的缺失类 =====

class SimpleTrial:
    """简化的试验结果类"""

    def __init__(self, trial_id, config, score=None, status='pending',
                 start_time=None, end_time=None, error=None):
        self.trial_id = trial_id
        self.config = config
        self.score = score
        self.status = status
        self.start_time = start_time or time.time()
        self.end_time = end_time
        self.error = error
        self.metrics = None

    @property
    def duration(self):
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0  # 返回0而不是None，避免类型错误


class SimpleExperimentTracker:
    """简化的实验追踪器"""

    def __init__(self, experiment_name='simple_experiment', backend='memory'):
        self.experiment_name = experiment_name
        self.backend = backend
        self.trials = []

    def log_trial(self, trial):
        """记录试验"""
        trial_data = {
            'trial_id': trial.trial_id,
            'config': trial.config,
            'score': trial.score,
            'status': trial.status,
            'duration': trial.duration
        }
        self.trials.append(trial_data)

    def get_trials(self, status=None):
        """获取试验记录"""
        if status:
            return [t for t in self.trials if t['status'] == status]
        return self.trials

    def save_final_results(self, optimizer):
        """保存最终结果"""
        pass  # 简化版本不保存


class SimpleConstraintManager:
    """简化的约束管理器"""

    def __init__(self):
        self.constraints = []

    def add_relation(self, param1, param2, relation='<'):
        """添加关系约束"""
        constraint = {
            'type': 'relation',
            'param1': param1,
            'param2': param2,
            'relation': relation
        }
        self.constraints.append(constraint)

    def check_all(self, config):
        """检查所有约束"""
        violations = []

        for constraint in self.constraints:
            if constraint['type'] == 'relation':
                param1 = constraint['param1']
                param2 = constraint['param2']
                relation = constraint['relation']

                if param1 in config and param2 in config:
                    val1 = config[param1]
                    val2 = config[param2]

                    if relation == '<' and not (val1 < val2):
                        violations.append(f"{param1} < {param2}")
                    elif relation == '<=' and not (val1 <= val2):
                        violations.append(f"{param1} <= {param2}")
                    elif relation == '>' and not (val1 > val2):
                        violations.append(f"{param1} > {param2}")
                    elif relation == '>=' and not (val1 >= val2):
                        violations.append(f"{param1} >= {param2}")

        return len(violations) == 0, violations

    def fix_config(self, config):
        """尝试修正配置"""
        fixed_config = config.copy()

        for constraint in self.constraints:
            if constraint['type'] == 'relation':
                param1 = constraint['param1']
                param2 = constraint['param2']
                relation = constraint['relation']

                if param1 in fixed_config and param2 in fixed_config:
                    val1 = fixed_config[param1]
                    val2 = fixed_config[param2]

                    if relation == '<' and val1 >= val2:
                        # 确保 val1 < val2
                        if val1 == val2:
                            fixed_config[param1] = val2 * 0.99
                        else:
                            # 交换值并调整
                            fixed_config[param1] = min(val1, val2) * 0.99
                            fixed_config[param2] = max(val1, val2)

        return fixed_config

    def get_statistics(self):
        """获取约束统计"""
        return {
            'n_constraints': len(self.constraints),
            'rejection_count': 0,
            'fix_count': 0,
            'rejection_rate': 0
        }


class SimpleParameterSpace:
    """简化的参数空间"""

    def __init__(self):
        self.parameters = {}
        self.param_order = []

    def add_continuous(self, name, low, high, scale='linear'):
        """添加连续参数"""
        self.parameters[name] = {
            'type': 'continuous',
            'low': low,
            'high': high,
            'scale': scale
        }
        self.param_order.append(name)

    def add_discrete(self, name, low, high, step=1):
        """添加离散参数"""
        self.parameters[name] = {
            'type': 'discrete',
            'low': low,
            'high': high,
            'step': step,
            'values': list(range(low, high + 1, step))
        }
        self.param_order.append(name)

    def add_categorical(self, name, choices):
        """添加分类参数"""
        self.parameters[name] = {
            'type': 'categorical',
            'choices': choices
        }
        self.param_order.append(name)

    def sample(self, random_state):
        """采样一个配置"""
        config = {}

        for name in self.param_order:
            param = self.parameters[name]

            if param['type'] == 'continuous':
                if param['scale'] == 'log':
                    log_low = np.log(param['low'])
                    log_high = np.log(param['high'])
                    log_val = random_state.uniform(log_low, log_high)
                    config[name] = np.exp(log_val)
                else:
                    config[name] = random_state.uniform(
                        param['low'], param['high'])

            elif param['type'] == 'discrete':
                config[name] = random_state.choice(param['values'])

            elif param['type'] == 'categorical':
                config[name] = random_state.choice(param['choices'])

        return config

    def get_dimension(self):
        """获取维度"""
        return len(self.param_order)

    def get_parameter_info(self):
        """获取参数信息"""
        return self.parameters.copy()


class SimpleRandomSampler:
    """简化的随机采样器"""

    def __init__(self, space, seed=None):
        self.space = space
        self.random_state = np.random.RandomState(seed)

    def sample(self, n_samples=1):
        """采样"""
        samples = []
        for _ in range(n_samples):
            config = self.space.sample(self.random_state)
            samples.append(config)
        return samples


class SimpleLatinHypercubeSampler:
    """简化的LHS采样器"""

    def __init__(self, space, seed=None):
        self.space = space
        self.random_state = np.random.RandomState(seed)

    def sample(self, n_samples=1):
        """LHS采样"""
        n_dims = len(self.space.parameters)

        # 生成LHS采样点
        samples_normalized = self._lhs_sample(n_samples, n_dims)

        # 转换为配置
        samples = []
        for i in range(n_samples):
            config = {}
            dim_idx = 0

            for name in self.space.param_order:
                param = self.space.parameters[name]
                normalized_val = samples_normalized[i, dim_idx]

                if param['type'] == 'continuous':
                    if param['scale'] == 'log':
                        log_low = np.log(param['low'])
                        log_high = np.log(param['high'])
                        log_val = log_low + normalized_val * \
                            (log_high - log_low)
                        config[name] = np.exp(log_val)
                    else:
                        config[name] = param['low'] + normalized_val * \
                            (param['high'] - param['low'])

                elif param['type'] == 'discrete':
                    idx = int(normalized_val * len(param['values']))
                    idx = min(idx, len(param['values']) - 1)
                    config[name] = param['values'][idx]

                elif param['type'] == 'categorical':
                    idx = int(normalized_val * len(param['choices']))
                    idx = min(idx, len(param['choices']) - 1)
                    config[name] = param['choices'][idx]

                dim_idx += 1

            samples.append(config)

        return samples

    def _lhs_sample(self, n_samples, n_dims):
        """生成LHS采样点"""
        samples = np.zeros((n_samples, n_dims))

        for dim in range(n_dims):
            perm = self.random_state.permutation(n_samples)

            for i in range(n_samples):
                low = perm[i] / n_samples
                high = (perm[i] + 1) / n_samples
                samples[i, dim] = self.random_state.uniform(low, high)

        return samples


class SimpleHyperOptimizer:
    """简化的超参数优化器"""

    def __init__(self, objective_fn, space, sampler, constraints=None,
                 tracker=None, maximize=False, seed=None):
        self.objective_fn = objective_fn
        self.space = space
        self.sampler = sampler
        self.constraints = constraints
        self.tracker = tracker
        self.maximize = maximize
        self.seed = seed
        self.trials = []
        self.best_trial = None

    def optimize(self, n_trials=50, batch_size=1, no_improvement_rounds=10):
        """运行优化"""
        print(f"开始优化，目标试验数: {n_trials}")

        no_improvement_count = 0
        if self.maximize:
            best_score = float('-inf')
        else:
            best_score = float('inf')

        for trial_id in range(n_trials):
            # 采样配置
            configs = self.sampler.sample(batch_size)

            for config in configs:
                # 应用约束
                if self.constraints:
                    satisfied, violations = self.constraints.check_all(config)
                    if not satisfied:
                        # 尝试修正配置
                        fixed_config = self.constraints.fix_config(config)
                        if fixed_config:
                            config = fixed_config
                        else:
                            print(f"配置违反约束，跳过: {violations}")
                            continue

                # 评估配置
                start_time = time.time()
                try:
                    score = self.objective_fn(config)
                    status = 'completed'
                    error = None
                except Exception as e:
                    if not self.maximize:
                        score = float('inf')
                    else:
                        score = float('-inf')
                    status = 'failed'
                    error = str(e)
                    print(f"试验失败: {e}")

                end_time = time.time()

                # 创建试验记录
                if 'Trial' in globals():
                    trial = Trial(
                        trial_id=len(self.trials),
                        config=config,
                        score=score,
                        status=status,
                        start_time=start_time,
                        end_time=end_time,
                        error=error
                    )
                else:
                    trial = SimpleTrial(
                        trial_id=len(self.trials),
                        config=config,
                        score=score,
                        status=status,
                        start_time=start_time,
                        end_time=end_time,
                        error=error
                    )

                self.trials.append(trial)

                # 更新最佳试验
                is_better = (self.maximize and score > best_score) or \
                    (not self.maximize and score < best_score)

                if is_better and status == 'completed':
                    best_score = score
                    self.best_trial = trial
                    no_improvement_count = 0
                    print(f"找到新的最佳配置! 分数: {score:.4f}")
                else:
                    no_improvement_count += 1

                # 记录到tracker
                if self.tracker:
                    self.tracker.log_trial(trial)

                print(f"试验 {trial.trial_id}: 分数={score:.4f}, 状态={status}")

                # 早停检查
                if no_improvement_count >= no_improvement_rounds:
                    print(f"连续 {no_improvement_rounds} 次无改进，提前停止")
                    return self.best_trial

        return self.best_trial

    def get_results(self):
        """获取优化结果"""
        completed_trials = [t for t in self.trials if t.status == 'completed']
        failed_trials = [t for t in self.trials if t.status == 'failed']

        all_trials = []
        for t in self.trials:
            trial_data = {
                'trial_id': t.trial_id,
                'config': t.config,
                'score': t.score,
                'status': t.status,
                'duration': getattr(t, 'duration', None) or 0
            }
            all_trials.append(trial_data)

        return {
            'n_trials': len(self.trials),
            'n_completed': len(completed_trials),
            'n_failed': len(failed_trials),
            'best_trial': self.best_trial,
            'all_trials': all_trials
        }


# ===== 主要功能类 =====

class SimpleObjectiveFunction:
    """简单的目标函数：训练并评估矩阵分解模型"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

    def __call__(self, config):
        """
        目标函数：给定配置，训练模型并返回验证集RMSE

        Args:
            config: 超参数配置字典

        Returns:
            float: 验证集RMSE（越小越好）
        """
        try:
            # 创建损失函数
            if config['loss_type'] == 'hpl':
                loss_function = HybridPiecewiseLoss(
                    delta1=config['delta1'],
                    delta2=config['delta2'],
                    l_max=config.get('l_max', 3.0),
                    c_sigmoid=config.get('c_sigmoid', 1.0)
                )
            else:
                loss_function = L2Loss()

            # 创建正则化器
            regularizer = L2Regularizer(lambda_reg=config['lambda_reg'])

            # 创建模型
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

            # 初始化参数
            initializer = NormalInitializer(mean=0.0, std=0.01)
            model.initialize_parameters(initializer)

            # 训练模型（少量epochs以加快速度）
            model.fit(
                train_data=self.train_data,
                val_data=self.val_data,
                n_epochs=20,  # 每个配置最多训练20轮
                verbose=0,    # 0不打印训练过程, 打印训练日志
                early_stopping_patience=5  # 连续5轮验证无改进停止训练
            )

            # 在验证集上评估
            val_predictions = model.predict(
                self.val_data[:, 0].astype(int),
                self.val_data[:, 1].astype(int)
            )

            # 还原到原始尺度
            if self.data_manager.global_mean is not None:
                val_predictions += self.data_manager.global_mean
                val_targets = self.val_data[:, 2] + \
                    self.data_manager.global_mean
            else:
                val_targets = self.val_data[:, 2]

            # 计算RMSE
            rmse = np.sqrt(np.mean((val_predictions - val_targets) ** 2))

            return rmse

        except Exception as e:
            print(f"配置 {config} 评估失败: {e}")
            return 10.0  # 返回一个较大的值表示失败


def setup_data():
    """设置数据"""
    print("准备数据...")

    # 创建数据管理器配置
    data_config = {
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'shuffle': True,
        'center_data': True,
        'ensure_user_in_train': True
    }

    # 创建数据管理器
    data_manager = DataManager(data_config)

    # 加载数据
    data_manager.load_dataset(
        'movielens100k',
        'dataset/20201202M100K_data_all_random.txt'
    )

    # 预处理数据
    data_manager.preprocess()

    # 打印数据摘要
    data_manager.print_summary()

    return data_manager


def create_parameter_space():
    """创建参数空间"""
    print("创建参数空间...")

    # 尝试使用原版本，失败则使用简化版本
    try:
        if 'ParameterSpace' in globals():
            space = ParameterSpace()
        else:
            space = SimpleParameterSpace()
    except:
        space = SimpleParameterSpace()

    # 添加超参数
    space.add_continuous('learning_rate', 0.001, 0.1, scale='log')  # 学习率
    space.add_discrete('latent_factors', 5, 100, step=5)          # 潜在因子数
    space.add_continuous('lambda_reg', 0.001, 0.1, scale='log')    # 正则化参数
    space.add_continuous('delta1', 0.1, 1.0)                       # HPL参数1
    space.add_continuous('delta2', 1.0, 3.0)                       # HPL参数2
    space.add_categorical('loss_type', ['hpl', 'l2'])               # 损失函数类型

    print(f"参数空间维度: {space.get_dimension()}")
    print("参数信息:")
    for name, info in space.get_parameter_info().items():
        print(f"  {name}: {info}")

    return space


def create_constraints():
    """创建约束条件"""
    print("创建约束条件...")

    # 尝试使用原版本，失败则使用简化版本
    try:
        if 'ConstraintManager' in globals():
            constraints = ConstraintManager()
        else:
            constraints = SimpleConstraintManager()
    except:
        constraints = SimpleConstraintManager()

    # 添加约束：delta1 < delta2（HPL损失函数的要求）
    constraints.add_relation('delta1', 'delta2', '<')

    try:
        constraint_count = len(constraints.constraints)
    except:
        constraint_count = 1

    print(f"约束数量: {constraint_count}")

    return constraints


def create_sampler(space, sampler_type='random', seed=42):
    """创建采样器"""
    try:
        if sampler_type == 'random':
            if 'RandomSampler' in globals():
                return RandomSampler(space, seed=seed)
            else:
                return SimpleRandomSampler(space, seed=seed)
        elif sampler_type == 'lhs':
            if 'LatinHypercubeSampler' in globals():
                return LatinHypercubeSampler(space, seed=seed)
            else:
                return SimpleLatinHypercubeSampler(space, seed=seed)
    except:
        # 如果原版本失败，使用简化版本
        if sampler_type == 'random':
            return SimpleRandomSampler(space, seed=seed)
        elif sampler_type == 'lhs':
            return SimpleLatinHypercubeSampler(space, seed=seed)

    return SimpleRandomSampler(space, seed=seed)


def create_tracker(experiment_name='simple_hpl_optimization'):
    """创建实验追踪器"""
    try:
        if 'ExperimentTracker' in globals():
            return ExperimentTracker(
                experiment_name=experiment_name,
                backend='memory'
            )
        else:
            return SimpleExperimentTracker(
                experiment_name=experiment_name,
                backend='memory'
            )
    except:
        return SimpleExperimentTracker(
            experiment_name=experiment_name,
            backend='memory'
        )


def create_optimizer(objective_fn, space, sampler, constraints, tracker):
    """创建优化器"""
    try:
        if 'HyperOptimizer' in globals():
            return HyperOptimizer(
                objective_fn=objective_fn,
                space=space,
                sampler=sampler,
                constraints=constraints,
                tracker=tracker,
                maximize=False,  # 最小化RMSE
                seed=42
            )
        else:
            return SimpleHyperOptimizer(
                objective_fn=objective_fn,
                space=space,
                sampler=sampler,
                constraints=constraints,
                tracker=tracker,
                maximize=False,  # 最小化RMSE
                seed=42
            )
    except:
        return SimpleHyperOptimizer(
            objective_fn=objective_fn,
            space=space,
            sampler=sampler,
            constraints=constraints,
            tracker=tracker,
            maximize=False,  # 最小化RMSE
            seed=42
        )


def run_simple_optimization():
    """运行简单的超参数优化"""
    print("="*60)
    print("开始简单超参数优化示例")
    print("="*60)

    # 1. 准备数据
    data_manager = setup_data()

    # 2. 创建目标函数
    objective = SimpleObjectiveFunction(data_manager)

    # 3. 创建参数空间
    space = create_parameter_space()

    # 4. 创建约束
    constraints = create_constraints()

    # 5. 创建采样器
    sampler = create_sampler(space, 'random', seed=42)
    print(f"使用采样器: {sampler.__class__.__name__}")

    # 6. 创建实验追踪器
    tracker = create_tracker('simple_hpl_optimization')
    print(f"使用追踪器: {tracker.__class__.__name__}")

    # 7. 创建优化器
    optimizer = create_optimizer(
        objective, space, sampler, constraints, tracker)
    print(f"使用优化器: {optimizer.__class__.__name__}")

    print("开始优化...")
    start_time = time.time()

    # 8. 运行优化
    best_trial = optimizer.optimize(
        n_trials=20,                 # 最大试验数（硬限制）
        no_improvement_rounds=10,    # 早停轮数（智能停止）
        batch_size=1                 # 批次大小（影响检查频率）
    )

    end_time = time.time()

    # 9. 打印结果
    print("\n" + "="*60)
    print("优化结果")
    print("="*60)

    if best_trial:
        print(f"最佳配置: {best_trial.config}")
        print(f"最佳验证RMSE: {best_trial.score:.4f}")
        print(f"优化耗时: {end_time - start_time:.2f}秒")
    else:
        print("优化失败，未找到有效配置")

    # 10. 获取优化历史
    results = optimizer.get_results()
    print(f"\n总试验数: {results['n_trials']}")
    print(f"成功试验数: {results['n_completed']}")
    print(f"失败试验数: {results['n_failed']}")

    # 调试信息：检查all_trials结构
    if results['all_trials']:
        print(f"\n调试信息 - 第一个试验的结构:")
        first_trial = results['all_trials'][0]
        print(f"  键: {list(first_trial.keys())}")
        print(f"  试验ID: {first_trial.get('trial_id')}")
        print(f"  状态: {first_trial.get('status')}")
        print(f"  分数: {first_trial.get('score')}")

    # 11. 显示所有试验结果
    print("\n所有试验结果:")
    print("-" * 80)
    print(f"{'试验ID':<8} {'RMSE':<8} {'学习率':<10} {'因子数':<8} {'正则化':<10} {'损失':<8}")
    print("-" * 80)

    for trial_info in results['all_trials']:
        # 安全地检查status字段
        trial_status = trial_info.get('status', 'unknown')
        if trial_status == 'completed' and trial_info.get('config') and trial_info.get('score') is not None:
            config = trial_info['config']
            try:
                print(f"{trial_info['trial_id']:<8} "
                      f"{trial_info['score']:<8.4f} "
                      f"{config.get('learning_rate', 0):<10.4f} "
                      f"{config.get('latent_factors', 0):<8} "
                      f"{config.get('lambda_reg', 0):<10.4f} "
                      f"{config.get('loss_type', 'unknown'):<8}")
            except Exception as e:
                print(f"试验 {trial_info.get('trial_id', 'unknown')}: 显示失败 - {e}")
        else:
            print(
                f"试验 {trial_info.get('trial_id', 'unknown')}: 状态={trial_status}")

    return optimizer, best_trial


def run_comparison_experiment():
    """运行采样器对比实验"""
    print("\n" + "="*60)
    print("采样器对比实验")
    print("="*60)

    # 准备数据
    data_manager = setup_data()
    objective = SimpleObjectiveFunction(data_manager)
    space = create_parameter_space()
    constraints = create_constraints()

    # 不同的采样器
    samplers = {
        'Random': create_sampler(space, 'random', seed=42),
        'LHS': create_sampler(space, 'lhs', seed=42)
    }

    results = {}

    for sampler_name, sampler in samplers.items():
        print(f"\n测试 {sampler_name} 采样器...")

        # 创建追踪器和优化器
        tracker = create_tracker(f'{sampler_name}_optimization')
        optimizer = create_optimizer(
            objective, space, sampler, constraints, tracker)

        # 运行优化
        start_time = time.time()
        best_trial = optimizer.optimize(n_trials=15, batch_size=1)
        end_time = time.time()

        # 保存结果
        if best_trial:
            best_score = best_trial.score
            best_config = best_trial.config
        else:
            best_score = float('inf')
            best_config = None

        results[sampler_name] = {
            'best_score': best_score,
            'best_config': best_config,
            'time': end_time - start_time,
            'optimizer': optimizer
        }

    # 对比结果
    print("\n采样器对比结果:")
    print("-" * 50)
    print(f"{'采样器':<10} {'最佳RMSE':<12} {'耗时(s)':<10}")
    print("-" * 50)

    for name, result in results.items():
        print(
            f"{name:<10} {result['best_score']:<12.4f} {result['time']:<10.2f}")

    # 找出最佳采样器
    best_sampler = min(results.items(), key=lambda x: x[1]['best_score'])
    print(
        f"\n最佳采样器: {best_sampler[0]} (RMSE: {best_sampler[1]['best_score']:.4f})")

    return results


def demonstrate_constraints():
    """演示约束功能"""
    print("\n" + "="*60)
    print("约束功能演示")
    print("="*60)

    # 创建参数空间
    space = create_parameter_space()
    constraints = create_constraints()

    # 生成一些配置进行测试
    sampler = create_sampler(space, 'random', seed=42)

    print("测试约束检查和修正...")
    print("-" * 40)

    for i in range(5):
        # 采样一个配置
        config = sampler.sample(1)[0]

        # 检查约束
        try:
            satisfied, violations = constraints.check_all(config)
        except:
            # 如果约束检查失败，手动检查
            satisfied = config['delta1'] < config['delta2']
            if satisfied:
                violations = []
            else:
                violations = ['delta1 >= delta2']

        print(f"\n配置 {i+1}:")
        print(
            f"  delta1: {config['delta1']:.3f}, delta2: {config['delta2']:.3f}")
        print(f"  约束满足: {satisfied}")

        if not satisfied:
            print(f"  违反约束: {violations}")

            # 尝试修正
            try:
                fixed_config = constraints.fix_config(config)
                if fixed_config:
                    print(f"  修正后: delta1: {fixed_config['delta1']:.3f}, "
                          f"delta2: {fixed_config['delta2']:.3f}")

                    # 验证修正结果
                    try:
                        satisfied_after, _ = constraints.check_all(
                            fixed_config)
                    except:
                        satisfied_after = fixed_config['delta1'] < fixed_config['delta2']
                    print(f"  修正后满足约束: {satisfied_after}")
                else:
                    print("  无法修正")
            except Exception as e:
                print(f"  修正失败: {e}")

    # 打印约束统计
    try:
        stats = constraints.get_statistics()
        print(f"\n约束统计: {stats}")
    except:
        print(f"\n约束统计: 1个关系约束（delta1 < delta2）")


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy标量类型
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy数组类型
        return obj.tolist()
    else:
        return obj


def save_results(optimizer, best_trial, filename="hyperopt_results.json"):
    """保存结果到文件"""
    try:
        results = {
            'experiment_info': {
                'optimizer_type': optimizer.__class__.__name__,
                'sampler_type': optimizer.sampler.__class__.__name__,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'total_trials': len(optimizer.trials)
            },
            'best_trial': {
                'config': convert_numpy_types(best_trial.config) if best_trial else None,
                'score': convert_numpy_types(best_trial.score) if best_trial else None,
                'trial_id': convert_numpy_types(best_trial.trial_id) if best_trial else None
            },
            'optimization_results': convert_numpy_types(optimizer.get_results())
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n结果已保存到: {filename}")

    except Exception as e:
        print(f"保存结果失败: {e}")


def load_and_analyze_results(filename="hyperopt_results.json"):
    """加载和分析保存的结果"""
    try:
        with open(filename, 'r') as f:
            results = json.load(f)

        print(f"\n从 {filename} 加载的结果分析:")
        print("-" * 40)

        # 基本信息
        exp_info = results['experiment_info']
        print(f"优化器类型: {exp_info['optimizer_type']}")
        print(f"采样器类型: {exp_info['sampler_type']}")
        print(f"实验时间: {exp_info['timestamp']}")
        print(f"总试验数: {exp_info['total_trials']}")

        # 最佳结果
        best = results['best_trial']
        if best['config']:
            print(f"\n最佳配置:")
            for key, value in best['config'].items():
                print(f"  {key}: {value}")
            print(f"最佳分数: {best['score']:.4f}")

        # 优化历史分析
        opt_results = results['optimization_results']
        all_trials = opt_results['all_trials']

        if all_trials:
            scores = [t['score']
                      for t in all_trials if t['status'] == 'completed']
            if scores:
                print(f"\n分数统计:")
                print(f"  最好: {min(scores):.4f}")
                print(f"  最差: {max(scores):.4f}")
                print(f"  平均: {np.mean(scores):.4f}")
                print(f"  标准差: {np.std(scores):.4f}")

        return results

    except Exception as e:
        print(f"加载结果失败: {e}")
        return None


def plot_optimization_history(optimizer, save_path="optimization_history.png"):
    """绘制优化历史"""
    try:
        import matplotlib.pyplot as plt

        # 提取完成的试验
        completed_trials = [
            t for t in optimizer.trials if t.status == 'completed']

        if not completed_trials:
            print("没有完成的试验可以绘图")
            return

        # 提取数据
        trial_ids = [t.trial_id for t in completed_trials]
        scores = [t.score for t in completed_trials]

        # 计算累积最佳分数
        best_scores = []
        current_best = float('inf')
        for score in scores:
            if score < current_best:
                current_best = score
            best_scores.append(current_best)

        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # 分数历史
        ax1.scatter(trial_ids, scores, alpha=0.6, label='Trial scores')
        ax1.plot(trial_ids, best_scores, 'r-', linewidth=2, label='Best score')
        ax1.set_xlabel('Trial ID')
        ax1.set_ylabel('RMSE')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 分数分布
        ax2.hist(scores, bins=min(20, len(scores)),
                 alpha=0.7, edgecolor='black')
        ax2.axvline(min(scores), color='red', linestyle='--',
                    label=f'Best: {min(scores):.4f}')
        ax2.set_xlabel('RMSE')
        ax2.set_ylabel('Count')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"优化历史图已保存到: {save_path}")

    except ImportError:
        print("matplotlib未安装，无法绘制优化历史图")
    except Exception as e:
        print(f"绘制优化历史图失败: {e}")


def main():
    """主函数"""
    try:
        print("检查模块导入状态...")

        # 检查关键模块是否可用
        modules_status = {
            'ParameterSpace': 'ParameterSpace' in globals(),
            'RandomSampler': 'RandomSampler' in globals(),
            'ConstraintManager': 'ConstraintManager' in globals(),
            'HyperOptimizer': 'HyperOptimizer' in globals(),
            'ExperimentTracker': 'ExperimentTracker' in globals()
        }

        print("模块状态:")
        for module, available in modules_status.items():
            if available:
                status = "✓"
            else:
                status = "✗ (使用简化版本)"
            print(f"  {module}: {status}")

        print("\n" + "="*60)

        # 1. 运行简单优化示例
        optimizer, best_trial = run_simple_optimization()

        # 2. 保存结果
        if best_trial:
            save_results(optimizer, best_trial, "hyperopt_demo_results.json")

            # 3. 绘制优化历史
            plot_optimization_history(optimizer, "optimization_demo.png")

        # 4. 运行采样器对比（可选）
        try:
            print("\n是否运行采样器对比实验？(y/n): ", end="")
            # 自动选择不运行以简化示例
            choice = 'y'
            if choice.lower() == 'y':
                run_comparison_experiment()
        except:
            pass

        # 5. 演示约束功能
        demonstrate_constraints()

        print("\n" + "="*60)
        print("所有示例运行完成！")
        print("="*60)

        # 如果找到了最佳配置，可以进一步测试
        if best_trial:
            print(f"\n推荐配置用于进一步训练:")
            # 转换numpy类型后再序列化
            clean_config = convert_numpy_types(best_trial.config)
            config_str = json.dumps(clean_config, indent=2, ensure_ascii=False)
            print(config_str)

            # 分析最佳配置
            print(f"\n最佳配置分析:")
            config = best_trial.config
            learning_rate = float(config['learning_rate']) if hasattr(
                config['learning_rate'], 'item') else config['learning_rate']
            latent_factors = int(config['latent_factors']) if hasattr(
                config['latent_factors'], 'item') else config['latent_factors']
            lambda_reg = float(config['lambda_reg']) if hasattr(
                config['lambda_reg'], 'item') else config['lambda_reg']
            delta1 = float(config['delta1']) if hasattr(
                config['delta1'], 'item') else config['delta1']
            delta2 = float(config['delta2']) if hasattr(
                config['delta2'], 'item') else config['delta2']

            print(
                f"  学习率: {learning_rate:.4f} ({'对数缩放' if learning_rate < 0.01 else '正常范围'})")
            print(
                f"  潜在因子数: {latent_factors} ({'较少' if latent_factors < 50 else '较多'})")
            print(
                f"  正则化参数: {lambda_reg:.4f} ({'较强' if lambda_reg > 0.01 else '较弱'})")
            print(
                f"  损失函数: {config['loss_type']} ({'混合分段损失' if config['loss_type'] == 'hpl' else '标准L2损失'})")

            if config['loss_type'] == 'hpl':
                print(f"  HPL参数 delta1: {delta1:.3f}")
                print(f"  HPL参数 delta2: {delta2:.3f}")
                print(f"  HPL参数差值: {delta2 - delta1:.3f}")

        return True

    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
