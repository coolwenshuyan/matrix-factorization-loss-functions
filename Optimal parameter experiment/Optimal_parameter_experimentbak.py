#!/usr/bin/env python3
# Optimal parameter experiment\Optimal_parameter_experiment.py
"""
HPL损失函数专用优化实验 - 重构版本 (批量模式)

专门针对混合分段损失函数(HPL)的超参数优化
重构为统一的批量处理模式，删除重复代码，提高可维护性
"""
from pathlib import Path
import sys
import os
import numpy as np
import time
import json
import glob
import traceback

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

# 在文件顶部导入短信通知模块
from utils.sms_notification import send_sms_notification

class DatasetAwareResultManager:
    """数据集感知的结果管理器"""

    DATASET_CONFIGS = {
        'movielens100k': {
            'name': 'MovieLens100K',
            'size': '100K',
            'version': '2020-12-02',
            'short_name': 'ml100k'
        },
        'movielens1m': {
            'name': 'MovieLens1M',
            'size': '1M',
            'version': '2021-01-01',
            'short_name': 'ml1m'
        },
        'amazon_movies': {
            'name': 'Amazon Movies',
            'size': 'Various',
            'version': '2023',
            'short_name': 'amazon_mv'
        },
        'amazon_movies': {
            'name': 'Amazon Movies',
            'size': 'Various',
            'version': '2023',
            'short_name': 'amazon_mv'
        }
    }

    def __init__(self, dataset_name='movielens100k', dataset_file=None, base_dir=None):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file

        # 修改：设置base_dir为当前脚本同级目录下的results
        if base_dir is None:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_script_dir, 'results')

        self.base_dir = Path(base_dir)

        # 获取数据集信息
        self.dataset_info = self.DATASET_CONFIGS.get(dataset_name, {
            'name': dataset_name.title(),
            'size': 'Unknown',
            'version': 'Unknown',
            'short_name': dataset_name.lower()
        })

        # 创建数据集专用目录
        self.dataset_dir = self.base_dir / self.dataset_info['short_name']
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._create_subdirectories()

        # 打印保存路径信息
        print(f"📁 结果保存目录: {self.dataset_dir}")

    def _create_subdirectories(self):
        """创建必要的子目录"""
        subdirs = ['hpl_optimization', 'l2_optimization', 'comparison', 'models', 'reports', 'logs']
        for subdir in subdirs:
            (self.dataset_dir / subdir).mkdir(exist_ok=True)

    def generate_filename(self, experiment_type, file_format='json', include_timestamp=True):
        """生成数据集相关的文件名"""
        timestamp = time.strftime("%Y%m%d_%H%M%S") if include_timestamp else ""
        parts = [self.dataset_info['short_name'], experiment_type, timestamp]
        parts = [part for part in parts if part]
        filename = "_".join(parts)
        return f"{filename}.{file_format}"

    def get_save_path(self, experiment_type, file_format='json'):
        """获取完整的保存路径"""
        subdir_map = {
            'hpl_optimization': 'hpl_optimization',
            'enhanced_hpl_optimization': 'hpl_optimization',
            'l2_optimization': 'l2_optimization',
            'hpl_vs_l2_comparison': 'comparison',
            'model_evaluation': 'reports',
            'training_log': 'logs'
        }
        subdir = subdir_map.get(experiment_type, 'hpl_optimization')
        filename = self.generate_filename(experiment_type, file_format)
        return self.dataset_dir / subdir / filename

    def save_experiment_results(self, results, experiment_type, metadata=None):
        """保存实验结果，包含完整的数据集信息"""
        try:
            save_data = {
                'dataset_info': {
                    'dataset_name': self.dataset_name,
                    'dataset_file': self.dataset_file,
                    'dataset_display_name': self.dataset_info['name'],
                    'dataset_size': self.dataset_info['size'],
                    'dataset_version': self.dataset_info['version']
                },
                'experiment_info': {
                    'experiment_type': experiment_type,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'experiment_id': f"{experiment_type}_{time.strftime('%Y%m%d_%H%M%S')}"
                },
                'results': results
            }

            if metadata:
                save_data['metadata'] = metadata

            save_path = self.get_save_path(experiment_type, 'json')
            clean_data = self._convert_numpy_types(save_data)

            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(clean_data, f, indent=2, ensure_ascii=False)

            print(f"\n📁 实验结果已保存到: {save_path}")
            print(f"   数据集: {self.dataset_info['name']}")
            print(f"   实验类型: {experiment_type}")

            return save_path

        except Exception as e:
            print(f"保存实验结果失败: {e}")
            return None

    def _convert_numpy_types(self, obj):
        """递归转换numpy类型"""
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return obj


class OptimizationMonitor:
    """统一的优化监控器"""

    def __init__(self, result_manager, experiment_type='optimization'):
        self.result_manager = result_manager
        self.experiment_type = experiment_type
        self.trial_history = []
        self.best_score_history = []

        # 创建实时日志文件 - 修改：使用当前目录下的results
        log_filename = f"{experiment_type}_log_{time.strftime('%Y%m%d_%H%M%S')}.txt"
        self.log_path = result_manager.dataset_dir / 'logs' / log_filename

        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self.log_file.write(f"优化日志 - {result_manager.dataset_info['name']}\n")
        self.log_file.write(f"保存路径: {result_manager.dataset_dir}\n")
        self.log_file.write(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write("="*60 + "\n\n")
        self.log_file.flush()

    def update(self, trial):
        """更新监控信息"""
        self.trial_history.append(trial)

        if not self.best_score_history:
            self.best_score_history.append(trial.score)
        else:
            self.best_score_history.append(
                min(self.best_score_history[-1], trial.score)
            )

        # 记录到日志文件
        self.log_file.write(f"试验 {len(self.trial_history)}: 分数={trial.score:.4f}, 配置={trial.config}\n")
        self.log_file.flush()

        # 每10次试验打印进度
        if len(self.trial_history) % 10 == 0:
            progress_msg = f"进度: {len(self.trial_history)} 次试验完成，当前最佳: {min(self.best_score_history):.4f}"
            print(progress_msg)
            self.log_file.write(f"\n{progress_msg}\n\n")
            self.log_file.flush()

    def generate_final_report(self):
        """生成最终报告并保存"""
        report_data = {
            'total_trials': len(self.trial_history),
            'best_score': min(self.best_score_history) if self.best_score_history else None,
            'best_trial': min(self.trial_history, key=lambda x: x.score) if self.trial_history else None,
            'score_history': self.best_score_history,
            'convergence_analysis': self._analyze_convergence()
        }

        report_path = self.result_manager.save_experiment_results(
            results=report_data,
            experiment_type=f"{self.experiment_type}_report",
            metadata={'monitor_type': 'OptimizationMonitor'}
        )

        # 关闭日志文件
        self.log_file.write(f"\n结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        self.log_file.write(f"总试验次数: {len(self.trial_history)}\n")
        if self.best_score_history:
            self.log_file.write(f"最终最佳分数: {min(self.best_score_history):.4f}\n")
        self.log_file.close()

        print(f"\n📊 监控报告已保存至: {report_path}")
        print(f"📝 日志文件: {self.log_path}")
        print(f"总试验次数: {len(self.trial_history)}")
        if self.best_score_history:
            print(f"最终最佳分数: {min(self.best_score_history):.4f}")

    def _analyze_convergence(self):
        """分析收敛情况"""
        if len(self.best_score_history) < 10:
            return {'status': 'insufficient_data'}

        recent_scores = self.best_score_history[-10:]
        improvement = recent_scores[0] - recent_scores[-1]

        return {
            'recent_improvement': improvement,
            'converged': improvement < 0.001,
            'best_trial_index': self.best_score_history.index(min(self.best_score_history)),
            'plateau_length': len(self.best_score_history) - self.best_score_history.index(min(self.best_score_history))
        }


class BaseObjectiveFunction:
    """基础目标函数类"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        print(f"目标函数初始化:")
        print(f"  训练集: {len(self.train_data)} 条")
        print(f"  验证集: {len(self.val_data)} 条")
        print(f"  测试集: {len(self.test_data)} 条")

    def _create_model(self, config, loss_function):
        """创建模型的通用方法"""
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
        return model

    def _train_and_evaluate(self, model, n_epochs=50):
        """训练和评估模型的通用方法"""
        model.fit(
            train_data=self.train_data,
            val_data=self.val_data,
            n_epochs=n_epochs,
            verbose=0,
            early_stopping_patience=15
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
        return rmse

    def _format_config_output(self, config, rmse, loss_type):
        """格式化配置输出"""
        base_info = (f"{loss_type}配置: 学习率={config['learning_rate']:.4f}, "
                    f"因子数={config['latent_factors']}, "
                    f"正则化={config['lambda_reg']:.6f}")

        if loss_type == "HPL" and 'delta1' in config:
            hpl_info = (f", δ1={config['delta1']:.3f}, δ2={config['delta2']:.3f}, "
                       f"l_max={config.get('l_max', 4.0):.2f}, c_sig={config.get('c_sigmoid', 1.0):.2f}")
            base_info += hpl_info

        print(f"{base_info}, RMSE={rmse:.4f}")


class HPLObjectiveFunction(BaseObjectiveFunction):
    """HPL专用目标函数"""

    def __call__(self, config):
        try:
            loss_function = HybridPiecewiseLoss(
                delta1=config['delta1'],
                delta2=config['delta2'],
                l_max=config.get('l_max', 4.0),
                c_sigmoid=config.get('c_sigmoid', 1.0)
            )

            model = self._create_model(config, loss_function)
            rmse = self._train_and_evaluate(model)
            self._format_config_output(config, rmse, "HPL")
            return rmse

        except Exception as e:
            print(f"HPL配置评估失败 {config}: {e}")
            return 10.0


class L2ObjectiveFunction(BaseObjectiveFunction):
    """L2专用目标函数"""

    def __call__(self, config):
        try:
            loss_function = L2Loss()
            model = self._create_model(config, loss_function)
            rmse = self._train_and_evaluate(model)
            self._format_config_output(config, rmse, "L2")
            return rmse

        except Exception as e:
            print(f"L2配置评估失败 {config}: {e}")
            return 10.0


class ParameterSpaceFactory:
    """参数空间工厂类"""

    @staticmethod
    def create_hpl_space():
        """创建HPL专用参数空间"""
        try:
            space = ParameterSpace()
        except:
            space = SimpleParameterSpace()

        # 基础模型参数
        space.add_continuous('learning_rate', 0.01, 0.08, scale='log')
        space.add_discrete('latent_factors', 15, 75, step=5)
        space.add_continuous('lambda_reg', 0.001, 0.02, scale='log')

        # HPL专用参数
        space.add_continuous('delta1', 0.05, 1.5)
        space.add_continuous('delta2', 0.8, 4.0)
        space.add_continuous('l_max', 2.5, 6.0)
        space.add_continuous('c_sigmoid', 0.3, 3.0)

        print(f"HPL参数空间维度: {space.get_dimension()}")
        return space

    @staticmethod
    def create_l2_space():
        """创建L2专用参数空间"""
        try:
            space = ParameterSpace()
        except:
            space = SimpleParameterSpace()

        space.add_continuous('learning_rate', 0.01, 0.08, scale='log')
        space.add_discrete('latent_factors', 15, 75, step=5)
        space.add_continuous('lambda_reg', 0.001, 0.02, scale='log')

        return space

    @staticmethod
    def create_enhanced_hpl_space():
        """创建增强的HPL参数空间"""
        try:
            space = ParameterSpace()
        except:
            space = SimpleParameterSpace()

        # 基础模型参数 - 扩展范围
        space.add_continuous('learning_rate', 0.005, 0.15, scale='log')
        space.add_discrete('latent_factors', 8, 120, step=8)
        space.add_continuous('lambda_reg', 0.0001, 0.05, scale='log')

        # HPL专用参数 - 扩展范围
        space.add_continuous('delta1', 0.05, 1.0)
        space.add_continuous('delta2', 0.3, 3.5)
        space.add_continuous('l_max', 1.5, 6.0)
        space.add_continuous('c_sigmoid', 0.2, 3.0)

        print(f"增强HPL参数空间维度: {space.get_dimension()}")
        return space


class ConstraintFactory:
    """约束工厂类"""

    @staticmethod
    def create_hpl_constraints():
        """创建HPL专用约束条件"""
        try:
            constraints = ConstraintManager()
        except:
            constraints = SimpleConstraintManager()

        # 核心约束：delta1 < delta2
        constraints.add_relation('delta1', 'delta2', '<')
        return constraints


class OptimizerFactory:
    """优化器工厂类"""

    @staticmethod
    def create_optimizer(objective, space, constraints=None, name='optimization'):
        """创建优化器"""
        try:
            sampler = LatinHypercubeSampler(space, seed=42)
            tracker = ExperimentTracker(name, backend='memory')
            optimizer = HyperOptimizer(
                objective_fn=objective,
                space=space,
                sampler=sampler,
                constraints=constraints,
                tracker=tracker,
                maximize=False,
                seed=42
            )
            print("使用完整版hyperopt组件")
        except:
            sampler = SimpleLatinHypercubeSampler(space, seed=42)
            tracker = SimpleExperimentTracker(name, backend='memory')
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

        return optimizer


def setup_data_manager(dataset_name='movielens100k',
                      dataset_file='dataset/20201202M100K_data_all_random.txt'):
    """设置数据管理器"""
    print(f"准备数据集: {dataset_name}")

    # 确保数据集已注册
    from data.loader import DatasetLoader
    from data.dataset import MovieLens100K, MovieLens1M, Netflix, AmazonMI, CiaoDVD, Epinions, FilmTrust, MovieTweetings

    # 获取数据集类
    dataset_classes = {
        'movielens100k': MovieLens100K,
        'movielens1m': MovieLens1M,
        'netflix': Netflix,
        'amazonmi': AmazonMI,
        'ciaodvd': CiaoDVD,
        'epinions': Epinions,
        'filmtrust': FilmTrust,
        'movietweetings': MovieTweetings
    }

    # 仿照 example_data newDataset.py 的方式注册数据集
    # 直接注册，不检查是否已存在
    dataset_class = dataset_classes.get(dataset_name.lower(), MovieLens100K)
    DatasetLoader.register_dataset(dataset_name, dataset_class)
    print(f"✓ 已注册数据集: {dataset_name}")
    print(f"✓ 当前注册的数据集: {list(DatasetLoader.DATASET_REGISTRY.keys())}")

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

    data_manager = DataManager(data_config)

    # 打印详细的加载信息
    print(f"DEBUG: 尝试加载数据集 '{dataset_name}'")
    print(f"DEBUG: 数据文件路径: '{dataset_file}'")
    print(f"DEBUG: 当前注册的数据集: {list(DatasetLoader.DATASET_REGISTRY.keys())}")

    try:
        data_manager.load_dataset(dataset_name, dataset_file)
        data_manager.preprocess()
        print(f"✓ 数据集 {dataset_name} 加载和预处理成功")
        # 打印数据摘要
        data_manager.print_summary()
        return data_manager
    except Exception as e:
        print(f"加载 {dataset_name} 数据集失败: {e}")
        raise


def run_optimization(loss_type='hpl', enhanced=False, n_trials=40,
                    dataset_name='movielens100k',
                    dataset_file='dataset/20201202M100K_data_all_random.txt'):
    """统一的优化运行函数"""
    print(f"开始{loss_type.upper()}优化 ({'增强版' if enhanced else '基础版'})...")

    # 设置数据和结果管理器
    data_manager = setup_data_manager(dataset_name, dataset_file)
    result_manager = DatasetAwareResultManager(dataset_name, dataset_file)
    monitor = OptimizationMonitor(result_manager, f'{loss_type}_optimization')

    # 创建目标函数和参数空间
    if loss_type == 'hpl':
        objective = HPLObjectiveFunction(data_manager)
        if enhanced:
            space = ParameterSpaceFactory.create_enhanced_hpl_space()
        else:
            space = ParameterSpaceFactory.create_hpl_space()
        constraints = ConstraintFactory.create_hpl_constraints()
    else:  # l2
        objective = L2ObjectiveFunction(data_manager)
        space = ParameterSpaceFactory.create_l2_space()
        constraints = None

    # 创建优化器
    optimizer = OptimizerFactory.create_optimizer(objective, space, constraints, f'{loss_type}_optimization')

    # 运行优化
    start_time = time.time()
    best_trial = optimizer.optimize(
        n_trials=n_trials,
        no_improvement_rounds=15,
        batch_size=1
    )
    end_time = time.time()

    # 输出结果
    print(f"\n{loss_type.upper()}优化结果:")
    if best_trial:
        print(f"最佳配置: {best_trial.config}")
        print(f"最佳分数: {best_trial.score:.4f}")
        print(f"优化耗时: {end_time - start_time:.2f}秒")

        # 保存结果
        optimization_results = {
            'best_config': best_trial.config,
            'best_score': best_trial.score,
            'optimization_time': end_time - start_time,
            'optimizer_results': optimizer.get_results()
        }

        result_manager.save_experiment_results(
            results=optimization_results,
            experiment_type=f'{loss_type}_optimization',
            metadata={'algorithm': 'Latin Hypercube Sampling', 'enhanced': enhanced}
        )

    monitor.generate_final_report()
    return optimizer, best_trial, result_manager


def run_hpl_vs_l2_comparison(n_trials=40,
                            dataset_name='movielens100k',
                            dataset_file='dataset/20201202M100K_data_all_random.txt'):
    """HPL与L2损失函数的对比"""
    print("\n" + "="*60)
    print("HPL vs L2 对比实验")
    print("="*60)

    # 运行HPL优化
    print("\n1. 运行HPL优化...")
    hpl_optimizer, hpl_best, result_manager = run_optimization(
        'hpl', False, n_trials, dataset_name, dataset_file
    )

    # 运行L2优化
    print("\n2. 运行L2优化...")
    l2_optimizer, l2_best, _ = run_optimization(
        'l2', False, n_trials, dataset_name, dataset_file
    )

    # 对比分析
    print("\n" + "="*60)
    print("对比结果")
    print("="*60)

    hpl_rmse = hpl_best.score if hpl_best else float('inf')
    l2_rmse = l2_best.score if l2_best else float('inf')

    print(f"{'损失函数':<12} {'最佳RMSE':<12}")
    print("-" * 25)
    print(f"{'HPL':<12} {hpl_rmse:<12.4f}")
    print(f"{'L2':<12} {l2_rmse:<12.4f}")

    if hpl_rmse < l2_rmse:
        improvement = (l2_rmse - hpl_rmse) / l2_rmse * 100
        print(f"\n🎉 HPL表现更优! 改进了: {improvement:.2f}%")
        winner = 'HPL'
    elif l2_rmse < hpl_rmse:
        degradation = (hpl_rmse - l2_rmse) / l2_rmse * 100
        print(f"\n📊 L2表现更优，HPL差了: {degradation:.2f}%")
        winner = 'L2'
    else:
        print(f"\n🤝 两种损失函数表现相当")
        winner = 'Tie'

    # 保存对比结果
    comparison_results = {
        'winner': winner,
        'hpl_rmse': hpl_rmse,
        'l2_rmse': l2_rmse,
        'improvement_percentage': (l2_rmse - hpl_rmse) / l2_rmse * 100 if l2_rmse != 0 else 0,
        'hpl_config': hpl_best.config if hpl_best else None,
        'l2_config': l2_best.config if l2_best else None
    }

    result_manager.save_experiment_results(
        results=comparison_results,
        experiment_type='hpl_vs_l2_comparison',
        metadata={'comparison_type': 'HPL_vs_L2'}
    )

    return comparison_results


def guess_dataset_type(filename):
    """根据文件名猜测数据集类型"""
    filename = filename.lower()

    # 返回与数据集类对应的名称
    if 'm100k' in filename or 'movielens100k' in filename:
        return 'movielens100k'
    elif 'netflix' in filename:
        return 'netflix'
    elif 'm1m' in filename or 'movielens1m' in filename or 'moive1m' in filename:
        return 'movielens1m'
    elif 'amazon' in filename and ('musical' in filename or 'mi' in filename):
        return 'amazonmi'
    elif 'ciaodvd' in filename:
        return 'ciaodvd'
    elif 'epinions' in filename:
        return 'epinions'
    elif 'filmtrust' in filename or 'flimtrust' in filename:
        return 'filmtrust'
    elif 'movietweetings' in filename or 'moivetweetings' in filename:
        return 'movietweetings'
    else:
        # 如果无法识别，返回一个默认值
        return 'movielens100k'  # 未知类型


def check_file_format(file_path):
    """检查文件格式并显示前几行，尝试自动修复格式问题"""
    try:
        print(f"\n检查文件格式: {os.path.basename(file_path)}")
        # 文件的格式为  [user_id,movie_id,rating]
        print("1. 文件格式为每行: [user_id, movie_id, rating]")
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:10]]

        print("文件前10行内容:")
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line}")

        # 尝试检测文件格式
        format_type = "未知"
        format_issues = []

        # 检查是否为JSON格式
        if lines and lines[0].startswith('[') and lines[0].endswith(']'):
            format_type = "[user_id, item_id, rating]格式"
            # 检查是否有格式问题
            try:
                import ast
                for line in lines:
                    data = ast.literal_eval(line)
                    if len(data) != 3:
                        format_issues.append(f"数据项数量不是3: {line}")
            except:
                format_issues.append("JSON格式解析失败")

        # 检查是否为制表符分隔
        elif lines and '\t' in lines[0]:
            format_type = "制表符分隔格式"
            # 检查每行的列数是否一致
            cols = [len(line.split('\t')) for line in lines if line]
            if len(set(cols)) > 1:
                format_issues.append(f"列数不一致: {cols}")

        # 检查是否为逗号分隔
        elif lines and ',' in lines[0]:
            format_type = "逗号分隔格式"
            # 检查每行的列数是否一致
            cols = [len(line.split(',')) for line in lines if line]
            if len(set(cols)) > 1:
                format_issues.append(f"列数不一致: {cols}")

        print(f"推测文件格式: {format_type}")

        if format_issues:
            print("⚠️ 检测到潜在格式问题:")
            for issue in format_issues:
                print(f"  - {issue}")

            # 询问是否尝试修复
            fix = input("是否尝试自动修复格式问题? (y/n, 默认n): ").strip().lower()
            if fix == 'y':
                fixed_path = fix_file_format(file_path, format_type)
                if fixed_path:
                    print(f"✅ 已修复格式问题，新文件: {fixed_path}")
                    return fixed_path
                else:
                    print("❌ 修复失败")

        # 提示用户确认
        confirm = input("是否继续使用此文件? (y/n, 默认y): ").strip().lower()
        if confirm == 'n':
            return None
        return file_path
    except Exception as e:
        print(f"检查文件格式时出错: {e}")
        return None


def fix_file_format(file_path, format_type):
    """尝试修复文件格式问题"""
    try:
        base_name = os.path.basename(file_path)
        dir_name = os.path.dirname(file_path)
        fixed_path = os.path.join(dir_name, f"fixed_{base_name}")

        with open(file_path, 'r', encoding='utf-8') as f_in:
            with open(fixed_path, 'w', encoding='utf-8') as f_out:
                for line in f_in:
                    line = line.strip()
                    if not line:  # 跳过空行
                        continue

                    # 根据不同格式进行修复
                    if format_type == "[user_id, item_id, rating]格式":
                        try:
                            import ast
                            data = ast.literal_eval(line)
                            if len(data) >= 3:  # 确保至少有3个元素
                                # 只保留前3个元素
                                f_out.write(f"[{data[0]}, {data[1]}, {data[2]}]\n")
                        except:
                            # 如果解析失败，尝试简单的格式修复
                            line = line.replace('[', '').replace(']', '')
                            parts = [p.strip() for p in line.split(',')]
                            if len(parts) >= 3:
                                f_out.write(f"[{parts[0]}, {parts[1]}, {parts[2]}]\n")

                    elif format_type == "制表符分隔格式":
                        parts = [p.strip() for p in line.split('\t')]
                        if len(parts) >= 3:  # 确保至少有3个元素
                            # 只保留前3个元素
                            f_out.write(f"{parts[0]}\t{parts[1]}\t{parts[2]}\n")

                    elif format_type == "逗号分隔格式":
                        parts = [p.strip() for p in line.split(',')]
                        if len(parts) >= 3:  # 确保至少有3个元素
                            # 只保留前3个元素
                            f_out.write(f"{parts[0]},{parts[1]},{parts[2]}\n")

                    else:  # 未知格式，尝试通用修复
                        # 移除所有非数字、逗号、点、方括号和空格的字符
                        import re
                        cleaned = re.sub(r'[^\d\.,\[\]\s]', '', line)
                        # 尝试提取3个数字
                        numbers = re.findall(r'\d+(?:\.\d+)?', cleaned)
                        if len(numbers) >= 3:
                            f_out.write(f"[{numbers[0]}, {numbers[1]}, {numbers[2]}]\n")

        # 检查修复后的文件是否有内容
        if os.path.getsize(fixed_path) > 0:
            print(f"✅ 修复后的文件保存至: {fixed_path}")
            return fixed_path
        else:
            os.remove(fixed_path)  # 删除空文件
            return None
    except Exception as e:
        print(f"修复文件格式时出错: {e}")
        return None


def scan_available_datasets(dataset_dir):
    """扫描并列出所有可用数据集"""
    print(f"\n🔍 扫描数据集目录: {dataset_dir}")

    # 检查目录是否存在
    if not os.path.exists(dataset_dir):
        print(f"⚠️ 数据集目录不存在: {dataset_dir}")
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"✅ 已创建数据集目录: {dataset_dir}")
        except Exception as e:
            print(f"❌ 无法创建数据集目录: {e}")
            return []

    # 获取所有txt文件
    pattern = os.path.join(dataset_dir, '*.txt')
    dataset_files = glob.glob(pattern)

    if not dataset_files:
        print(f"❌ 在 {dataset_dir} 中未找到任何 .txt 数据集文件")
        print("请将数据集文件放入该目录后重新运行程序")
        return []

    print(f"✅ 找到 {len(dataset_files)} 个数据集文件")
    return dataset_files


def create_dataset_mapping(dataset_files):
    """创建文件名到数据集类型的映射"""
    dataset_map = {}

    print("\n📋 可用数据集列表:")
    print("-" * 80)
    print(f"{'序号':<4} {'文件名':<40} {'推测类型':<20} {'文件大小'}")
    print("-" * 80)

    for i, file_path in enumerate(dataset_files, 1):
        filename = os.path.basename(file_path)
        dataset_type = guess_dataset_type(filename)

        # 获取文件大小
        try:
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:  # MB
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
            elif file_size > 1024:  # KB
                size_str = f"{file_size / 1024:.1f}KB"
            else:
                size_str = f"{file_size}B"
        except:
            size_str = "未知"

        dataset_map[str(i)] = {
            'dataset_type': dataset_type,
            'file_path': file_path,
            'filename': filename,
            'file_size': size_str
        }

        print(f"{i:<4} {filename:<40} {dataset_type:<20} {size_str}")

    print("-" * 80)
    return dataset_map


def get_user_dataset_selection(dataset_map):
    """获取用户的数据集选择"""
    print("\n📝 数据集选择:")
    print("- 输入序号选择单个数据集 (例如: 1)")
    print("- 输入多个序号选择多个数据集 (例如: 1,3,5)")
    print("- 输入 'all' 选择所有数据集")

    while True:
        choice = input("\n请选择数据集: ").strip()

        if not choice:
            print("❌ 请输入有效选择")
            continue

        selected_datasets = []

        if choice.lower() == 'all':
            selected_datasets = list(dataset_map.values())
            print(f"✅ 已选择所有 {len(selected_datasets)} 个数据集")
            break

        # 解析用户输入
        try:
            choices = [c.strip() for c in choice.split(',')]
            for c in choices:
                if c in dataset_map:
                    selected_datasets.append(dataset_map[c])
                else:
                    print(f"❌ 无效选择: {c}")
                    raise ValueError("无效选择")

            if selected_datasets:
                print(f"✅ 已选择 {len(selected_datasets)} 个数据集:")
                for ds in selected_datasets:
                    print(f"   - {ds['filename']} ({ds['dataset_type']})")
                break
            else:
                print("❌ 未选择任何数据集")

        except ValueError:
            print("❌ 请输入有效的序号")
            continue

    return selected_datasets


def batch_check_file_formats(selected_datasets):
    """批量检查和修复文件格式"""
    print("\n🔧 批量检查文件格式...")

    processed_datasets = []
    skipped_datasets = []

    for i, dataset_info in enumerate(selected_datasets, 1):
        print(f"\n处理 {i}/{len(selected_datasets)}: {dataset_info['filename']}")

        try:
            # 检查文件格式
            fixed_file = check_file_format(dataset_info['file_path'])

            if fixed_file:
                # 更新文件路径（如果文件被修复）
                if fixed_file != dataset_info['file_path']:
                    print(f"✅ 使用修复后的文件: {os.path.basename(fixed_file)}")
                    dataset_info['file_path'] = fixed_file
                    dataset_info['format_fixed'] = True
                else:
                    dataset_info['format_fixed'] = False

                processed_datasets.append(dataset_info)
                print(f"✅ {dataset_info['filename']} 格式检查通过")
            else:
                print(f"❌ {dataset_info['filename']} 格式检查失败，跳过")
                skipped_datasets.append(dataset_info)

        except Exception as e:
            print(f"❌ 处理 {dataset_info['filename']} 时出错: {e}")
            skipped_datasets.append(dataset_info)

    print(f"\n📊 格式检查结果:")
    print(f"   ✅ 成功: {len(processed_datasets)} 个")
    print(f"   ❌ 跳过: {len(skipped_datasets)} 个")

    if skipped_datasets:
        print("\n⚠️ 跳过的数据集:")
        for ds in skipped_datasets:
            print(f"   - {ds['filename']}")

    return processed_datasets


def get_experiment_type_selection():
    """获取用户的实验类型选择"""
    print("\n🧪 实验类型选择:")
    print("1. HPL基础优化 (40次试验)")
    print("2. HPL增强优化 (150次试验，扩展参数空间)")
    print("3. HPL vs L2 对比实验 (分别优化后对比)")

    while True:
        choice = input("\n选择实验类型 (1-3, 默认1): ").strip()

        if not choice:
            choice = '1'

        if choice in ['1', '2', '3']:
            experiment_types = {
                '1': ('hpl_basic', 'HPL基础优化'),
                '2': ('hpl_enhanced', 'HPL增强优化'),
                '3': ('hpl_vs_l2', 'HPL vs L2 对比')
            }

            exp_type, exp_name = experiment_types[choice]
            print(f"✅ 已选择: {exp_name}")
            return exp_type
        else:
            print("❌ 请输入有效选择 (1-3)")


def execute_single_dataset_experiment(dataset_info, experiment_type):
    """执行单个数据集的实验"""
    dataset_name = dataset_info['dataset_type']
    dataset_file = dataset_info['file_path']
    filename = dataset_info['filename']

    print(f"\n🚀 开始实验: {filename}")
    print(f"   数据集类型: {dataset_name}")
    print(f"   实验类型: {experiment_type}")

    start_time = time.time()

    try:
        # 动态注册数据集
        if not register_dataset(dataset_name):
            raise Exception(f"数据集注册失败: {dataset_name}")

        # 设置数据管理器
        data_manager = setup_data_manager(dataset_name, dataset_file)

        # 根据实验类型执行相应的实验
        if experiment_type == 'hpl_basic':
            optimizer, best_trial, result_manager = run_optimization(
                'hpl', False, 40, dataset_name, dataset_file
            )
            result = {
                'type': 'hpl_basic',
                'best_trial': best_trial,
                'best_score': best_trial.score if best_trial else None,
                'best_config': best_trial.config if best_trial else None
            }

        elif experiment_type == 'hpl_enhanced':
            optimizer, best_trial, result_manager = run_optimization(
                'hpl', True, 150, dataset_name, dataset_file
            )
            result = {
                'type': 'hpl_enhanced',
                'best_trial': best_trial,
                'best_score': best_trial.score if best_trial else None,
                'best_config': best_trial.config if best_trial else None
            }

        elif experiment_type == 'hpl_vs_l2':
            comparison_result = run_hpl_vs_l2_comparison(40, dataset_name, dataset_file)
            result = {
                'type': 'hpl_vs_l2',
                'comparison_result': comparison_result,
                'winner': comparison_result.get('winner'),
                'hpl_score': comparison_result.get('hpl_rmse'),
                'l2_score': comparison_result.get('l2_rmse'),
                'improvement': comparison_result.get('improvement_percentage')
            }

        else:
            raise ValueError(f"未知实验类型: {experiment_type}")

        end_time = time.time()
        duration = end_time - start_time

        result.update({
            'status': 'success',
            'duration': duration,
            'dataset_info': dataset_info
        })

        print(f"✅ {filename} 实验完成 ({duration:.1f}秒)")
        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        print(f"❌ {filename} 实验失败: {e}")
        traceback.print_exc()

        return {
            'status': 'failed',
            'error': str(e),
            'duration': duration,
            'dataset_info': dataset_info
        }


def generate_multi_dataset_summary(all_results, experiment_type):
    """生成多数据集实验汇总报告"""
    print("\n" + "="*80)
    print("多数据集实验汇总报告")
    print("="*80)

    # 统计信息
    total_datasets = len(all_results)
    successful_datasets = len([r for r in all_results.values() if r['status'] == 'success'])
    failed_datasets = total_datasets - successful_datasets
    total_duration = sum(r['duration'] for r in all_results.values())

    print(f"\n📊 实验统计:")
    print(f"   总数据集数: {total_datasets}")
    print(f"   成功数量: {successful_datasets}")
    print(f"   失败数量: {failed_datasets}")
    print(f"   总耗时: {total_duration:.1f}秒 ({total_duration/60:.1f}分钟)")
    print(f"   平均耗时: {total_duration/total_datasets:.1f}秒/数据集")

    # 详细结果
    print(f"\n📋 详细结果:")
    print("-" * 80)

    if experiment_type in ['hpl_basic', 'hpl_enhanced']:
        print(f"{'数据集':<30} {'状态':<8} {'最佳RMSE':<12} {'耗时(秒)':<10}")
        print("-" * 80)

        successful_results = []
        for dataset_name, result in all_results.items():
            status = "✅成功" if result['status'] == 'success' else "❌失败"

            if result['status'] == 'success':
                rmse = result.get('best_score', 'N/A')
                rmse_str = f"{rmse:.4f}" if rmse and rmse != 'N/A' else 'N/A'
                successful_results.append((dataset_name, rmse))
            else:
                rmse_str = f"错误: {result.get('error', 'Unknown')[:20]}..."

            duration = result.get('duration', 0)
            print(f"{dataset_name:<30} {status:<8} {rmse_str:<12} {duration:<10.1f}")

        # 排序显示最佳结果
        if successful_results:
            print(f"\n🏆 最佳表现排序:")
            successful_results.sort(key=lambda x: x[1])
            for i, (dataset_name, rmse) in enumerate(successful_results, 1):
                print(f"   {i}. {dataset_name}: {rmse:.4f}")

    elif experiment_type == 'hpl_vs_l2':
        print(f"{'数据集':<25} {'状态':<8} {'胜出方':<8} {'HPL RMSE':<12} {'L2 RMSE':<12} {'改进%':<10}")
        print("-" * 80)

        hpl_wins = 0
        l2_wins = 0
        ties = 0

        for dataset_name, result in all_results.items():
            status = "✅成功" if result['status'] == 'success' else "❌失败"

            if result['status'] == 'success':
                winner = result.get('winner', 'N/A')
                hpl_score = result.get('hpl_score', 'N/A')
                l2_score = result.get('l2_score', 'N/A')
                improvement = result.get('improvement', 0)

                hpl_str = f"{hpl_score:.4f}" if hpl_score != 'N/A' else 'N/A'
                l2_str = f"{l2_score:.4f}" if l2_score != 'N/A' else 'N/A'
                imp_str = f"{improvement:+.2f}" if improvement else '0.00'

                if winner == 'HPL':
                    hpl_wins += 1
                elif winner == 'L2':
                    l2_wins += 1
                else:
                    ties += 1
            else:
                winner = "失败"
                hpl_str = l2_str = imp_str = "N/A"

            print(f"{dataset_name:<25} {status:<8} {winner:<8} {hpl_str:<12} {l2_str:<12} {imp_str:<10}")

        # 对比总结
        if successful_datasets > 0:
            print(f"\n🏆 对比总结:")
            print(f"   HPL胜出: {hpl_wins} 次")
            print(f"   L2胜出: {l2_wins} 次")
            print(f"   平局: {ties} 次")

            if hpl_wins > l2_wins:
                print(f"   🎉 HPL整体表现更优!")
            elif l2_wins > hpl_wins:
                print(f"   📊 L2整体表现更优!")
            else:
                print(f"   🤝 两种损失函数表现相当!")

    # 保存汇总报告
    summary_data = {
        'experiment_info': {
            'experiment_type': experiment_type,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_datasets': total_datasets,
            'successful_datasets': successful_datasets,
            'failed_datasets': failed_datasets,
            'total_duration': total_duration
        },
        'detailed_results': all_results
    }

    # 保存到 results 目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = Path(os.path.join(current_script_dir, 'results'))
    results_dir.mkdir(exist_ok=True)

    summary_filename = f"multi_dataset_summary_{experiment_type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = results_dir / summary_filename

    try:
        # 转换numpy类型
        clean_data = convert_numpy_types(summary_data)

        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(clean_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 汇总报告已保存: {summary_path}")
        print(f"📍 完整路径: {summary_path.absolute()}")
    except Exception as e:
        print(f"⚠️ 保存汇总报告失败: {e}")

    return summary_data


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    else:
        return obj


def run_multiple_datasets(processed_datasets, experiment_type):
    """重构后的多数据集批量实验函数"""
    print("\n" + "="*80)
    print("🚀 开始批量执行实验")
    print("="*80)

    print(f"将对 {len(processed_datasets)} 个数据集执行 {experiment_type} 实验")

    all_results = {}

    for i, dataset_info in enumerate(processed_datasets, 1):
        print(f"\n{'='*60}")
        print(f"🔄 进度: {i}/{len(processed_datasets)}")
        print(f"{'='*60}")

        # 执行单个数据集实验
        result = execute_single_dataset_experiment(dataset_info, experiment_type)

        # 存储结果
        dataset_key = f"{dataset_info['dataset_type']}_{os.path.basename(dataset_info['file_path'])}"
        all_results[dataset_key] = result

        # 显示进度
        if result['status'] == 'success':
            print(f"✅ 完成 {i}/{len(processed_datasets)}: {dataset_info['filename']}")
        else:
            print(f"❌ 失败 {i}/{len(processed_datasets)}: {dataset_info['filename']}")

    # 生成汇总报告
    print("\n📊 生成汇总报告...")
    summary_data = generate_multi_dataset_summary(all_results, experiment_type)

    print(f"\n🎉 多数据集实验完成!")
    print(f"   成功: {len([r for r in all_results.values() if r['status'] == 'success'])}/{len(all_results)}")

    return True


def register_dataset(dataset_name):
    """注册指定的数据集"""
    from data.loader import DatasetLoader
    from data.dataset import MovieLens100K, MovieLens1M, Netflix, AmazonMI, CiaoDVD, Epinions, FilmTrust, MovieTweetings

    # 检查数据集是否已注册
    if dataset_name.lower() in [k.lower() for k in DatasetLoader.DATASET_REGISTRY.keys()]:
        return True

    # 根据数据集类型选择合适的类
    if 'filmtrust' in dataset_name.lower() or 'flimtrust' in dataset_name.lower():
        DatasetLoader.register_dataset(dataset_name, FilmTrust)
    elif 'netflix' in dataset_name.lower():
        DatasetLoader.register_dataset(dataset_name, Netflix)
    elif 'amazon' in dataset_name.lower():
        DatasetLoader.register_dataset(dataset_name, AmazonMI)
    elif 'movielens1m' in dataset_name.lower() or 'ml1m' in dataset_name.lower():
        DatasetLoader.register_dataset(dataset_name, MovieLens1M)
    else:
        DatasetLoader.register_dataset(dataset_name, MovieLens100K)

    return True


def main():
    """重构后的主函数 - 统一批量处理模式"""
    try:
        print("HPL损失函数优化实验 - 批量处理模式")
        print("="*60)

        # 显示当前脚本路径和结果保存路径
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        results_save_dir = os.path.join(current_script_dir, 'results')
        print(f"📂 当前脚本目录: {current_script_dir}")
        print(f"💾 结果保存目录: {results_save_dir}")

        # 第一步：扫描并获取可用数据集
        print("\n📂 第一步: 数据集扫描与选择")

        # 修改：获取数据集目录路径（保持原逻辑，从项目根目录获取）
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(base_dir, 'dataset')
        print(f"🔍 数据集搜索目录: {dataset_dir}")

        # 扫描可用数据集
        dataset_files = scan_available_datasets(dataset_dir)
        if not dataset_files:
            print("❌ 没有找到可用的数据集文件")
            return False

        # 创建数据集映射
        dataset_map = create_dataset_mapping(dataset_files)

        # 用户选择数据集
        selected_datasets = get_user_dataset_selection(dataset_map)
        if not selected_datasets:
            print("❌ 没有选择任何数据集")
            return False

        # 第二步：批量检查文件格式
        print("\n🔧 第二步: 文件格式检查与修复")
        processed_datasets = batch_check_file_formats(selected_datasets)

        if not processed_datasets:
            print("❌ 没有可用的数据集，退出实验")
            return False

        # 第三步：选择实验类型
        print("\n🧪 第三步: 实验类型选择")
        experiment_type = get_experiment_type_selection()

        # 第四步：执行批量实验
        print("\n🏃 第四步: 批量执行实验")
        success = run_multiple_datasets(processed_datasets, experiment_type)

        if success:
            print("\n✅ 所有实验已完成!")
            send_sms_notification(f"最优参数实验已完成，数据集: {', '.join([d.name for d in processed_datasets])}")
        else:
            print("\n❌ 实验执行过程中出现问题")
            send_sms_notification("最优参数实验执行过程中出现问题")

        return success

    except Exception as e:
        print(f"主程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        send_sms_notification(f"最优参数实验失败: {str(e)[:50]}...")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

