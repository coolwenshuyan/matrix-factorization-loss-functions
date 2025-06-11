#!/usr/bin/env python3
# 修复版本：Optimal parameter experiment\Optimal_parameter_experiment.py
"""
HPL损失函数专用优化实验 - 修复版本

修复内容：
1. 修复 best_trial 对象属性访问问题
2. 改进结果保存逻辑
3. 增强错误处理
"""
from pathlib import Path
import sys
import os
import numpy as np
import time
import json
import glob
import traceback
from pathlib import Path
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
    from src.evaluation.metrics import (
        RMSE, MAE, MSE, R2Score, HitRate, Precision, Recall,
        MAP, NDCG, MRR, CatalogCoverage, UserCoverage,
        Diversity, Novelty, MetricFactory
    )
    print("成功导入数据和模型模块")
except ImportError as e:
    print(f"导入数据/模型模块失败: {e}")
    sys.exit(1)

# 在文件顶部导入短信通知模块
try:
    from utils.sms_notification import send_sms_notification
except ImportError:
    print("警告：无法导入短信通知模块")
    def send_sms_notification(message):
        print(f"通知: {message}")


class SafeTrial:
    """安全的试验结果包装器"""
    def __init__(self, config, score):
        self.config = config
        self.score = score

    def to_dict(self):
        return {'config': self.config, 'score': self.score}


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
        }
    }

    def __init__(self, dataset_name='movielens100k', dataset_file=None, base_dir=None):
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file

        if base_dir is None:
            current_script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.join(current_script_dir, 'results')

        self.base_dir = Path(base_dir)

        self.dataset_info = self.DATASET_CONFIGS.get(dataset_name, {
            'name': dataset_name.title(),
            'size': 'Unknown',
            'version': 'Unknown',
            'short_name': dataset_name.lower()
        })

        self.dataset_dir = self.base_dir / self.dataset_info['short_name']
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self._create_subdirectories()

        print(f"📁 结果保存目录: {self.dataset_dir}")

    def _create_subdirectories(self):
        """创建必要的子目录"""
        subdirs = ['hpl_optimization', 'models', 'reports', 'logs']
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
            'model_evaluation': 'reports',
            'training_log': 'logs'
        }
        subdir = subdir_map.get(experiment_type, 'hpl_optimization')
        filename = self.generate_filename(experiment_type, file_format)
        return self.dataset_dir / subdir / filename

    def save_experiment_results(self, results, experiment_type, metadata=None):
        """保存实验结果，包含完整的数据集信息和所有配置参数"""
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
                    'experiment_id': f"{experiment_type}_{time.strftime('%Y%m%d_%H%M%S')}",
                    'python_version': sys.version,
                    'numpy_version': np.__version__,
                    'random_seed': 42
                },
                'results': results,
                'reproduction_info': {
                    'command': f"python {os.path.basename(__file__)}",
                    'working_directory': os.getcwd(),
                    'environment': {
                        'PYTHONPATH': os.environ.get('PYTHONPATH', ''),
                        'project_root': project_root
                    }
                }
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
            traceback.print_exc()
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
        # 确保trial是SafeTrial对象
        if not isinstance(trial, SafeTrial):
            if hasattr(trial, 'config') and hasattr(trial, 'score'):
                trial = SafeTrial(trial.config, trial.score)
            elif isinstance(trial, dict):
                trial = SafeTrial(trial.get('config', {}), trial.get('score', float('inf')))
            else:
                print(f"警告：无法处理的trial类型: {type(trial)}")
                return

        self.trial_history.append(trial)

        if not self.best_score_history:
            self.best_score_history.append(trial.score)
        else:
            self.best_score_history.append(
                min(self.best_score_history[-1], trial.score)
            )

        self.log_file.write(f"试验 {len(self.trial_history)}: 分数={trial.score:.4f}, 配置={trial.config}\n")
        self.log_file.flush()

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
            'best_trial': min(self.trial_history, key=lambda x: x.score).to_dict() if self.trial_history else None,
            'score_history': self.best_score_history,
            'convergence_analysis': self._analyze_convergence(),
            'all_trials': [trial.to_dict() for trial in self.trial_history]
        }

        # 只有当有试验数据时才保存监控报告
        if len(self.trial_history) > 0:
            report_path = self.result_manager.save_experiment_results(
                results=report_data,
                experiment_type=f"{self.experiment_type}_monitor_report",  # 改名避免覆盖
                metadata={'monitor_type': 'OptimizationMonitor'}
            )
        else:
            print("⚠️ 监控器没有记录到任何试验，跳过监控报告生成")
            report_path = None

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


class HPLObjectiveFunction:
    """HPL专用目标函数"""

    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        self.rmse_metric = RMSE()

        print(f"目标函数初始化:")
        print(f"  训练集: {len(self.train_data)} 条")
        print(f"  验证集: {len(self.val_data)} 条")
        print(f"  测试集: {len(self.test_data)} 条")

    def _create_model(self, config):
        """创建模型"""
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
        return model

    def _train_and_evaluate(self, model, n_epochs=100):
    # def _train_and_evaluate(self, model, n_epochs=10): # 快速
        """训练和评估模型"""
        # 从全局获取模型早停参数，默认为12
        model_patience = getattr(self, 'model_early_stopping', 12)

        model.fit(
            train_data=self.train_data,
            val_data=self.val_data,
            n_epochs=n_epochs,
            verbose=0,
            early_stopping_patience=model_patience
        )

        val_predictions = model.predict(
            self.val_data[:, 0].astype(int),
            self.val_data[:, 1].astype(int)
        )

        if self.data_manager.global_mean is not None:
            val_predictions += self.data_manager.global_mean
            val_targets = self.val_data[:, 2] + self.data_manager.global_mean
        else:
            val_targets = self.val_data[:, 2]

        rmse = self.rmse_metric.calculate(val_targets, val_predictions)
        return rmse

    def __call__(self, config):
        try:
            model = self._create_model(config)
            rmse = self._train_and_evaluate(model)

            print(f"HPL配置: 学习率={config['learning_rate']:.4f}, "
                  f"因子数={config['latent_factors']}, "
                  f"正则化={config['lambda_reg']:.6f}, "
                  f"δ1={config['delta1']:.3f}, δ2={config['delta2']:.3f}, "
                  f"l_max={config.get('l_max', 4.0):.2f}, "
                  f"c_sig={config.get('c_sigmoid', 1.0):.2f}, "
                  f"RMSE={rmse:.4f}")

            return rmse

        except Exception as e:
            print(f"HPL配置评估失败 {config}: {e}")
            return 10.0


class BestParameterEvaluator:
    """最佳参数性能评估器"""

    def __init__(self, data_manager, best_config):
        self.data_manager = data_manager
        self.best_config = best_config
        self.n_users = data_manager.get_statistics()['n_users']
        self.n_items = data_manager.get_statistics()['n_items']
        self.train_data, self.val_data, self.test_data = data_manager.get_splits()

        print(f"🎯 最佳参数评估器初始化:")
        print(f"   最佳配置: {best_config}")
        print(f"   用户数: {self.n_users}, 物品数: {self.n_items}")
        print(f"   测试集大小: {len(self.test_data)}")

    def create_best_model(self):
        """使用最佳配置创建模型"""
        loss_function = HybridPiecewiseLoss(
            delta1=self.best_config['delta1'],
            delta2=self.best_config['delta2'],
            l_max=self.best_config.get('l_max', 4.0),
            c_sigmoid=self.best_config.get('c_sigmoid', 1.0)
        )

        regularizer = L2Regularizer(lambda_reg=self.best_config['lambda_reg'])

        model = MatrixFactorizationSGD(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.best_config['latent_factors'],
            learning_rate=self.best_config['learning_rate'],
            regularizer=regularizer,
            loss_function=loss_function,
            use_bias=True,
            global_mean=self.data_manager.global_mean or 0.0
        )

        # 打印所有模型配置参数
        print(f"\n📋 模型配置参数:")
        print(f"   学习率: {self.best_config['learning_rate']:.6f}")
        print(f"   潜在因子数: {self.best_config['latent_factors']}")
        print(f"   正则化系数: {self.best_config['lambda_reg']:.6f}")
        print(f"   HPL参数 δ1: {self.best_config['delta1']:.4f}")
        print(f"   HPL参数 δ2: {self.best_config['delta2']:.4f}")
        print(f"   HPL参数 l_max: {self.best_config.get('l_max', 4.0):.2f}")
        print(f"   HPL参数 c_sigmoid: {self.best_config.get('c_sigmoid', 1.0):.2f}")
        print(f"   使用偏置: True")
        print(f"   全局均值: {self.data_manager.global_mean or 0.0:.4f}")

        initializer = NormalInitializer(mean=0.0, std=0.01)
        model.initialize_parameters(initializer)
        return model

    def train_best_model(self, n_epochs=100):
        """使用最佳参数训练模型"""
        print(f"\n🚀 开始使用最佳参数训练模型 (训练轮数: {n_epochs})")

        model = self.create_best_model()

        start_time = time.time()
        model.fit(
            train_data=self.train_data,
            val_data=self.val_data,
            n_epochs=n_epochs,
            verbose=1,
            early_stopping_patience=15  # 使用早停避免过拟合
        )
        train_time = time.time() - start_time

        print(f"✅ 模型训练完成，耗时: {train_time:.2f}秒")
        return model, train_time

    def evaluate_all_metrics(self, model):
        """计算所有性能指标"""
        print(f"\n📊 开始计算所有性能指标...")

        results = {}

        # 获取测试数据
        test_users = self.test_data[:, 0].astype(int)
        test_items = self.test_data[:, 1].astype(int)
        test_ratings = self.test_data[:, 2]

        # 预测评分
        print("   正在预测测试集评分...")
        predictions = model.predict(test_users, test_items)

        # 还原到原始尺度
        if hasattr(self.data_manager, 'global_mean') and self.data_manager.global_mean is not None:
            predictions += self.data_manager.global_mean
            test_ratings_adjusted = test_ratings + self.data_manager.global_mean
        else:
            test_ratings_adjusted = test_ratings

        # 1. 评分预测指标
        print("   计算评分预测指标...")
        rating_metrics = ['RMSE', 'MAE', 'MSE', 'R2']
        for metric_name in rating_metrics:
            try:
                metric = MetricFactory.create(metric_name.lower())
                results[metric_name] = metric.calculate(test_ratings_adjusted, predictions)
                print(f"     {metric_name}: {results[metric_name]:.4f}")
            except Exception as e:
                print(f"     ⚠️ 计算{metric_name}失败: {e}")
                results[metric_name] = None

        # 2. 排序质量指标
        print("   计算排序质量指标...")
        ranking_metrics = ['HitRate', 'Precision', 'Recall', 'MAP', 'NDCG']
        k_values = [5, 10, 20]

        for k in k_values:
            print(f"     计算 @{k} 指标...")
            for metric_name in ranking_metrics:
                try:
                    metric = MetricFactory.create(f"{metric_name}@{k}")
                    # 为排序指标准备数据
                    user_items, recommendations = self._prepare_ranking_data(model, k)

                    if user_items and recommendations:
                        score = metric.calculate(
                            None, None,  # 不使用矩阵格式
                            user_items=user_items,
                            recommendations=recommendations
                        )
                        results[f'{metric_name}@{k}'] = score
                        print(f"       {metric_name}@{k}: {score:.4f}")
                    else:
                        results[f'{metric_name}@{k}'] = None
                        print(f"       {metric_name}@{k}: 数据准备失败")
                except Exception as e:
                    print(f"       ⚠️ 计算{metric_name}@{k}失败: {e}")
                    results[f'{metric_name}@{k}'] = None

        # 3. MRR指标
        print("   计算MRR指标...")
        try:
            mrr_metric = MetricFactory.create('mrr')
            user_items, recommendations = self._prepare_ranking_data(model, 50)  # 使用更大的K值计算MRR
            if user_items and recommendations:
                results['MRR'] = mrr_metric.calculate(
                    None, None,
                    user_items=user_items,
                    recommendations=recommendations
                )
                print(f"     MRR: {results['MRR']:.4f}")
            else:
                results['MRR'] = None
        except Exception as e:
            print(f"     ⚠️ 计算MRR失败: {e}")
            results['MRR'] = None

        # 4. 覆盖度指标
        print("   计算覆盖度指标...")
        try:
            user_items, recommendations = self._prepare_ranking_data(model, 10)
            if recommendations:
                # 目录覆盖率
                catalog_coverage = MetricFactory.create('catalog_coverage')
                results['CatalogCoverage'] = catalog_coverage.calculate(
                    None, None,
                    recommendations=recommendations,
                    n_items=self.n_items
                )
                print(f"     目录覆盖率: {results['CatalogCoverage']:.4f}")

                # 用户覆盖率
                user_coverage = MetricFactory.create('user_coverage')
                results['UserCoverage'] = user_coverage.calculate(
                    None, None,
                    recommendations=recommendations,
                    n_users=self.n_users
                )
                print(f"     用户覆盖率: {results['UserCoverage']:.4f}")
            else:
                results['CatalogCoverage'] = None
                results['UserCoverage'] = None
        except Exception as e:
            print(f"     ⚠️ 计算覆盖度指标失败: {e}")
            results['CatalogCoverage'] = None
            results['UserCoverage'] = None

        print(f"✅ 性能指标计算完成")
        return results

    def _prepare_ranking_data(self, model, k=10):
        """为排序指标准备数据"""
        try:
            # 获取测试集中的用户-物品交互
            user_items = {}
            for row in self.test_data:
                user_id = int(row[0])
                item_id = int(row[1])
                if user_id not in user_items:
                    user_items[user_id] = []
                user_items[user_id].append(item_id)

            # 为每个用户生成Top-K推荐
            recommendations = {}
            unique_users = list(user_items.keys())[:100]  # 限制用户数量以提高计算效率

            for user_id in unique_users:
                try:
                    # 获取该用户的所有物品预测分数
                    all_items = list(range(self.n_items))
                    user_predictions = model.predict([user_id] * len(all_items), all_items)

                    # 排序并获取Top-K
                    item_scores = list(zip(all_items, user_predictions))
                    item_scores.sort(key=lambda x: x[1], reverse=True)
                    recommendations[user_id] = [item for item, score in item_scores[:k]]

                except Exception as e:
                    print(f"       ⚠️ 为用户{user_id}生成推荐失败: {e}")
                    continue

            # 只保留有推荐的用户
            filtered_user_items = {uid: items for uid, items in user_items.items() if uid in recommendations}

            return filtered_user_items, recommendations

        except Exception as e:
            print(f"     ⚠️ 准备排序数据失败: {e}")
            return {}, {}


class ParameterSpaceFactory:
    """参数空间工厂类"""

    @staticmethod
    def create_enhanced_hpl_space():
        """创建增强的HPL参数空间"""
        space = ParameterSpace()

        space.add_continuous('learning_rate', 0.005, 0.15, scale='log')
        space.add_discrete('latent_factors', 5, 120, step=5)  # 完整测试
        # space.add_discrete('latent_factors', 5, 15, step=5)  # 快速
        space.add_continuous('lambda_reg', 0.0001, 0.05, scale='log')

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
        constraints = ConstraintManager()
        constraints.add_relation('delta1', 'delta2', '<')
        return constraints


class OptimizerFactory:
    """优化器工厂类"""

    @staticmethod
    def create_optimizer(objective, space, constraints=None, name='optimization'):
        """创建优化器"""
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
        return optimizer


def setup_data_manager(dataset_name='movielens100k',
                      dataset_file='dataset/20201202M100K_data_all_random.txt'):
    """设置数据管理器"""
    print(f"准备数据集: {dataset_name}")

    from data.loader import DatasetLoader
    from data.dataset import MovieLens100K, MovieLens1M, Netflix, AmazonMI, CiaoDVD, Epinions, FilmTrust, MovieTweetings

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

    dataset_class = dataset_classes.get(dataset_name.lower(), MovieLens100K)
    DatasetLoader.register_dataset(dataset_name, dataset_class)
    print(f"✓ 已注册数据集: {dataset_name}")

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

    print(f"DEBUG: 尝试加载数据集 '{dataset_name}'")
    print(f"DEBUG: 数据文件路径: '{dataset_file}'")

    try:
        data_manager.load_dataset(dataset_name, dataset_file)
        data_manager.preprocess()
        print(f"✓ 数据集 {dataset_name} 加载和预处理成功")
        data_manager.print_summary()
        return data_manager
    except Exception as e:
        print(f"加载 {dataset_name} 数据集失败: {e}")
        raise


def safe_get_trials(optimizer):
    """安全地获取试验数据"""
    try:
        if hasattr(optimizer, 'tracker'):
            tracker = optimizer.tracker

            if hasattr(tracker, 'trials'):
                trials = tracker.trials
                return [SafeTrial(t.config if hasattr(t, 'config') else t.get('config', {}),
                                t.score if hasattr(t, 'score') else t.get('score', float('inf')))
                       for t in trials]

            if hasattr(tracker, 'get_trials'):
                trials = tracker.get_trials()
                return [SafeTrial(t.config if hasattr(t, 'config') else t.get('config', {}),
                                t.score if hasattr(t, 'score') else t.get('score', float('inf')))
                       for t in trials]

            if hasattr(tracker, 'get_all_trials'):
                trials = tracker.get_all_trials()
                return [SafeTrial(t.config if hasattr(t, 'config') else t.get('config', {}),
                                t.score if hasattr(t, 'score') else t.get('score', float('inf')))
                       for t in trials]

            if hasattr(tracker, '_trials'):
                trials = tracker._trials
                return [SafeTrial(t.config if hasattr(t, 'config') else t.get('config', {}),
                                t.score if hasattr(t, 'score') else t.get('score', float('inf')))
                       for t in trials]

            if hasattr(tracker, 'results'):
                results = tracker.results
                if isinstance(results, dict):
                    trials = []
                    for trial_id, trial_data in results.items():
                        if 'config' in trial_data and 'score' in trial_data:
                            trials.append(SafeTrial(trial_data['config'], trial_data['score']))
                    return trials

        print("⚠️ 无法获取trials数据，将使用空列表")
        return []

    except Exception as e:
        print(f"⚠️ 获取trials数据时出错: {e}")
        return []


def safe_extract_trial_info(best_trial):
    """安全地提取试验信息"""
    try:
        if best_trial is None:
            return None, None

        # 如果是SafeTrial对象
        if isinstance(best_trial, SafeTrial):
            return best_trial.config, best_trial.score

        # 如果有config和score属性
        if hasattr(best_trial, 'config') and hasattr(best_trial, 'score'):
            return best_trial.config, best_trial.score

        # 如果是字典
        if isinstance(best_trial, dict):
            config = best_trial.get('config', {})
            score = best_trial.get('score', float('inf'))
            return config, score

        # 其他情况
        print(f"警告：无法识别的best_trial类型: {type(best_trial)}")
        print(f"best_trial内容: {best_trial}")
        return {}, float('inf')

    except Exception as e:
        print(f"提取试验信息时出错: {e}")
        return {}, float('inf')


# def run_enhanced_hpl_optimization(n_trials=150, dataset_name='movielens100k', dataset_file='dataset/20201202M100K_data_all_random.txt'):
def run_enhanced_hpl_optimization(n_trials=5, dataset_name='movielens100k', dataset_file='dataset/20201202M100K_data_all_random.txt'):
    """运行HPL增强优化"""
    print("开始HPL增强优化 (扩展参数空间，150次试验)...")

    # 🎯 根据数据集和模型特征设置早停参数
    def calculate_early_stopping_params(dataset_name, n_trials, data_stats=None):
        """根据数据集特征动态计算早停参数"""

        # 🔧 修改：尝试从data_stats获取真实数据大小
        if data_stats is not None:
            try:
                # 如果data_stats包含训练集大小信息
                if hasattr(data_stats, 'get_splits'):
                    train_data, _, _ = data_stats.get_splits()
                    estimated_size = len(train_data)
                    print(f"📊 使用真实训练集大小: {estimated_size:,}")
                elif hasattr(data_stats, 'train_data'):
                    estimated_size = len(data_stats.train_data)
                    print(f"📊 使用真实训练集大小: {estimated_size:,}")
                elif isinstance(data_stats, dict) and 'train_size' in data_stats:
                    estimated_size = data_stats['train_size']
                    print(f"📊 使用真实训练集大小: {estimated_size:,}")
                else:
                    # 回退到预估值
                    raise AttributeError("无法从data_stats获取训练集大小")
            except Exception as e:
                print(f"⚠️ 无法获取真实数据大小({e})，使用预估值")
                # 数据集规模映射 (预估) - 作为后备方案
                dataset_size_map = {
                    'movielens100k': 100000,
                    'movielens1m': 1000000,
                    'netflix': 100000000,
                    'amazonmi': 50000,
                    'filmtrust': 35000,
                }
                estimated_size = dataset_size_map.get(dataset_name.lower(), 100000)
        else:
            # 原有的预估逻辑
            dataset_size_map = {
                'movielens100k': 100000,
                'movielens1m': 1000000,
                'netflix': 100000000,
                'amazonmi': 50000,
                'filmtrust': 35000,
            }
            estimated_size = dataset_size_map.get(dataset_name.lower(), 100000)
            print(f"📊 使用预估数据集大小: {estimated_size:,}")

        # 模型训练早停 - 基于数据规模和HPL特性
        if estimated_size < 10000:  # 小数据集
            model_patience = 8
            # model_patience = 3  # 快速
        elif estimated_size < 100000:  # 中等数据集
            # model_patience = 5  # 快速
            model_patience = 12  # 标准设置
        else:  # 大数据集
            model_patience = 15
            # model_patience = 6 # 快速

        # HPL损失函数通常收敛较慢，增加20%
        model_patience = int(model_patience * 1.2)
        # model_patience = int(model_patience * 0.8) # 快速

        # 超参数优化早停 - 基于搜索预算
        if n_trials <= 50:
            hyperopt_patience = max(10, n_trials // 8)    # 小预算：12.5%
        elif n_trials <= 200:
            hyperopt_patience = max(16, n_trials // 12)   # 中预算：8.3%
        else:
            hyperopt_patience = max(20, n_trials // 15)  # 大预算：6.7%

        return model_patience, hyperopt_patience

    # 计算早停参数
    # 首先加载数据管理器
    data_manager = setup_data_manager(dataset_name, dataset_file)

    # 计算早停参数（使用真实数据）
    MODEL_EARLY_STOPPING, HYPEROPT_EARLY_STOPPING = calculate_early_stopping_params(
        dataset_name, n_trials, data_manager  # 传入data_manager
    )

    print(f"📊 早停参数设置:")
    print(f"   模型训练早停: {MODEL_EARLY_STOPPING} epochs")
    print(f"   优化搜索早停: {HYPEROPT_EARLY_STOPPING} trials")

    # data_manager = setup_data_manager(dataset_name, dataset_file)
    result_manager = DatasetAwareResultManager(dataset_name, dataset_file)
    monitor = OptimizationMonitor(result_manager, 'enhanced_hpl_optimization')

    # 使用实际数据统计更新早停参数
    data_stats = data_manager.get_statistics()
    actual_data_size = len(data_manager.get_splits()[0])  # 训练集大小

    if actual_data_size < 1000:
        MODEL_EARLY_STOPPING = max(6, MODEL_EARLY_STOPPING - 3)
        print(f"   🔧 检测到小数据集({actual_data_size}条)，调整模型早停为: {MODEL_EARLY_STOPPING}")
    elif actual_data_size > 500000:
        MODEL_EARLY_STOPPING = min(25, MODEL_EARLY_STOPPING + 3)
        print(f"   🔧 检测到大数据集({actual_data_size}条)，调整模型早停为: {MODEL_EARLY_STOPPING}")

    objective = HPLObjectiveFunction(data_manager)
    space = ParameterSpaceFactory.create_enhanced_hpl_space()
    constraints = ConstraintFactory.create_hpl_constraints()

    optimizer = OptimizerFactory.create_optimizer(
        objective, space, constraints, 'enhanced_hpl_optimization'
    )

    # 测试目标函数是否正常工作
    print(f"\n🧪 测试目标函数...")
    try:
        # 修改这里：使用numpy的随机数生成器
        import numpy as np
        random_state = np.random.RandomState(42)  # 使用固定种子以便复现
        # 直接使用sample返回的配置，不要尝试索引访问
        test_config = space.sample(random_state)
        test_score = objective(test_config)
        print(f"   测试配置: {test_config}")
        print(f"   测试分数: {test_score}")
        if test_score == 10.0:
            print("   ⚠️ 警告：目标函数返回默认错误值，可能存在问题")
        else:
            print("   ✅ 目标函数工作正常")
    except Exception as e:
        print(f"   ❌ 目标函数测试失败: {e}")
        import traceback
        traceback.print_exc()

    start_time = time.time()
    print(f"\n🚀 开始优化，目标试验次数: {n_trials}")
    print(f"   早停轮数: {HYPEROPT_EARLY_STOPPING}")

    try:
        best_trial = optimizer.optimize(
            n_trials=n_trials,
            no_improvement_rounds=HYPEROPT_EARLY_STOPPING,
            batch_size=1
        )
        print(f"✅ 优化完成，返回的best_trial类型: {type(best_trial)}")
        print(f"   best_trial内容: {best_trial}")
    except Exception as e:
        print(f"❌ 优化过程失败: {e}")
        import traceback
        traceback.print_exc()
        best_trial = None

    end_time = time.time()

    # 安全地提取结果
    best_config, best_score = safe_extract_trial_info(best_trial)
    print(f"📊 提取的最佳配置: {best_config}")
    print(f"📊 提取的最佳分数: {best_score}")

    print(f"\nHPL增强优化结果:")
    if best_config:
        print(f"最佳配置: {best_config}")
        print(f"最佳RMSE: {best_score:.4f}")
        print(f"优化耗时: {end_time - start_time:.2f}秒")

        # 获取试验结果
        trials = safe_get_trials(optimizer)
        print(f"📊 获取到 {len(trials)} 个试验结果")

        # 获取优化器结果
        optimizer_results = {}
        if hasattr(optimizer, 'get_results'):
            optimizer_results = optimizer.get_results()
        elif hasattr(optimizer, 'tracker') and hasattr(optimizer.tracker, 'get_trials'):
            tracker_trials = optimizer.tracker.get_trials()
            optimizer_results = {'trials': tracker_trials}

        # 计算实际执行的试验次数
        actual_trials = len(trials)
        if actual_trials == 0:
            print("⚠️ 警告：没有获取到任何试验结果！")
            # 尝试从优化器直接获取
            if hasattr(optimizer, 'tracker') and hasattr(optimizer.tracker, 'trials'):
                direct_trials = optimizer.tracker.trials
                print(f"   从tracker直接获取到 {len(direct_trials)} 个试验")
                trials = [SafeTrial(t.config if hasattr(t, 'config') else {},
                                  t.score if hasattr(t, 'score') else float('inf'))
                         for t in direct_trials]
                actual_trials = len(trials)

        optimization_results = {
            'best_config': best_config,
            'best_score': best_score,
            'optimization_time': end_time - start_time,
            'n_trials': n_trials,
            'actual_trials': actual_trials,  # 实际执行的试验次数
            'early_stopping_config': {
                'model_early_stopping': MODEL_EARLY_STOPPING,
                'hyperopt_early_stopping': HYPEROPT_EARLY_STOPPING,
                'actual_data_size': actual_data_size,
                'estimated_data_size': 100000  # 默认估计大小
            },
            'parameter_space': {
                'learning_rate': {'min': 0.005, 'max': 0.15, 'scale': 'log'},
                'latent_factors': {'min': 8, 'max': 120, 'step': 8},
                'lambda_reg': {'min': 0.0001, 'max': 0.05, 'scale': 'log'},
                'delta1': {'min': 0.05, 'max': 1.0},
                'delta2': {'min': 0.3, 'max': 3.5},
                'l_max': {'min': 1.5, 'max': 6.0},
                'c_sigmoid': {'min': 0.2, 'max': 3.0}
            },
            'constraints': ['delta1 < delta2'],
            'optimizer_results': optimizer_results,
            'all_trials': [trial.to_dict() for trial in trials]
        }

        result_manager.save_experiment_results(
            results=optimization_results,
            experiment_type='enhanced_hpl_optimization',
            metadata={
                'algorithm': 'Latin Hypercube Sampling',
                'n_trials': n_trials,
                'early_stopping_strategy': {
                    'model_patience': MODEL_EARLY_STOPPING,
                    'hyperopt_patience': HYPEROPT_EARLY_STOPPING,
                    'strategy': 'adaptive_based_on_dataset_size',
                    'hpl_adjustment': '+20% for slow convergence'
                },
                'data_characteristics': {
                    'dataset_name': dataset_name,
                    'training_samples': actual_data_size,
                    'n_users': data_stats.get('n_users', 'unknown'),
                    'n_items': data_stats.get('n_items', 'unknown')
                }
            }
        )

        # 🎯 使用最佳参数进行完整性能评估
        print(f"\n🔍 开始使用最佳参数进行完整性能评估...")
        try:
            evaluator = BestParameterEvaluator(data_manager, best_config)
            best_model, training_time = evaluator.train_best_model(n_epochs=100)

            # 计算所有性能指标
            performance_metrics = evaluator.evaluate_all_metrics(best_model)

            # 将性能指标添加到结果中
            optimization_results['performance_evaluation'] = {
                'training_time': training_time,
                'metrics': performance_metrics,
                'evaluation_timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }

            # 重新保存包含性能评估的结果
            result_manager.save_experiment_results(
                results=optimization_results,
                experiment_type='enhanced_hpl_optimization_with_metrics',
                metadata={
                    'algorithm': 'Latin Hypercube Sampling',
                    'n_trials': n_trials,
                    'includes_performance_evaluation': True,
                    'evaluation_metrics': list(performance_metrics.keys()),
                    'early_stopping_strategy': {
                        'model_patience': MODEL_EARLY_STOPPING,
                        'hyperopt_patience': HYPEROPT_EARLY_STOPPING,
                        'strategy': 'adaptive_based_on_dataset_size',
                        'hpl_adjustment': '+20% for slow convergence'
                    },
                    'data_characteristics': {
                        'dataset_name': dataset_name,
                        'training_samples': actual_data_size,
                        'n_users': data_stats.get('n_users', 'unknown'),
                        'n_items': data_stats.get('n_items', 'unknown')
                    }
                }
            )

            print(f"\n📊 性能评估结果摘要:")
            print(f"   训练时间: {training_time:.2f}秒")

            # 显示主要指标
            key_metrics = ['RMSE', 'MAE', 'HitRate@10', 'Precision@10', 'Recall@10', 'NDCG@10']
            for metric in key_metrics:
                if metric in performance_metrics and performance_metrics[metric] is not None:
                    print(f"   {metric}: {performance_metrics[metric]:.4f}")

        except Exception as e:
            print(f"⚠️ 性能评估失败: {e}")
            import traceback
            traceback.print_exc()
            # 即使评估失败，也保存优化结果
            result_manager.save_experiment_results(
                results=optimization_results,
                experiment_type='enhanced_hpl_optimization',
                metadata={
                    'algorithm': 'Latin Hypercube Sampling',
                    'n_trials': n_trials,
                    'performance_evaluation_failed': True,
                    'early_stopping_strategy': {
                        'model_patience': MODEL_EARLY_STOPPING,
                        'hyperopt_patience': HYPEROPT_EARLY_STOPPING,
                        'strategy': 'adaptive_based_on_dataset_size',
                        'hpl_adjustment': '+20% for slow convergence'
                    },
                    'data_characteristics': {
                        'dataset_name': dataset_name,
                        'training_samples': actual_data_size,
                        'n_users': data_stats.get('n_users', 'unknown'),
                        'n_items': data_stats.get('n_items', 'unknown')
                    }
                }
            )

        safe_best_trial = SafeTrial(best_config, best_score)
    else:
        print("优化失败，未获得有效结果")
        safe_best_trial = None

    # 生成监控报告（但不覆盖主要结果）
    try:
        monitor.generate_final_report()
    except Exception as e:
        print(f"⚠️ 监控报告生成失败: {e}")

    return optimizer, safe_best_trial, result_manager


def guess_dataset_type(filename):
    """根据文件名猜测数据集类型"""
    filename = filename.lower()

    if 'm100k' in filename or 'movielens100k' in filename:
        return 'movielens100k'
    elif 'netflix' in filename:
        return 'netflix'
    elif 'm1m' in filename or 'movielens1m' in filename:
        return 'movielens1m'
    elif 'amazon' in filename and ('musical' in filename or 'mi' in filename):
        return 'amazonmi'
    elif 'ciaodvd' in filename:
        return 'ciaodvd'
    elif 'epinions' in filename:
        return 'epinions'
    elif 'filmtrust' in filename:
        return 'filmtrust'
    elif 'movietweetings' in filename:
        return 'movietweetings'
    else:
        return filename


def check_file_format(file_path):
    """检查文件格式并显示前几行"""
    try:
        print(f"\n检查文件格式: {os.path.basename(file_path)}")
        print("1. 文件格式为每行: [user_id, movie_id, rating]")

        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()[:10]]

        print("文件前10行内容:")
        for i, line in enumerate(lines, 1):
            print(f"{i}: {line}")

        confirm = input("是否继续使用此文件? (y/n, 默认y): ").strip().lower()
        if confirm == 'n':
            return None
        return file_path
    except Exception as e:
        print(f"检查文件格式时出错: {e}")
        return None


def scan_available_datasets(dataset_dir):
    """扫描并列出所有可用数据集"""
    print(f"\n🔍 扫描数据集目录: {dataset_dir}")

    if not os.path.exists(dataset_dir):
        print(f"⚠️ 数据集目录不存在: {dataset_dir}")
        try:
            os.makedirs(dataset_dir, exist_ok=True)
            print(f"✅ 已创建数据集目录: {dataset_dir}")
        except Exception as e:
            print(f"❌ 无法创建数据集目录: {e}")
            return []

    pattern = os.path.join(dataset_dir, '*.txt')
    dataset_files = glob.glob(pattern)

    if not dataset_files:
        print(f"❌ 在 {dataset_dir} 中未找到任何 .txt 数据集文件")
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

        try:
            file_size = os.path.getsize(file_path)
            if file_size > 1024 * 1024:
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
            elif file_size > 1024:
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
            return selected_datasets

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
                return selected_datasets
            else:
                print("❌ 未选择任何数据集")

        except ValueError:
            print("❌ 请输入有效的序号")
            continue


def execute_single_dataset_experiment(dataset_info):
    """执行单个数据集的HPL增强优化实验"""
    dataset_name = dataset_info['dataset_type']
    dataset_file = dataset_info['file_path']
    filename = dataset_info['filename']

    print(f"\n🚀 开始实验: {filename}")
    print(f"   数据集类型: {dataset_name}")
    print(f"   实验类型: HPL增强优化")

    start_time = time.time()

    try:
        optimizer, best_trial, result_manager = run_enhanced_hpl_optimization(
            n_trials= 20, # 测试用 20 次试验
            dataset_name=dataset_name,
            dataset_file=dataset_file
        )

        end_time = time.time()
        duration = end_time - start_time

        # 安全地提取试验信息
        if best_trial:
            best_config, best_score = safe_extract_trial_info(best_trial)

            # 提取性能评估结果（如果存在）
            performance_metrics = {}
            if hasattr(best_trial, 'performance_evaluation'):
                performance_metrics = best_trial.performance_evaluation.get('metrics', {})
            elif isinstance(best_trial, dict) and 'performance_evaluation' in best_trial:
                performance_metrics = best_trial['performance_evaluation'].get('metrics', {})

            completion_message = f"✅ 单数据集HPL优化完成\n" \
                               f"数据集: {dataset_info['filename']}\n" \
                               f"最佳RMSE: {best_score:.4f}\n" \
                               f"耗时: {duration:.1f}秒"

            # 如果有性能指标，添加到通知中
            if performance_metrics:
                if 'RMSE' in performance_metrics and performance_metrics['RMSE'] is not None:
                    completion_message += f"\n测试RMSE: {performance_metrics['RMSE']:.4f}"
                if 'HitRate@10' in performance_metrics and performance_metrics['HitRate@10'] is not None:
                    completion_message += f"\nHR@10: {performance_metrics['HitRate@10']:.4f}"

            try:
                # send_sms_notification(completion_message)
                print(f"✅ 已发送完成通知: {dataset_info['filename']}")
            except Exception as e:
                print(f"⚠️ 发送通知失败: {e}")

            result = {
                'status': 'success',
                'best_trial': best_trial,
                'best_score': best_score,
                'best_config': best_config,
                'duration': duration,
                'dataset_info': dataset_info,
                'performance_metrics': performance_metrics  # 添加性能指标
            }
        else:
            result = {
                'status': 'failed',
                'error': 'No valid results obtained',
                'duration': duration,
                'dataset_info': dataset_info
            }

        print(f"✅ {filename} 实验完成 ({duration:.1f}秒)")
        return result

    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time

        error_message = f"❌ HPL优化实验失败\n" \
                       f"数据集: {dataset_info['filename']}\n" \
                       f"错误: {str(e)[:50]}...\n" \
                       f"耗时: {duration:.1f}秒"
        try:
            # send_sms_notification(error_message)
            print(f"✅ 已发送错误通知: {dataset_info['filename']}")
        except Exception as sms_e:
            print(f"⚠️ 发送错误通知失败: {sms_e}")

        print(f"❌ {filename} 实验失败: {e}")
        traceback.print_exc()

        return {
            'status': 'failed',
            'error': str(e),
            'duration': duration,
            'dataset_info': dataset_info
        }


def generate_multi_dataset_summary(all_results):
    """生成多数据集实验汇总报告 - 修复版本"""
    print("\n" + "="*80)
    print("多数据集HPL增强优化汇总报告")
    print("="*80)

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

    print(f"\n📋 详细结果:")
    print("-" * 120)
    print(f"{'数据集':<30} {'状态':<8} {'最佳RMSE':<12} {'测试RMSE':<12} {'HR@10':<10} {'耗时(秒)':<10}")
    print("-" * 120)

    successful_results = []

    # 🔧 修复：处理结果时进行安全转换
    processed_results = {}
    for dataset_name, result in all_results.items():
        status = "✅成功" if result['status'] == 'success' else "❌失败"

        if result['status'] == 'success':
            rmse = result.get('best_score', 'N/A')
            rmse_str = f"{rmse:.4f}" if rmse and rmse != 'N/A' else 'N/A'

            # 提取性能指标
            performance_metrics = result.get('performance_metrics', {})
            test_rmse = performance_metrics.get('RMSE', 'N/A')
            test_rmse_str = f"{test_rmse:.4f}" if test_rmse and test_rmse != 'N/A' else 'N/A'

            hr10 = performance_metrics.get('HitRate@10', 'N/A')
            hr10_str = f"{hr10:.4f}" if hr10 and hr10 != 'N/A' else 'N/A'

            # 🔧 安全地提取配置信息
            best_config = result.get('best_config', {})
            if not isinstance(best_config, dict):
                best_config = {}

            successful_results.append((dataset_name, rmse, best_config, performance_metrics))

            # 🔧 安全地处理best_trial
            best_trial = result.get('best_trial')
            if hasattr(best_trial, 'to_dict'):
                processed_best_trial = best_trial.to_dict()
            elif isinstance(best_trial, dict):
                processed_best_trial = best_trial
            else:
                processed_best_trial = {
                    'config': best_config,
                    'score': rmse
                }

            processed_results[dataset_name] = {
                'status': result['status'],
                'best_score': rmse,
                'best_config': best_config,
                'best_trial_dict': processed_best_trial,  # 使用处理后的版本
                'duration': result.get('duration', 0),
                'dataset_info': result.get('dataset_info', {}),
                'performance_metrics': performance_metrics  # 添加性能指标
            }
        else:
            rmse_str = f"错误: {result.get('error', 'Unknown')[:20]}..."
            test_rmse_str = 'N/A'
            hr10_str = 'N/A'
            processed_results[dataset_name] = {
                'status': result['status'],
                'error': result.get('error', 'Unknown'),
                'duration': result.get('duration', 0),
                'dataset_info': result.get('dataset_info', {})
            }

        duration = result.get('duration', 0)
        print(f"{dataset_name:<30} {status:<8} {rmse_str:<12} {test_rmse_str:<12} {hr10_str:<10} {duration:<10.1f}")

    if successful_results:
        print(f"\n🏆 最佳表现排序 (按验证RMSE):")
        successful_results.sort(key=lambda x: x[1])
        for i, (dataset_name, rmse, config, metrics) in enumerate(successful_results, 1):
            print(f"   {i}. {dataset_name}: 验证RMSE={rmse:.4f}")

            # 显示测试性能指标
            if metrics:
                test_rmse = metrics.get('RMSE', 'N/A')
                hr10 = metrics.get('HitRate@10', 'N/A')
                precision10 = metrics.get('Precision@10', 'N/A')
                recall10 = metrics.get('Recall@10', 'N/A')
                ndcg10 = metrics.get('NDCG@10', 'N/A')

                print(f"      测试性能: RMSE={test_rmse:.4f if test_rmse != 'N/A' else 'N/A'}, "
                      f"HR@10={hr10:.4f if hr10 != 'N/A' else 'N/A'}, "
                      f"P@10={precision10:.4f if precision10 != 'N/A' else 'N/A'}")
                print(f"                R@10={recall10:.4f if recall10 != 'N/A' else 'N/A'}, "
                      f"NDCG@10={ndcg10:.4f if ndcg10 != 'N/A' else 'N/A'}")

            print(f"      最佳配置: learning_rate={config.get('learning_rate', 'N/A'):.4f}, "
                  f"latent_factors={config.get('latent_factors', 'N/A')}, "
                  f"lambda_reg={config.get('lambda_reg', 'N/A'):.6f}")
            print(f"      HPL参数: δ1={config.get('delta1', 'N/A'):.3f}, δ2={config.get('delta2', 'N/A'):.3f}, "
                  f"l_max={config.get('l_max', 'N/A'):.2f}, c_sigmoid={config.get('c_sigmoid', 'N/A'):.2f}")
            print()  # 空行分隔

    # 🔧 使用处理后的结果构建汇总数据
    summary_data = {
        'experiment_info': {
            'experiment_type': 'enhanced_hpl_optimization_batch',
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'total_datasets': total_datasets,
            'successful_datasets': successful_datasets,
            'failed_datasets': failed_datasets,
            'total_duration': total_duration
        },
        'detailed_results': processed_results,  # 使用处理后的结果
        'parameter_space': {
            'learning_rate': {'min': 0.005, 'max': 0.15, 'scale': 'log'},
            'latent_factors': {'min': 8, 'max': 120, 'step': 8},
            'lambda_reg': {'min': 0.0001, 'max': 0.05, 'scale': 'log'},
            'delta1': {'min': 0.05, 'max': 1.0},
            'delta2': {'min': 0.3, 'max': 3.5},
            'l_max': {'min': 1.5, 'max': 6.0},
            'c_sigmoid': {'min': 0.2, 'max': 3.0}
        }
    }

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = Path(os.path.join(current_script_dir, 'results'))
    results_dir.mkdir(exist_ok=True)

    summary_filename = f"multi_dataset_hpl_enhanced_summary_{time.strftime('%Y%m%d_%H%M%S')}.json"
    summary_path = results_dir / summary_filename

    # 🔧 使用安全的JSON保存函数
    success = safe_json_dump(summary_data, summary_path)

    if success:
        print(f"\n💾 汇总报告已保存: {summary_path}")
        print(f"📍 完整路径: {summary_path.absolute()}")
    else:
        print(f"⚠️ 汇总报告保存失败，请检查文件权限和磁盘空间")

    return summary_data


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，增强版本"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    # 🔧 新增：处理SafeTrial对象
    elif hasattr(obj, 'to_dict'):
        return convert_numpy_types(obj.to_dict())
    # 🔧 新增：处理其他自定义对象
    elif hasattr(obj, '__dict__'):
        return convert_numpy_types(obj.__dict__)
    elif hasattr(obj, 'item'):
        return obj.item()
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    # 🔧 新增：处理numpy标量
    elif str(type(obj)).startswith("<class 'numpy."):
        try:
            return obj.item()
        except (ValueError, AttributeError):
            return float(obj) if hasattr(obj, '__float__') else str(obj)
    # 🔧 新增：处理不可序列化的对象
    elif not isinstance(obj, (str, int, float, bool, type(None))):
        try:
            return str(obj)
        except:
            return "unserializable_object"
    else:
        return obj


def safe_json_dump(data, file_path):
    """安全的JSON保存函数"""
    try:
        # 步骤1: 先转换数据类型
        clean_data = convert_numpy_types(data)

        # 步骤2: 测试序列化是否成功
        json_str = json.dumps(clean_data, indent=2, ensure_ascii=False)

        # 步骤3: 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            f.flush()  # 强制刷新缓冲区

        print(f"✅ JSON文件保存成功: {file_path}")
        return True

    except Exception as e:
        print(f"❌ JSON保存失败: {e}")
        print(f"尝试保存备用格式...")

        # 备用方案：保存为文本格式
        try:
            backup_path = str(file_path).replace('.json', '_backup.txt')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(f"JSON保存失败，原始数据结构：\n")
                f.write(str(data))
            print(f"⚠️ 已保存备用文件: {backup_path}")
        except:
            print(f"❌ 备用保存也失败")

        return False


def run_multiple_datasets(processed_datasets):
    """批量执行HPL增强优化实验"""
    print("\n" + "="*80)
    print("🚀 开始批量执行HPL增强优化实验")
    print("="*80)

    print(f"将对 {len(processed_datasets)} 个数据集执行HPL增强优化 (150次试验)")

    all_results = {}

    for i, dataset_info in enumerate(processed_datasets, 1):
        print(f"\n{'='*60}")
        print(f"🔄 进度: {i}/{len(processed_datasets)}")
        print(f"{'='*60}")

        result = execute_single_dataset_experiment(dataset_info)

        dataset_key = f"{dataset_info['dataset_type']}_{os.path.basename(dataset_info['file_path'])}"
        all_results[dataset_key] = result

        if result['status'] == 'success':
            print(f"✅ 完成 {i}/{len(processed_datasets)}: {dataset_info['filename']}")
        else:
            print(f"❌ 失败 {i}/{len(processed_datasets)}: {dataset_info['filename']}")

    print("\n📊 生成汇总报告...")
    summary_data = generate_multi_dataset_summary(all_results)

    try:
        successful_count = len([r for r in all_results.values() if r['status'] == 'success'])
        success_rate = successful_count / len(all_results) * 100

        successful_rmses = [r.get('best_score', float('inf')) for r in all_results.values()
                           if r['status'] == 'success' and r.get('best_score') is not None]
        avg_rmse = sum(successful_rmses) / len(successful_rmses) if successful_rmses else float('inf')

        best_dataset = None
        best_rmse = float('inf')
        for dataset_name, result in all_results.items():
            if result['status'] == 'success' and result.get('best_score', float('inf')) < best_rmse:
                best_rmse = result.get('best_score')
                best_dataset = dataset_name

        summary_message = f"🎉 HPL增强优化批量实验完成!\n" \
                         f"总数据集: {len(all_results)}\n" \
                         f"成功率: {success_rate:.1f}%\n" \
                         f"平均RMSE: {avg_rmse:.4f}\n" \
                         f"最佳数据集: {best_dataset}\n" \
                         f"最佳RMSE: {best_rmse:.4f}"

        # send_sms_notification(summary_message)
        print("✅ 已发送批量实验总结通知")
    except Exception as e:
        print(f"⚠️ 发送总结通知失败: {e}")

    print(f"\n🎉 多数据集HPL增强优化实验完成!")
    print(f"   成功: {len([r for r in all_results.values() if r['status'] == 'success'])}/{len(all_results)}")

    return True


def main():
    """主函数 - HPL增强优化批量处理"""
    try:
        print("HPL损失函数增强优化实验")
        print("="*60)

        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        results_save_dir = os.path.join(current_script_dir, 'results')
        print(f"📂 当前脚本目录: {current_script_dir}")
        print(f"💾 结果保存目录: {results_save_dir}")

        print("\n📂 第一步: 数据集扫描与选择")
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_dir = os.path.join(base_dir, 'dataset')
        print(f"🔍 数据集搜索目录: {dataset_dir}")

        dataset_files = scan_available_datasets(dataset_dir)
        if not dataset_files:
            print("❌ 没有找到可用的数据集文件")
            return False

        dataset_map = create_dataset_mapping(dataset_files)

        selected_datasets = get_user_dataset_selection(dataset_map)
        if not selected_datasets:
            print("❌ 没有选择任何数据集")
            return False

        print("\n🔧 第二步: 文件格式检查")
        processed_datasets = []
        for dataset_info in selected_datasets:
            print(f"\n检查: {dataset_info['filename']}")
            checked_file = check_file_format(dataset_info['file_path'])
            if checked_file:
                processed_datasets.append(dataset_info)
            else:
                print(f"跳过: {dataset_info['filename']}")

        if not processed_datasets:
            print("❌ 没有可用的数据集，退出实验")
            return False

        print(f"\n🏃 第三步: 批量执行HPL增强优化实验")
        print(f"确认执行: 将对 {len(processed_datasets)} 个数据集进行150次试验的HPL增强优化")
        confirm = input("是否继续? (y/n, 默认y): ").strip().lower()
        if confirm == 'n':
            print("实验已取消")
            return False

        success = run_multiple_datasets(processed_datasets)

        if success:
            print("\n✅ 所有实验已完成!")
            # send_sms_notification(f"HPL增强优化实验已完成，共处理{len(processed_datasets)}个数据集")
        else:
            print("\n❌ 实验执行过程中出现问题")
            # send_sms_notification("HPL增强优化实验执行失败")

        return success

    except Exception as e:
        print(f"主程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        # send_sms_notification(f"HPL增强优化实验失败: {str(e)[:50]}...")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)


