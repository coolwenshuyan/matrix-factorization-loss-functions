#!/usr/bin/env python3
"""
实验执行器模块
负责协调和执行完整的鲁棒性测试实验
"""

import sys
import os
import time
import json
import logging
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入自定义模块
from dataset_selector import DatasetInfo
from config_matcher import OptimalConfig
from result_manager import ResultManager
from data.data_manager import DataManager
from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.initializers import NormalInitializer, XavierInitializer, UniformInitializer
from src.models.regularizers import L2Regularizer, L1Regularizer
from src.losses.standard import L1Loss, L2Loss
from src.losses.hpl import HybridPiecewiseLoss
from src.losses.robust import HuberLoss, LogcoshLoss
from src.losses.sigmoid import SigmoidLikeLoss

# 假设已有的噪声注入系统
from noise_injection_system import NoiseConfig, RobustnessEvaluator

logger = logging.getLogger(__name__)


class ExperimentStatus:
    """实验状态跟踪器"""

    def __init__(self):
        """初始化状态跟踪器"""
        self.start_time = None
        self.total_tasks = 0
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.current_task = ""
        self.task_start_time = None

    def start(self, total_tasks: int):
        """开始实验"""
        self.start_time = time.time()
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.current_task = "初始化"
        self.task_start_time = time.time()

    def update(self, task_description: str):
        """更新当前任务"""
        self.current_task = task_description
        self.task_start_time = time.time()

    def complete_task(self, success: bool, message: str = ""):
        """完成任务"""
        self.completed_tasks += 1
        if success:
            self.successful_tasks += 1
        else:
            self.failed_tasks += 1
            logger.warning(f"任务失败: {message}")

    def get_progress(self) -> Tuple[float, str]:
        """获取进度信息"""
        if self.total_tasks == 0:
            return 0.0, "未开始"

        progress = self.completed_tasks / self.total_tasks
        elapsed = time.time() - self.start_time

        if progress > 0:
            estimated_total = elapsed / progress
            remaining = estimated_total - elapsed
            time_info = f"已用时: {self._format_time(elapsed)} | 剩余: {self._format_time(remaining)}"
        else:
            time_info = f"已用时: {self._format_time(elapsed)}"

        return progress, time_info

    def get_summary(self) -> str:
        """获取实验摘要"""
        elapsed = time.time() - self.start_time
        return (f"实验完成: {self.completed_tasks}/{self.total_tasks} 任务 | "
                f"成功: {self.successful_tasks} | 失败: {self.failed_tasks} | "
                f"总用时: {self._format_time(elapsed)}")

    def _format_time(self, seconds: float) -> str:
        """格式化时间"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ExperimentExecutor:
    """实验执行器"""

    def __init__(self, base_config: Dict[str, Any] = None):
        """
        初始化实验执行器

        Args:
            base_config: 基础配置
        """
        self.base_config = base_config or self._get_default_base_config()
        self.status = ExperimentStatus()

        # 初始化时不创建ResultManager，而是在需要时设置
        self.result_manager = None  # 不要在这里创建ResultManager实例

        # 噪声实验配置
        self.noise_experiment_configs = self._get_default_noise_configs()

    def _get_default_base_config(self) -> Dict[str, Any]:
        """获取默认基础配置"""
        return {
            'random_seed': 42,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'center_data': True,
            'ensure_user_in_train': True
        }

    def _get_default_noise_configs(self) -> List[Dict[str, Any]]:
        """获取默认噪声实验配置"""
        return [
            # 高斯噪声测试
            {
                'name': 'gaussian_test',
                'description': '高斯噪声测试',
                'noise_types': ['gaussian'],
                'noise_strengths': [0.1, 0.2, 0.3, 0.5, 0.8],
                'noise_ratios': [0.1, 0.5, 1.0],
                'random_seed': 42
            },
            # 异常值测试
            {
                'name': 'outlier_test',
                'description': '异常值噪声测试',
                'noise_types': ['outlier'],
                'noise_strengths': [2.0, 3.0, 5.0],
                'noise_ratios': [0.01, 0.05, 0.1],
                'random_seed': 42
            },
            # 椒盐噪声测试
            {
                'name': 'salt_pepper_test',
                'description': '椒盐噪声测试',
                'noise_types': ['salt_pepper'],
                'noise_strengths': [0],
                'noise_ratios': [0.01, 0.05, 0.1, 0.2],
                'random_seed': 42
            },
            # 系统性偏差测试
            {
                'name': 'systematic_test',
                'description': '系统性偏差测试',
                'noise_types': ['systematic'],
                'noise_strengths': [-0.5, -0.2, 0.2, 0.5],
                'noise_ratios': [0.5, 1.0],
                'random_seed': 42
            }
        ]

    def execute_robustness_experiment(self,
                                    datasets: List[DatasetInfo],
                                    configs: List[OptimalConfig],
                                    save_dir: str = None) -> Dict[str, Any]:
        """
        执行鲁棒性实验

        Args:
            datasets: 数据集列表
            configs: 优化配置列表
            save_dir: 保存目录

        Returns:
            实验结果摘要
        """
        # 如果没有指定保存目录，则使用绝对路径
        if save_dir is None:
            # 获取当前文件的绝对路径
            current_file = Path(__file__).resolve()
            # 构建保存路径 - 使用正确的目录名称
            save_dir = str(current_file.parent / "robustness_results")
            print(f"保存目录设置为: {save_dir}")

        # 计算总任务数
        total_tasks = len(datasets) * len(configs) * len(self.noise_experiment_configs)
        self.status.start(total_tasks)

        experiment_results = {}
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 如果没有设置result_manager，则创建一个
        if self.result_manager is None:
            from result_manager import ResultManager
            self.result_manager = ResultManager(str(save_path))
            logger.info(f"为实验创建了新的ResultManager，保存目录: {save_path}")

        # 保存实验元信息
        experiment_meta = {
            'start_time': datetime.now().isoformat(),
            'datasets': [d.name for d in datasets],
            'total_tasks': total_tasks,
            'base_config': self.base_config
        }

        # 对每个数据集运行实验
        for dataset_idx, dataset in enumerate(datasets):
            dataset_results = self._execute_dataset_experiment(
                dataset, configs, save_path, dataset_idx
            )
            experiment_results[dataset.name] = dataset_results

            # 显示进度
            progress, time_info = self.status.get_progress()
            print(f"\n📊 总进度: {progress*100:.1f}% | {time_info}")

        # 生成最终摘要
        summary = self._generate_experiment_summary(experiment_results)

        print(f"\n{self.status.get_summary()}")
        return summary

    def _execute_dataset_experiment(self,
                                   dataset: DatasetInfo,
                                   configs: List[OptimalConfig],
                                   save_path: Path,
                                   dataset_idx: int) -> Dict[str, Any]:
        """执行单个数据集的实验"""

        dataset_results = {
            'dataset_info': {
                'name': dataset.name,
                'display_name': dataset.display_name,
                'file_path': dataset.file_path,
                'file_size': dataset.file_size
            },
            'model_results': {},
            'experiment_summary': {}
        }

        print(f"\n{'='*80}")
        print(f"📊 开始数据集实验: {dataset.display_name}")
        print(f"{'='*80}")

        try:
            # 准备数据
            self.status.update(f"准备数据集: {dataset.display_name}")
            data_manager = self._prepare_data(dataset)

            # 对每个配置运行实验
            for config_idx, config in enumerate(configs):
                model_results = self._execute_model_experiment(
                    data_manager, config, save_path, dataset, config_idx
                )
                dataset_results['model_results'][config.dataset_name] = model_results

            # 如果有结果管理器，保存数据集结果
            if self.result_manager:
                result_path = self.result_manager.save_experiment_results(
                    dataset_results,
                    f"{dataset.name}_experiment",
                    dataset.name,
                    "robustness"
                )
                logger.info(f"数据集 {dataset.name} 实验结果已保存: {result_path}")

            return dataset_results

        except Exception as e:
            error_msg = f"数据集 {dataset.name} 实验失败: {e}"
            logger.error(error_msg)
            self.status.complete_task(False, error_msg)
            return {'error': error_msg}

    def _execute_model_experiment(self,
                                 data_manager: DataManager,
                                 config: OptimalConfig,
                                 save_path: Path,
                                 dataset: DatasetInfo,
                                 config_idx: int) -> Dict[str, Any]:
        """执行单个模型配置的实验"""

        model_results = {
            'config_info': {
                'dataset_name': config.dataset_name,
                'confidence': config.confidence,
                'source_file': config.source_file
            },
            'loss_function_results': {},
            'summary': {}
        }

        try:
            # 构建模型配置
            model_configs = config.build_model_configs(config) if hasattr(config, 'build_model_configs') else self._build_model_configs_from_optimal(config)

            # 对每个损失函数运行实验
            for loss_idx, model_config in enumerate(model_configs):
                self.status.update(f"训练模型: {dataset.display_name} - {model_config['name']}")

                loss_results = self._execute_loss_function_experiment(
                    data_manager, model_config, save_path, dataset, config_idx, loss_idx
                )
                model_results['loss_function_results'][model_config['name']] = loss_results

            return model_results

        except Exception as e:
            error_msg = f"模型配置实验失败: {e}"
            logger.error(error_msg)
            self.status.complete_task(False, error_msg)
            return {'error': error_msg}

    def _execute_loss_function_experiment(self,
                                        data_manager: DataManager,
                                        model_config: Dict[str, Any],
                                        save_path: Path,
                                        dataset: DatasetInfo,
                                        config_idx: int,
                                        loss_idx: int) -> Dict[str, Any]:
        """执行单个损失函数的实验"""

        loss_results = {
            'model_config': model_config,
            'noise_experiment_results': {},
            'training_info': {}
        }

        try:
            # 训练模型
            model, training_info = self._train_model(data_manager, model_config)
            loss_results['training_info'] = training_info

            # 创建鲁棒性评估器
            evaluator = RobustnessEvaluator(data_manager)

            # 对每个噪声实验配置进行测试
            for noise_exp_idx, noise_exp_config in enumerate(self.noise_experiment_configs):
                self.status.update(f"噪声测试: {dataset.display_name} - {model_config['name']} - {noise_exp_config['name']}")

                # 创建噪声配置列表
                noise_configs = self._create_noise_configs(noise_exp_config)

                # 评估鲁棒性
                experiment_name = f"{dataset.name}_{model_config['name']}_{noise_exp_config['name']}"
                robustness_results = evaluator.evaluate_model_robustness(
                    model, noise_configs, experiment_name
                )

                loss_results['noise_experiment_results'][noise_exp_config['name']] = robustness_results

                # 保存单个实验结果
                result_file = save_path / f"{experiment_name}_results.json"
                evaluator.save_results(result_file)

                # 保存报告
                report = evaluator.generate_robustness_report()
                report_file = save_path / f"{experiment_name}_report.txt"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)

                self.status.complete_task(True)

            return loss_results

        except Exception as e:
            error_msg = f"损失函数 {model_config['name']} 实验失败: {e}"
            logger.error(error_msg)
            self.status.complete_task(False, error_msg)
            return {'error': error_msg}

    def _prepare_data(self, dataset: DatasetInfo) -> DataManager:
        """准备数据"""
        config = self.base_config.copy()
        data_manager = DataManager(config)
        data_manager.load_dataset(dataset.name, dataset.file_path).preprocess()
        return data_manager

    def _train_model(self, data_manager: DataManager, model_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        """训练模型"""
        start_time = time.time()

        try:
            # 获取数据维度
            stats = data_manager.get_statistics()
            n_users = stats['n_users']
            n_items = stats['n_items']

            # 创建损失函数
            loss_function = self._create_loss_function(model_config['loss_function'])

            # 创建正则化器
            regularizer = self._create_regularizer(model_config['regularizer'])

            # 创建初始化器
            initializer = self._create_initializer(model_config['initializer'])

            # 创建模型
            model = MatrixFactorizationSGD(
                n_users=n_users,
                n_items=n_items,
                n_factors=model_config['n_factors'],
                learning_rate=model_config['learning_rate'],
                regularizer=regularizer,
                loss_function=loss_function,
                use_bias=model_config['use_bias'],
                global_mean=data_manager.global_mean or 0.0
            )

            # 初始化参数
            model.initialize_parameters(initializer)

            # 获取训练数据
            train_data, val_data, _ = data_manager.get_splits()

            # 训练模型
            model.fit(
                train_data=train_data,
                val_data=val_data,
                n_epochs=model_config['n_epochs'],
                verbose=0  # 减少输出
            )

            training_time = time.time() - start_time
            training_info = {
                'training_time': training_time,
                'n_epochs': model_config['n_epochs'],
                'final_loss': getattr(model, 'train_history', {}).get('loss', [])[-1:],
                'final_val_loss': getattr(model, 'train_history', {}).get('val_loss', [])[-1:]
            }

            return model, training_info

        except Exception as e:
            raise Exception(f"模型训练失败: {e}")

    def _create_loss_function(self, loss_config: Dict[str, Any]):
        """创建损失函数"""
        loss_type = loss_config['type']

        if loss_type == 'L2':
            return L2Loss()
        elif loss_type == 'L1':
            return L1Loss()
        elif loss_type == 'Huber':
            return HuberLoss(delta=loss_config.get('delta', 1.0))
        elif loss_type == 'HPL':
            return HybridPiecewiseLoss(
                delta1=loss_config.get('delta1', 0.5),
                delta2=loss_config.get('delta2', 2.0),
                l_max=loss_config.get('l_max', 3.0)
            )
        elif loss_type == 'Logcosh':
            return LogcoshLoss()
        elif loss_type == 'SigmoidLike':
            return SigmoidLikeLoss(
                alpha=loss_config.get('alpha', 1.0),
                l_max=loss_config.get('l_max', 3.0)
            )
        else:
            raise ValueError(f"不支持的损失函数: {loss_type}")

    def _create_regularizer(self, reg_config: Dict[str, Any]):
        """创建正则化器"""
        reg_type = reg_config['type']

        if reg_type == 'L2':
            return L2Regularizer(lambda_reg=reg_config.get('lambda_reg', 0.01))
        elif reg_type == 'L1':
            return L1Regularizer(lambda_reg=reg_config.get('lambda_reg', 0.01))
        else:
            raise ValueError(f"不支持的正则化器: {reg_type}")

    def _create_initializer(self, init_config: Dict[str, Any]):
        """创建初始化器"""
        init_type = init_config['type']

        if init_type == 'Normal':
            return NormalInitializer(
                mean=init_config.get('mean', 0.0),
                std=init_config.get('std', 0.01)
            )
        elif init_type == 'Xavier':
            return XavierInitializer()
        elif init_type == 'Uniform':
            return UniformInitializer(
                low=init_config.get('low', -0.01),
                high=init_config.get('high', 0.01)
            )
        else:
            raise ValueError(f"不支持的初始化器: {init_type}")

    def _create_noise_configs(self, exp_config: Dict[str, Any]) -> List[NoiseConfig]:
        """创建噪声配置列表"""
        noise_configs = []

        for noise_type in exp_config['noise_types']:
            for strength in exp_config['noise_strengths']:
                for ratio in exp_config['noise_ratios']:
                    config = NoiseConfig(
                        noise_type=noise_type,
                        noise_strength=strength,
                        noise_ratio=ratio,
                        random_seed=exp_config.get('random_seed', 42)
                    )
                    noise_configs.append(config)

        return noise_configs

    def _build_model_configs_from_optimal(self, optimal_config: OptimalConfig) -> List[Dict[str, Any]]:
        """从最优配置构建模型配置（兼容方法）"""
        base_config = optimal_config.get_best_config()

        # 提取基础参数
        base_params = {
            'n_factors': base_config.get('latent_factors', 50),
            'learning_rate': base_config.get('learning_rate', 0.01),
            'n_epochs': 100,
            'use_bias': True,
            'regularizer': {
                'type': 'L2',
                'lambda_reg': base_config.get('lambda_reg', 0.01)
            },
            'initializer': {'type': 'Normal', 'mean': 0.0, 'std': 0.01}
        }

        # 构建不同损失函数的配置
        model_configs = []

        # HPL优化模型
        hpl_config = base_params.copy()
        hpl_config.update({
            'name': 'HPL_optimized',
            'description': '优化后的HPL损失函数模型',
            'loss_function': {
                'type': 'HPL',
                'delta1': base_config.get('delta1', 0.5),
                'delta2': base_config.get('delta2', 2.0),
                'l_max': base_config.get('l_max', 3.0)
            }
        })
        model_configs.append(hpl_config)

        # L2基线模型
        l2_config = base_params.copy()
        l2_config.update({
            'name': 'L2_baseline',
            'description': 'L2损失函数基线模型',
            'loss_function': {'type': 'L2'}
        })
        model_configs.append(l2_config)

        return model_configs

    def _generate_experiment_summary(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成实验摘要"""
        summary = {
            'experiment_info': {
                'total_datasets': len(experiment_results),
                'completion_time': datetime.now().isoformat(),
                'status': 'completed',
                'total_tasks': self.status.total_tasks,
                'successful_tasks': self.status.successful_tasks,
                'failed_tasks': self.status.failed_tasks
            },
            'dataset_summaries': {},
            'overall_findings': {}
        }

        # 为每个数据集生成摘要
        for dataset_name, dataset_result in experiment_results.items():
            if 'error' not in dataset_result:
                summary['dataset_summaries'][dataset_name] = {
                    'dataset_info': dataset_result.get('dataset_info', {}),
                    'model_count': len(dataset_result.get('model_results', {})),
                    'status': 'completed'
                }
            else:
                summary['dataset_summaries'][dataset_name] = {
                    'status': 'failed',
                    'error': dataset_result['error']
                }

        return summary

    def _generate_partial_summary(self, experiment_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成部分完成的实验摘要"""
        summary = self._generate_experiment_summary(experiment_results)
        summary['experiment_info']['status'] = 'interrupted'
        summary['experiment_info']['completion_time'] = datetime.now().isoformat()
        return summary

    def _generate_error_summary(self, error_message: str) -> Dict[str, Any]:
        """生成错误摘要"""
        return {
            'experiment_info': {
                'status': 'failed',
                'error_time': datetime.now().isoformat(),
                'error_message': error_message,
                'successful_tasks': self.status.successful_tasks,
                'failed_tasks': self.status.failed_tasks
            }
        }


def main():
    """MovieLens 100K 数据集实验"""
    # 导入必要的模块
    from dataset_selector import DatasetInfo
    from config_matcher import OptimalConfig

    # 创建 MovieLens 100K 数据集
    ml100k_dataset = DatasetInfo(
        name='movielens100k',
        file_path='dataset\small_20201202M100K_data_all_random_1percent.txt',
        display_name='MovieLens 100K Small'
    )

    # 创建优化配置
    ml100k_config_data = {
        'results': {
            'best_config': {
                'learning_rate': 0.01,
                'latent_factors': 20,
                'lambda_reg': 0.01,
                'delta1': 0.3,
                'delta2': 1.5,
                'l_max': 3.0
            },
            'best_score': 0.95
        }
    }
    ml100k_config = OptimalConfig('movielens100k', ml100k_config_data)

    # 运行 MovieLens 100K 实验
    executor = ExperimentExecutor()
    results = executor.execute_robustness_experiment(
        datasets=[ml100k_dataset],
        configs=[ml100k_config],
        save_dir=None  # 使用默认路径
    )

    print("MovieLens 100K 实验完成！结果摘要:")
    print(results)


if __name__ == "__main__":
    main()























