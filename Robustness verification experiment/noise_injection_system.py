#!/usr/bin/env python3
"""
噪声注入测试方案
核心思路：在数据加载后、模型评估前，对测试集中的评分数据注入噪声
使用在原始干净训练集上训练好的模型，在带噪声的测试集上进行评估
通过修改实验配置来控制噪声的类型和强度
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import seaborn as sns
# 配置matplotlib以避免中文字体警告和美化样式
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#E5E5E5'
plt.rcParams['grid.linewidth'] = 0.5
# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")
import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入项目模块
from data.data_manager import DataManager
from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.initializers import NormalInitializer, XavierInitializer, UniformInitializer
from src.models.regularizers import L2Regularizer, L1Regularizer
from src.losses.standard import L1Loss, L2Loss
from src.losses.hpl import HybridPiecewiseLoss
from src.losses.robust import HuberLoss

# 导入短信通知模块
from utils.sms_notification import send_sms_notification

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NoiseConfig:
    """噪声配置类"""
    noise_type: str  # 'gaussian', 'uniform', 'salt_pepper', 'outlier', 'systematic'
    noise_strength: float  # 噪声强度
    noise_ratio: float = 1.0  # 受噪声影响的样本比例 (0-1)
    random_seed: int = 42
    # 特定噪声类型的参数
    outlier_scale: float = 3.0  # 异常值的缩放倍数
    systematic_bias: float = 0.5  # 系统性偏差


class NoiseInjector(ABC):
    """噪声注入器基类"""

    def __init__(self, config: NoiseConfig):
        self.config = config
        self.rng = np.random.RandomState(config.random_seed)

    @abstractmethod
    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        """注入噪声到评分数据"""
        pass

    def _select_affected_samples(self, n_samples: int) -> np.ndarray:
        """选择受噪声影响的样本索引"""
        n_affected = int(n_samples * self.config.noise_ratio)
        return self.rng.choice(n_samples, n_affected, replace=False)


class GaussianNoiseInjector(NoiseInjector):
    """高斯噪声注入器"""

    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        noisy_ratings = ratings.copy()
        affected_indices = self._select_affected_samples(len(ratings))

        # 生成高斯噪声
        noise = self.rng.normal(0, self.config.noise_strength, len(affected_indices))
        noisy_ratings[affected_indices] += noise

        # 裁剪到有效范围
        min_rating, max_rating = rating_scale
        noisy_ratings = np.clip(noisy_ratings, min_rating, max_rating)

        return noisy_ratings


class UniformNoiseInjector(NoiseInjector):
    """均匀噪声注入器"""

    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        noisy_ratings = ratings.copy()
        affected_indices = self._select_affected_samples(len(ratings))

        # 生成均匀噪声 [-strength, +strength]
        noise = self.rng.uniform(-self.config.noise_strength,
                               self.config.noise_strength,
                               len(affected_indices))
        noisy_ratings[affected_indices] += noise

        # 裁剪到有效范围
        min_rating, max_rating = rating_scale
        noisy_ratings = np.clip(noisy_ratings, min_rating, max_rating)

        return noisy_ratings


class SaltPepperNoiseInjector(NoiseInjector):
    """椒盐噪声注入器"""

    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        noisy_ratings = ratings.copy()
        affected_indices = self._select_affected_samples(len(ratings))

        min_rating, max_rating = rating_scale

        # 随机选择一半设为最小值，一半设为最大值
        n_salt = len(affected_indices) // 2
        salt_indices = affected_indices[:n_salt]
        pepper_indices = affected_indices[n_salt:]

        noisy_ratings[salt_indices] = max_rating  # "盐"
        noisy_ratings[pepper_indices] = min_rating  # "胡椒"

        return noisy_ratings


class OutlierNoiseInjector(NoiseInjector):
    """异常值噪声注入器"""

    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        noisy_ratings = ratings.copy()
        affected_indices = self._select_affected_samples(len(ratings))

        # 计算评分的标准差
        rating_std = np.std(ratings)

        # 生成异常值：原值 ± outlier_scale * std
        outlier_noise = self.rng.choice([-1, 1], len(affected_indices)) * \
                       self.config.outlier_scale * rating_std

        noisy_ratings[affected_indices] += outlier_noise

        # 裁剪到有效范围
        min_rating, max_rating = rating_scale
        noisy_ratings = np.clip(noisy_ratings, min_rating, max_rating)

        return noisy_ratings


class SystematicNoiseInjector(NoiseInjector):
    """系统性偏差噪声注入器"""

    def inject_noise(self, ratings: np.ndarray, rating_scale: Tuple[float, float]) -> np.ndarray:
        noisy_ratings = ratings.copy()
        affected_indices = self._select_affected_samples(len(ratings))

        # 添加系统性偏差（所有受影响的样本都加上相同的偏差）
        noisy_ratings[affected_indices] += self.config.systematic_bias

        # 裁剪到有效范围
        min_rating, max_rating = rating_scale
        noisy_ratings = np.clip(noisy_ratings, min_rating, max_rating)

        return noisy_ratings


class NoiseInjectionManager:
    """噪声注入管理器"""

    def __init__(self):
        self.injectors = {
            'gaussian': GaussianNoiseInjector,
            'uniform': UniformNoiseInjector,
            'salt_pepper': SaltPepperNoiseInjector,
            'outlier': OutlierNoiseInjector,
            'systematic': SystematicNoiseInjector
        }

    def create_injector(self, config: NoiseConfig) -> NoiseInjector:
        """创建噪声注入器"""
        if config.noise_type not in self.injectors:
            raise ValueError(f"不支持的噪声类型: {config.noise_type}")

        return self.injectors[config.noise_type](config)

    def inject_noise_to_test_data(self, test_data: np.ndarray,
                                 noise_config: NoiseConfig,
                                 rating_scale: Tuple[float, float] = (1, 5)) -> np.ndarray:
        """
        对测试数据注入噪声

        Args:
            test_data: 测试数据 [user_id, item_id, rating]
            noise_config: 噪声配置
            rating_scale: 评分范围

        Returns:
            带噪声的测试数据
        """
        # 创建噪声注入器
        injector = self.create_injector(noise_config)

        # 复制测试数据
        noisy_test_data = test_data.copy()

        # 注入噪声到评分列
        original_ratings = test_data[:, 2]
        noisy_ratings = injector.inject_noise(original_ratings, rating_scale)
        noisy_test_data[:, 2] = noisy_ratings

        # 记录噪声统计信息
        rating_diff = noisy_ratings - original_ratings
        logger.info(f"噪声注入完成:")
        logger.info(f"  类型: {noise_config.noise_type}")
        logger.info(f"  强度: {noise_config.noise_strength}")
        logger.info(f"  影响比例: {noise_config.noise_ratio}")
        logger.info(f"  平均变化: {np.mean(rating_diff):.4f}")
        logger.info(f"  变化标准差: {np.std(rating_diff):.4f}")
        logger.info(f"  最大变化: {np.max(np.abs(rating_diff)):.4f}")

        return noisy_test_data


class RobustnessEvaluator:
    """鲁棒性评估器"""

    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.noise_manager = NoiseInjectionManager()
        self.results = {}

    def evaluate_model_robustness(self, model, noise_configs: List[NoiseConfig],
                                dataset_name: str = "test_dataset") -> Dict[str, Any]:
        """
        评估模型在不同噪声条件下的鲁棒性

        Args:
            model: 训练好的模型
            noise_configs: 噪声配置列表
            dataset_name: 数据集名称

        Returns:
            评估结果字典
        """
        # 获取原始测试数据
        _, _, original_test_data = self.data_manager.get_splits()

        # 获取评分范围
        stats = self.data_manager.get_statistics()
        rating_scale = (stats['rating_min'], stats['rating_max'])

        # 存储结果
        results = {
            'dataset_name': dataset_name,
            'original_performance': {},
            'noisy_performance': {},
            'robustness_metrics': {}
        }

        # 评估原始测试集性能
        logger.info("评估原始测试集性能...")
        original_performance = self._evaluate_model_on_data(model, original_test_data)
        results['original_performance'] = original_performance

        # 评估每种噪声条件下的性能
        for i, noise_config in enumerate(noise_configs):
            logger.info(f"评估噪声条件 {i+1}/{len(noise_configs)}: {noise_config.noise_type}")

            # 注入噪声
            noisy_test_data = self.noise_manager.inject_noise_to_test_data(
                original_test_data, noise_config, rating_scale
            )

            # 评估模型性能
            noisy_performance = self._evaluate_model_on_data(model, noisy_test_data)

            # 存储结果
            config_key = f"{noise_config.noise_type}_strength_{noise_config.noise_strength}_ratio_{noise_config.noise_ratio}"
            results['noisy_performance'][config_key] = {
                'noise_config': noise_config.__dict__,
                'performance': noisy_performance
            }

            # 计算鲁棒性指标
            robustness_metrics = self._calculate_robustness_metrics(
                original_performance, noisy_performance
            )
            results['robustness_metrics'][config_key] = robustness_metrics

        self.results[dataset_name] = results
        return results

    def _evaluate_model_on_data(self, model, test_data: np.ndarray) -> Dict[str, float]:
        """在给定数据上评估模型性能"""
        # 预测
        user_ids = test_data[:, 0].astype(int)
        item_ids = test_data[:, 1].astype(int)
        true_ratings = test_data[:, 2]

        predicted_ratings = model.predict(user_ids, item_ids)

        # 如果数据被中心化，需要还原到原始尺度
        if self.data_manager.global_mean is not None:
            predicted_ratings += self.data_manager.global_mean
            true_ratings = true_ratings + self.data_manager.global_mean

        # 计算评估指标
        mae = np.mean(np.abs(predicted_ratings - true_ratings))
        mse = np.mean((predicted_ratings - true_ratings) ** 2)
        rmse = np.sqrt(mse)

        # 计算MAPE（避免除零错误）
        non_zero_mask = true_ratings != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((predicted_ratings[non_zero_mask] - true_ratings[non_zero_mask])
                                / true_ratings[non_zero_mask])) * 100
        else:
            mape = float('inf')

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }

    def _calculate_robustness_metrics(self, original_perf: Dict[str, float],
                                    noisy_perf: Dict[str, float]) -> Dict[str, float]:
        """计算鲁棒性指标"""
        robustness_metrics = {}

        for metric in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            original_value = original_perf[metric]
            noisy_value = noisy_perf[metric]

            # 性能降解（相对变化）
            if original_value != 0:
                relative_change = (noisy_value - original_value) / original_value
            else:
                relative_change = float('inf') if noisy_value != 0 else 0

            # 绝对变化
            absolute_change = noisy_value - original_value

            robustness_metrics[f'{metric}_relative_change'] = relative_change
            robustness_metrics[f'{metric}_absolute_change'] = absolute_change
            robustness_metrics[f'{metric}_noisy'] = noisy_value

        return robustness_metrics

    def save_results(self, save_path: str):
        """保存评估结果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"评估结果已保存到: {save_path}")

    def generate_robustness_report(self) -> str:
        """生成鲁棒性评估报告"""
        if not self.results:
            return "无评估结果"

        report = []
        report.append("=" * 80)
        report.append("模型鲁棒性评估报告")
        report.append("=" * 80)

        for dataset_name, dataset_results in self.results.items():
            report.append(f"\n数据集: {dataset_name}")
            report.append("-" * 60)

            # 原始性能
            original_perf = dataset_results['original_performance']
            report.append(f"原始测试集性能:")
            for metric, value in original_perf.items():
                report.append(f"  {metric}: {value:.4f}")

            # 噪声条件下的性能
            report.append(f"\n噪声条件下的性能变化:")
            report.append(f"{'噪声类型':<12} {'强度':<8} {'比例':<8} {'RMSE变化':<12} {'MAE变化':<12}")
            report.append("-" * 60)

            for config_key, result in dataset_results['robustness_metrics'].items():
                parts = config_key.split('_')
                noise_type = parts[0]
                strength = parts[2]
                ratio = parts[4]

                rmse_change = result.get('RMSE_relative_change', 0) * 100
                mae_change = result.get('MAE_relative_change', 0) * 100

                report.append(f"{noise_type:<12} {strength:<8} {ratio:<8} "
                            f"{rmse_change:+7.2f}%    {mae_change:+7.2f}%")

        report.append("=" * 80)
        return "\n".join(report)


class ExperimentRunner:
    """实验运行器"""

    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = {}
        self.model_configs = []  # 存储当前实验的模型配置

    def run_robustness_experiment(self,
                                 dataset_name: str,
                                 data_path: str,
                                 noise_experiment_configs: List[Dict[str, Any]],
                                 model_configs: List[Dict[str, Any]],
                                 save_dir: str = "robustness_experiments"):
        """
        运行完整的鲁棒性实验

        Args:
            dataset_name: 数据集名称
            data_path: 数据文件路径
            noise_experiment_configs: 噪声实验配置列表
            model_configs: 模型配置列表
            save_dir: 保存目录
        """
        logger.info(f"开始鲁棒性实验: {dataset_name}")

        # 存储模型配置以供可视化使用
        self.model_configs = model_configs

        # 创建保存目录
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 准备数据
        data_manager = self._prepare_data(dataset_name, data_path)

        # 创建鲁棒性评估器（在模型循环外创建，以便收集所有模型的结果）
        evaluator = RobustnessEvaluator(data_manager)

        # 对每个模型配置运行实验
        for model_idx, model_config in enumerate(model_configs):
            logger.info(f"训练模型配置 {model_idx + 1}/{len(model_configs)}")

            # 训练模型
            model = self._train_model(data_manager, model_config)

            # 对每个噪声实验配置运行评估
            for exp_idx, exp_config in enumerate(noise_experiment_configs):
                logger.info(f"噪声实验 {exp_idx + 1}/{len(noise_experiment_configs)}")

                # 创建噪声配置列表
                noise_configs = self._create_noise_configs(exp_config)

                # 评估鲁棒性
                _ = evaluator.evaluate_model_robustness(
                    model, noise_configs,
                    f"{dataset_name}_model_{model_idx}_exp_{exp_idx}"
                )

                # 保存单个实验结果
                exp_save_path = save_path / f"model_{model_idx}_exp_{exp_idx}_results.json"
                evaluator.save_results(exp_save_path)

                # 保存报告
                report = evaluator.generate_robustness_report()
                report_path = save_path / f"model_{model_idx}_exp_{exp_idx}_report.txt"
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)

        # 在所有模型训练完成后生成可视化
        self._visualize_results(evaluator, save_path)

        logger.info("鲁棒性实验完成")

    def _prepare_data(self, dataset_name: str, data_path: str) -> DataManager:
        """准备数据"""
        config = self.base_config.copy()
        data_manager = DataManager(config)
        data_manager.load_dataset(dataset_name, data_path).preprocess()
        return data_manager

    def _train_model(self, data_manager: DataManager, model_config: Dict[str, Any]):
        """训练模型"""
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
            verbose=1
        )

        return model

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

    def _visualize_results(self, evaluator: RobustnessEvaluator, save_dir: Path):
        """可视化实验结果"""
        save_dir.mkdir(parents=True, exist_ok=True)

        # 为每个模型单独生成可视化
        for model_idx, model_config in enumerate(self.model_configs):
            model_name = model_config['name']
            loss_type = model_config['loss_function']['type']

            model_save_dir = save_dir / f"model_{model_idx}_{model_name}"
            model_save_dir.mkdir(parents=True, exist_ok=True)

            # 提取该模型的数据
            model_data = self._extract_model_data(evaluator, model_idx)

            if not model_data:
                continue

            # 生成单模型可视化
            self._create_single_model_plots(model_data, model_name, loss_type, model_save_dir)

        # 生成多模型对比图
        self._create_model_comparison_plots(evaluator, save_dir)

    def _extract_model_data(self, evaluator: RobustnessEvaluator, model_idx: int):
        """提取特定模型的数据"""
        model_data = {}

        for dataset_name, dataset_results in evaluator.results.items():
            noise_types = []
            strengths = []
            rmse_changes = []
            mae_changes = []

            for config_key, result in dataset_results['robustness_metrics'].items():
                # 检查是否属于当前模型
                if f'model_{model_idx}' not in config_key:
                    continue

                parts = config_key.split('_')
                noise_type = parts[0]
                strength = float(parts[2])

                noise_types.append(noise_type)
                strengths.append(strength)
                rmse_changes.append(result.get('RMSE_relative_change', 0) * 100)
                mae_changes.append(result.get('MAE_relative_change', 0) * 100)

            if noise_types:  # 只有当有数据时才添加
                model_data[dataset_name] = {
                    'noise_types': noise_types,
                    'strengths': strengths,
                    'rmse_changes': rmse_changes,
                    'mae_changes': mae_changes
                }

        return model_data

    def _create_single_model_plots(self, model_data: dict, model_name: str, loss_type: str, save_dir: Path):
        """为单个模型创建可视化图表"""
        for dataset_name, data in model_data.items():
            noise_types = data['noise_types']
            strengths = data['strengths']
            rmse_changes = data['rmse_changes']
            mae_changes = data['mae_changes']

            # 创建可视化
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{model_name} ({loss_type} Loss) - {dataset_name}', fontsize=16)

            # 1. RMSE变化 vs 噪声强度
            ax1 = axes[0, 0]
            unique_types = list(set(noise_types))
            for noise_type in unique_types:
                mask = [nt == noise_type for nt in noise_types]
                type_strengths = [s for s, m in zip(strengths, mask) if m]
                type_rmse = [r for r, m in zip(rmse_changes, mask) if m]
                ax1.plot(type_strengths, type_rmse, 'o-', label=noise_type)

            ax1.set_xlabel('Noise Strength')
            ax1.set_ylabel('RMSE Relative Change (%)')
            ax1.set_title('RMSE Change vs Noise Strength')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 2. MAE变化 vs 噪声强度
            ax2 = axes[0, 1]
            for noise_type in unique_types:
                mask = [nt == noise_type for nt in noise_types]
                type_strengths = [s for s, m in zip(strengths, mask) if m]
                type_mae = [r for r, m in zip(mae_changes, mask) if m]
                ax2.plot(type_strengths, type_mae, 's-', label=noise_type)

            ax2.set_xlabel('Noise Strength')
            ax2.set_ylabel('MAE Relative Change (%)')
            ax2.set_title('MAE Change vs Noise Strength')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 3. 不同噪声类型的鲁棒性对比
            ax3 = axes[1, 0]
            avg_rmse_by_type = {}
            avg_mae_by_type = {}

            for noise_type in unique_types:
                mask = [nt == noise_type for nt in noise_types]
                type_rmse = [r for r, m in zip(rmse_changes, mask) if m]
                type_mae = [r for r, m in zip(mae_changes, mask) if m]
                avg_rmse_by_type[noise_type] = np.mean(type_rmse)
                avg_mae_by_type[noise_type] = np.mean(type_mae)

            x_pos = np.arange(len(unique_types))
            width = 0.35

            ax3.bar(x_pos - width/2, list(avg_rmse_by_type.values()),
                   width, label='RMSE Change (%)', alpha=0.8)
            ax3.bar(x_pos + width/2, list(avg_mae_by_type.values()),
                   width, label='MAE Change (%)', alpha=0.8)

            ax3.set_xlabel('Noise Type')
            ax3.set_ylabel('Average Relative Change (%)')
            ax3.set_title('Average Impact of Different Noise Types')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(unique_types)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # 4. 鲁棒性热力图
            ax4 = axes[1, 1]

            # 创建热力图数据
            if len(unique_types) > 1 and len(set(strengths)) > 1:
                heatmap_data = np.zeros((len(unique_types), len(set(strengths))))
                strength_values = sorted(set(strengths))

                for i, noise_type in enumerate(unique_types):
                    for j, strength in enumerate(strength_values):
                        # 找到对应的RMSE变化值
                        for nt, s, rmse_change in zip(noise_types, strengths, rmse_changes):
                            if nt == noise_type and s == strength:
                                heatmap_data[i, j] = rmse_change
                                break

                im = ax4.imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
                ax4.set_xticks(range(len(strength_values)))
                ax4.set_xticklabels([f'{s:.2f}' for s in strength_values])
                ax4.set_yticks(range(len(unique_types)))
                ax4.set_yticklabels(unique_types)
                ax4.set_xlabel('Noise Strength')
                ax4.set_ylabel('Noise Type')
                ax4.set_title('RMSE Change Heatmap (%)')

                # 添加颜色条
                plt.colorbar(im, ax=ax4)
            else:
                ax4.text(0.5, 0.5, 'Insufficient data for heatmap',
                        transform=ax4.transAxes, ha='center', va='center')
                ax4.set_title('RMSE Change Heatmap')

            plt.tight_layout()

            # 保存图像
            plot_path = save_dir / f'{dataset_name}_robustness_analysis.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"单模型可视化结果已保存到: {plot_path}")

    def _create_model_comparison_plots(self, evaluator: RobustnessEvaluator, save_dir: Path):
        """创建多模型对比图表"""
        if len(self.model_configs) < 2:
            logger.info("模型数量少于2个，跳过模型对比图生成")
            return

        comparison_dir = save_dir / "model_comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)

        # 为每个数据集创建模型对比图
        for dataset_name in evaluator.results.keys():
            self._create_dataset_comparison_plot(evaluator, dataset_name, comparison_dir)

    def _create_dataset_comparison_plot(self, evaluator: RobustnessEvaluator, dataset_name: str, save_dir: Path):
        """为特定数据集创建美观的模型对比图"""
        # 收集所有模型的数据
        model_data = {}
        loss_types = {}

        for model_idx, model_config in enumerate(self.model_configs):
            model_name = model_config['name']
            loss_type = model_config['loss_function']['type']
            loss_types[model_name] = loss_type

            # 提取该模型的数据
            data = self._extract_model_data(evaluator, model_idx)
            if dataset_name in data:
                model_data[model_name] = data[dataset_name]

        if not model_data:
            logger.warning(f"没有找到数据集 {dataset_name} 的模型数据")
            return

        # 创建专业的配色方案
        loss_color_map = {
            'L2': '#E74C3C',      # 红色 - 传统方法
            'L1': '#F39C12',      # 橙色 - 传统方法
            'Huber': '#3498DB',   # 蓝色 - 鲁棒方法
            'HPL': '#27AE60',     # 绿色 - 我们的方法（突出显示）
            'MAE': '#9B59B6',     # 紫色
            'MSE': '#E67E22'      # 深橙色
        }

        loss_marker_map = {
            'L2': 'o',      # 圆形
            'L1': 's',      # 方形
            'Huber': '^',   # 三角形
            'HPL': 'D',     # 钻石形（突出显示）
            'MAE': 'v',     # 倒三角
            'MSE': '<'      # 左三角
        }

        # 创建美观的对比图
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        fig.patch.set_facecolor('white')

        # 主标题 - 使用更专业的样式
        fig.suptitle(f'Loss Function Robustness Comparison\nDataset: {dataset_name}',
                    fontsize=18, fontweight='bold', y=0.95,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.3))

        # 1. RMSE鲁棒性对比 - 高斯噪声
        ax1 = axes[0, 0]
        for model_name, data in model_data.items():
            loss_type = loss_types[model_name]

            # 按高斯噪声绘制
            gaussian_mask = [nt == 'gaussian' for nt in data['noise_types']]
            if any(gaussian_mask):
                gaussian_strengths = [s for s, m in zip(data['strengths'], gaussian_mask) if m]
                gaussian_rmse = [r for r, m in zip(data['rmse_changes'], gaussian_mask) if m]

                # 排序以便绘制连续线条
                sorted_data = sorted(zip(gaussian_strengths, gaussian_rmse))
                strengths_sorted, rmse_sorted = zip(*sorted_data) if sorted_data else ([], [])

                # 使用专业配色
                color = loss_color_map.get(loss_type, '#7F8C8D')
                marker = loss_marker_map.get(loss_type, 'o')

                # HPL使用更粗的线条突出显示
                linewidth = 3.5 if loss_type == 'HPL' else 2.5
                markersize = 10 if loss_type == 'HPL' else 8
                alpha = 1.0 if loss_type == 'HPL' else 0.8

                ax1.plot(strengths_sorted, rmse_sorted,
                        color=color, marker=marker,
                        linewidth=linewidth, markersize=markersize,
                        alpha=alpha, label=f'{loss_type} Loss',
                        markeredgecolor='white', markeredgewidth=1)

        # 美化坐标轴
        ax1.set_xlabel('Gaussian Noise Strength', fontsize=12, fontweight='bold')
        ax1.set_ylabel('RMSE Relative Change (%)', fontsize=12, fontweight='bold')
        ax1.set_title('RMSE Robustness vs Noise Strength', fontsize=14, fontweight='bold', pad=20)
        ax1.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # 2. MAE鲁棒性对比 - 高斯噪声
        ax2 = axes[0, 1]
        for model_name, data in model_data.items():
            loss_type = loss_types[model_name]

            gaussian_mask = [nt == 'gaussian' for nt in data['noise_types']]
            if any(gaussian_mask):
                gaussian_strengths = [s for s, m in zip(data['strengths'], gaussian_mask) if m]
                gaussian_mae = [r for r, m in zip(data['mae_changes'], gaussian_mask) if m]

                sorted_data = sorted(zip(gaussian_strengths, gaussian_mae))
                strengths_sorted, mae_sorted = zip(*sorted_data) if sorted_data else ([], [])

                # 使用专业配色
                color = loss_color_map.get(loss_type, '#7F8C8D')
                marker = loss_marker_map.get(loss_type, 'o')

                # HPL使用更粗的线条突出显示
                linewidth = 3.5 if loss_type == 'HPL' else 2.5
                markersize = 10 if loss_type == 'HPL' else 8
                alpha = 1.0 if loss_type == 'HPL' else 0.8

                ax2.plot(strengths_sorted, mae_sorted,
                        color=color, marker=marker,
                        linewidth=linewidth, markersize=markersize,
                        alpha=alpha, label=f'{loss_type} Loss',
                        markeredgecolor='white', markeredgewidth=1)

        # 美化坐标轴
        ax2.set_xlabel('Gaussian Noise Strength', fontsize=12, fontweight='bold')
        ax2.set_ylabel('MAE Relative Change (%)', fontsize=12, fontweight='bold')
        ax2.set_title('MAE Robustness vs Noise Strength', fontsize=14, fontweight='bold', pad=20)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # 3. 整体鲁棒性对比 - 美观的柱状图
        ax3 = axes[1, 0]
        loss_names = []
        avg_rmse_changes = []
        avg_mae_changes = []
        bar_colors = []

        for model_name, data in model_data.items():
            loss_type = loss_types[model_name]
            loss_names.append(loss_type)
            avg_rmse_changes.append(np.mean([abs(r) for r in data['rmse_changes']]))
            avg_mae_changes.append(np.mean([abs(r) for r in data['mae_changes']]))
            bar_colors.append(loss_color_map.get(loss_type, '#7F8C8D'))

        x_pos = np.arange(len(loss_names))
        width = 0.35

        # 创建渐变效果的柱状图
        bars1 = ax3.bar(x_pos - width/2, avg_rmse_changes, width,
                       label='RMSE Change (%)', alpha=0.8,
                       color=[c for c in bar_colors], edgecolor='white', linewidth=1.5)
        bars2 = ax3.bar(x_pos + width/2, avg_mae_changes, width,
                       label='MAE Change (%)', alpha=0.6,
                       color=[c for c in bar_colors], edgecolor='white', linewidth=1.5)

        # 添加数值标签
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax3.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.1,
                    f'{height1:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax3.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.1,
                    f'{height2:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # 美化坐标轴
        ax3.set_xlabel('Loss Function', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Average Performance Change (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Overall Robustness Comparison', fontsize=14, fontweight='bold', pad=20)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(loss_names, fontsize=11, fontweight='bold')
        ax3.legend(frameon=True, fancybox=True, shadow=True, fontsize=10)
        ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # 4. 鲁棒性排名 - 专业的排名图
        ax4 = axes[1, 1]

        # 计算综合鲁棒性得分（越小越好）
        robustness_scores = []
        for model_name, data in model_data.items():
            loss_type = loss_types[model_name]
            # 使用RMSE和MAE变化的平均绝对值作为鲁棒性得分
            rmse_score = np.mean([abs(r) for r in data['rmse_changes']])
            mae_score = np.mean([abs(r) for r in data['mae_changes']])
            combined_score = (rmse_score + mae_score) / 2
            robustness_scores.append((loss_type, combined_score))

        # 按得分排序（越小越好）
        robustness_scores.sort(key=lambda x: x[1])

        loss_names, scores = zip(*robustness_scores)
        y_pos = np.arange(len(loss_names))

        # 创建渐变色彩的水平柱状图
        colors_for_ranking = []
        for i, (_, _) in enumerate(robustness_scores):
            if i == 0:  # 最佳
                colors_for_ranking.append('#27AE60')  # 绿色
            elif i == 1:  # 第二
                colors_for_ranking.append('#3498DB')  # 蓝色
            elif i == 2:  # 第三
                colors_for_ranking.append('#F39C12')  # 橙色
            else:  # 其他
                colors_for_ranking.append('#E74C3C')  # 红色

        ax4.barh(y_pos, scores, alpha=0.8, color=colors_for_ranking,
                edgecolor='white', linewidth=2)

        # 添加排名标记
        rank_labels = ['🥇', '🥈', '🥉'] + [f'{i+1}' for i in range(3, len(loss_names))]

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels([f'{rank} {name}' for rank, name in zip(rank_labels, loss_names)],
                           fontsize=11, fontweight='bold')
        ax4.set_xlabel('Robustness Score (lower = better)', fontsize=12, fontweight='bold')
        ax4.set_title('Loss Function Robustness Ranking', fontsize=14, fontweight='bold', pad=20)
        ax4.grid(True, alpha=0.3, linestyle='--', axis='x')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # 添加精美的数值标签
        for i, (_, score) in enumerate(robustness_scores):
            ax4.text(score + max(scores) * 0.02, i, f'{score:.2f}',
                    va='center', ha='left', fontweight='bold', fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        plt.tight_layout()

        # 保存图像
        plot_path = save_dir / f'{dataset_name}_model_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"模型对比图已保存到: {plot_path}")

        # 创建额外的总结图
        self._create_summary_robustness_plot(evaluator, dataset_name, save_dir)

    def _create_summary_robustness_plot(self, evaluator: RobustnessEvaluator, dataset_name: str, save_dir: Path):
        """创建简洁美观的鲁棒性总结图"""
        # 收集数据
        model_data = {}
        loss_types = {}

        for model_idx, model_config in enumerate(self.model_configs):
            model_name = model_config['name']
            loss_type = model_config['loss_function']['type']
            loss_types[model_name] = loss_type

            data = self._extract_model_data(evaluator, model_idx)
            if dataset_name in data:
                model_data[model_name] = data[dataset_name]

        if not model_data:
            return

        # 创建专业的单图总结
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.patch.set_facecolor('white')

        # 配色方案
        loss_color_map = {
            'L2': '#E74C3C', 'L1': '#F39C12', 'Huber': '#3498DB',
            'HPL': '#27AE60', 'MAE': '#9B59B6', 'MSE': '#E67E22'
        }
        loss_marker_map = {
            'L2': 'o', 'L1': 's', 'Huber': '^',
            'HPL': 'D', 'MAE': 'v', 'MSE': '<'
        }

        # 绘制高斯噪声下的RMSE变化
        for model_name, data in model_data.items():
            loss_type = loss_types[model_name]

            gaussian_mask = [nt == 'gaussian' for nt in data['noise_types']]
            if any(gaussian_mask):
                gaussian_strengths = [s for s, m in zip(data['strengths'], gaussian_mask) if m]
                gaussian_rmse = [r for r, m in zip(data['rmse_changes'], gaussian_mask) if m]

                sorted_data = sorted(zip(gaussian_strengths, gaussian_rmse))
                strengths_sorted, rmse_sorted = zip(*sorted_data) if sorted_data else ([], [])

                color = loss_color_map.get(loss_type, '#7F8C8D')
                marker = loss_marker_map.get(loss_type, 'o')

                # HPL特殊处理
                if loss_type == 'HPL':
                    ax.plot(strengths_sorted, rmse_sorted, color=color, marker=marker,
                           linewidth=4, markersize=12, alpha=1.0, label=f'{loss_type} (Proposed)',
                           markeredgecolor='white', markeredgewidth=2, zorder=10)
                else:
                    ax.plot(strengths_sorted, rmse_sorted, color=color, marker=marker,
                           linewidth=2.5, markersize=8, alpha=0.8, label=f'{loss_type}',
                           markeredgecolor='white', markeredgewidth=1)

        # 美化图表
        ax.set_xlabel('Gaussian Noise Strength', fontsize=14, fontweight='bold')
        ax.set_ylabel('RMSE Relative Change (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Loss Function Robustness Comparison\n{dataset_name} Dataset',
                    fontsize=16, fontweight='bold', pad=25)

        # 图例美化
        legend = ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12,
                          loc='upper left', bbox_to_anchor=(0.02, 0.98))
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        # 网格和边框
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        # 添加性能说明文本
        textstr = 'Lower curves indicate\nbetter robustness'
        props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
        ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)

        plt.tight_layout()

        # 保存总结图
        summary_path = save_dir / f'{dataset_name}_robustness_summary.png'
        plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        logger.info(f"鲁棒性总结图已保存到: {summary_path}")


def run_demo_experiment():
    """运行演示实验"""
    logger.info("开始演示实验...")

    # 基础配置
    base_config = {
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'center_data': True,
        'ensure_user_in_train': True
    }

    # 简单的噪声实验配置
    noise_exp_config = {
        'noise_types': ['gaussian', 'uniform'],
        'noise_strengths': [0.1, 0.3, 0.5],
        'noise_ratios': [0.5, 1.0],
        'random_seed': 42
    }

    # 简单的模型配置
    model_config = {
        'name': 'L2_baseline_demo',
        'n_factors': 20,
        'learning_rate': 0.01,
        'n_epochs': 50,
        'use_bias': True,
        'loss_function': {'type': 'L2'},
        'regularizer': {'type': 'L2', 'lambda_reg': 0.01},
        'initializer': {'type': 'Normal', 'mean': 0.0, 'std': 0.01}
    }

    # 创建实验运行器
    runner = ExperimentRunner(base_config)

    # 运行实验
    try:
        runner.run_robustness_experiment(
            dataset_name='MovieLens100K',
            data_path='dataset/20201202M100K_data_all_random.txt',
            noise_experiment_configs=[noise_exp_config],
            model_configs=[model_config],
            save_dir='demo_results'
        )
        logger.info("演示实验完成")
    except Exception as e:
        logger.error(f"演示实验失败: {e}")
        import traceback
        traceback.print_exc()


def run_full_experiment():
    """运行完整实验"""
    try:
        # 导入配置文件
        from noise_injection_config import get_experiment_config

        # 获取完整实验配置
        config = get_experiment_config('comprehensive_robustness')

        # 创建实验运行器
        runner = ExperimentRunner(config['base_config'])

        # 对每个数据集运行实验
        for dataset_name in config['datasets']:
            # 获取数据集路径
            dataset_configs = {
                'movielens100k': 'dataset/20201202M100K_data_all_random.txt',
                'netflix': 'dataset/20201202NetFlix_data_all_random.txt',
                'filmtrust': 'dataset/flimtrust20220604random.txt'
            }

            if dataset_name not in dataset_configs:
                logger.warning(f"跳过未知数据集: {dataset_name}")
                continue

            data_path = dataset_configs[dataset_name]

            runner.run_robustness_experiment(
                dataset_name=dataset_name,
                data_path=data_path,
                noise_experiment_configs=config['noise_experiments'],
                model_configs=config['models'],
                save_dir=f'full_results/{dataset_name}'
            )

        logger.info("完整实验完成")
        send_sms_notification(f"噪声注入鲁棒性实验完成，数据集: {', '.join(config['datasets'])}")

    except ImportError:
        logger.error("无法导入配置文件，请确保 noise_injection_config.py 存在")
        send_sms_notification("噪声注入鲁棒性实验失败: 无法导入配置文件")
    except Exception as e:
        logger.error(f"完整实验失败: {e}")
        import traceback
        traceback.print_exc()
        send_sms_notification(f"噪声注入鲁棒性实验失败: {str(e)[:50]}...")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='噪声注入鲁棒性测试系统')
    parser.add_argument('--mode', type=str, default='demo',
                       choices=['demo', 'full'],
                       help='运行模式: demo(演示) 或 full(完整实验)')

    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo_experiment()
    elif args.mode == 'full':
        run_full_experiment()
    else:
        logger.error(f"未知的运行模式: {args.mode}")


if __name__ == "__main__":
    main()

