#!/usr/bin/env python3
"""
完整的矩阵分解训练示例
使用 data、models 和 losses 模块进行端到端训练
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
from pathlib import Path
import time

# 添加项目根目录到Python路径
# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 导入项目模块
from data.data_manager import DataManager
from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.initializers import NormalInitializer, XavierInitializer, UniformInitializer, TruncatedNormalInitializer
from src.models.regularizers import L2Regularizer, L1Regularizer, ElasticNetRegularizer
from src.losses.standard import L1Loss, L2Loss
from src.losses.robust import HuberLoss, LogcoshLoss
from src.losses.hpl import HybridPiecewiseLoss, HPLVariants
from src.losses.sigmoid import SigmoidLikeLoss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MatrixFactorizationTrainer:
    """矩阵分解训练器"""
    
    def __init__(self, config: dict):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.data_manager = None
        self.model = None
        self.results = {}
        
    def prepare_data(self, dataset_name: str, data_path: str):
        """
        准备数据
        
        Args:
            dataset_name: 数据集名称
            data_path: 数据文件路径
        """
        logger.info("开始准备数据...")
        
        # 创建数据管理器
        data_config = {
            'random_seed': self.config['random_seed'],
            'train_ratio': self.config['train_ratio'],
            'val_ratio': self.config['val_ratio'],
            'test_ratio': self.config['test_ratio'],
            'batch_size': self.config['batch_size'],
            'shuffle': True,
            'center_data': self.config['center_data'],
            'ensure_user_in_train': True
        }
        
        self.data_manager = DataManager(data_config)
        
        # 加载和预处理数据
        self.data_manager.load_dataset(dataset_name, data_path)
        self.data_manager.preprocess()
        
        # 打印数据摘要
        self.data_manager.print_summary()
        
        logger.info("数据准备完成")
    
    def create_model(self):
        """创建模型"""
        logger.info("创建模型...")
        
        # 获取数据维度
        stats = self.data_manager.get_statistics()
        n_users = stats['n_users']
        n_items = stats['n_items']
        
        # 创建损失函数
        loss_function = self._create_loss_function()
        
        # 创建正则化器
        regularizer = self._create_regularizer()
        
        # 创建初始化器
        initializer = self._create_initializer()
        
        # 创建模型
        self.model = MatrixFactorizationSGD(
            n_users=n_users,
            n_items=n_items,
            n_factors=self.config['n_factors'],
            learning_rate=self.config['learning_rate'],
            regularizer=regularizer,
            loss_function=loss_function,
            use_bias=self.config['use_bias'],
            global_mean=self.data_manager.global_mean or 0.0,
            clip_gradient=self.config.get('clip_gradient'),
            momentum=self.config.get('momentum', 0.0),
            lr_schedule=self.config.get('lr_schedule')
        )
        
        # 初始化参数
        self.model.initialize_parameters(initializer)
        
        logger.info(f"模型创建完成: {n_users} 用户, {n_items} 物品, {self.config['n_factors']} 因子")
    
    def _create_loss_function(self):
        """创建损失函数"""
        loss_config = self.config['loss_function']
        loss_type = loss_config['type']
        
        if loss_type == 'L2':
            return L2Loss()
        elif loss_type == 'L1':
            return L1Loss()
        elif loss_type == 'Huber':
            return HuberLoss(delta=loss_config.get('delta', 1.0))
        elif loss_type == 'Logcosh':
            return LogcoshLoss()
        elif loss_type == 'HPL':
            return HybridPiecewiseLoss(
                delta1=loss_config.get('delta1', 0.5),
                delta2=loss_config.get('delta2', 2.0),
                l_max=loss_config.get('l_max', 3.0),
                c_sigmoid=loss_config.get('c_sigmoid', 1.0)
            )
        elif loss_type == 'HPL_NoSat':
            return HPLVariants.no_saturation(
                delta1=loss_config.get('delta1', 0.5)
            )
        elif loss_type == 'HPL_NoLin':
            return HPLVariants.no_linear(
                delta1=loss_config.get('delta1', 0.5),
                l_max=loss_config.get('l_max', 3.0),
                c_sigmoid=loss_config.get('c_sigmoid', 1.0)
            )
        elif loss_type == 'SigmoidLike':
            return SigmoidLikeLoss(
                alpha=loss_config.get('alpha', 1.0),
                l_max=loss_config.get('l_max', 3.0)
            )
        else:
            raise ValueError(f"未知的损失函数类型: {loss_type}")
    
    def _create_regularizer(self):
        """创建正则化器"""
        reg_config = self.config['regularizer']
        reg_type = reg_config['type']
        
        if reg_type == 'L2':
            return L2Regularizer(
                lambda_reg=reg_config.get('lambda_reg', 0.01),
                lambda_user=reg_config.get('lambda_user'),
                lambda_item=reg_config.get('lambda_item'),
                lambda_bias=reg_config.get('lambda_bias')
            )
        elif reg_type == 'L1':
            return L1Regularizer(
                lambda_reg=reg_config.get('lambda_reg', 0.01)
            )
        elif reg_type == 'ElasticNet':
            return ElasticNetRegularizer(
                lambda_reg=reg_config.get('lambda_reg', 0.01),
                l1_ratio=reg_config.get('l1_ratio', 0.5)
            )
        else:
            raise ValueError(f"未知的正则化器类型: {reg_type}")
    
    def _create_initializer(self):
        """创建初始化器 - 支持所有初始化类型"""
        init_config = self.config['initializer']
        init_type = init_config['type']
        
        if init_type == 'Normal':
            return NormalInitializer(
                mean=init_config.get('mean', 0.0),
                std=init_config.get('std', 0.01),
                random_seed=self.config['random_seed']
            )
        elif init_type == 'Xavier':
            return XavierInitializer(
                mode=init_config.get('mode', 'fan_avg'),
                random_seed=self.config['random_seed']
            )
        elif init_type == 'Uniform':
            return UniformInitializer(
                low=init_config.get('low', -0.01),
                high=init_config.get('high', 0.01),
                random_seed=self.config['random_seed']
            )
        elif init_type == 'TruncatedNormal':
            return TruncatedNormalInitializer(
                mean=init_config.get('mean', 0.0),
                std=init_config.get('std', 0.01),
                num_std=init_config.get('num_std', 2.0),
                random_seed=self.config['random_seed']
            )
        else:
            raise ValueError(f"未知的初始化器类型: {init_type}。"
                            f"支持的类型: Normal, Xavier, Uniform, TruncatedNormal")
    
    def train(self):
        """训练模型"""
        logger.info("开始训练...")
        
        # 获取数据
        train_data, val_data, test_data = self.data_manager.get_splits()
        
        # 记录训练开始时间
        start_time = time.time()
        
        # 训练模型
        self.model.fit(
            train_data=train_data,
            val_data=val_data,
            n_epochs=self.config['n_epochs'],
            verbose=self.config.get('verbose', 1),
            early_stopping_patience=self.config.get('early_stopping_patience', 10),
            shuffle=True
        )
        
        # 记录训练时间
        training_time = time.time() - start_time
        
        logger.info(f"训练完成，耗时: {training_time:.2f}秒")
        
        # 保存训练结果
        self.results['training_time'] = training_time
        self.results['train_history'] = self.model.train_history
    
    def evaluate(self):
        """评估模型"""
        logger.info("开始评估...")
        
        # 获取测试数据
        _, _, test_data = self.data_manager.get_splits()
        
        # 预测
        test_predictions = self.model.predict(
            test_data[:, 0].astype(int),
            test_data[:, 1].astype(int)
        )
        
        # 如果数据被中心化，需要还原到原始尺度
        if self.data_manager.global_mean is not None:
            test_predictions += self.data_manager.global_mean
            test_targets = test_data[:, 2] + self.data_manager.global_mean
        else:
            test_targets = test_data[:, 2]
        
        # 计算评估指标
        metrics = self._calculate_metrics(test_predictions, test_targets)
        
        # 保存评估结果
        self.results['test_metrics'] = metrics
        
        # 打印结果
        logger.info("评估结果:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
    
    def _calculate_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> dict:
        """计算评估指标"""
        # 基本误差指标
        mae = np.mean(np.abs(predictions - targets))
        mse = np.mean((predictions - targets) ** 2)
        rmse = np.sqrt(mse)
        
        # 计算MAPE（平均绝对百分比误差）
        # 避免除零错误
        non_zero_mask = targets != 0
        if np.any(non_zero_mask):
            mape = np.mean(np.abs((predictions[non_zero_mask] - targets[non_zero_mask]) / targets[non_zero_mask])) * 100
        else:
            mape = float('inf')
        
        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'MAPE': mape
        }
    
    def visualize_results(self, save_path: str = None):
        """可视化结果"""
        logger.info("生成可视化结果...")
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 训练损失曲线
        ax1 = axes[0, 0]
        history = self.results['train_history']
        
        if 'loss' in history and len(history['loss']) > 0:
            ax1.plot(history['loss'], label='Training Loss', color='blue')

        if 'val_loss' in history and len(history['val_loss']) > 0:
            ax1.plot(history['val_loss'], label='Validation Loss', color='red')

        ax1.set_title('Training Loss Curve')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 预测vs真实值散点图
        ax2 = axes[0, 1]
        _, _, test_data = self.data_manager.get_splits()
        
        # 采样部分数据用于可视化
        sample_size = min(1000, len(test_data))
        sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
        sample_data = test_data[sample_indices]
        
        sample_predictions = self.model.predict(
            sample_data[:, 0].astype(int),
            sample_data[:, 1].astype(int)
        )
        
        # 还原到原始尺度
        if self.data_manager.global_mean is not None:
            sample_predictions += self.data_manager.global_mean
            sample_targets = sample_data[:, 2] + self.data_manager.global_mean
        else:
            sample_targets = sample_data[:, 2]
        
        ax2.scatter(sample_targets, sample_predictions, alpha=0.5, s=1)
        
        # 添加理想线
        min_val = min(np.min(sample_targets), np.min(sample_predictions))
        max_val = max(np.max(sample_targets), np.max(sample_predictions))
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        
        ax2.set_title('Predictions vs True Values')
        ax2.set_xlabel('True Values')
        ax2.set_ylabel('Predictions')
        ax2.grid(True, alpha=0.3)

        # 3. 误差分布直方图
        ax3 = axes[1, 0]
        errors = sample_predictions - sample_targets
        ax3.hist(errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_title('Prediction Error Distribution')
        ax3.set_xlabel('Error')
        ax3.set_ylabel('Frequency')
        ax3.grid(True, alpha=0.3)
        
        # 4. 评估指标柱状图
        ax4 = axes[1, 1]
        metrics = self.results['test_metrics']
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax4.bar(metric_names, metric_values, alpha=0.7)
        ax4.set_title('Evaluation Metrics')
        ax4.set_ylabel('Value')
        
        # 在柱子上添加数值标签
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存图表
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化结果已保存至: {save_path}")
        
        plt.show()
    
    def save_results(self, save_dir: str):
        """保存实验结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # 保存结果
        with open(save_path / 'results.json', 'w') as f:
            # 转换numpy数组和numpy数据类型为Python原生类型以便JSON序列化
            def convert_numpy_types(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.floating)):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj

            serializable_results = convert_numpy_types(self.results)
            json.dump(serializable_results, f, indent=2)
        
        # 保存模型
        self.model.save_model(str(save_path / 'model.npz'))
        
        # 保存数据统计
        stats = self.data_manager.get_statistics()
        with open(save_path / 'data_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"实验结果已保存至: {save_path}")


def create_example_config():
    """创建示例配置"""
    return {
        # 数据配置
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'center_data': True,
        
        # 模型配置
        'n_factors': 50,
        'use_bias': True,
        
        # 训练配置
        'n_epochs': 100,
        'batch_size': 128,
        'learning_rate': 0.01,
        'clip_gradient': 5.0,
        'momentum': 0.9,
        'lr_schedule': 'exponential',
        'early_stopping_patience': 10,
        'verbose': 1,
        
        # 损失函数配置
        'loss_function': {
            'type': 'HPL',  # 使用HPL损失函数
            'delta1': 0.5,
            'delta2': 2.0,
            'l_max': 3.0,
            'c_sigmoid': 1.0
        },
        
        # 正则化配置
        'regularizer': {
            'type': 'L2',
            'lambda_reg': 0.01,
            'lambda_bias': 0.001
        },
        
        # 初始化配置
        'initializer': {
            'type': 'Normal',
            'mean': 0.0,
            'std': 0.01
        }
    }


def run_initializer_comparison_experiment():
    """运行初始化器对比实验"""
    logger.info("开始初始化器对比实验...")
    
    # 基础配置
    base_config = create_example_config()
    base_config['n_epochs'] = 30  # 减少训练轮数以加快实验
    
    # 选择代表性的初始化器进行对比
    from configs.initializer_configs import get_initializer_configs
    
    # 获取所有预定义的初始化器配置
    all_initializer_configs = get_initializer_configs()
    
    # 使用所有配置进行测试
    test_initializers = {name.replace('_', ' ').title(): config 
                         for name, config in all_initializer_configs.items()}
    
    results = {}
    
    for init_name, init_config in test_initializers.items():
        logger.info(f"训练 {init_name} 初始化模型...")
        
        # 创建配置
        config = base_config.copy()
        config['initializer'] = init_config
        
        # 创建训练器
        trainer = MatrixFactorizationTrainer(config)
        
        try:
            # 准备数据（第一次运行时）
            if 'data_prepared_init' not in globals():
                trainer.prepare_data(
                    'movielens100k',
                    'dataset/20201202M100K_data_all_random.txt'
                )
                global data_prepared_init
                data_prepared_init = True
                global shared_data_manager_init
                shared_data_manager_init = trainer.data_manager
            else:
                # 复用已准备的数据
                trainer.data_manager = shared_data_manager_init
            
            # 创建模型并训练
            trainer.create_model()
            trainer.train()
            trainer.evaluate()
            
            # 保存结果
            results[init_name] = {
                'test_metrics': trainer.results['test_metrics'],
                'final_train_loss': trainer.results['train_history']['loss'][-1],
                'final_val_loss': trainer.results['train_history']['val_loss'][-1] if trainer.results['train_history']['val_loss'] else None,
                'training_time': trainer.results['training_time'],
                'convergence_epochs': len(trainer.results['train_history']['loss'])
            }
            
            # 检查是否出现NaN
            if np.isnan(trainer.results['test_metrics']['RMSE']):
                results[init_name]['status'] = 'NaN_detected'
            else:
                results[init_name]['status'] = 'success'
            
        except Exception as e:
            logger.error(f"{init_name} 训练失败: {e}")
            results[init_name] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # 打印对比结果
    print("\n" + "="*100)
    print("初始化器对比实验结果")
    print("="*100)
    
    print(f"{'初始化器':<20} {'状态':<10} {'RMSE':<8} {'MAE':<8} {'训练时间(s)':<12} {'收敛轮数':<10} {'最终损失':<10}")
    print("-"*100)
    
    # 按RMSE排序显示结果
    successful_results = {k: v for k, v in results.items() if v.get('status') == 'success'}
    if successful_results:
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['test_metrics']['RMSE'])
        
        for init_name, result in sorted_results:
            metrics = result['test_metrics']
            print(f"{init_name:<20} "
                  f"{'SUCCESS':<10} "
                  f"{metrics['RMSE']:<8.4f} "
                  f"{metrics['MAE']:<8.4f} "
                  f"{result['training_time']:<12.2f} "
                  f"{result['convergence_epochs']:<10} "
                  f"{result['final_train_loss']:<10.4f}")
    
    # 显示失败的实验
    failed_results = {k: v for k, v in results.items() if v.get('status') != 'success'}
    for init_name, result in failed_results.items():
        status = result.get('status', 'UNKNOWN')
        error_msg = result.get('error', 'Unknown error')[:30]
        print(f"{init_name:<20} "
              f"{status:<10} "
              f"{'ERROR':<8} "
              f"{'ERROR':<8} "
              f"{'ERROR':<12} "
              f"{'ERROR':<10} "
              f"{'ERROR':<10}")
        print(f"    错误信息: {error_msg}")
    
    print("="*100)
    
    # 分析和建议
    if successful_results:
        best_init = min(successful_results.items(), key=lambda x: x[1]['test_metrics']['RMSE'])
        fastest_init = min(successful_results.items(), key=lambda x: x[1]['training_time'])
        
        print(f"\n最佳性能: {best_init[0]} (RMSE: {best_init[1]['test_metrics']['RMSE']:.4f})")
        print(f"最快训练: {fastest_init[0]} (时间: {fastest_init[1]['training_time']:.2f}s)")
    
    return results


def run_experiment_comparison():
    """运行损失函数对比实验"""
    logger.info("开始损失函数对比实验...")
    
    # 基础配置
    base_config = create_example_config()
    base_config['n_epochs'] = 50  # 减少训练轮数以加快实验
    
    # 不同损失函数配置
    loss_configs = {
        'L2': {'type': 'L2'},
        'L1': {'type': 'L1'},
        'Huber': {'type': 'Huber', 'delta': 1.0},
        'HPL': {
            'type': 'HPL',
            'delta1': 0.5,
            'delta2': 2.0,
            'l_max': 3.0,
            'c_sigmoid': 1.0
        },
        'HPL_NoSat': {
            'type': 'HPL_NoSat',
            'delta1': 0.5
        },
        'SigmoidLike': {
            'type': 'SigmoidLike',
            'alpha': 1.0,
            'l_max': 3.0
        }
    }
    
    results = {}
    
    for loss_name, loss_config in loss_configs.items():
        logger.info(f"训练 {loss_name} 模型...")
        
        # 创建配置
        config = base_config.copy()
        config['loss_function'] = loss_config
        
        # 创建训练器
        trainer = MatrixFactorizationTrainer(config)
        
        try:
            # 准备数据（第一次运行时）
            if 'data_prepared' not in globals():
                trainer.prepare_data(
                    'movielens100k',
                    'dataset/20201202M100K_data_all_random.txt'
                )
                global data_prepared
                data_prepared = True
                global shared_data_manager
                shared_data_manager = trainer.data_manager
            else:
                # 复用已准备的数据
                trainer.data_manager = shared_data_manager
            
            # 创建模型并训练
            trainer.create_model()
            trainer.train()
            trainer.evaluate()
            
            # 保存结果
            results[loss_name] = {
                'test_metrics': trainer.results['test_metrics'],
                'final_train_loss': trainer.results['train_history']['loss'][-1],
                'final_val_loss': trainer.results['train_history']['val_loss'][-1] if trainer.results['train_history']['val_loss'] else None,
                'training_time': trainer.results['training_time']
            }
            
        except Exception as e:
            logger.error(f"{loss_name} 训练失败: {e}")
            results[loss_name] = {'error': str(e)}
    
    # 打印对比结果
    print("\n" + "="*80)
    print("损失函数对比实验结果")
    print("="*80)
    
    print(f"{'损失函数':<15} {'RMSE':<8} {'MAE':<8} {'训练时间(s)':<12} {'训练损失':<10} {'验证损失':<10}")
    print("-"*80)
    
    for loss_name, result in results.items():
        if 'error' in result:
            print(f"{loss_name:<15} {'ERROR':<8} {'ERROR':<8} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10}")
        else:
            metrics = result['test_metrics']
            print(f"{loss_name:<15} "
                  f"{metrics['RMSE']:<8.4f} "
                  f"{metrics['MAE']:<8.4f} "
                  f"{result['training_time']:<12.2f} "
                  f"{result['final_train_loss']:<10.4f} "
                  f"{result['final_val_loss'] or 'N/A':<10}")
    
    print("="*80)


def main():
    """主函数 - 单个模型训练示例"""
    logger.info("开始矩阵分解训练示例...")
    
    # 创建配置
    config = create_example_config()
    
    # 创建训练器
    trainer = MatrixFactorizationTrainer(config)
    
    try:
        # 1. 准备数据
        trainer.prepare_data(
            dataset_name='movielens100k',
            data_path='dataset/20201202M100K_data_all_random.txt'
        )
        
        # 2. 创建模型
        trainer.create_model()
        
        # 3. 训练模型
        trainer.train()
        
        # 4. 评估模型
        trainer.evaluate()
        
        # 5. 可视化结果
        trainer.visualize_results('results/training_results.png')
        
        # 6. 保存结果
        trainer.save_results('results/experiment_1')
        
        logger.info("训练完成！")
        
    except Exception as e:
        logger.error(f"训练失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    # 选择运行模式
    import argparse
    
    parser = argparse.ArgumentParser(description='矩阵分解训练示例')
    parser.add_argument('--mode', choices=['single', 'comparison', 'initializer_comparison'], 
                       default='initializer_comparison', help='运行模式')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        main()
    elif args.mode == 'comparison':
        run_experiment_comparison()
    elif args.mode == 'initializer_comparison':
        run_initializer_comparison_experiment()

