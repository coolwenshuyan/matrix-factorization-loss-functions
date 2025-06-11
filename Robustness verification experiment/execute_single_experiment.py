#!/usr/bin/env python3
"""
单个实验执行模块
负责执行单个数据集、单个模型配置和一组噪声配置的实验
"""

import sys
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
from dataset_selector import DatasetInfo
from data.data_manager import DataManager
from noise_injection_system import NoiseConfig, RobustnessEvaluator
from result_manager import ResultManager

logger = logging.getLogger(__name__)

def execute_single_experiment(
    dataset: DatasetInfo,
    model_config: Dict[str, Any],
    noise_configs: List[NoiseConfig],
    experiment_name: str,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    执行单个实验
    
    Args:
        dataset: 数据集信息
        model_config: 模型配置
        noise_configs: 噪声配置列表
        experiment_name: 实验名称
        save_dir: 保存目录（可选）
        
    Returns:
        实验结果字典
    """
    logger.info(f"开始执行实验: {experiment_name}")
    start_time = time.time()
    
    # 如果没有指定保存目录，则使用默认目录
    if save_dir is None:
        save_dir = str(current_dir / "experiment_results" / experiment_name)
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # 准备数据
        logger.info(f"准备数据集: {dataset.name}")
        data_manager = prepare_data(dataset)
        
        # 训练模型
        logger.info(f"训练模型: {model_config['name']}")
        model, training_info = train_model(data_manager, model_config)
        
        # 创建鲁棒性评估器
        evaluator = RobustnessEvaluator(data_manager)
        
        # 评估鲁棒性
        logger.info(f"评估鲁棒性: {len(noise_configs)} 个噪声配置")
        robustness_results = evaluator.evaluate_model_robustness(
            model, noise_configs, experiment_name
        )
        
        # 保存结果
        result_file = save_path / f"{experiment_name}_results.json"
        evaluator.save_results(result_file)
        
        # 生成并保存报告
        report = evaluator.generate_robustness_report()
        report_file = save_path / f"{experiment_name}_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 创建结果管理器并保存结果
        result_manager = ResultManager(str(save_path))
        result_path = result_manager.save_experiment_results(
            {
                'dataset_info': {
                    'name': dataset.name,
                    'display_name': dataset.display_name,
                    'file_path': dataset.file_path
                },
                'model_config': model_config,
                'training_info': training_info,
                'robustness_results': robustness_results,
                'execution_time': time.time() - start_time
            },
            experiment_name,
            dataset.name,
            "robustness"
        )
        
        logger.info(f"实验完成，结果已保存: {result_path}")
        
        return {
            'status': 'success',
            'execution_time': time.time() - start_time,
            'result_path': str(result_path),
            'robustness_summary': robustness_results.get('robustness_metrics', {})
        }
        
    except Exception as e:
        logger.error(f"实验执行失败: {e}", exc_info=True)
        return {
            'status': 'failed',
            'error': str(e),
            'execution_time': time.time() - start_time
        }

def prepare_data(dataset: DatasetInfo) -> DataManager:
    """准备数据"""
    config = {
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'center_data': True,
        'ensure_user_in_train': True
    }
    
    data_manager = DataManager(config)
    data_manager.load_dataset(dataset.name, dataset.file_path).preprocess()
    return data_manager

def train_model(data_manager: DataManager, model_config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """训练模型"""
    from src.models.mf_sgd import MatrixFactorizationSGD
    from src.models.initializers import NormalInitializer, XavierInitializer, UniformInitializer
    from src.models.regularizers import L2Regularizer, L1Regularizer
    from src.losses.standard import L1Loss, L2Loss
    from src.losses.hpl import HybridPiecewiseLoss
    from src.losses.robust import HuberLoss, LogcoshLoss
    from src.losses.sigmoid import SigmoidLikeLoss
    
    start_time = time.time()
    
    try:
        # 获取数据维度
        stats = data_manager.get_statistics()
        n_users = stats['n_users']
        n_items = stats['n_items']
        
        # 创建损失函数
        loss_function = create_loss_function(model_config['loss_function'])
        
        # 创建正则化器
        regularizer = create_regularizer(model_config['regularizer'])
        
        # 创建初始化器
        initializer = create_initializer(model_config['initializer'])
        
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
            verbose=1  # 显示进度
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

def create_loss_function(loss_config: Dict[str, Any]):
    """创建损失函数"""
    from src.losses.standard import L1Loss, L2Loss
    from src.losses.hpl import HybridPiecewiseLoss
    from src.losses.robust import HuberLoss, LogcoshLoss
    from src.losses.sigmoid import SigmoidLikeLoss
    
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

def create_regularizer(reg_config: Dict[str, Any]):
    """创建正则化器"""
    from src.models.regularizers import L2Regularizer, L1Regularizer
    
    reg_type = reg_config['type']
    
    if reg_type == 'L2':
        return L2Regularizer(lambda_reg=reg_config.get('lambda', 0.01))
    elif reg_type == 'L1':
        return L1Regularizer(lambda_reg=reg_config.get('lambda', 0.01))
    else:
        raise ValueError(f"不支持的正则化器: {reg_type}")

def create_initializer(init_config: Dict[str, Any]):
    """创建初始化器"""
    from src.models.initializers import NormalInitializer, XavierInitializer, UniformInitializer
    
    init_type = init_config['type']
    
    if init_type == 'Normal':
        return NormalInitializer(mean=init_config.get('mean', 0.0), std=init_config.get('std', 0.1))
    elif init_type == 'Xavier':
        return XavierInitializer()
    elif init_type == 'Uniform':
        return UniformInitializer(low=init_config.get('low', -0.1), high=init_config.get('high', 0.1))
    else:
        raise ValueError(f"不支持的初始化器: {init_type}")

if __name__ == "__main__":
    # 测试代码
    print("单个实验执行模块")