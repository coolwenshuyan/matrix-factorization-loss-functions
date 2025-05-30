# src/training/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path
import seaborn as sns
from datetime import datetime


def plot_training_history(history: Dict, save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 10)):
    """
    绘制训练历史
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
        figsize: 图像大小
    """
    # 确定需要绘制的子图数量
    n_metrics = len([k for k in history.keys() if k != 'learning_rates'])
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.ravel() if n_metrics > 1 else [axes]
    
    # 绘制每个指标
    metric_idx = 0
    for key, values in history.items():
        if key == 'learning_rates':
            continue
            
        ax = axes[metric_idx]
        epochs = range(1, len(values) + 1)
        
        # 绘制曲线
        ax.plot(epochs, values, 'b-', label=key, linewidth=2)
        
        # 如果有对应的验证指标，一起绘制
        val_key = key.replace('train_', 'val_')
        if val_key in history and val_key != key:
            val_values = history[val_key]
            ax.plot(range(1, len(val_values) + 1), val_values, 
                   'r--', label=val_key, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel(key.replace('_', ' ').title())
        ax.set_title(f'{key.replace("_", " ").title()} History')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        metric_idx += 1
    
    # 隐藏多余的子图
    for idx in range(metric_idx, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_learning_rate_schedule(history: Dict, save_path: Optional[str] = None):
    """
    绘制学习率变化曲线
    
    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    if 'learning_rates' not in history:
        print("历史记录中没有学习率信息")
        return
    
    lrs = history['learning_rates']
    epochs = range(1, len(lrs) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lrs, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def save_training_summary(trainer, save_dir: str):
    """
    保存训练总结
    
    Args:
        trainer: 训练器对象
        save_dir: 保存目录
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 保存训练配置
    config = {
        'model': trainer.model.__class__.__name__,
        'optimizer': trainer.optimizer.__class__.__name__,
        'loss_function': trainer.loss_function.__class__.__name__ if hasattr(trainer.loss_function, '__class__') else str(trainer.loss_function),
        'learning_rate': trainer.optimizer.get_learning_rate(),
        'n_epochs': trainer.current_epoch,
        'metrics': list(trainer.metrics.keys())
    }
    
    with open(save_path / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # 保存训练历史
    with open(save_path / 'history.json', 'w') as f:
        json.dump(trainer.training_history, f, indent=2)
    
    # 生成训练报告
    report = generate_training_report(trainer)
    with open(save_path / 'report.txt', 'w') as f:
        f.write(report)
    
    # 绘制并保存图表
    plot_training_history(
        trainer.training_history,
        save_path=save_path / 'training_history.png'
    )
    
    plot_learning_rate_schedule(
        trainer.training_history,
        save_path=save_path / 'learning_rate.png'
    )


def generate_training_report(trainer) -> str:
    """
    生成训练报告
    
    Args:
        trainer: 训练器对象
        
    Returns:
        报告字符串
    """
    report = []
    report.append("="*50)
    report.append("训练报告")
    report.append("="*50)
    report.append("")

    # 基本信息
    report.append("基本信息:")
    report.append(f"  - 模型: {trainer.model.__class__.__name__}")
    report.append(f"  - 优化器: {trainer.optimizer.__class__.__name__}")
    report.append(f"  - 损失函数: {trainer.loss_function.__class__.__name__ if hasattr(trainer.loss_function, '__class__') else str(trainer.loss_function)}")
    report.append(f"  - 训练轮数: {trainer.current_epoch}")
    report.append("")

    # 最终性能
    report.append("最终性能:")
    history = trainer.training_history

    if history['train_loss']:
        final_train_loss = history['train_loss'][-1]
        report.append(f"  - 训练损失: {final_train_loss:.4f}")

    if history['val_loss']:
        final_val_loss = history['val_loss'][-1]
        report.append(f"  - 验证损失: {final_val_loss:.4f}")

    for metric_name, metric_values in history['metrics'].items():
        if metric_values:
            final_value = metric_values[-1]
            report.append(f"  - {metric_name}: {final_value:.4f}")

    report.append("")

    # 最佳性能
    if history['val_loss']:
        best_epoch = np.argmin(history['val_loss'])
        best_val_loss = history['val_loss'][best_epoch]
        report.append("最佳性能:")
        report.append(f"  - 最佳epoch: {best_epoch + 1}")
        report.append(f"  - 最佳验证损失: {best_val_loss:.4f}")
        
        for metric_name, metric_values in history['metrics'].items():
            if metric_values and len(metric_values) > best_epoch:
                best_value = metric_values[best_epoch]
                report.append(f"  - 最佳{metric_name}: {best_value:.4f}")

    report.append("")
    report.append("="*50)

    return "\n".join(report)


def create_experiment_summary(experiment_results: List[Dict],
                           save_path: Optional[str] = None) -> pd.DataFrame:
   """
   创建实验总结表格
   
   Args:
       experiment_results: 实验结果列表
       save_path: 保存路径
       
   Returns:
       总结DataFrame
   """
   import pandas as pd
   
   # 创建总结数据
   summary_data = []
   
   for result in experiment_results:
       row = {
           'experiment_name': result['name'],
           'model': result['model'],
           'loss_function': result['loss_function'],
           'final_train_loss': result['final_train_loss'],
           'final_val_loss': result['final_val_loss'],
           'best_val_loss': result['best_val_loss'],
           'best_epoch': result['best_epoch'],
           'total_epochs': result['total_epochs'],
           'training_time': result['training_time']
       }
       
       # 添加其他指标
       for metric_name, metric_value in result.get('final_metrics', {}).items():
           row[f'final_{metric_name}'] = metric_value
       
       summary_data.append(row)
   
   # 创建DataFrame
   df = pd.DataFrame(summary_data)
   
   # 排序（按验证损失）
   if 'best_val_loss' in df.columns:
       df = df.sort_values('best_val_loss')
   
   # 保存
   if save_path:
       df.to_csv(save_path, index=False)
   
   return df


def analyze_gradient_flow(model, save_path: Optional[str] = None):
   """
   分析梯度流
   
   Args:
       model: 模型对象
       save_path: 保存路径
   """
   # 获取所有参数的梯度范数
   grad_norms = {}
   
   params = model.get_parameters()
   for name, param in params.items():
       if hasattr(param, 'grad') and param.grad is not None:
           grad_norms[name] = np.linalg.norm(param.grad)
   
   if not grad_norms:
       print("没有找到梯度信息")
       return
   
   # 绘制梯度范数
   plt.figure(figsize=(12, 6))
   
   names = list(grad_norms.keys())
   values = list(grad_norms.values())
   
   plt.bar(range(len(names)), values)
   plt.xticks(range(len(names)), names, rotation=45)
   plt.xlabel('Parameter')
   plt.ylabel('Gradient Norm')
   plt.title('Gradient Flow Analysis')
   plt.yscale('log')
   plt.grid(True, alpha=0.3)
   
   plt.tight_layout()
   
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   plt.show()


def compare_experiments(experiment_dirs: List[str], 
                      metric: str = 'val_loss',
                      save_path: Optional[str] = None):
   """
   比较多个实验的结果
   
   Args:
       experiment_dirs: 实验目录列表
       metric: 要比较的指标
       save_path: 保存路径
   """
   plt.figure(figsize=(12, 8))
   
   for exp_dir in experiment_dirs:
       exp_path = Path(exp_dir)
       
       # 加载历史记录
       history_file = exp_path / 'history.json'
       if history_file.exists():
           with open(history_file, 'r') as f:
               history = json.load(f)
           
           # 获取指标值
           if metric in history:
               values = history[metric]
           elif metric in history.get('metrics', {}):
               values = history['metrics'][metric]
           else:
               print(f"指标 {metric} 在 {exp_dir} 中未找到")
               continue
           
           # 绘制曲线
           epochs = range(1, len(values) + 1)
           plt.plot(epochs, values, label=exp_path.name, linewidth=2)
   
   plt.xlabel('Epoch')
   plt.ylabel(metric.replace('_', ' ').title())
   plt.title(f'{metric.replace("_", " ").title()} Comparison')
   plt.legend()
   plt.grid(True, alpha=0.3)
   
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   plt.show()


def calculate_model_size(model) -> Dict[str, int]:
   """
   计算模型大小
   
   Args:
       model: 模型对象
       
   Returns:
       模型大小信息
   """
   total_params = 0
   param_counts = {}
   
   params = model.get_parameters()
   for name, param in params.items():
       count = param.size
       param_counts[name] = count
       total_params += count
   
   # 计算内存占用（假设float32）
   memory_mb = (total_params * 4) / (1024 * 1024)
   
   return {
       'total_parameters': total_params,
       'parameter_counts': param_counts,
       'memory_mb': memory_mb
   }


def profile_training_speed(trainer, n_batches: int = 100) -> Dict[str, float]:
   """
   分析训练速度
   
   Args:
       trainer: 训练器对象
       n_batches: 测试的批次数
       
   Returns:
       性能统计
   """
   import time
   
   # 记录不同阶段的时间
   forward_times = []
   backward_times = []
   update_times = []
   total_times = []
   
   # 测试n个批次
   batch_count = 0
   for batch_data in trainer.train_loader:
       if batch_count >= n_batches:
           break
       
       # 总时间
       total_start = time.time()
       
       # 前向传播时间
       forward_start = time.time()
       user_ids, item_ids, ratings = batch_data
       predictions = trainer.model.predict(user_ids, item_ids)
       loss = trainer.loss_function.forward(predictions, ratings)
       forward_time = time.time() - forward_start
       
       # 反向传播时间
       backward_start = time.time()
       loss_grad = trainer.loss_function.gradient(predictions, ratings)
       backward_time = time.time() - backward_start
       
       # 参数更新时间
       update_start = time.time()
       for i in range(len(user_ids)):
           trainer.model.sgd_update(
               user_ids[i], 
               item_ids[i], 
               ratings[i],
               0
           )
       update_time = time.time() - update_start
       
       total_time = time.time() - total_start
       
       # 记录时间
       forward_times.append(forward_time)
       backward_times.append(backward_time)
       update_times.append(update_time)
       total_times.append(total_time)
       
       batch_count += 1
   
   # 计算统计信息
   stats = {
       'forward_mean': np.mean(forward_times),
       'forward_std': np.std(forward_times),
       'backward_mean': np.mean(backward_times),
       'backward_std': np.std(backward_times),
       'update_mean': np.mean(update_times),
       'update_std': np.std(update_times),
       'total_mean': np.mean(total_times),
       'total_std': np.std(total_times),
       'samples_per_second': len(user_ids) / np.mean(total_times)
   }
   
   return stats


class ExperimentTracker:
   """实验追踪器"""
   
   def __init__(self, base_dir: str = './experiments'):
       """
       初始化实验追踪器
       
       Args:
           base_dir: 基础目录
       """
       self.base_dir = Path(base_dir)
       self.base_dir.mkdir(parents=True, exist_ok=True)
       self.experiments = []
   
   def create_experiment(self, name: str, config: Dict) -> str:
       """
       创建新实验
       
       Args:
           name: 实验名称
           config: 实验配置
           
       Returns:
           实验目录路径
       """
       # 创建实验目录
       timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
       exp_dir = self.base_dir / f"{name}_{timestamp}"
       exp_dir.mkdir(parents=True, exist_ok=True)
       
       # 保存配置
       with open(exp_dir / 'config.json', 'w') as f:
           json.dump(config, f, indent=2)
       
       # 记录实验
       self.experiments.append({
           'name': name,
           'timestamp': timestamp,
           'path': str(exp_dir),
           'config': config
       })
       
       return str(exp_dir)
   
   def log_metric(self, exp_dir: str, metric_name: str, 
                  value: float, step: int):
       """
       记录指标
       
       Args:
           exp_dir: 实验目录
           metric_name: 指标名称
           value: 指标值
           step: 步数
       """
       exp_path = Path(exp_dir)
       metrics_file = exp_path / 'metrics.json'
       
       # 加载现有指标
       if metrics_file.exists():
           with open(metrics_file, 'r') as f:
               metrics = json.load(f)
       else:
           metrics = {}
       
       # 添加新指标
       if metric_name not in metrics:
           metrics[metric_name] = []
       
       metrics[metric_name].append({
           'step': step,
           'value': value,
           'timestamp': datetime.now().isoformat()
       })
       
       # 保存
       with open(metrics_file, 'w') as f:
           json.dump(metrics, f, indent=2)
   
   def get_best_experiment(self, metric: str = 'val_loss',
                          mode: str = 'min') -> Dict:
       """
       获取最佳实验
       
       Args:
           metric: 评估指标
           mode: 'min'或'max'
           
       Returns:
           最佳实验信息
       """
       best_exp = None
       best_value = float('inf') if mode == 'min' else float('-inf')
       
       for exp in self.experiments:
           # 加载指标
           metrics_file = Path(exp['path']) / 'metrics.json'
           if metrics_file.exists():
               with open(metrics_file, 'r') as f:
                   metrics = json.load(f)
               
               if metric in metrics and metrics[metric]:
                   # 获取最后的值
                   last_value = metrics[metric][-1]['value']
                   
                   # 更新最佳值
                   if mode == 'min' and last_value < best_value:
                       best_value = last_value
                       best_exp = exp
                   elif mode == 'max' and last_value > best_value:
                       best_value = last_value
                       best_exp = exp
       
       return best_exp
