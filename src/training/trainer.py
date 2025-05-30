# src/training/trainer.py
import numpy as np
import logging
import time
from typing import Optional, Dict, List, Tuple, Any, Callable
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class Trainer:
    """
    统一的训练器类，管理整个训练流程
    """
    
    def __init__(self, 
                 model,
                 optimizer,
                 loss_function,
                 train_data_loader,
                 val_data_loader=None,
                 test_data_loader=None,
                 metrics: Optional[Dict[str, Callable]] = None,
                 callbacks: Optional[List] = None,
                 device: str = 'cpu',
                 log_dir: Optional[str] = None):
        """
        初始化训练器
        
        Args:
            model: 模型实例
            optimizer: 优化器
            loss_function: 损失函数
            train_data_loader: 训练数据加载器
            val_data_loader: 验证数据加载器
            test_data_loader: 测试数据加载器
            metrics: 评估指标字典
            callbacks: 回调函数列表
            device: 设备类型
            log_dir: 日志目录
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.train_loader = train_data_loader
        self.val_loader = val_data_loader
        self.test_loader = test_data_loader
        self.metrics = metrics or {}
        self.callbacks = callbacks or []
        self.device = device
        
        # 设置日志目录
        if log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = f"logs/train_{timestamp}"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': {name: [] for name in self.metrics.keys()},
            'learning_rates': []
        }
        
        # 设置日志
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        log_file = self.log_dir / 'training.log'
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        # 添加到logger
        logger.addHandler(file_handler)
        
    def train(self, 
              n_epochs: int,
              val_frequency: int = 1,
              save_frequency: int = 5,
              resume_from: Optional[str] = None):
        """
        主训练循环
        
        Args:
            n_epochs: 训练轮数
            val_frequency: 验证频率
            save_frequency: 保存频率
            resume_from: 恢复训练的检查点路径
        """
        # 恢复训练状态
        if resume_from:
            self.load_checkpoint(resume_from)
        
        # 训练开始回调
        self._run_callbacks('on_train_begin')
        
        logger.info(f"开始训练，共 {n_epochs} 个epoch")
        start_time = time.time()
        
        try:
            for epoch in range(self.current_epoch, n_epochs):
                self.current_epoch = epoch
                
                # Epoch开始回调
                self._run_callbacks('on_epoch_begin', epoch=epoch)
                
                # 训练一个epoch
                train_metrics = self.train_epoch()
                
                # 记录学习率
                current_lr = self.optimizer.get_learning_rate()
                self.training_history['learning_rates'].append(current_lr)
                
                # 验证
                val_metrics = {}
                if self.val_loader and (epoch + 1) % val_frequency == 0:
                    val_metrics = self.validate()
                
                # 更新历史记录
                self._update_history(train_metrics, val_metrics)
                
                # Epoch结束回调
                self._run_callbacks('on_epoch_end', 
                                  epoch=epoch,
                                  train_metrics=train_metrics,
                                  val_metrics=val_metrics)
                
                # 保存检查点
                if (epoch + 1) % save_frequency == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pt')
                
                # 记录日志
                self._log_epoch_metrics(epoch, train_metrics, val_metrics)
                
        except KeyboardInterrupt:
            logger.info("训练被用户中断")
        except Exception as e:
            logger.error(f"训练过程中发生错误: {str(e)}")
            raise
        finally:
            # 训练结束回调
            self._run_callbacks('on_train_end')
            
            # 保存最终模型
            self.save_checkpoint('final_model.pt')
            
            # 记录训练总时间
            total_time = time.time() - start_time
            logger.info(f"训练完成，总用时: {total_time/3600:.2f} 小时")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            训练指标字典
        """
        self.model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # 批次级别的指标
        batch_metrics = {name: 0.0 for name in self.metrics.keys()}
        
        for batch_idx, batch_data in enumerate(self.train_loader):
            # 批次开始回调
            self._run_callbacks('on_batch_begin', batch=batch_idx)
            
            # 执行训练步骤
            loss, metrics = self.train_step(batch_data)
            
            # 累积损失和指标
            epoch_loss += loss
            for name, value in metrics.items():
                batch_metrics[name] += value
            
            n_batches += 1
            self.global_step += 1
            
            # 批次结束回调
            self._run_callbacks('on_batch_end', 
                              batch=batch_idx,
                              loss=loss,
                              metrics=metrics)
        
        # 计算平均值
        avg_loss = epoch_loss / n_batches
        avg_metrics = {name: value / n_batches 
                      for name, value in batch_metrics.items()}
        
        # 添加损失到指标
        avg_metrics['loss'] = avg_loss
        
        return avg_metrics
    
    def train_step(self, batch_data) -> Tuple[float, Dict[str, float]]:
        """
        单个批次的训练步骤
        
        Args:
            batch_data: 批次数据
            
        Returns:
            损失值和指标字典
        """
        # 解析批次数据
        user_ids, item_ids, ratings = batch_data
        
        # 前向传播
        predictions = self.model.predict(user_ids, item_ids)
        
        # 计算损失
        loss = self.loss_function.forward(predictions, ratings)
        
        # 计算梯度
        loss_grad = self.loss_function.gradient(predictions, ratings)
        
        # 反向传播和参数更新（这里调用模型的SGD更新）
        for i in range(len(user_ids)):
            self.model.sgd_update(
                user_ids[i], 
                item_ids[i], 
                ratings[i],
                self.current_epoch
            )
        
        # 计算指标
        metrics = {}
        for name, metric_fn in self.metrics.items():
            metrics[name] = metric_fn(predictions, ratings)
        
        return float(loss), metrics
    
    def validate(self) -> Dict[str, float]:
        """
        在验证集上评估
        
        Returns:
            验证指标字典
        """
        self.model.eval()
        val_loss = 0.0
        n_batches = 0
        
        # 收集所有预测和真实值
        all_predictions = []
        all_targets = []
        
        with np.errstate(all='ignore'):  # 忽略数值警告
            for batch_data in self.val_loader:
                user_ids, item_ids, ratings = batch_data
                
                # 预测
                predictions = self.model.predict(user_ids, item_ids)
                
                # 计算损失
                loss = self.loss_function.forward(predictions, ratings)
                val_loss += float(loss)
                
                # 收集结果
                all_predictions.extend(predictions)
                all_targets.extend(ratings)
                
                n_batches += 1
        
        # 转换为数组
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # 计算指标
        metrics = {}
        for name, metric_fn in self.metrics.items():
            metrics[name] = metric_fn(all_predictions, all_targets)
        
        # 添加平均损失
        metrics['loss'] = val_loss / n_batches
        
        return metrics
    
    def test(self) -> Dict[str, float]:
        """
        在测试集上评估
        
        Returns:
            测试指标字典
        """
        if self.test_loader is None:
            raise ValueError("未提供测试数据加载器")
        
        # 临时替换验证加载器
        original_val_loader = self.val_loader
        self.val_loader = self.test_loader
        
        # 执行评估
        test_metrics = self.validate()
        
        # 恢复原始验证加载器
        self.val_loader = original_val_loader
        
        return test_metrics
    
    def save_checkpoint(self, filename: str):
        """
        保存训练检查点
        
        Args:
            filename: 保存文件名
        """
        checkpoint_path = self.log_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state': self.model.get_parameters(),
            'optimizer_state': self.optimizer.get_state(),
            'training_history': self.training_history,
            'best_metric': self.best_metric,
            'config': {
                'model_config': self.model.get_config() if hasattr(self.model, 'get_config') else {},
                'optimizer_config': self.optimizer.get_config()
            }
        }
        
        # 保存检查点
        np.savez_compressed(checkpoint_path, **checkpoint)
        
        # 同时保存为JSON格式的元信息
        meta_path = checkpoint_path.with_suffix('.json')
        meta_info = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'training_history': self.training_history,
            'best_metric': self.best_metric
        }
        
        with open(meta_path, 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        logger.info(f"检查点已保存至: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载训练检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch'].item()
        self.global_step = checkpoint['global_step'].item()
        self.training_history = checkpoint['training_history'].item()
        self.best_metric = checkpoint['best_metric'].item()
        
        # 恢复模型参数
        self.model.set_parameters(checkpoint['model_state'].item())
        
        # 恢复优化器状态
        self.optimizer.set_state(checkpoint['optimizer_state'].item())
        
        logger.info(f"从检查点恢复训练: epoch {self.current_epoch}")
    
    def _update_history(self, train_metrics: Dict, val_metrics: Dict):
        """更新训练历史"""
        self.training_history['train_loss'].append(train_metrics.get('loss', 0))
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics.get('loss', 0))
            
            for name in self.metrics.keys():
                if name in val_metrics:
                    self.training_history['metrics'][name].append(val_metrics[name])
    
    def _log_epoch_metrics(self, epoch: int, 
                          train_metrics: Dict, 
                          val_metrics: Dict):
        """记录epoch指标"""
        log_str = f"Epoch {epoch + 1}/{self.current_epoch + 1}"
        
        # 训练指标
        log_str += f" - train_loss: {train_metrics['loss']:.4f}"
        for name, value in train_metrics.items():
            if name != 'loss':
                log_str += f" - train_{name}: {value:.4f}"
        
        # 验证指标
        if val_metrics:
            log_str += f" - val_loss: {val_metrics['loss']:.4f}"
            for name, value in val_metrics.items():
                if name != 'loss':
                    log_str += f" - val_{name}: {value:.4f}"
        
        # 学习率
        current_lr = self.training_history['learning_rates'][-1]
        log_str += f" - lr: {current_lr:.6f}"
        
        logger.info(log_str)
    
    def _run_callbacks(self, hook_name: str, **kwargs):
        """运行回调函数"""
        for callback in self.callbacks:
            if hasattr(callback, hook_name):
                getattr(callback, hook_name)(self, **kwargs)