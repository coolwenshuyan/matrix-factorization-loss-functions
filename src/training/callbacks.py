# src/training/callbacks.py
import numpy as np
import json
import csv
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging
from datetime import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Callback:
    """回调函数基类"""
    
    def on_train_begin(self, trainer, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer, **kwargs):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """epoch开始时调用"""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, 
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """epoch结束时调用"""
        pass
    
    def on_batch_begin(self, trainer, batch: int, **kwargs):
        """批次开始时调用"""
        pass
    
    def on_batch_end(self, trainer, batch: int, 
                     loss: float, metrics: Dict, **kwargs):
        """批次结束时调用"""
        pass


class ModelCheckpoint(Callback):
    """模型检查点回调"""
    
    def __init__(self, filepath: str, monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = True,
                 save_weights_only: bool = False, period: int = 1,
                 verbose: int = 1):
        """
        初始化模型检查点
        
        Args:
            filepath: 保存路径
            monitor: 监控的指标
            mode: 'min'或'max'
            save_best_only: 是否只保存最佳模型
            save_weights_only: 是否只保存权重
            period: 保存周期
            verbose: 日志详细程度
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.verbose = verbose
        
        self.epochs_since_last_save = 0
        self.best = np.inf if mode == 'min' else -np.inf
    
    def on_epoch_end(self, trainer, epoch: int,
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """在epoch结束时保存模型"""
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            # 获取监控指标的值
            if self.monitor.startswith('val_'):
                current = val_metrics.get(self.monitor[4:], None)
            else:
                current = train_metrics.get(self.monitor, None)
            
            if current is None:
                logger.warning(f"监控指标 {self.monitor} 未找到")
                return
            
            # 检查是否需要保存
            if self.save_best_only:
                if self.mode == 'min' and current < self.best:
                    if self.verbose > 0:
                        logger.info(f"Epoch {epoch+1}: {self.monitor} 改善从 {self.best:.4f} 到 {current:.4f}")
                    self.best = current
                    self._save_model(trainer, epoch)
                elif self.mode == 'max' and current > self.best:
                    if self.verbose > 0:
                        logger.info(f"Epoch {epoch+1}: {self.monitor} 改善从 {self.best:.4f} 到 {current:.4f}")
                    self.best = current
                    self._save_model(trainer, epoch)
            else:
                self._save_model(trainer, epoch)
    
    def _save_model(self, trainer, epoch: int):
        """保存模型"""
        filepath = str(self.filepath).format(epoch=epoch+1)
        
        if self.save_weights_only:
            # 只保存权重
            np.savez_compressed(filepath, **trainer.model.get_parameters())
        else:
            # 保存完整检查点
            trainer.save_checkpoint(filepath)


class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(self, monitor: str = 'val_loss', patience: int = 10,
                 mode: str = 'min', min_delta: float = 0.0001,
                 restore_best_weights: bool = True, verbose: int = 1):
        """
        初始化早停
        
        Args:
            monitor: 监控的指标
            patience: 耐心值
            mode: 'min'或'max'
            min_delta: 最小改善阈值
            restore_best_weights: 是否恢复最佳权重
            verbose: 日志详细程度
        """
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if mode == 'min' else -np.inf
        self.best_weights = None
    
    def on_train_begin(self, trainer, **kwargs):
        """训练开始时重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf if self.mode == 'min' else -np.inf
    
    def on_epoch_end(self, trainer, epoch: int,
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """检查是否需要早停"""
        # 获取监控指标的值
        if self.monitor.startswith('val_'):
            current = val_metrics.get(self.monitor[4:], None)
        else:
            current = train_metrics.get(self.monitor, None)
        
        if current is None:
            logger.warning(f"早停监控指标 {self.monitor} 未找到")
            return
        
        # 检查是否改善
        if self.mode == 'min':
            if current < self.best - self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = trainer.model.get_parameters()
            else:
                self.wait += 1
        else:  # mode == 'max'
            if current > self.best + self.min_delta:
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = trainer.model.get_parameters()
            else:
                self.wait += 1
        
        # 检查是否需要停止
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            trainer.stop_training = True
            
            if self.verbose > 0:
                logger.info(f"Epoch {epoch+1}: 早停触发")
            
            if self.restore_best_weights and self.best_weights is not None:
                if self.verbose > 0:
                    logger.info("恢复最佳模型权重")
                trainer.model.set_parameters(self.best_weights)


class CSVLogger(Callback):
    """CSV日志记录器"""
    
    def __init__(self, filename: str, separator: str = ',',
                 append: bool = False):
        """
        初始化CSV日志记录器
        
        Args:
            filename: 文件名
            separator: 分隔符
            append: 是否追加模式
        """
        self.filename = filename
        self.separator = separator
        self.append = append
        self.writer = None
        self.file = None
        self.keys = None
    
    def on_train_begin(self, trainer, **kwargs):
        """训练开始时打开文件"""
        if self.append:
            mode = 'a'
        else:
            mode = 'w'
        
        self.file = open(self.filename, mode)
    
    def on_epoch_end(self, trainer, epoch: int,
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """记录epoch结果"""
        # 准备记录的数据
        row_dict = {'epoch': epoch + 1}
        
        # 添加训练指标
        for key, value in train_metrics.items():
            row_dict[f'train_{key}'] = value
        
        # 添加验证指标
        for key, value in val_metrics.items():
            row_dict[f'val_{key}'] = value
        
        # 添加学习率
        row_dict['lr'] = trainer.optimizer.get_learning_rate()
        
        # 第一次写入时创建header
        if self.writer is None:
            self.keys = sorted(row_dict.keys())
            self.writer = csv.DictWriter(
                self.file, 
                fieldnames=self.keys,
                delimiter=self.separator
            )
            if not self.append:
                self.writer.writeheader()
        
        # 写入数据
        self.writer.writerow(row_dict)
        self.file.flush()
    
    def on_train_end(self, trainer, **kwargs):
        """训练结束时关闭文件"""
        if self.file:
            self.file.close()


class ProgressBar(Callback):
    """进度条回调"""
    
    def __init__(self, verbose: int = 1):
        """
        初始化进度条
        
        Args:
            verbose: 显示详细程度
        """
        self.verbose = verbose
        self.epoch_bar = None
        self.batch_bar = None
    
    def on_epoch_begin(self, trainer, epoch: int, **kwargs):
        """创建epoch进度条"""
        if self.verbose > 0:
            desc = f'Epoch {epoch+1}/{trainer.current_epoch+1}'
            self.epoch_bar = tqdm(
                total=len(trainer.train_loader),
                desc=desc,
                unit='batch'
            )
    
    def on_batch_end(self, trainer, batch: int,
                     loss: float, metrics: Dict, **kwargs):
        """更新批次进度"""
        if self.verbose > 0 and self.epoch_bar is not None:
            # 更新进度条
            self.epoch_bar.update(1)
            
            # 更新显示的指标
            postfix = {'loss': f'{loss:.4f}'}
            for key, value in metrics.items():
                postfix[key] = f'{value:.4f}'
            self.epoch_bar.set_postfix(postfix)
    
    def on_epoch_end(self, trainer, epoch: int,
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """关闭epoch进度条"""
        if self.verbose > 0 and self.epoch_bar is not None:
            self.epoch_bar.close()
            
            # 打印epoch总结
            summary = f"Epoch {epoch+1}"
            for key, value in train_metrics.items():
                summary += f" - {key}: {value:.4f}"
            for key, value in val_metrics.items():
                summary += f" - val_{key}: {value:.4f}"
            print(summary)


class TensorBoard(Callback):
    """TensorBoard回调（简化版）"""
    
    def __init__(self, log_dir: str = './logs', 
                 histogram_freq: int = 0,
                 write_graph: bool = True,
                 write_images: bool = False):
        """
        初始化TensorBoard
        
        Args:
            log_dir: 日志目录
            histogram_freq: 直方图频率
            write_graph: 是否写入图
            write_images: 是否写入图像
        """
        self.log_dir = Path(log_dir)
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建事件文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_file = self.log_dir / f'events_{timestamp}.json'
        self.events = []
    
    def on_epoch_end(self, trainer, epoch: int,
                     train_metrics: Dict, val_metrics: Dict, **kwargs):
        """记录epoch指标"""
        event = {
            'epoch': epoch + 1,
            'step': trainer.global_step,
            'timestamp': datetime.now().isoformat(),
            'metrics': {}
        }
        
        # 记录训练指标
        for key, value in train_metrics.items():
            event['metrics'][f'train/{key}'] = float(value)
        
        # 记录验证指标
        for key, value in val_metrics.items():
            event['metrics'][f'val/{key}'] = float(value)
        
        # 记录学习率
        event['metrics']['learning_rate'] = trainer.optimizer.get_learning_rate()
        
        # 添加到事件列表
        self.events.append(event)
        
        # 保存到文件
        with open(self.event_file, 'w') as f:
            json.dump(self.events, f, indent=2)
    
    def on_train_end(self, trainer, **kwargs):
        """训练结束时的清理"""
        # 最终保存
        with open(self.event_file, 'w') as f:
            json.dump(self.events, f, indent=2)
        
        logger.info(f"TensorBoard日志已保存至: {self.event_file}")