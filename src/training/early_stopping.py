# src/training/early_stopping.py
import numpy as np
import logging
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    早停机制的独立实现
    """
    
    def __init__(self, patience: int = 10, mode: str = 'min',
                 min_delta: float = 0.0001, restore_best: bool = True,
                 verbose: bool = True):
        """
        初始化早停机制
        
        Args:
            patience: 等待改善的epoch数
            mode: 监控模式 ('min' 或 'max')
            min_delta: 认为有改善的最小变化量
            restore_best: 是否恢复最佳模型
            verbose: 是否打印信息
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_params = None
        
        # 根据模式设置比较函数
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
            self.min_delta *= 1
    
    def __call__(self, score: float, model) -> bool:
        """
        检查是否需要早停
        
        Args:
            score: 当前分数
            model: 模型对象
            
        Returns:
            是否需要停止训练
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
            return False
        
        # 检查是否有改善
        if self.monitor_op(score - self.min_delta, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
            
            if self.verbose:
                logger.info(f'验证分数改善到 {score:.4f}')
        else:
            self.counter += 1
            
            if self.verbose:
                logger.info(f'早停计数器: {self.counter}/{self.patience}')
            
            if self.counter >= self.patience:
                self.early_stop = True
                
                if self.restore_best and self.best_model_params is not None:
                    if self.verbose:
                        logger.info('早停触发，恢复最佳模型参数')
                    model.set_parameters(self.best_model_params)
                
                return True
        
        return False
    
    def save_checkpoint(self, model):
        """保存最佳模型参数"""
        if self.restore_best:
            self.best_model_params = model.get_parameters().copy()
    
    def reset(self):
        """重置早停状态"""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_params = None