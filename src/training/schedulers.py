# src/training/schedulers.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


class LRScheduler(ABC):
    """学习率调度器基类"""
    
    def __init__(self, optimizer, last_epoch: int = -1):
        """
        初始化学习率调度器
        
        Args:
            optimizer: 优化器
            last_epoch: 上一个epoch数
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.base_lr = optimizer.get_learning_rate()
        
        # 初始化时执行一次
        if last_epoch == -1:
            self.last_epoch = 0
            self.step()
    
    @abstractmethod
    def get_lr(self) -> float:
        """计算当前学习率"""
        pass
    
    def step(self, epoch: Optional[int] = None):
        """更新学习率"""
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        
        new_lr = self.get_lr()
        self.optimizer.set_learning_rate(new_lr)
    
    def get_last_lr(self) -> float:
        """获取最后的学习率"""
        return self.optimizer.get_learning_rate()


class StepLR(LRScheduler):
    """步进学习率调度器"""
    
    def __init__(self, optimizer, step_size: int, gamma: float = 0.1, last_epoch: int = -1):
        """
        初始化步进学习率调度器
        
        Args:
            optimizer: 优化器
            step_size: 步长
            gamma: 衰减因子
            last_epoch: 上一个epoch数
        """
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        """计算步进学习率"""
        return self.base_lr * (self.gamma ** (self.last_epoch // self.step_size))


class ExponentialLR(LRScheduler):
    """指数衰减学习率调度器"""
    
    def __init__(self, optimizer, gamma: float, last_epoch: int = -1):
        """
        初始化指数衰减学习率调度器
        
        Args:
            optimizer: 优化器
            gamma: 衰减因子
            last_epoch: 上一个epoch数
        """
        self.gamma = gamma
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        """计算指数衰减学习率"""
        return self.base_lr * (self.gamma ** self.last_epoch)


class CosineAnnealingLR(LRScheduler):
    """余弦退火学习率调度器"""
    
    def __init__(self, optimizer, T_max: int, eta_min: float = 0, last_epoch: int = -1):
        """
        初始化余弦退火学习率调度器
        
        Args:
            optimizer: 优化器
            T_max: 最大迭代次数
            eta_min: 最小学习率
            last_epoch: 上一个epoch数
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> float:
        """计算余弦退火学习率"""
        if self.last_epoch == 0:
            return self.base_lr
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.get_last_lr() + (self.base_lr - self.eta_min) * \
                   (1 - np.cos(np.pi / self.T_max)) / 2
        else:
            return (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / \
                   (1 + np.cos(np.pi * (self.last_epoch - 1) / self.T_max)) * \
                   (self.get_last_lr() - self.eta_min) + self.eta_min


class ReduceLROnPlateau(LRScheduler):
   """在指标停滞时降低学习率"""
   
   def __init__(self, optimizer, mode: str = 'min', factor: float = 0.1,
                patience: int = 10, threshold: float = 1e-4,
                threshold_mode: str = 'rel', cooldown: int = 0,
                min_lr: float = 0, eps: float = 1e-8):
       """
       初始化ReduceLROnPlateau
       
       Args:
           optimizer: 优化器
           mode: 'min'或'max'
           factor: 学习率降低因子
           patience: 等待的epoch数
           threshold: 判断是否改善的阈值
           threshold_mode: 'rel'或'abs'
           cooldown: 降低学习率后的冷却期
           min_lr: 最小学习率
           eps: 最小衰减量
       """
       self.mode = mode
       self.factor = factor
       self.patience = patience
       self.threshold = threshold
       self.threshold_mode = threshold_mode
       self.cooldown = cooldown
       self.min_lr = min_lr
       self.eps = eps
       
       self.cooldown_counter = 0
       self.wait = 0
       self.best = None
       self.mode_worse = np.inf if mode == 'min' else -np.inf
       
       super().__init__(optimizer, -1)
   
   def get_lr(self) -> float:
       """获取当前学习率"""
       return self.optimizer.get_learning_rate()
   
   def step(self, metrics: float, epoch: Optional[int] = None):
       """基于指标更新学习率"""
       current = float(metrics)
       
       if self.best is None:
           self.best = current
       
       if self.is_better(current, self.best):
           self.best = current
           self.wait = 0
       else:
           self.wait += 1
           
       if self.in_cooldown:
           self.cooldown_counter -= 1
           self.wait = 0
       
       if self.wait >= self.patience:
           self.reduce_lr()
           self.cooldown_counter = self.cooldown
           self.wait = 0
   
   def is_better(self, a: float, best: float) -> bool:
       """判断是否改善"""
       if self.mode == 'min' and self.threshold_mode == 'rel':
           rel_epsilon = 1. - self.threshold
           return a < best * rel_epsilon
       
       elif self.mode == 'min' and self.threshold_mode == 'abs':
           return a < best - self.threshold
       
       elif self.mode == 'max' and self.threshold_mode == 'rel':
           rel_epsilon = self.threshold + 1.
           return a > best * rel_epsilon
       
       else:  # mode == 'max' and threshold_mode == 'abs':
           return a > best + self.threshold
   
   def reduce_lr(self):
       """降低学习率"""
       old_lr = self.optimizer.get_learning_rate()
       new_lr = max(old_lr * self.factor, self.min_lr)
       
       if old_lr - new_lr > self.eps:
           self.optimizer.set_learning_rate(new_lr)
   
   @property
   def in_cooldown(self) -> bool:
       """是否在冷却期"""
       return self.cooldown_counter > 0


class CyclicLR(LRScheduler):
   """循环学习率调度器"""
   
   def __init__(self, optimizer, base_lr: float, max_lr: float,
                step_size_up: int = 2000, step_size_down: Optional[int] = None,
                mode: str = 'triangular', gamma: float = 1.,
                scale_fn: Optional[callable] = None, scale_mode: str = 'cycle',
                last_epoch: int = -1):
       """
       初始化循环学习率调度器
       
       Args:
           optimizer: 优化器
           base_lr: 基础学习率
           max_lr: 最大学习率
           step_size_up: 上升步数
           step_size_down: 下降步数
           mode: 循环模式
           gamma: 衰减因子
           scale_fn: 缩放函数
           scale_mode: 缩放模式
           last_epoch: 上一个epoch数
       """
       self.base_lr = base_lr
       self.max_lr = max_lr
       self.step_size_up = step_size_up
       self.step_size_down = step_size_down or step_size_up
       self.mode = mode
       self.gamma = gamma
       
       if scale_fn is None:
           if self.mode == 'triangular':
               self.scale_fn = lambda x: 1.
               self.scale_mode = 'cycle'
           elif self.mode == 'triangular2':
               self.scale_fn = lambda x: 1 / (2. ** (x - 1))
               self.scale_mode = 'cycle'
           elif self.mode == 'exp_range':
               self.scale_fn = lambda x: gamma ** x
               self.scale_mode = 'iterations'
       else:
           self.scale_fn = scale_fn
           self.scale_mode = scale_mode
       
       super().__init__(optimizer, last_epoch)
   
   def get_lr(self) -> float:
       """计算循环学习率"""
       cycle = np.floor(1 + self.last_epoch / (self.step_size_up + self.step_size_down))
       x = 1. + self.last_epoch / (self.step_size_up + self.step_size_down) - cycle
       
       if x <= self.step_size_up / (self.step_size_up + self.step_size_down):
           scale_factor = x / (self.step_size_up / (self.step_size_up + self.step_size_down))
       else:
           scale_factor = (x - 1) / (self.step_size_down / (self.step_size_up + self.step_size_down)) + 1
       
       base_height = (self.max_lr - self.base_lr) * scale_factor
       
       if self.scale_mode == 'cycle':
           return self.base_lr + base_height * self.scale_fn(cycle)
       else:
           return self.base_lr + base_height * self.scale_fn(self.last_epoch)