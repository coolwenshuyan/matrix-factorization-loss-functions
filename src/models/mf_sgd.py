# src/models/mf_sgd.py
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
import logging
from tqdm import tqdm

from base_mf import BaseMatrixFactorization
from initializers import NormalInitializer
from regularizers import L2Regularizer

logger = logging.getLogger(__name__)


class MatrixFactorizationSGD(BaseMatrixFactorization):
    """
    使用SGD优化的矩阵分解模型
    """
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 10,
                 learning_rate: float = 0.01, regularizer=None,
                 loss_function=None, use_bias: bool = True,
                 global_mean: float = 0.0, clip_gradient: Optional[float] = None,
                 momentum: float = 0.0, lr_schedule: Optional[str] = None,
                 dtype: np.dtype = np.float32):
        """
        初始化SGD矩阵分解模型
        
        Args:
            n_users: 用户数量
            n_items: 物品数量
            n_factors: 潜在因子数量
            learning_rate: 学习率
            regularizer: 正则化器
            loss_function: 损失函数
            use_bias: 是否使用偏差项
            global_mean: 全局均值
            clip_gradient: 梯度裁剪阈值
            momentum: 动量系数
            lr_schedule: 学习率调度策略
            dtype: 数据类型
        """
        super().__init__(n_users, n_items, n_factors, use_bias, global_mean, dtype)
        
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.regularizer = regularizer or L2Regularizer(lambda_reg=0.01)
        self.loss_function = loss_function
        self.clip_gradient = clip_gradient
        self.momentum = momentum
        self.lr_schedule = lr_schedule
        
        # 动量项（如果使用）
        if self.momentum > 0:
            self.velocity = {}
        
        # 学习率调度器参数
        self.lr_decay_rate = 0.95
        self.lr_decay_steps = 10
        
    def initialize_parameters(self, initializer=None):
        """初始化模型参数"""
        if initializer is None:
            initializer = NormalInitializer(mean=0.0, std=1.0/np.sqrt(self.n_factors))
        
        # 初始化因子矩阵
        self.user_factors = initializer.initialize((self.n_users, self.n_factors))
        self.item_factors = initializer.initialize((self.n_items, self.n_factors))
        
        # 初始化偏差项
        if self.use_bias:
            self.user_bias = np.zeros(self.n_users, dtype=self.dtype)
            self.item_bias = np.zeros(self.n_items, dtype=self.dtype)
        
        # 初始化动量项
        if self.momentum > 0:
            self.velocity['user_factors'] = np.zeros_like(self.user_factors)
            self.velocity['item_factors'] = np.zeros_like(self.item_factors)
            if self.use_bias:
                self.velocity['user_bias'] = np.zeros_like(self.user_bias)
                self.velocity['item_bias'] = np.zeros_like(self.item_bias)
    
    def compute_loss(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        计算总损失（数据损失 + 正则化）
        
        Args:
            predictions: 预测值
            targets: 真实值
            
        Returns:
            总损失
        """
        # 数据损失
        if self.loss_function is not None:
            data_loss = self.loss_function.forward(predictions, targets)
        else:
            # 默认使用MSE
            errors = predictions - targets
            data_loss = 0.5 * np.mean(errors ** 2)
        
        # 正则化损失
        reg_loss = self.regularizer.compute_penalty(self.get_parameters())
        
        return data_loss + reg_loss
    
    def sgd_update(self, user_id: int, item_id: int, rating: float,
                   epoch: int) -> float:
        """
        执行单个样本的SGD更新
        
        Args:
            user_id: 用户ID
            item_id: 物品ID
            rating: 真实评分
            epoch: 当前epoch
            
        Returns:
            样本损失
        """
        # 预测评分
        prediction = self.predict(user_id, item_id)[0]
        
        # 计算误差
        error = prediction - rating
        
        # 计算损失梯度
        if self.loss_function is not None:
            loss_grad = self.loss_function.gradient(
                np.array([prediction]), np.array([rating])
            )[0]
        else:
            # 默认MSE梯度
            loss_grad = error
        
        # 获取当前学习率
        current_lr = self._get_learning_rate(epoch)
        
        # 计算参数梯度
        user_vec = self.user_factors[user_id].copy()
        item_vec = self.item_factors[item_id].copy()
        
        # 用户因子梯度
        user_grad = loss_grad * item_vec + \
                   self.regularizer.compute_gradient('user_factors', user_vec)
        
        # 物品因子梯度
        item_grad = loss_grad * user_vec + \
                   self.regularizer.compute_gradient('item_factors', item_vec)
        
        # 梯度裁剪
        if self.clip_gradient is not None:
            user_grad = self._clip_gradient(user_grad)
            item_grad = self._clip_gradient(item_grad)
        
        # 更新参数（带动量）
        if self.momentum > 0:
            # 更新速度
            self.velocity['user_factors'][user_id] = \
                self.momentum * self.velocity['user_factors'][user_id] - current_lr * user_grad
            self.velocity['item_factors'][item_id] = \
                self.momentum * self.velocity['item_factors'][item_id] - current_lr * item_grad
            
            # 更新参数
            self.user_factors[user_id] += self.velocity['user_factors'][user_id]
            self.item_factors[item_id] += self.velocity['item_factors'][item_id]
        else:
            # 标准SGD更新
            self.user_factors[user_id] -= current_lr * user_grad
            self.item_factors[item_id] -= current_lr * item_grad
        
        # 更新偏差项
        if self.use_bias:
            bias_grad_user = loss_grad + \
                           self.regularizer.compute_gradient('user_bias', self.user_bias[user_id])
            bias_grad_item = loss_grad + \
                           self.regularizer.compute_gradient('item_bias', self.item_bias[item_id])
            
            if self.clip_gradient is not None:
                bias_grad_user = np.clip(bias_grad_user, -self.clip_gradient, self.clip_gradient)
                bias_grad_item = np.clip(bias_grad_item, -self.clip_gradient, self.clip_gradient)
            
            if self.momentum > 0:
                self.velocity['user_bias'][user_id] = \
                    self.momentum * self.velocity['user_bias'][user_id] - current_lr * bias_grad_user
                self.velocity['item_bias'][item_id] = \
                    self.momentum * self.velocity['item_bias'][item_id] - current_lr * bias_grad_item
                
                self.user_bias[user_id] += self.velocity['user_bias'][user_id]
                self.item_bias[item_id] += self.velocity['item_bias'][item_id]
            else:
                self.user_bias[user_id] -= current_lr * bias_grad_user
                self.item_bias[item_id] -= current_lr * bias_grad_item
        
        # 计算样本损失
        if self.loss_function is not None:
            sample_loss = self.loss_function.forward(
                np.array([prediction]), np.array([rating])
            )
        else:
            sample_loss = 0.5 * error ** 2
        
        return float(sample_loss)
    
    def _clip_gradient(self, gradient: np.ndarray) -> np.ndarray:
        """
        梯度裁剪
        
        Args:
            gradient: 梯度
            
        Returns:
            裁剪后的梯度
        """
        grad_norm = np.linalg.norm(gradient)
        
        if grad_norm > self.clip_gradient:
            gradient = gradient * self.clip_gradient / grad_norm
        
        return gradient
    
    def _get_learning_rate(self, epoch: int) -> float:
        """
        获取当前学习率
        
        Args:
            epoch: 当前epoch
            
        Returns:
            学习率
        """
        if self.lr_schedule is None:
            return self.learning_rate
        
        if self.lr_schedule == 'exponential':
            # 指数衰减
            decay_factor = self.lr_decay_rate ** (epoch // self.lr_decay_steps)
            return self.initial_learning_rate * decay_factor
        
        elif self.lr_schedule == 'inverse':
            # 反比例衰减
            return self.initial_learning_rate / (1 + 0.01 * epoch)
        
        elif self.lr_schedule == 'cosine':
            # 余弦退火
            return self.initial_learning_rate * 0.5 * (1 + np.cos(np.pi * epoch / 100))
        
        else:
            return self.learning_rate
    
    def fit(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
            n_epochs: int = 100, batch_size: Optional[int] = None,
            verbose: int = 1, early_stopping_patience: int = 10,
            shuffle: bool = True, callbacks: Optional[list] = None):
        """
        训练模型
        
        Args:
            train_data: 训练数据 [user_id, item_id, rating]
            val_data: 验证数据（可选）
            n_epochs: 训练轮数
            batch_size: 批大小（None表示单样本SGD）
            verbose: 日志详细程度
            early_stopping_patience: 早停耐心值
            shuffle: 是否打乱数据
            callbacks: 回调函数列表
        """
        # 初始化参数（如果尚未初始化）
        if self.user_factors is None:
            self.initialize_parameters()
        
        # 准备训练
        n_samples = len(train_data)
        indices = np.arange(n_samples)
        
        best_val_loss = np.inf
        patience_counter = 0
        
        # 训练循环
        for epoch in range(n_epochs):
            # 打乱数据
            if shuffle:
                np.random.shuffle(indices)
            
            # 训练一个epoch
            train_losses = []
            
            # 进度条
            if verbose > 0:
                pbar = tqdm(total=n_samples, desc=f'Epoch {epoch+1}/{n_epochs}')
            
            # 遍历训练数据
            for idx in indices:
                user_id = int(train_data[idx, 0])
                item_id = int(train_data[idx, 1])
                rating = train_data[idx, 2]
                
                # SGD更新
                loss = self.sgd_update(user_id, item_id, rating, epoch)
                train_losses.append(loss)
                
                if verbose > 0:
                    pbar.update(1)
                    if len(train_losses) % 1000 == 0:
                        pbar.set_postfix({'loss': np.mean(train_losses[-1000:])})
            
            if verbose > 0:
                pbar.close()
            
            # 计算epoch平均损失
            epoch_train_loss = np.mean(train_losses)
            self.train_history['loss'].append(epoch_train_loss)
            
            # 验证
            if val_data is not None:
                val_predictions = self.predict(
                    val_data[:, 0].astype(int),
                    val_data[:, 1].astype(int)
                )
                val_loss = self.compute_loss(val_predictions, val_data[:, 2])
                self.train_history['val_loss'].append(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # 保存最佳模型参数
                    self._best_params = self.get_parameters().copy()
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    if verbose > 0:
                        logger.info(f"早停在epoch {epoch+1}")
                    # 恢复最佳参数
                    if hasattr(self, '_best_params'):
                        self.set_parameters(self._best_params)
                    break
            
            # 打印进度
            if verbose > 0:
                log_msg = f"Epoch {epoch+1}/{n_epochs} - loss: {epoch_train_loss:.4f}"
                if val_data is not None:
                    log_msg += f" - val_loss: {val_loss:.4f}"
                logger.info(log_msg)
            
            # 执行回调
            if callbacks:
                for callback in callbacks:
                    callback(self, epoch)
    
    def predict_proba(self, user_ids: np.ndarray, item_ids: np.ndarray,
                      rating_scale: Tuple[float, float] = (1, 5)) -> np.ndarray:
        """
        预测评分概率分布
        
        Args:
            user_ids: 用户ID数组
            item_ids: 物品ID数组
            rating_scale: 评分范围
            
        Returns:
            预测评分
        """
        predictions = self.predict(user_ids, item_ids)
        
        # 裁剪到有效范围
        predictions = np.clip(predictions, rating_scale[0], rating_scale[1])
        
        return predictions