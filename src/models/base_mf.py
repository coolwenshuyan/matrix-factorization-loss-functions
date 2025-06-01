# src/models/base_mf.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, Union, Any
import json
import os


class BaseMatrixFactorization(ABC):
    """矩阵分解模型的基类"""
    
    def __init__(self, n_users: int, n_items: int, n_factors: int = 10,
                 use_bias: bool = True, global_mean: float = 0.0,
                 dtype: np.dtype = np.float32):
        """
        初始化基础矩阵分解模型
        
        Args:
            n_users: 用户数量
            n_items: 物品数量
            n_factors: 潜在因子数量
            use_bias: 是否使用偏差项
            global_mean: 全局均值（用于中心化数据）
            dtype: 数据类型
        """
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.use_bias = use_bias
        self.global_mean = global_mean
        self.dtype = dtype
        
        # 模型参数（子类中初始化）
        self.user_factors: Optional[np.ndarray] = None  # P矩阵
        self.item_factors: Optional[np.ndarray] = None  # Q矩阵
        self.user_bias: Optional[np.ndarray] = None
        self.item_bias: Optional[np.ndarray] = None
        
        # 训练历史
        self.train_history = {
            'loss': [],
            'val_loss': [],
            'metrics': {}
        }
    
    @abstractmethod
    def initialize_parameters(self, initializer=None):
        """
        初始化模型参数
        
        Args:
            initializer: 参数初始化器
        """
        pass
    
    def predict(self, user_ids: Union[int, np.ndarray],
                item_ids: Union[int, np.ndarray]) -> np.ndarray:
        """
        预测评分
        
        Args:
            user_ids: 用户ID（单个或数组）
            item_ids: 物品ID（单个或数组）
            
        Returns:
            预测评分
        """
        # 确保输入是数组
        user_ids = np.atleast_1d(user_ids)
        item_ids = np.atleast_1d(item_ids)
        
        # 验证输入
        if len(user_ids) != len(item_ids):
            raise ValueError("用户ID和物品ID数量必须相同")
        
        # 获取用户和物品因子
        user_vecs = self.user_factors[user_ids]
        item_vecs = self.item_factors[item_ids]
        
        # 计算点积
        predictions = np.sum(user_vecs * item_vecs, axis=1)
        
        # 添加偏差项
        if self.use_bias:
            predictions += self.user_bias[user_ids]
            predictions += self.item_bias[item_ids]
        
        # 添加全局均值
        predictions += self.global_mean
        
        return predictions
    
    def predict_all(self) -> np.ndarray:
        """
        预测所有用户-物品对的评分
        
        Returns:
            完整的预测评分矩阵
        """
        # 计算所有评分 R = P @ Q^T
        predictions = self.user_factors @ self.item_factors.T
        
        # 添加偏差项
        if self.use_bias:
            predictions += self.user_bias[:, np.newaxis]
            predictions += self.item_bias[np.newaxis, :]
        
        # 添加全局均值
        predictions += self.global_mean
        
        return predictions
    
    @abstractmethod
    def fit(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None,
            **kwargs):
        """
        训练模型
        
        Args:
            train_data: 训练数据 [user_id, item_id, rating]
            val_data: 验证数据（可选）
            **kwargs: 其他训练参数
        """
        pass
    
    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        获取所有模型参数
        
        Returns:
            参数字典
        """
        params = {
            'user_factors': self.user_factors,
            'item_factors': self.item_factors
        }
        
        if self.use_bias:
            params['user_bias'] = self.user_bias
            params['item_bias'] = self.item_bias
        
        return params
    
    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        设置模型参数
        
        Args:
            parameters: 参数字典
        """
        self.user_factors = parameters['user_factors'].astype(self.dtype)
        self.item_factors = parameters['item_factors'].astype(self.dtype)
        
        if self.use_bias:
            self.user_bias = parameters.get('user_bias', np.zeros(self.n_users)).astype(self.dtype)
            self.item_bias = parameters.get('item_bias', np.zeros(self.n_items)).astype(self.dtype)
    
    def save_model(self, filepath: str):
        """
        保存模型到文件
        
        Args:
            filepath: 保存路径
        """
        # 准备保存数据
        save_dict = {
            'config': {
                'n_users': self.n_users,
                'n_items': self.n_items,
                'n_factors': self.n_factors,
                'use_bias': self.use_bias,
                'global_mean': self.global_mean,
                'dtype': np.dtype(self.dtype).name  # 确保是dtype实例
            },
            'parameters': self.get_parameters(),
            'train_history': self.train_history
        }
        
        # 保存为.npz文件
        np.savez_compressed(filepath, **save_dict)
        
        # 同时保存配置为JSON（便于查看）
        config_path = filepath.replace('.npz', '_config.json')
        with open(config_path, 'w') as f:
            config_copy = save_dict['config'].copy()
            # dtype已经是字符串了，不需要再转换
            json.dump(config_copy, f, indent=2)
    
    def load_model(self, filepath: str):
        """
        从文件加载模型
        
        Args:
            filepath: 加载路径
        """
        # 加载数据
        data = np.load(filepath, allow_pickle=True)
        
        # 恢复配置
        config = data['config'].item()
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.n_factors = config['n_factors']
        self.use_bias = config['use_bias']
        self.global_mean = config['global_mean']
        self.dtype = np.dtype(config['dtype'])
        
        # 恢复参数
        parameters = data['parameters'].item()
        self.set_parameters(parameters)
        
        # 恢复训练历史
        self.train_history = data['train_history'].item()
    
    def get_user_embeddings(self, user_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取用户嵌入
        
        Args:
            user_ids: 用户ID列表（None表示所有用户）
            
        Returns:
            用户嵌入矩阵
        """
        if user_ids is None:
            return self.user_factors
        else:
            return self.user_factors[user_ids]
    
    def get_item_embeddings(self, item_ids: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取物品嵌入
        
        Args:
            item_ids: 物品ID列表（None表示所有物品）
            
        Returns:
            物品嵌入矩阵
        """
        if item_ids is None:
            return self.item_factors
        else:
            return self.item_factors[item_ids]
    
    def get_similar_items(self, item_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相似物品
        
        Args:
            item_id: 查询物品ID
            n_similar: 返回的相似物品数量
            
        Returns:
            相似物品ID和相似度分数
        """
        item_vec = self.item_factors[item_id]
        
        # 计算余弦相似度
        similarities = self.item_factors @ item_vec
        similarities = similarities / (np.linalg.norm(self.item_factors, axis=1) * np.linalg.norm(item_vec) + 1e-8)
        
        # 排除自身
        similarities[item_id] = -np.inf
        
        # 获取最相似的物品
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores
    
    def get_similar_users(self, user_id: int, n_similar: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取相似用户
        
        Args:
            user_id: 查询用户ID
            n_similar: 返回的相似用户数量
            
        Returns:
            相似用户ID和相似度分数
        """
        user_vec = self.user_factors[user_id]
        
        # 计算余弦相似度
        similarities = self.user_factors @ user_vec
        similarities = similarities / (np.linalg.norm(self.user_factors, axis=1) * np.linalg.norm(user_vec) + 1e-8)
        
        # 排除自身
        similarities[user_id] = -np.inf
        
        # 获取最相似的用户
        top_indices = np.argsort(similarities)[::-1][:n_similar]
        top_scores = similarities[top_indices]
        
        return top_indices, top_scores