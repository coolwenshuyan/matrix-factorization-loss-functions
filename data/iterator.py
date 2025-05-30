# src/data/iterator.py
import numpy as np
from typing import Optional, Tuple, Generator
import logging

logger = logging.getLogger(__name__)


class BatchIterator:
    """批处理迭代器"""
    
    def __init__(self, data: np.ndarray, 
                 batch_size: int = 128,
                 shuffle: bool = True,
                 random_seed: int = 42):
        """
        初始化批处理迭代器
        
        Args:
            data: 数据数组 [user_id, item_id, rating]
            batch_size: 批大小
            shuffle: 是否打乱数据
            random_seed: 随机种子
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.RandomState(random_seed)
        
        self.n_samples = len(data)
        self.n_batches = int(np.ceil(self.n_samples / batch_size))
        self._current_position = 0
        self._index_array = None
        
    def __iter__(self):
        """返回迭代器自身"""
        self._current_position = 0
        if self.shuffle:
            self._shuffle_data()
        else:
            self._index_array = np.arange(self.n_samples)
        return self
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回下一批数据
        
        Returns:
            user_ids, item_ids, ratings
        """
        if self._current_position >= self.n_samples:
            raise StopIteration
        
        # 计算批次的起始和结束位置
        start_idx = self._current_position
        end_idx = min(start_idx + self.batch_size, self.n_samples)
        
        # 获取批次索引
        batch_indices = self._index_array[start_idx:end_idx]
        
        # 提取批次数据
        batch_data = self.data[batch_indices]
        
        # 更新位置
        self._current_position = end_idx
        
        # 返回分离的数据
        return (batch_data[:, 0].astype(int),  # user_ids
                batch_data[:, 1].astype(int),  # item_ids
                batch_data[:, 2])               # ratings
    
    def _shuffle_data(self):
        """打乱数据索引"""
        self._index_array = np.arange(self.n_samples)
        self.rng.shuffle(self._index_array)
    
    def __len__(self):
        """返回批次数量"""
        return self.n_batches
    
    def reset(self):
        """重置迭代器"""
        self._current_position = 0
        if self.shuffle:
            self._shuffle_data()


class NegativeSamplingIterator(BatchIterator):
    """支持负采样的批处理迭代器"""
    
    def __init__(self, data: np.ndarray,
                 n_users: int,
                 n_items: int,
                 batch_size: int = 128,
                 n_negative: int = 4,
                 shuffle: bool = True,
                 random_seed: int = 42):
        """
        初始化负采样迭代器
        
        Args:
            data: 正样本数据
            n_users: 用户总数
            n_items: 物品总数
            batch_size: 批大小
            n_negative: 每个正样本对应的负样本数
            shuffle: 是否打乱
            random_seed: 随机种子
        """
        super().__init__(data, batch_size, shuffle, random_seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_negative = n_negative
        
        # 构建用户-物品交互集合，用于负采样
        self.user_items = self._build_user_items()
    
    def _build_user_items(self) -> dict:
        """构建用户-物品交互字典"""
        user_items = {}
        for user_id, item_id, _ in self.data:
            user_id = int(user_id)
            item_id = int(item_id)
            if user_id not in user_items:
                user_items[user_id] = set()
            user_items[user_id].add(item_id)
        return user_items
    
    def __next__(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        返回下一批数据（包含负样本）
        
        Returns:
            user_ids, item_ids, ratings, labels (1 for positive, 0 for negative)
        """
        # 获取正样本
        pos_users, pos_items, pos_ratings = super().__next__()
        batch_size = len(pos_users)
        
        # 生成负样本
        neg_users = []
        neg_items = []
        
        for i in range(batch_size):
            user_id = pos_users[i]
            user_items = self.user_items.get(user_id, set())
            
            # 为该用户采样负样本
            neg_samples = []
            attempts = 0
            max_attempts = self.n_negative * 10  # 防止无限循环
            
            while len(neg_samples) < self.n_negative and attempts < max_attempts:
                item_id = self.rng.randint(0, self.n_items)
                if item_id not in user_items:
                    neg_samples.append(item_id)
                attempts += 1
            
            # 如果没有足够的负样本，用随机物品填充
            while len(neg_samples) < self.n_negative:
                neg_samples.append(self.rng.randint(0, self.n_items))
            
            neg_users.extend([user_id] * self.n_negative)
            neg_items.extend(neg_samples)
        
        # 组合正负样本
        all_users = np.concatenate([pos_users, np.array(neg_users)])
        all_items = np.concatenate([pos_items, np.array(neg_items)])
        all_ratings = np.concatenate([pos_ratings, np.zeros(len(neg_users))])
        all_labels = np.concatenate([np.ones(batch_size), np.zeros(len(neg_users))])
        
        return all_users, all_items, all_ratings, all_labels