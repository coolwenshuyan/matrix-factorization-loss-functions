# src/data/data_manager.py  统一的数据管理器
import numpy as np
from typing import Dict, Tuple, Optional, Union
import logging
import json
from pathlib import Path

from .dataset import BaseDataset
from .loader import DatasetLoader
from .preprocessor import DataPreprocessor
from .iterator import BatchIterator, NegativeSamplingIterator

logger = logging.getLogger(__name__)


class DataManager:
    """统一的数据管理器"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据管理器
        
        Args:
            config: 配置字典，包含数据相关的所有参数
        """
        self.config = config or self._get_default_config()
        self.dataset: Optional[BaseDataset] = None
        self.preprocessor = DataPreprocessor(self.config['random_seed'])
        
        # 数据集
        self.train_data: Optional[np.ndarray] = None
        self.val_data: Optional[np.ndarray] = None
        self.test_data: Optional[np.ndarray] = None
        
        # 映射关系
        self.user_id_map: Optional[Dict] = None
        self.user_id_inverse_map: Optional[Dict] = None
        self.item_id_map: Optional[Dict] = None
        self.item_id_inverse_map: Optional[Dict] = None
        
        # 统计信息
        self.statistics: Dict = {}
        self.global_mean: Optional[float] = None
        
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'random_seed': 42,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'batch_size': 128,
            'shuffle': True,
            'center_data': True,
            'ensure_user_in_train': True,
            'negative_sampling': False,
            'n_negative': 4
        }
    
    def load_dataset(self, name: str, path: str) -> 'DataManager':
        """
        加载数据集
        
        Args:
            name: 数据集名称
            path: 数据文件路径
            
        Returns:
            self，支持链式调用
        """
        logger.info(f"加载数据集: {name} from {path}")
        
        # 使用加载器加载数据集
        self.dataset = DatasetLoader.load_dataset(name, path)
        
        # 获取原始数据
        raw_data = self.dataset.raw_data
        
        # 验证数据
        if hasattr(self.dataset, 'rating_scale'):
            is_valid = self.preprocessor.validate_data(raw_data, self.dataset.rating_scale)
            if not is_valid:
                raise ValueError("数据验证失败")
        
        return self
    
    def preprocess(self) -> 'DataManager':
        """
        执行所有预处理步骤
        
        Returns:
            self，支持链式调用
        """
        if self.dataset is None:
            raise ValueError("请先加载数据集")
        
        logger.info("开始数据预处理...")
        
        # 1. 重新索引ID
        reindexed_data, self.user_id_map, self.user_id_inverse_map, \
            self.item_id_map, self.item_id_inverse_map = \
            self.preprocessor.reindex_ids(self.dataset.raw_data)
        
        # 2. 划分数据集
        self.train_data, self.val_data, self.test_data = \
            self.preprocessor.split_data(
                reindexed_data,
                self.config['train_ratio'],
                self.config['val_ratio'],
                self.config['test_ratio'],
                self.config['ensure_user_in_train']
            )
        
        # 3. 数据中心化（如果启用）
        if self.config['center_data']:
            # 计算训练集均值并中心化所有数据
            all_data = np.vstack([self.train_data, self.val_data, self.test_data])
            centered_data, self.global_mean = self.preprocessor.center_ratings(
                self.train_data, all_data
            )
            
            # 更新数据集
            train_size = len(self.train_data)
            val_size = len(self.val_data)
            self.train_data = centered_data[:train_size]
            self.val_data = centered_data[train_size:train_size + val_size]
            self.test_data = centered_data[train_size + val_size:]
        
        # 4. 计算统计信息
        self._calculate_statistics()
        
        logger.info("数据预处理完成")
        return self
    
    def _calculate_statistics(self):
        """计算数据集统计信息"""
        # 基础统计
        self.statistics = {
            'n_users': len(self.user_id_map),
            'n_items': len(self.item_id_map),
            'n_train': len(self.train_data),
            'n_val': len(self.val_data),
            'n_test': len(self.test_data),
            'n_total': len(self.train_data) + len(self.val_data) + len(self.test_data),
            'sparsity': 1 - (len(self.train_data) + len(self.val_data) + len(self.test_data)) / 
                           (len(self.user_id_map) * len(self.item_id_map))
        }
        
        # 评分统计（使用原始评分，非中心化后的）
        if self.global_mean is not None:
            train_ratings = self.train_data[:, 2] + self.global_mean
        else:
            train_ratings = self.train_data[:, 2]
        
        self.statistics.update({
            'rating_mean': float(np.mean(train_ratings)),
            'rating_std': float(np.std(train_ratings)),
            'rating_min': float(np.min(train_ratings)),
            'rating_max': float(np.max(train_ratings))
        })
        
        # 用户和物品的活跃度
        train_users = self.train_data[:, 0].astype(int)
        train_items = self.train_data[:, 1].astype(int)
        
        user_counts = np.bincount(train_users)
        item_counts = np.bincount(train_items)
        
        self.statistics.update({
            'user_activity': {
                'mean': float(np.mean(user_counts[user_counts > 0])),
                'std': float(np.std(user_counts[user_counts > 0])),
                'min': int(np.min(user_counts[user_counts > 0])),
                'max': int(np.max(user_counts))
            },
            'item_popularity': {
                'mean': float(np.mean(item_counts[item_counts > 0])),
                'std': float(np.std(item_counts[item_counts > 0])),
                'min': int(np.min(item_counts[item_counts > 0])),
                'max': int(np.max(item_counts))
            }
        })
    
    def get_splits(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        返回训练/验证/测试集
        
        Returns:
            训练集、验证集、测试集
        """
        if self.train_data is None:
            raise ValueError("数据尚未预处理，请先调用 preprocess()")
        
        return self.train_data, self.val_data, self.test_data
    
    def get_statistics(self) -> Dict:
        """返回数据统计信息"""
        return self.statistics
    
    def get_batch_iterator(self, data_type: str = 'train', 
                          batch_size: Optional[int] = None,
                          shuffle: Optional[bool] = None,
                          negative_sampling: Optional[bool] = None) -> Union[BatchIterator, NegativeSamplingIterator]:
        """
        获取批处理迭代器
        
        Args:
            data_type: 'train', 'val', 或 'test'
            batch_size: 批大小（默认使用配置中的值）
            shuffle: 是否打乱（默认使用配置中的值）
            negative_sampling: 是否使用负采样（默认使用配置中的值）
            
        Returns:
            批处理迭代器
        """
        # 获取对应的数据集
        if data_type == 'train':
            data = self.train_data
        elif data_type == 'val':
            data = self.val_data
        elif data_type == 'test':
            data = self.test_data
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        # 使用默认参数
        if batch_size is None:
            batch_size = self.config['batch_size']
        if shuffle is None:
            shuffle = self.config['shuffle'] and data_type == 'train'
        if negative_sampling is None:
            negative_sampling = self.config['negative_sampling']
        
        # 创建迭代器
        if negative_sampling:
            return NegativeSamplingIterator(
                data,
                n_users=self.statistics['n_users'],
                n_items=self.statistics['n_items'],
                batch_size=batch_size,
                n_negative=self.config['n_negative'],
                shuffle=shuffle,
                random_seed=self.config['random_seed']
            )
        else:
            return BatchIterator(
                data,
                batch_size=batch_size,
                shuffle=shuffle,
                random_seed=self.config['random_seed']
            )
    
    def save_preprocessed_data(self, save_dir: str):
        """
        保存预处理后的数据
        
        Args:
            save_dir: 保存目录
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存数据集
        np.save(save_path / 'train_data.npy', self.train_data)
        np.save(save_path / 'val_data.npy', self.val_data)
        np.save(save_path / 'test_data.npy', self.test_data)
        
        # 保存映射关系
        mappings = {
            'user_id_map': self.user_id_map,
            'user_id_inverse_map': self.user_id_inverse_map,
            'item_id_map': self.item_id_map,
            'item_id_inverse_map': self.item_id_inverse_map
        }
        with open(save_path / 'mappings.json', 'w') as f:
            # 转换键为字符串（JSON要求）并确保值是Python原生类型
            json_mappings = {}
            for key, mapping in mappings.items():
                if mapping is not None:
                    json_mappings[key] = {str(k): int(v) if isinstance(v, np.integer) else v 
                                         for k, v in mapping.items()}
            json.dump(json_mappings, f)
        
        # 保存配置和统计信息
        metadata = {
            'config': self.config,
            'statistics': {k: int(v) if isinstance(v, np.integer) else float(v) if isinstance(v, np.floating) else v 
                          for k, v in self.statistics.items()},
            'global_mean': float(self.global_mean) if self.global_mean is not None else None,
            'dataset_name': self.dataset.name if self.dataset else None
        }
        with open(save_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"预处理数据已保存至: {save_path}")
    
    def load_preprocessed_data(self, load_dir: str) -> 'DataManager':
        """
        加载预处理后的数据
        
        Args:
            load_dir: 加载目录
            
        Returns:
            self，支持链式调用
        """
        load_path = Path(load_dir)
        
        # 加载数据集
        self.train_data = np.load(load_path / 'train_data.npy')
        self.val_data = np.load(load_path / 'val_data.npy')
        self.test_data = np.load(load_path / 'test_data.npy')
        
        # 加载映射关系
        with open(load_path / 'mappings.json', 'r') as f:
            json_mappings = json.load(f)
            
        # 转换键回整数
        self.user_id_map = {int(k): v for k, v in json_mappings.get('user_id_map', {}).items()}
        self.user_id_inverse_map = {int(k): v for k, v in json_mappings.get('user_id_inverse_map', {}).items()}
        self.item_id_map = {int(k): v for k, v in json_mappings.get('item_id_map', {}).items()}
        self.item_id_inverse_map = {int(k): v for k, v in json_mappings.get('item_id_inverse_map', {}).items()}
        
        # 加载元数据
        with open(load_path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.config = metadata['config']
        self.statistics = metadata['statistics']
        self.global_mean = metadata['global_mean']
        
        logger.info(f"预处理数据已从 {load_path} 加载")
        return self
    
    def get_rating_matrix(self, data_type: str = 'train', dense: bool = False) -> np.ndarray:
        """
        获取评分矩阵
        
        Args:
            data_type: 'train', 'val', 或 'test'
            dense: 是否返回密集矩阵（默认返回稀疏矩阵的坐标格式）
            
        Returns:
            评分矩阵或坐标数据
        """
        # 获取数据
        if data_type == 'train':
            data = self.train_data
        elif data_type == 'val':
            data = self.val_data
        elif data_type == 'test':
            data = self.test_data
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        if dense:
            # 创建密集矩阵
            n_users = self.statistics['n_users']
            n_items = self.statistics['n_items']
            matrix = np.zeros((n_users, n_items))
            
            for user_id, item_id, rating in data:
                matrix[int(user_id), int(item_id)] = rating
            
            return matrix
        else:
            # 返回坐标格式
            return data
    
    def print_summary(self):
        """打印数据集摘要"""
        if self.statistics:
            print(f"\n{'='*50}")
            print(f"数据集摘要")
            print(f"{'='*50}")
            print(f"用户数: {self.statistics['n_users']}")
            print(f"物品数: {self.statistics['n_items']}")
            print(f"训练集: {self.statistics['n_train']} 条评分")
            print(f"验证集: {self.statistics['n_val']} 条评分")
            print(f"测试集: {self.statistics['n_test']} 条评分")
            print(f"总评分数: {self.statistics['n_total']}")
            print(f"稀疏度: {self.statistics['sparsity']:.2%}")
            print(f"\n评分统计:")
            print(f"  均值: {self.statistics['rating_mean']:.2f}")
            print(f"  标准差: {self.statistics['rating_std']:.2f}")
            print(f"  范围: [{self.statistics['rating_min']}, {self.statistics['rating_max']}]")
            print(f"\n用户活跃度:")
            print(f"  平均评分数: {self.statistics['user_activity']['mean']:.2f}")
            print(f"  标准差: {self.statistics['user_activity']['std']:.2f}")
            print(f"  范围: [{self.statistics['user_activity']['min']}, {self.statistics['user_activity']['max']}]")
            print(f"\n物品流行度:")
            print(f"  平均评分数: {self.statistics['item_popularity']['mean']:.2f}")
            print(f"  标准差: {self.statistics['item_popularity']['std']:.2f}")
            print(f"  范围: [{self.statistics['item_popularity']['min']}, {self.statistics['item_popularity']['max']}]")
            print(f"{'='*50}\n")
        else:
            print("尚未加载或预处理数据")


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建数据管理器
    config = {
        'random_seed': 42,
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'batch_size': 128,
        'shuffle': True,
        'center_data': True,
        'ensure_user_in_train': True
    }
    
    data_manager = DataManager(config)
    
    # 加载和预处理数据
    data_manager.load_dataset('movielens100k', 'E:\工作资料\科研\论文\写作\使用混合分段损失函数增强矩阵分解用于推荐\矩阵分解损失函数code\dataset\\20201202M100K_data_all_random.txt')
    data_manager.preprocess()
    
    # 打印摘要
    data_manager.print_summary()
    
    # 获取批处理迭代器
    train_iterator = data_manager.get_batch_iterator('train')
    print(f"训练批次数: {len(train_iterator)}")
    
    # 迭代第一个批次
    for i, (user_ids, item_ids, ratings) in enumerate(train_iterator):
        print(f"批次 {i}: 用户 {user_ids[:5]}, 物品 {item_ids[:5]}, 评分 {ratings[:5]}")
        if i >= 2:  # 只打印前3个批次
            break
