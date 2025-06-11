# src/data/preprocessor.py
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self.global_mean: Optional[float] = None

    def reindex_ids(self, data: np.ndarray) -> Tuple[np.ndarray, Dict, Dict, Dict, Dict]:
        """
        重新索引用户和物品ID

        Args:
            data: 原始数据 [user_id, item_id, rating]

        Returns:
            重新索引后的数据和映射关系
        """
        logger.info("开始重新索引ID...")

        # 提取用户和物品ID
        user_ids = data[:, 0].astype(int)
        item_ids = data[:, 1].astype(int)

        # 获取唯一ID并排序
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)

        # 创建映射关系
        user_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_users)}
        item_id_map = {old_id: new_id for new_id, old_id in enumerate(unique_items)}

        # 创建反向映射
        user_id_inverse_map = {new_id: old_id for old_id, new_id in user_id_map.items()}
        item_id_inverse_map = {new_id: old_id for old_id, new_id in item_id_map.items()}

        # 应用映射
        new_data = data.copy()
        new_data[:, 0] = np.array([user_id_map[uid] for uid in user_ids])
        new_data[:, 1] = np.array([item_id_map[iid] for iid in item_ids])

        logger.info(f"重新索引完成: {len(unique_users)} 用户, {len(unique_items)} 物品")

        return new_data, user_id_map, user_id_inverse_map, item_id_map, item_id_inverse_map

    def center_ratings(self, train_data: np.ndarray, all_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        中心化评分数据

        Args:
            train_data: 训练数据
            all_data: 所有数据

        Returns:
            中心化后的数据和全局均值
        """
        # 计算训练集的全局均值
        self.global_mean = float(np.mean(train_data[:, 2]))
        logger.info(f"训练集全局均值: {self.global_mean:.4f}")

        # 中心化所有数据
        centered_data = all_data.copy()
        centered_data[:, 2] = centered_data[:, 2] - self.global_mean

        return centered_data, self.global_mean

    def split_data(self, data: np.ndarray,
                   train_ratio: float = 0.8,
                   val_ratio: float = 0.1,
                   test_ratio: float = 0.1,
                   ensure_user_in_train: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        划分数据集
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

        n_samples = len(data)
        indices = np.arange(n_samples)

        if ensure_user_in_train:
            # 确保每个用户在训练集中至少有一个评分
            train_indices, val_indices, test_indices = self._split_with_user_constraint(
                data, train_ratio, val_ratio, test_ratio
            )

        else:
            # 简单随机划分
            self.rng.shuffle(indices)
            train_size = int(n_samples * train_ratio)
            val_size = int(n_samples * val_ratio)

            train_indices = indices[:train_size]
            val_indices = indices[train_size:train_size + val_size]
            test_indices = indices[train_size + val_size:]

        logger.info(f"数据集划分: 训练集 {len(train_indices)}, "
                   f"验证集 {len(val_indices)}, 测试集 {len(test_indices)}")

        # 确保索引是整数类型，避免索引错误
        train_indices = np.asarray(train_indices, dtype=int)
        val_indices = np.asarray(val_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)

        return data[train_indices], data[val_indices], data[test_indices]

    def _split_with_user_constraint(self, data: np.ndarray,
                                   train_ratio: float,
                                   val_ratio: float,
                                   test_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """确保每个用户在训练集中至少有一个评分的划分方法"""
        # 按用户分组
        user_groups = defaultdict(list)
        for idx, row in enumerate(data):
            user_id = int(row[0])
            user_groups[user_id].append(idx)

        train_indices = []
        val_indices = []
        test_indices = []

        for user_id, user_indices in user_groups.items():
            # 随机打乱该用户的评分
            self.rng.shuffle(user_indices)

            if len(user_indices) == 1:
                # 如果用户只有一个评分，放入训练集
                train_indices.extend(user_indices)
            else:
                # 确保至少一个评分在训练集
                train_indices.append(user_indices[0])

                # 剩余的按比例分配，但仍然考虑整体比例
                remaining = user_indices[1:]
                n_remaining = len(remaining)

                if n_remaining == 0:
                    continue

                # 计算应该分配给训练集的额外数量
                # 考虑到已经分配了一个，调整比例
                adjusted_train_ratio = (train_ratio * (n_remaining + 1) - 1) / n_remaining
                adjusted_train_ratio = max(0, min(1, adjusted_train_ratio))  # 确保在[0,1]范围内

                # 确保所有计算结果都是整数
                n_train_extra = int(np.round(n_remaining * adjusted_train_ratio))

                # 计算验证集和测试集的数量
                remaining_after_train = n_remaining - n_train_extra
                if remaining_after_train > 0:
                    val_test_ratio = val_ratio / (val_ratio + test_ratio) if (val_ratio + test_ratio) > 0 else 0.5
                    n_val = int(np.round(remaining_after_train * val_test_ratio))
                    n_test = remaining_after_train - n_val
                else:
                    n_val = 0
                    n_test = 0

                # 确保索引分配不超出范围
                n_train_extra = min(n_train_extra, n_remaining)
                n_val = min(n_val, n_remaining - n_train_extra)
                n_test = n_remaining - n_train_extra - n_val

                # 分配额外的训练样本
                if n_train_extra > 0:
                    train_indices.extend(remaining[:n_train_extra])

                # 分配验证样本
                if n_val > 0:
                    val_indices.extend(remaining[n_train_extra:n_train_extra + n_val])

                # 分配测试样本
                if n_test > 0:
                    test_indices.extend(remaining[n_train_extra + n_val:n_train_extra + n_val + n_test])

        # 显式转换为整数类型的numpy数组
        train_indices_array = np.array(train_indices, dtype=int)
        val_indices_array = np.array(val_indices, dtype=int)
        test_indices_array = np.array(test_indices, dtype=int)

        return train_indices_array, val_indices_array, test_indices_array

    def validate_data(self, data: np.ndarray, rating_scale: Tuple[float, float]) -> bool:
        """
        验证数据的有效性

        Args:
            data: 输入数据
            rating_scale: 评分范围

        Returns:
            是否有效
        """
        # 检查数据形状
        if data.shape[1] != 3:
            logger.error("数据格式错误: 期望3列 [user_id, item_id, rating]")
            return False

        # 检查评分范围
        ratings = data[:, 2]
        min_rating, max_rating = rating_scale

        if np.any(ratings < min_rating) or np.any(ratings > max_rating):
            logger.warning(f"发现超出范围的评分: [{np.min(ratings)}, {np.max(ratings)}]")
            # 可以选择剪裁或拒绝
            data[:, 2] = np.clip(ratings, min_rating, max_rating)

        # 检查是否有缺失值
        if np.any(np.isnan(data)):
            logger.error("数据包含缺失值")
            return False

        return True
