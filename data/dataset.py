# src/data/dataset.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """基础数据集抽象类"""

    def __init__(self, name: str, data_path: str):
        self.name = name
        self.data_path = Path(data_path)
        self.raw_data: Optional[np.ndarray] = None
        self.user_id_map: Dict[int, int] = {}
        self.user_id_inverse_map: Dict[int, int] = {}
        self.item_id_map: Dict[int, int] = {}
        self.item_id_inverse_map: Dict[int, int] = {}
        self.statistics: Dict = {}

    def _parse_line(self, line: str) -> Optional[List[float]]:
        """解析数据行，处理可能的浮点数ID"""
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            # 去除方括号并分割
            parts = line[1:-1].split(',')
            if len(parts) == 3:
                try:
                    # 处理可能的浮点数格式
                    user_id_str = parts[0].strip()
                    item_id_str = parts[1].strip()
                    rating_str = parts[2].strip()

                    # 如果是浮点数格式，先转为浮点数再转为整数
                    if '.' in user_id_str:
                        user_id = int(float(user_id_str))
                    else:
                        user_id = int(user_id_str)

                    if '.' in item_id_str:
                        item_id = int(float(item_id_str))
                    else:
                        item_id = int(item_id_str)

                    rating = float(rating_str)
                    return [user_id, item_id, rating]
                except (ValueError, TypeError) as e:
                    logger.warning(f"跳过无效行: {line}, 错误: {e}")
                    return None
        return None

    @abstractmethod
    def load_raw_data(self) -> np.ndarray:
        """加载原始数据，返回格式为 [user_id, item_id, rating] 的数组"""
        pass

    def get_statistics(self) -> Dict:
        """计算数据统计信息"""
        if self.raw_data is None:
            raise ValueError("数据尚未加载，请先调用 load_raw_data()")

        # 提取数据
        users = self.raw_data[:, 0].astype(int)
        items = self.raw_data[:, 1].astype(int)
        ratings = self.raw_data[:, 2]

        # 计算统计信息
        self.statistics = {
            'n_users': len(np.unique(users)),
            'n_items': len(np.unique(items)),
            'n_ratings': len(ratings),
            'rating_min': float(np.min(ratings)),
            'rating_max': float(np.max(ratings)),
            'rating_mean': float(np.mean(ratings)),
            'rating_std': float(np.std(ratings)),
            'sparsity': 1 - len(ratings) / (len(np.unique(users)) * len(np.unique(items))),
            'user_activity': self._calculate_user_activity(users),
            'item_popularity': self._calculate_item_popularity(items)
        }

        return self.statistics

    def _calculate_user_activity(self, users: np.ndarray) -> Dict:
        """计算用户活跃度分布"""
        unique, counts = np.unique(users, return_counts=True)
        return {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts))
        }

    def _calculate_item_popularity(self, items: np.ndarray) -> Dict:
        """计算物品流行度分布"""
        unique, counts = np.unique(items, return_counts=True)
        return {
            'mean': float(np.mean(counts)),
            'std': float(np.std(counts)),
            'min': int(np.min(counts)),
            'max': int(np.max(counts))
        }

    def get_data_info(self) -> Dict:
        """返回数据集元信息"""
        return {
            'name': self.name,
            'path': str(self.data_path),
            'loaded': self.raw_data is not None,
            'statistics': self.statistics if self.statistics else None
        }


class MovieLens100K(BaseDataset):
    """MovieLens 100K 数据集"""

    def __init__(self, data_path: str):
        super().__init__("MovieLens100K", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 MovieLens 100K 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    parsed_data = self._parse_line(line)
                    if parsed_data:
                        data_list.append(parsed_data)

            if not data_list:
                raise ValueError(f"未能从 {self.data_path} 加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class MovieLens1M(BaseDataset):
    """MovieLens 1M 数据集"""

    def __init__(self, data_path: str):
        super().__init__("MovieLens1M", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 MovieLens 1M 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            user_id = int(float(parts[0].strip()))
                            item_id = int(float(parts[1].strip()))
                            rating = float(parts[2].strip())
                            data_list.append([user_id, item_id, rating])

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class Netflix(BaseDataset):
    """Netflix 数据集"""

    def __init__(self, data_path: str):
        super().__init__("Netflix", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 Netflix 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            user_id = int(float(parts[0].strip()))
                            item_id = int(float(parts[1].strip()))
                            rating = float(parts[2].strip())
                            data_list.append([user_id, item_id, rating])

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class AmazonMI(BaseDataset):
    """Amazon Musical Instruments 数据集"""

    def __init__(self, data_path: str):
        super().__init__("AmazonMI", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 Amazon Musical Instruments 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # Amazon数据集可能包含浮点数格式的ID
                            try:
                                user_id = int(float(parts[0].strip()))
                                item_id = int(float(parts[1].strip()))
                                rating = float(parts[2].strip())
                                data_list.append([user_id, item_id, rating])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                                continue

            if not data_list:
                raise ValueError("未能加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class CiaoDVD(BaseDataset):
    """CiaoDVD 数据集"""

    def __init__(self, data_path: str):
        super().__init__("CiaoDVD", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 CiaoDVD 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            try:
                                user_id = int(float(parts[0].strip()))
                                item_id = int(float(parts[1].strip()))
                                rating = float(parts[2].strip())
                                data_list.append([user_id, item_id, rating])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                                continue

            if not data_list:
                raise ValueError("未能加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class Epinions(BaseDataset):
    """Epinions 数据集"""

    def __init__(self, data_path: str):
        super().__init__("Epinions", data_path)
        self.rating_scale = (1, 5)

    def load_raw_data(self) -> np.ndarray:
        """加载 Epinions 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            try:
                                user_id = int(float(parts[0].strip()))
                                item_id = int(float(parts[1].strip()))
                                rating = float(parts[2].strip())
                                data_list.append([user_id, item_id, rating])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                                continue

            if not data_list:
                raise ValueError("未能加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class FilmTrust(BaseDataset):
    """FilmTrust 数据集"""

    def __init__(self, data_path: str):
        super().__init__("FilmTrust", data_path)
        self.rating_scale = (0.5, 4.0)

    def load_raw_data(self) -> np.ndarray:
        """加载 FilmTrust 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            try:
                                user_id = int(float(parts[0].strip()))
                                item_id = int(float(parts[1].strip()))
                                rating = float(parts[2].strip())
                                data_list.append([user_id, item_id, rating])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                                continue

            if not data_list:
                raise ValueError("未能加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise


class MovieTweetings(BaseDataset):
    """MovieTweetings 数据集"""

    def __init__(self, data_path: str):
        super().__init__("MovieTweetings", data_path)
        self.rating_scale = (1, 10)

    def load_raw_data(self) -> np.ndarray:
        """加载 MovieTweetings 数据"""
        try:
            data_list = []
            with open(self.data_path, 'r') as f:
                for line in f:
                    # 解析 [user_id, item_id, rating] 格式
                    line = line.strip()
                    if line.startswith('[') and line.endswith(']'):
                        # 去除方括号并分割
                        parts = line[1:-1].split(',')
                        if len(parts) == 3:
                            # 支持浮点数格式的ID转换
                            try:
                                user_id = int(float(parts[0].strip()))
                                item_id = int(float(parts[1].strip()))
                                rating = float(parts[2].strip())
                                data_list.append([user_id, item_id, rating])
                            except (ValueError, TypeError) as e:
                                logger.warning(f"跳过无效行: {line.strip()}, 错误: {e}")
                                continue

            if not data_list:
                raise ValueError("未能加载任何有效数据")

            self.raw_data = np.array(data_list, dtype=np.float32)
            logger.info(f"成功加载 {self.name} 数据集: {len(self.raw_data)} 条评分")
            return self.raw_data

        except Exception as e:
            logger.error(f"加载 {self.name} 数据集失败: {str(e)}")
            raise

