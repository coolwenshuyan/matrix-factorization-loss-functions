# src/data/loader.py
import logging
from typing import Dict, Type, Optional
from pathlib import Path
from .dataset import BaseDataset, MovieLens100K, MovieLens1M, Netflix, AmazonMI, CiaoDVD, Epinions, FilmTrust, MovieTweetings

logger = logging.getLogger(__name__)


class DatasetLoader:
    """数据集加载器"""
    
    # 数据集注册表
    DATASET_REGISTRY: Dict[str, Type[BaseDataset]] = {
        'movielens100k': MovieLens100K,
        'ml100k': MovieLens100K,
        'movielens1m': MovieLens1M,
        'ml1m': MovieLens1M,
        'netflix': Netflix,
        'amazonmi': AmazonMI,
        'amazon_musical_instruments': AmazonMI,
        'ciaodvd': CiaoDVD,
        'epinions': Epinions,
        'filmtrust': FilmTrust,
        'movietweetings': MovieTweetings
    }
    
    @classmethod
    def load_dataset(cls, name: str, path: str) -> BaseDataset:
        """
        加载指定数据集
        
        Args:
            name: 数据集名称
            path: 数据文件路径
            
        Returns:
            BaseDataset: 加载的数据集对象
        """
        # 标准化数据集名称
        name_lower = name.lower().replace('-', '').replace('_', '')
        
        # 调试信息
        print(f"DEBUG: 尝试加载数据集 '{name_lower}'")
        print(f"DEBUG: 当前注册的数据集: {list(cls.DATASET_REGISTRY.keys())}")
        
        # 自动识别数据集类型
        dataset_class = cls._identify_dataset_type(name_lower, path)
        
        if dataset_class is None:
            raise ValueError(f"未知的数据集类型: {name}")
        
        # 创建数据集实例
        dataset = dataset_class(path)
        
        # 加载数据
        try:
            dataset.load_raw_data()
            dataset.get_statistics()
            logger.info(f"成功加载数据集 {name}")
            return dataset
        except Exception as e:
            logger.error(f"加载数据集 {name} 失败: {str(e)}")
            raise
    
    @classmethod
    def _identify_dataset_type(cls, name: str, path: str) -> Optional[Type[BaseDataset]]:
        """识别数据集类型"""
        # 首先尝试从注册表中查找
        if name in cls.DATASET_REGISTRY:
            return cls.DATASET_REGISTRY[name]
        
        # 尝试从文件名推断
        path_obj = Path(path)
        filename = path_obj.name.lower()
        
        for key, dataset_class in cls.DATASET_REGISTRY.items():
            if key in filename:
                return dataset_class
        
        return None
    
    @classmethod
    def register_dataset(cls, name: str, dataset_class: Type[BaseDataset]):
        """注册新的数据集类型"""
        cls.DATASET_REGISTRY[name.lower()] = dataset_class
        logger.info(f"注册新数据集类型: {name}")
