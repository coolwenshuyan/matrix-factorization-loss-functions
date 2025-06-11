# src/data/loader.py
import logging
from typing import Dict, Type, Optional, Tuple
from pathlib import Path
from .dataset import BaseDataset, MovieLens100K, MovieLens1M, Netflix, AmazonMI, CiaoDVD, Epinions, FilmTrust, MovieTweetings, GenericDataset

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
        'movietweetings': MovieTweetings,
        # 添加通用类型
        'generic': GenericDataset,
        'auto': GenericDataset
    }

    @classmethod
    def load_dataset_from_path(cls, path: str,
                            name: str = None,
                            rating_scale: Tuple[float, float] = None,
                            format_type: str = "auto") -> BaseDataset:
        """
        直接从路径加载数据集，无需预定义类型

        Args:
            path: 数据文件路径
            name: 数据集名称（可选，默认使用文件名）
            rating_scale: 评分范围（可选，会自动检测）
            format_type: 数据格式类型（可选，支持自动检测）

        Returns:
            BaseDataset: 加载的数据集对象
        """
        from pathlib import Path

        # 验证文件路径
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"数据文件不存在: {path}")

        # 设置默认名称
        if name is None:
            name = path_obj.stem

        # 创建通用数据集实例
        if format_type == "auto":
            # 先创建临时实例进行格式检测
            temp_dataset = GenericDataset(path, name)
            detected_format = temp_dataset.auto_detect_format()
            logger.info(f"自动检测到数据格式: {detected_format}")

            # 使用检测到的格式创建最终实例
            dataset = GenericDataset(path, name,
                                rating_scale or (1, 5),
                                detected_format)
        else:
            dataset = GenericDataset(path, name,
                                rating_scale or (1, 5),
                                format_type)

        # 加载数据
        try:
            dataset.load_raw_data()

            # 自动检测评分范围（如果未指定）
            if rating_scale is None:
                detected_scale = dataset.auto_detect_rating_scale()
                dataset.rating_scale = detected_scale
                logger.info(f"自动检测到评分范围: {detected_scale}")

            dataset.get_statistics()
            logger.info(f"成功加载数据集 {name} from {path}")
            return dataset

        except Exception as e:
            logger.error(f"加载数据集失败: {str(e)}")
            raise

    @classmethod
    def quick_load(cls, path: str) -> BaseDataset:
        """
        快速加载方法，只需要提供路径

        Args:
            path: 数据文件路径

        Returns:
            BaseDataset: 加载的数据集对象
        """
        return cls.load_dataset_from_path(path)

    # 4. 修改现有的 load_dataset 方法
    # 将现有的 load_dataset 方法替换为：

    @classmethod
    def load_dataset(cls, name: str, path: str) -> BaseDataset:
        """
        加载指定数据集（支持路径直接加载）

        Args:
            name: 数据集名称或文件路径
            path: 数据文件路径

        Returns:
            BaseDataset: 加载的数据集对象
        """
        from pathlib import Path

        # 如果name是一个有效的文件路径，直接使用路径加载
        if Path(name).exists():
            return cls.load_dataset_from_path(name)
        elif Path(path).exists():
            # 如果path存在但name不是路径，可能是想要使用路径加载
            # 检查name是否是预定义的数据集类型
            name_lower = name.lower().replace('-', '').replace('_', '')
            if name_lower not in cls.DATASET_REGISTRY:
                logger.info(f"未找到预定义数据集类型 {name}，尝试从路径直接加载")
                return cls.load_dataset_from_path(path, name)

        # 标准化数据集名称
        name_lower = name.lower().replace('-', '').replace('_', '')

        # 自动识别数据集类型
        dataset_class = cls._identify_dataset_type(name_lower, path)

        if dataset_class is None:
            # 如果无法识别类型，尝试使用通用类
            logger.warning(f"未知数据集类型 {name}，尝试使用通用加载器")
            return cls.load_dataset_from_path(path, name)

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

