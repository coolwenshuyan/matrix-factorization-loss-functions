# 推荐系统数据处理模块使用示例

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# 现在导入其他模块
import logging
import numpy as np
from pathlib import Path
from data.data_manager import DataManager
from data.loader import DatasetLoader
from data.iterator import BatchIterator, NegativeSamplingIterator

# 配置日志
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    print("=" * 60)
    print("推荐系统数据处理模块使用示例")
    print("=" * 60)

    # 示例1：使用数据管理器（推荐方式）
    print("\n1. 使用统一数据管理器")
    print("-" * 40)

    # 配置参数
    config = {
        'random_seed': 42,          # 随机种子
        'train_ratio': 0.8,         # 训练集比例
        'val_ratio': 0.1,           # 验证集比例
        'test_ratio': 0.1,          # 测试集比例
        'batch_size': 128,          # 批大小
        'shuffle': True,            # 是否打乱数据
        'center_data': True,        # 是否中心化数据
        'ensure_user_in_train': True,  # 确保每个用户在训练集中至少有一个评分
        'negative_sampling': False,  # 是否使用负采样
        'n_negative': 4             # 负样本数量
    }

    # 创建数据管理器
    data_manager = DataManager(config)

    # 数据文件路径（请根据实际情况修改）
    data_path = 'dataset/20201202M100K_data_all_random.txt'

    try:
        # 加载和预处理数据（链式调用）
        data_manager.load_dataset('movielens100k', data_path).preprocess()

        # 打印数据摘要
        data_manager.print_summary()

        # 获取数据集划分
        train_data, val_data, test_data = data_manager.get_splits()
        print(f"\n数据划分结果:")
        print(f"训练集: {len(train_data)} 条评分")
        print(f"验证集: {len(val_data)} 条评分")
        print(f"测试集: {len(test_data)} 条评分")

        # 获取统计信息
        stats = data_manager.get_statistics()
        print(f"\n关键统计信息:")
        print(f"用户数: {stats['n_users']}")
        print(f"物品数: {stats['n_items']}")
        print(f"稀疏度: {stats['sparsity']:.4f}")
        print(f"平均评分: {stats['rating_mean']:.2f}")

        # 示例2：批处理迭代器使用
        print("\n2. 批处理迭代器使用示例")
        print("-" * 40)

        # 获取训练集的批处理迭代器
        train_iterator = data_manager.get_batch_iterator(
            'train', batch_size=64)
        print(f"训练集总批次数: {len(train_iterator)}")

        # 迭代前几个批次
        for i, (user_ids, item_ids, ratings) in enumerate(train_iterator):
            print(f"批次 {i+1}: 批大小={len(user_ids)}")
            print(f"  用户ID范围: [{user_ids.min()}, {user_ids.max()}]")
            print(f"  物品ID范围: [{item_ids.min()}, {item_ids.max()}]")
            print(f"  评分范围: [{ratings.min():.2f}, {ratings.max():.2f}]")

            if i >= 2:  # 只显示前3个批次
                break

        # 示例3：负采样迭代器
        print("\n3. 负采样迭代器使用示例")
        print("-" * 40)

        neg_iterator = data_manager.get_batch_iterator(
            'train',
            batch_size=32,
            negative_sampling=True
        )

        for i, (user_ids, item_ids, ratings, labels) in enumerate(neg_iterator):
            pos_samples = (labels == 1).sum()
            neg_samples = (labels == 0).sum()
            print(f"批次 {i+1}: 正样本={pos_samples}, 负样本={neg_samples}")

            if i >= 1:  # 只显示前2个批次
                break

        # 示例4：保存和加载预处理数据
        print("\n4. 保存和加载预处理数据")
        print("-" * 40)

        # 保存预处理后的数据
        save_dir = './preprocessed_data'
        data_manager.save_preprocessed_data(save_dir)
        print(f"数据已保存到: {save_dir}")

        # 创建新的数据管理器并加载数据
        new_data_manager = DataManager()
        new_data_manager.load_preprocessed_data(save_dir)
        print("预处理数据加载成功!")

        # 验证加载的数据
        new_stats = new_data_manager.get_statistics()
        print(f"加载后的用户数: {new_stats['n_users']}")
        print(f"加载后的物品数: {new_stats['n_items']}")

    except FileNotFoundError:
        print(f"错误: 找不到数据文件 {data_path}")
        print("请确保数据文件路径正确")
    except Exception as e:
        print(f"错误: {str(e)}")

    # 示例5：单独使用各个模块
    print("\n5. 单独使用各个模块")
    print("-" * 40)

    try:
        # 直接使用数据加载器
        from data.dataset import MovieLens100K
        dataset = MovieLens100K(data_path)
        raw_data = dataset.load_raw_data()
        print(f"原始数据形状: {raw_data.shape}")

        # 直接使用预处理器
        from data.preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor(random_seed=42)

        # ID重新索引
        reindexed_data, user_map, user_inv_map, item_map, item_inv_map = preprocessor.reindex_ids(
            raw_data)
        print(f"重新索引后用户数: {len(user_map)}")
        print(f"重新索引后物品数: {len(item_map)}")

        # 数据划分
        train, val, test = preprocessor.split_data(reindexed_data)
        print(f"划分结果: 训练={len(train)}, 验证={len(val)}, 测试={len(test)}")

        # 直接使用批处理迭代器
        batch_iter = BatchIterator(train, batch_size=64, shuffle=True)
        first_batch = next(iter(batch_iter))
        print(f"第一个批次大小: {len(first_batch[0])}")

    except FileNotFoundError:
        print("跳过单独模块示例（数据文件未找到）")
    except Exception as e:
        print(f"单独模块示例出错: {str(e)}")


def example_usage_for_different_datasets():
    """展示如何处理不同数据集的示例"""
    print("\n6. 不同数据集处理示例")
    print("-" * 40)

    # 支持的数据集类型
    supported_datasets = [
        'movielens100k', 'movielens1m', 'netflix',
        'amazonmi', 'ciaodvd', 'epinions',
        'filmtrust', 'movietweetings'
    ]

    print("支持的数据集类型:")
    for i, dataset in enumerate(supported_datasets, 1):
        print(f"  {i}. {dataset}")

    # 配置不同数据集的示例
    dataset_configs = {
        'movielens100k': {
            'rating_scale': (1, 5),
            'description': 'MovieLens 100K数据集，包含10万条电影评分'
        },
        'filmtrust': {
            'rating_scale': (0.5, 4.0),
            'description': 'FilmTrust数据集，评分范围0.5-4.0'
        },
        'movietweetings': {
            'rating_scale': (1, 10),
            'description': 'MovieTweetings数据集，评分范围1-10'
        }
    }

    print("\n数据集配置示例:")
    for name, config in dataset_configs.items():
        print(f"{name}:")
        print(f"  评分范围: {config['rating_scale']}")
        print(f"  描述: {config['description']}")


if __name__ == "__main__":
    main()
    example_usage_for_different_datasets()

    print("\n" + "=" * 60)
    print("使用说明:")
    print("1. 修改 data_path 为你的实际数据文件路径")
    print("2. 确保数据文件格式为 [user_id, item_id, rating]")
    print("3. 根据需要调整配置参数")
    print("4. 运行后会自动完成数据加载、预处理和批处理")
    print("=" * 60)
