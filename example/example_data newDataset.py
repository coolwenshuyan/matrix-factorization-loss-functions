# 豆瓣电影数据集使用 - 完整代码

import logging
import sys
import os

# 添加上级目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from data.loader import DatasetLoader
from data.dataset import FilmTrust
from data.data_manager import DataManager

# 配置日志
logging.basicConfig(level=logging.INFO)

def main():
    """
    使用豆瓣电影数据集的完整流程
    """
    print("=" * 50)
    print("豆瓣电影数据集使用示例")
    print("=" * 50)

    # 步骤1: 注册豆瓣电影数据集
    print("1. 注册豆瓣电影数据集...")
    # 确保注册成功
    DatasetLoader.register_dataset('doubanmovies', FilmTrust)
    # 验证注册是否成功
    print(f"✓ 注册成功，当前注册的数据集: {list(DatasetLoader.DATASET_REGISTRY.keys())}")

    # 确认douban_movies是否在注册表中
    if 'doubanmovies' in DatasetLoader.DATASET_REGISTRY:
        print("✓ 确认'doubanmovies'已在注册表中")
    else:
        print("✗ 警告: 'doubanmovies'未在注册表中找到")

    # 步骤2: 配置参数
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

    # 步骤3: 创建数据管理器
    print("\n2. 创建数据管理器...")
    data_manager = DataManager(config)
    print("✓ 创建成功")

    # 步骤4: 加载和预处理数据
    print("\n3. 加载豆瓣电影数据...")
    data_path = 'E:\工作资料\科研\论文\写作\使用混合分段损失函数增强矩阵分解用于推荐\矩阵分解损失函数code\dataset\small_flimtrust20220604random_10percent.txt'  # 你的数据文件路径

    try:
        # 一行代码完成加载和预处理
        data_manager.load_dataset('doubanmovies', data_path).preprocess()
        print("✓ 数据加载和预处理成功")

        # 步骤5: 查看数据摘要
        print("\n4. 数据摘要:")
        data_manager.print_summary()

        # 步骤6: 获取数据集
        print("\n5. 获取数据集...")
        train_data, val_data, test_data = data_manager.get_splits()
        print(f"✓ 训练集: {len(train_data)} 条")
        print(f"✓ 验证集: {len(val_data)} 条")
        print(f"✓ 测试集: {len(test_data)} 条")

        # 步骤7: 获取批处理迭代器
        print("\n6. 创建训练批处理迭代器...")
        train_iterator = data_manager.get_batch_iterator('train')
        print(f"✓ 总批次数: {len(train_iterator)}")

        # 步骤8: 示例 - 迭代前几个批次
        print("\n7. 示例 - 前3个训练批次:")
        for i, (user_ids, item_ids, ratings) in enumerate(train_iterator):
            print(f"批次 {i+1}: 批大小={len(user_ids)}, "
                  f"用户ID范围=[{user_ids.min()}-{user_ids.max()}], "
                  f"电影ID范围=[{item_ids.min()}-{item_ids.max()}], "
                  f"评分范围=[{ratings.min():.2f}-{ratings.max():.2f}]")

            if i >= 2:  # 只显示前3个批次
                break

        print("\n✓ 豆瓣电影数据集使用完成!")
        print("现在你可以使用这些数据训练推荐模型了")

    except FileNotFoundError:
        print(f"✗ 错误: 找不到数据文件 '{data_path}'")
        print("请确保:")
        print("1. 文件路径正确")
        print("2. 文件存在")
        print("3. 文件格式为每行: [user_id, item_id, rating]")

    except Exception as e:
        print(f"✗ 错误: {e}")
        print("请检查数据文件格式是否正确")

def quick_usage():
    """
    最简化的使用方式
    """
    print("\n" + "=" * 50)
    print("最简化使用方式（3行代码）")
    print("=" * 50)

    # 注册
    DatasetLoader.register_dataset('doubanmovies', FilmTrust)

    # 使用
    data_manager = DataManager()
    data_manager.load_dataset('doubanmovies', 'E:\工作资料\科研\论文\写作\使用混合分段损失函数增强矩阵分解用于推荐\矩阵分解损失函数code\dataset\small_flimtrust20220604random_10percent.txt').preprocess()

    # 获取训练迭代器
    train_iterator = data_manager.get_batch_iterator('train')

    print(f"✓ 完成! 可用于训练的批次数: {len(train_iterator)}")

if __name__ == "__main__":
    # 运行完整示例
    # main()

    # 运行简化示例
    try:
        quick_usage()
    except:
        print("简化示例需要数据文件存在")

    print("\n" + "=" * 50)
    print("重要提醒:")
    print("1. 确保 douban_movies_data.txt 文件存在")
    print("2. 数据格式必须是每行: [user_id, item_id, rating]")
    print("3. 例如: [1, 2501, 4.5]")
    print("=" * 50)
