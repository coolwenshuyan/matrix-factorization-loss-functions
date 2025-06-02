# 配置化数据集管理 - 简单易用版本


import logging
import sys

# 添加data模块路径（根据你的文件结构调整）
sys.path.append('data')

import logging
from pathlib import Path
from data.loader import DatasetLoader
from data.dataset import MovieLens100K
from data.data_manager import DataManager

# 配置日志
logging.basicConfig(level=logging.INFO)

# =====================================================
# 第一步：数据集配置
# =====================================================

# 数据集配置字典 - 在这里添加你的所有数据集
DATASETS_CONFIG = {
    'M100K_test': {
        'raw_data_path': 'dataset/20201202M100K_data_all_random.txt',  # 更新为相对路径
        'preprocessed_dir': './preprocessed_data/M100K_test',
        'description': 'M100K_test评分数据集'
    },
    'Netflix': {
        'raw_data_path': 'dataset/20201202NetFlix_data_all_random.txt',  # 更新为相对路径
        'preprocessed_dir': './preprocessed_data/Netflix',
        'description': 'Netflix评分数据集'
    },
    'MovieLens1M': {
        'raw_data_path': 'dataset/moive1M20221009randombigthan20bigthan20userbigandeq300.txt',
        'preprocessed_dir': './preprocessed_data/MovieLens1M',
        'description': 'MovieLens 1M评分数据集'
    },
    'AmazonMI': {
        'raw_data_path': 'dataset/Amazon_Musical_Instruments20220608random.txt',
        'preprocessed_dir': './preprocessed_data/AmazonMI',
        'description': '亚马逊乐器评分数据集'
    },
    'CiaoDVD': {
        'raw_data_path': 'dataset/ciaodvd20220530random.txt',
        'preprocessed_dir': './preprocessed_data/CiaoDVD',
        'description': 'CiaoDVD评分数据集'
    },
    'Epinions': {
        'raw_data_path': 'dataset/Epinions20220531random.txt',
        'preprocessed_dir': './preprocessed_data/Epinions',
        'description': 'Epinions评分数据集'
    },
    'FilmTrust': {
        'raw_data_path': 'dataset/flimtrust20220604random.txt',
        'preprocessed_dir': './preprocessed_data/FilmTrust',
        'description': 'FilmTrust评分数据集'
    },
    'MovieTweetings': {
        'raw_data_path': 'dataset/moivetweetings20220511random.txt',
        'preprocessed_dir': './preprocessed_data/MovieTweetings',
        'description': 'MovieTweetings评分数据集'
    }
}


# =====================================================
# 第二步：注册所有数据集
# =====================================================

def register_all_datasets():
    """一次性注册所有数据集"""
    for dataset_name in DATASETS_CONFIG.keys():
        DatasetLoader.register_dataset(dataset_name, MovieLens100K)
    print(f"✓ 已注册 {len(DATASETS_CONFIG)} 个数据集")

# =====================================================
# 第三步：核心管理函数
# =====================================================

def preprocess_dataset(dataset_name):
    """
    预处理指定数据集

    Args:
        dataset_name: 数据集名称（在DATASETS_CONFIG中定义）
    """
    if dataset_name not in DATASETS_CONFIG:
        print(f"❌ 错误：未知数据集 '{dataset_name}'")
        print(f"可用数据集：{list(DATASETS_CONFIG.keys())}")
        return False

    config = DATASETS_CONFIG[dataset_name]
    print(f"🔄 开始预处理：{config['description']}")

    try:
        # 检查原始数据文件是否存在
        if not Path(config['raw_data_path']).exists():
            print(f"❌ 原始数据文件不存在：{config['raw_data_path']}")
            return False

        # 创建数据管理器并预处理
        data_manager = DataManager()
        data_manager.load_dataset(dataset_name, config['raw_data_path'])
        data_manager.preprocess()

        # 保存预处理结果
        data_manager.save_preprocessed_data(config['preprocessed_dir'])

        print(f"✅ 预处理完成：{config['preprocessed_dir']}")
        return True

    except Exception as e:
        print(f"❌ 预处理失败：{e}")
        return False

def load_dataset(dataset_name):
    """
    加载数据集（智能加载：优先使用预处理文件，不存在则自动预处理）

    Args:
        dataset_name: 数据集名称

    Returns:
        DataManager对象，失败返回None
    """
    if dataset_name not in DATASETS_CONFIG:
        print(f"❌ 错误：未知数据集 '{dataset_name}'")
        print(f"可用数据集：{list(DATASETS_CONFIG.keys())}")
        return None

    config = DATASETS_CONFIG[dataset_name]
    preprocessed_dir = config['preprocessed_dir']

    # 检查预处理文件是否存在
    if Path(preprocessed_dir).exists():
        try:
            print(f"📁 发现预处理文件，直接加载：{config['description']}")
            data_manager = DataManager()
            data_manager.load_preprocessed_data(preprocessed_dir)
            print(f"✅ 加载成功：{dataset_name}")
            return data_manager
        except Exception as e:
            print(f"⚠️  预处理文件损坏，重新预处理：{e}")

    # 预处理文件不存在或损坏，重新预处理
    print(f"🔄 预处理文件不存在，开始预处理：{config['description']}")
    if preprocess_dataset(dataset_name):
        return load_dataset(dataset_name)  # 递归调用，加载刚预处理的数据
    else:
        return None

def list_datasets():
    """列出所有可用的数据集"""
    print("\n📋 可用数据集列表：")
    print("=" * 60)

    for i, (name, config) in enumerate(DATASETS_CONFIG.items(), 1):
        status = "✅ 已预处理" if Path(config['preprocessed_dir']).exists() else "⏳ 未预处理"
        print(f"{i:2d}. {name:<15} - {config['description']} [{status}]")

    print("=" * 60)

def remove_preprocessed_data(dataset_name):
    """删除指定数据集的预处理文件"""
    if dataset_name not in DATASETS_CONFIG:
        print(f"❌ 错误：未知数据集 '{dataset_name}'")
        return False

    import shutil
    preprocessed_dir = DATASETS_CONFIG[dataset_name]['preprocessed_dir']

    if Path(preprocessed_dir).exists():
        shutil.rmtree(preprocessed_dir)
        print(f"🗑️  已删除预处理文件：{dataset_name}")
        return True
    else:
        print(f"📁 预处理文件不存在：{dataset_name}")
        return False

# =====================================================
# 第四步：便捷使用函数
# =====================================================

def quick_start(dataset_name):
    """
    一键启动：注册→加载→返回可用的数据管理器

    Args:
        dataset_name: 数据集名称

    Returns:
        (data_manager, train_iterator, val_iterator, test_iterator)
    """
    print(f"🚀 快速启动数据集：{dataset_name}")

    # 注册数据集
    register_all_datasets()

    # 加载数据集
    data_manager = load_dataset(dataset_name)
    if data_manager is None:
        return None, None, None, None

    # 创建迭代器
    train_iterator = data_manager.get_batch_iterator('train')
    val_iterator = data_manager.get_batch_iterator('val')
    test_iterator = data_manager.get_batch_iterator('test')

    print(f"✅ 数据集 {dataset_name} 准备就绪！")
    return data_manager, train_iterator, val_iterator, test_iterator

def get_dataset_info(dataset_name):
    """获取数据集统计信息"""
    data_manager = load_dataset(dataset_name)
    if data_manager:
        stats = data_manager.get_statistics()
        print(f"\n📊 数据集 {dataset_name} 统计信息：")
        print(f"   用户数：{stats['n_users']}")
        print(f"   物品数：{stats['n_items']}")
        print(f"   评分数：{stats['n_total']}")
        print(f"   平均评分：{stats['rating_mean']:.2f}")
        print(f"   稀疏度：{stats['sparsity']:.2%}")
        return stats
    return None

# =====================================================
# 第五步：使用示例
# =====================================================

def main():
    """主函数 - 展示所有功能"""
    print("🎯 配置化数据集管理系统")
    print("=" * 50)

    # 1. 注册所有数据集
    register_all_datasets()

    # 2. 列出所有数据集
    list_datasets()

    # 3. 演示加载不同数据集
    test_datasets = ['Netflix', 'Amazon_Musical']  # 选择要测试的数据集

    for dataset_name in test_datasets:
        print(f"\n🔍 测试数据集：{dataset_name}")
        print("-" * 30)

        # 方式1：逐步操作
        data_manager = load_dataset(dataset_name)
        if data_manager:
            # 获取统计信息
            get_dataset_info(dataset_name)

            # 获取训练迭代器
            train_iterator = data_manager.get_batch_iterator('train')
            print(f"   训练批次数：{len(train_iterator)}")

            # 示例：迭代第一个批次
            first_batch = next(iter(train_iterator))
            print(f"   第一批次大小：{len(first_batch[0])}")

def quick_usage_examples():
    """快速使用示例"""
    print("\n" + "=" * 50)
    print("🚀 快速使用示例")
    print("=" * 50)

    # 示例1：一键启动
    print("\n1️⃣ 一键启动数据集：")
    dm, train_iter, val_iter, test_iter = quick_start('M100K_test')
    if dm:
        print(f"✅ 训练批次：{len(train_iter)}, 验证批次：{len(val_iter)}, 测试批次：{len(test_iter)}")

    # 示例2：切换数据集
    print("\n2️⃣ 切换到不同数据集：")
    for dataset in ['M100K_test', 'amazon_musical_instruments']:
        print(f"   切换到：{dataset}")
        dm = load_dataset(dataset)
        if dm:
            stats = dm.get_statistics()
            print(f"   ✅ 用户数：{stats['n_users']}, 物品数：{stats['n_items']}")

    # 示例3：批量预处理
    print("\n3️⃣ 批量预处理所有数据集：")
    for dataset_name in DATASETS_CONFIG.keys():
        print(f"   预处理：{dataset_name}")
        success = preprocess_dataset(dataset_name)
        print(f"   结果：{'✅ 成功' if success else '❌ 失败'}")

# =====================================================
# 实际使用模板
# =====================================================

def your_training_code():
    """
    你的训练代码模板
    """
    print("\n🎓 训练代码示例：")

    # 1. 加载数据集
    dataset_name = 'M100K_test'  # 修改这里切换数据集
    data_manager = load_dataset(dataset_name)

    if data_manager is None:
        print("❌ 数据集加载失败")
        return

    # 2. 获取数据迭代器
    train_iterator = data_manager.get_batch_iterator('train', batch_size=128)
    val_iterator = data_manager.get_batch_iterator('val', batch_size=128)
    test_iterator = data_manager.get_batch_iterator('test', batch_size=128)

    # 3. 开始训练（示例）
    print(f"🔥 开始训练，使用数据集：{dataset_name}")
    print(f"   训练批次：{len(train_iterator)}")
    print(f"   验证批次：{len(val_iterator)}")
    print(f"   测试批次：{len(test_iterator)}")

    # 4. 训练循环示例
    for epoch in range(3):  # 示例：训练3个epoch
        print(f"   Epoch {epoch + 1}:")

        # 训练
        for i, (user_ids, item_ids, ratings) in enumerate(train_iterator):
            # 这里放你的模型训练代码
            # model.train_step(user_ids, item_ids, ratings)
            if i >= 2:  # 只演示前3个批次
                break

        # 验证
        for i, (user_ids, item_ids, ratings) in enumerate(val_iterator):
            # 这里放你的模型验证代码
            # loss = model.validate_step(user_ids, item_ids, ratings)
            if i >= 1:  # 只演示前2个批次
                break

        print(f"     ✅ Epoch {epoch + 1} 完成")

if __name__ == "__main__":
    # 运行主程序
    main()

    # 快速使用示例
    # quick_usage_examples()

    # 训练代码示例
    # your_training_code()

    print("\n" + "=" * 50)
    print("📝 使用说明：")
    print("1. 在 DATASETS_CONFIG 中添加你的数据集配置")
    print("2. 使用 load_dataset('数据集名') 加载数据")
    print("3. 使用 quick_start('数据集名') 一键启动")
    print("4. 数据会自动预处理和缓存，下次加载更快")
    print("=" * 50)
