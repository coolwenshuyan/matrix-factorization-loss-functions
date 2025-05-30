# src/models/utils.py
import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from src.models.base_mf import BaseMatrixFactorization  # 现在可以使用绝对导入
from src.models.mf_sgd import MatrixFactorizationSGD  # 添加这行导入语句


def visualize_embeddings(model: BaseMatrixFactorization,
                        embedding_type: str = 'user',
                        method: str = 'pca',
                        n_components: int = 2,
                        n_samples: Optional[int] = None,
                        labels: Optional[np.ndarray] = None,
                        save_path: Optional[str] = None):
    """
    可视化用户或物品的嵌入
    
    Args:
        model: 矩阵分解模型
        embedding_type: 'user' 或 'item'
        method: 降维方法 ('pca', 'tsne')
        n_components: 降维后的维度
        n_samples: 采样数量（None表示全部）
        labels: 标签（用于着色）
        save_path: 保存路径
    """
    # 获取嵌入
    if embedding_type == 'user':
        embeddings = model.user_factors
        title = 'User Embeddings'
    else:
        embeddings = model.item_factors
        title = 'Item Embeddings'
    
    # 采样
    if n_samples is not None and n_samples < len(embeddings):
        indices = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[indices]
        if labels is not None:
            labels = labels[indices]
    
    # 降维
    if method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=42)
    else:
        raise ValueError(f"未知的降维方法: {method}")
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    
    if labels is not None:
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                            c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
    else:
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
    
    plt.title(f'{title} ({method.upper()})')
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_history(model: BaseMatrixFactorization,
                         metrics: List[str] = ['loss', 'val_loss'],
                         save_path: Optional[str] = None):
    """
    绘制训练历史
    
    Args:
        model: 矩阵分解模型
        metrics: 要绘制的指标列表
        save_path: 保存路径
    """
    history = model.train_history
    
    plt.figure(figsize=(12, 5))
    
    for i, metric in enumerate(metrics):
        if metric in history and len(history[metric]) > 0:
            plt.subplot(1, len(metrics), i+1)
            plt.plot(history[metric])
            plt.title(f'Model {metric}')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_factors(model: BaseMatrixFactorization) -> Dict[str, any]:
    """
    分析潜在因子的统计特性
    
    Args:
        model: 矩阵分解模型
        
    Returns:
        分析结果字典
    """
    results = {}
    
    # 用户因子分析
    user_norms = np.linalg.norm(model.user_factors, axis=1)
    results['user_factors'] = {
        'mean_norm': np.mean(user_norms),
        'std_norm': np.std(user_norms),
        'min_norm': np.min(user_norms),
        'max_norm': np.max(user_norms),
        'sparsity': np.mean(np.abs(model.user_factors) < 1e-3)
    }
    
    # 物品因子分析
    item_norms = np.linalg.norm(model.item_factors, axis=1)
    results['item_factors'] = {
        'mean_norm': np.mean(item_norms),
        'std_norm': np.std(item_norms),
        'min_norm': np.min(item_norms),
        'max_norm': np.max(item_norms),
        'sparsity': np.mean(np.abs(model.item_factors) < 1e-3)
    }
    
    # 因子相关性
    user_corr = np.corrcoef(model.user_factors.T)
    item_corr = np.corrcoef(model.item_factors.T)
    
    results['factor_correlation'] = {
        'user_mean_corr': np.mean(np.abs(user_corr[np.triu_indices_from(user_corr, k=1)])),
        'item_mean_corr': np.mean(np.abs(item_corr[np.triu_indices_from(item_corr, k=1)]))
    }
    
    # 偏差项分析
    if model.use_bias:
        results['bias'] = {
            'user_bias_mean': np.mean(model.user_bias),
            'user_bias_std': np.std(model.user_bias),
            'item_bias_mean': np.mean(model.item_bias),
            'item_bias_std': np.std(model.item_bias)
        }
    
    return results


def plot_factor_heatmap(model: BaseMatrixFactorization,
                       factor_type: str = 'user',
                       n_samples: int = 50,
                       save_path: Optional[str] = None):
    """
    绘制因子热力图
    
    Args:
        model: 矩阵分解模型
        factor_type: 'user' 或 'item'
        n_samples: 显示的样本数量
        save_path: 保存路径
    """
    if factor_type == 'user':
        factors = model.user_factors[:n_samples]
        title = 'User Factor Heatmap'
    else:
        factors = model.item_factors[:n_samples]
        title = 'Item Factor Heatmap'
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(factors, cmap='coolwarm', center=0, 
                xticklabels=range(model.n_factors),
                yticklabels=range(n_samples))
    plt.title(title)
    plt.xlabel('Factor Dimension')
    plt.ylabel(f'{factor_type.capitalize()} Index')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


# 测试代码
if __name__ == "__main__":
    # 创建测试数据
    n_users, n_items = 1000, 500
    n_factors = 10
    
    # 初始化模型
    from src.losses.standard import L2Loss
    from src.models.regularizers import L2Regularizer  # 添加这行导入语句
    
    model = MatrixFactorizationSGD(
        n_users=n_users,
        n_items=n_items,
        n_factors=n_factors,
        learning_rate=0.01,
        regularizer=L2Regularizer(lambda_reg=0.01),
        loss_function=L2Loss(),
        use_bias=True,
        clip_gradient=5.0,
        momentum=0.9
    )
    
    # 初始化参数
    model.initialize_parameters()
    
    # 创建模拟训练数据
    n_ratings = 10000
    train_data = np.column_stack([
        np.random.randint(0, n_users, n_ratings),
        np.random.randint(0, n_items, n_ratings),
        np.random.normal(3.5, 1.0, n_ratings)
    ])
    
    # 训练模型
    model.fit(train_data, n_epochs=10, verbose=1)
    
    # 预测
    test_users = np.array([0, 1, 2])
    test_items = np.array([0, 1, 2])
    predictions = model.predict(test_users, test_items)
    print(f"预测评分: {predictions}")
    
    # 保存模型
    model.save_model('test_model.npz')
    
    # 分析因子
    analysis = analyze_factors(model)
    print(f"因子分析结果: {analysis}")
