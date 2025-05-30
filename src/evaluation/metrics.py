# src/evaluation/metrics.py
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import warnings
from collections import defaultdict


class Metric(ABC):
    """评估指标基类"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """计算指标值"""
        pass
    
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        return self.calculate(y_true, y_pred, **kwargs)


# ============= 预测准确性指标 =============

class MAE(Metric):
    """平均绝对误差 (Mean Absolute Error)"""
    
    def __init__(self):
        super().__init__("MAE")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算MAE
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MAE值
        """
        return float(np.mean(np.abs(y_true - y_pred)))


class RMSE(Metric):
    """均方根误差 (Root Mean Square Error)"""
    
    def __init__(self):
        super().__init__("RMSE")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算RMSE
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            RMSE值
        """
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


class MSE(Metric):
    """均方误差 (Mean Square Error)"""
    
    def __init__(self):
        super().__init__("MSE")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算MSE
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            MSE值
        """
        return float(np.mean((y_true - y_pred) ** 2))


class R2Score(Metric):
    """决定系数 (R-squared)"""
    
    def __init__(self):
        super().__init__("R2")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算R²分数
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            R²值
        """
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return float(1 - (ss_res / ss_tot))


# ============= 排序质量指标 =============

class RankingMetric(Metric):
    """排序指标基类"""
    
    def __init__(self, name: str, k: int = 10):
        super().__init__(name)
        self.k = k


class HitRate(RankingMetric):
    """命中率 (Hit Rate @ K)"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"HR@{k}", k)
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算Hit Rate@K
        
        Args:
            y_true: 用户-物品交互矩阵或测试集列表
            y_pred: 预测的Top-K推荐列表
            
        Returns:
            HR@K值
        """
        if 'user_items' in kwargs:
            # 处理用户-物品字典格式
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations']
            )
        else:
            # 处理矩阵格式
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]]) -> float:
        """从字典格式计算HR@K"""
        hits = 0
        total = 0
        
        for user_id, true_items in user_items.items():
            if user_id in recommendations:
                rec_items = recommendations[user_id][:self.k]
                if any(item in true_items for item in rec_items):
                    hits += 1
            total += 1
        
        return hits / total if total > 0 else 0.0
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算HR@K"""
        n_users = y_true.shape[0]
        hits = 0
        
        for user_idx in range(n_users):
            true_items = np.where(y_true[user_idx] > 0)[0]
            if len(true_items) > 0:
                # 获取Top-K推荐
                top_k_items = np.argsort(y_pred[user_idx])[::-1][:self.k]
                if np.any(np.isin(top_k_items, true_items)):
                    hits += 1
        
        return hits / n_users if n_users > 0 else 0.0


class Precision(RankingMetric):
    """精确率 (Precision @ K)"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"Precision@{k}", k)
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算Precision@K
        
        Args:
            y_true: 用户-物品交互矩阵或测试集列表
            y_pred: 预测的Top-K推荐列表
            
        Returns:
            Precision@K值
        """
        if 'user_items' in kwargs:
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations']
            )
        else:
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]]) -> float:
        """从字典格式计算Precision@K"""
        precisions = []
        
        for user_id, true_items in user_items.items():
            if user_id in recommendations:
                rec_items = recommendations[user_id][:self.k]
                n_relevant = sum(1 for item in rec_items if item in true_items)
                precisions.append(n_relevant / len(rec_items) if rec_items else 0)
            else:
                precisions.append(0)
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算Precision@K"""
        n_users = y_true.shape[0]
        precisions = []
        
        for user_idx in range(n_users):
            true_items = np.where(y_true[user_idx] > 0)[0]
            if len(true_items) > 0:
                top_k_items = np.argsort(y_pred[user_idx])[::-1][:self.k]
                n_relevant = np.sum(np.isin(top_k_items, true_items))
                precisions.append(n_relevant / self.k)
        
        return np.mean(precisions) if precisions else 0.0


class Recall(RankingMetric):
    """召回率 (Recall @ K)"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"Recall@{k}", k)
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算Recall@K
        
        Args:
            y_true: 用户-物品交互矩阵或测试集列表
            y_pred: 预测的Top-K推荐列表
            
        Returns:
            Recall@K值
        """
        if 'user_items' in kwargs:
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations']
            )
        else:
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]]) -> float:
        """从字典格式计算Recall@K"""
        recalls = []
        
        for user_id, true_items in user_items.items():
            if len(true_items) > 0 and user_id in recommendations:
                rec_items = recommendations[user_id][:self.k]
                n_relevant = sum(1 for item in rec_items if item in true_items)
                recalls.append(n_relevant / len(true_items))
            else:
                recalls.append(0)
        
        return np.mean(recalls) if recalls else 0.0
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算Recall@K"""
        n_users = y_true.shape[0]
        recalls = []
        
        for user_idx in range(n_users):
            true_items = np.where(y_true[user_idx] > 0)[0]
            if len(true_items) > 0:
                top_k_items = np.argsort(y_pred[user_idx])[::-1][:self.k]
                n_relevant = np.sum(np.isin(top_k_items, true_items))
                recalls.append(n_relevant / len(true_items))
        
        return np.mean(recalls) if recalls else 0.0


class MAP(RankingMetric):
    """平均精度均值 (Mean Average Precision @ K)"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"MAP@{k}", k)
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算MAP@K
        
        Args:
            y_true: 用户-物品交互矩阵或测试集列表
            y_pred: 预测的Top-K推荐列表
            
        Returns:
            MAP@K值
        """
        if 'user_items' in kwargs:
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations']
            )
        else:
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _calculate_average_precision(self, true_items: List[int], 
                                   rec_items: List[int]) -> float:
        """计算单个用户的Average Precision"""
        if not true_items or not rec_items:
            return 0.0
        
        ap = 0.0
        n_relevant = 0
        
        for i, item in enumerate(rec_items[:self.k]):
            if item in true_items:
                n_relevant += 1
                ap += n_relevant / (i + 1)
        
        return ap / min(len(true_items), self.k)
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]]) -> float:
        """从字典格式计算MAP@K"""
        aps = []
        
        for user_id, true_items in user_items.items():
            if user_id in recommendations:
                rec_items = recommendations[user_id]
                ap = self._calculate_average_precision(true_items, rec_items)
                aps.append(ap)
            else:
                aps.append(0)
        
        return np.mean(aps) if aps else 0.0
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算MAP@K"""
        n_users = y_true.shape[0]
        aps = []
        
        for user_idx in range(n_users):
            true_items = np.where(y_true[user_idx] > 0)[0].tolist()
            if true_items:
                top_k_indices = np.argsort(y_pred[user_idx])[::-1][:self.k].tolist()
                ap = self._calculate_average_precision(true_items, top_k_indices)
                aps.append(ap)
        
        return np.mean(aps) if aps else 0.0


class NDCG(RankingMetric):
    """归一化折损累积增益 (Normalized Discounted Cumulative Gain @ K)"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"NDCG@{k}", k)
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算NDCG@K
        
        Args:
            y_true: 用户-物品交互矩阵或评分矩阵
            y_pred: 预测的评分或排序
            
        Returns:
            NDCG@K值
        """
        if 'user_items' in kwargs:
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations'],
                kwargs.get('ratings', None)
            )
        else:
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _dcg_at_k(self, scores: np.ndarray, k: int) -> float:
        """计算DCG@K"""
        scores = scores[:k]
        if scores.size == 0:
            return 0.0
        
        # 使用标准DCG公式: sum(score_i / log2(i+2))
        discounts = np.log2(np.arange(len(scores)) + 2)
        return np.sum(scores / discounts)
    
    def _ndcg_at_k(self, true_scores: np.ndarray, predicted_order: np.ndarray, k: int) -> float:
        """计算单个用户的NDCG@K"""
        # 获取预测排序的真实分数
        dcg = self._dcg_at_k(true_scores[predicted_order], k)
        
        # 计算理想排序的DCG
        ideal_order = np.argsort(true_scores)[::-1]
        idcg = self._dcg_at_k(true_scores[ideal_order], k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算NDCG@K"""
        n_users = y_true.shape[0]
        ndcgs = []
        
        for user_idx in range(n_users):
            true_ratings = y_true[user_idx]
            nonzero_indices = np.where(true_ratings > 0)[0]
            
            if len(nonzero_indices) > 0:
                # 获取预测排序
                predicted_order = np.argsort(y_pred[user_idx])[::-1]
                
                # 创建真实评分数组（未交互的为0）
                true_scores = np.zeros(len(y_pred[user_idx]))
                true_scores[nonzero_indices] = true_ratings[nonzero_indices]
                
                ndcg = self._ndcg_at_k(true_scores, predicted_order, self.k)
                ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]],
                            ratings: Optional[Dict[int, Dict[int, float]]] = None) -> float:
        """从字典格式计算NDCG@K"""
        ndcgs = []
        
        for user_id, true_items in user_items.items():
            if user_id in recommendations and true_items:
                rec_items = recommendations[user_id][:self.k]
                
                # 创建相关性分数（如果没有评分，使用二值相关性）
                if ratings and user_id in ratings:
                    scores = [ratings[user_id].get(item, 0) for item in rec_items]
                else:
                    scores = [1 if item in true_items else 0 for item in rec_items]
                
                # 计算DCG
                dcg = self._dcg_at_k(np.array(scores), self.k)
                
                # 计算理想DCG
                ideal_scores = [1] * min(len(true_items), self.k) + \
                              [0] * max(0, self.k - len(true_items))
                idcg = self._dcg_at_k(np.array(ideal_scores), self.k)
                
                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0


class MRR(Metric):
    """平均倒数排名 (Mean Reciprocal Rank)"""
    
    def __init__(self):
        super().__init__("MRR")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算MRR
        
        Args:
            y_true: 用户-物品交互矩阵
            y_pred: 预测的排序
            
        Returns:
            MRR值
        """
        if 'user_items' in kwargs:
            return self._calculate_from_dict(
                kwargs['user_items'], 
                kwargs['recommendations']
            )
        else:
            return self._calculate_from_matrix(y_true, y_pred)
    
    def _calculate_from_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """从矩阵格式计算MRR"""
        n_users = y_true.shape[0]
        reciprocal_ranks = []
        
        for user_idx in range(n_users):
            true_items = np.where(y_true[user_idx] > 0)[0]
            if len(true_items) > 0:
                # 获取排序
                sorted_indices = np.argsort(y_pred[user_idx])[::-1]
                
                # 找到第一个相关物品的位置
                for rank, item_idx in enumerate(sorted_indices):
                    if item_idx in true_items:
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        break
                else:
                    reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def _calculate_from_dict(self, user_items: Dict[int, List[int]], 
                            recommendations: Dict[int, List[int]]) -> float:
        """从字典格式计算MRR"""
        reciprocal_ranks = []
        
        for user_id, true_items in user_items.items():
            if user_id in recommendations and true_items:
                rec_items = recommendations[user_id]
                
                # 找到第一个相关物品的位置
                for rank, item in enumerate(rec_items):
                    if item in true_items:
                        reciprocal_ranks.append(1.0 / (rank + 1))
                        break
                else:
                    reciprocal_ranks.append(0.0)
            else:
                reciprocal_ranks.append(0.0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


# ============= 覆盖度和多样性指标 =============

class CatalogCoverage(Metric):
    """目录覆盖率"""
    
    def __init__(self):
        super().__init__("CatalogCoverage")
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算推荐系统的目录覆盖率
        
        Args:
            y_true: 不使用（为了接口一致性）
            y_pred: 预测的推荐列表
            
        Returns:
            覆盖率（0-1之间）
        """
        if 'recommendations' in kwargs:
            recommendations = kwargs['recommendations']
            n_items = kwargs.get('n_items', None)
            
            # 收集所有被推荐的物品
            recommended_items = set()
            for rec_list in recommendations.values():
                recommended_items.update(rec_list)
            
            # 如果提供了物品总数，使用它；否则从数据推断
            if n_items is None:
                all_items = set()
                if 'user_items' in kwargs:
                    for items in kwargs['user_items'].values():
                        all_items.update(items)
                all_items.update(recommended_items)
                n_items = len(all_items)
            
            return len(recommended_items) / n_items if n_items > 0 else 0.0
        else:
            # 矩阵格式
            n_items = y_pred.shape[1]
            # 找出至少被推荐一次的物品
            recommended = np.any(y_pred > np.percentile(y_pred, 90), axis=0)
            return np.sum(recommended) / n_items


class UserCoverage(Metric):
    """用户覆盖率"""
    
    def __init__(self, min_recommendations: int = 1):
        super().__init__("UserCoverage")
        self.min_recommendations = min_recommendations
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算能够获得有效推荐的用户比例
        
        Args:
            y_true: 不使用
            y_pred: 预测的推荐
            
        Returns:
            用户覆盖率
        """
        if 'recommendations' in kwargs:
            recommendations = kwargs['recommendations']
            n_users = kwargs.get('n_users', len(recommendations))
            
            users_with_recs = sum(1 for rec_list in recommendations.values() 
                                 if len(rec_list) >= self.min_recommendations)
            
            return users_with_recs / n_users if n_users > 0 else 0.0
        else:
            # 矩阵格式
            n_users = y_pred.shape[0]
            users_with_recs = 0
            
            for user_idx in range(n_users):
                # 检查用户是否有足够的高分预测
                high_scores = y_pred[user_idx] > np.mean(y_pred[user_idx])
                if np.sum(high_scores) >= self.min_recommendations:
                    users_with_recs += 1
            
            return users_with_recs / n_users


class Diversity(Metric):
    """推荐多样性（基于物品相似度）"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"Diversity@{k}")
        self.k = k
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算推荐列表的多样性
        
        Args:
            y_true: 不使用
            y_pred: 预测
            
        Returns:
            多样性分数
        """
        if 'item_features' not in kwargs:
            warnings.warn("未提供物品特征，无法计算多样性")
            return 0.0
        
        item_features = kwargs['item_features']
        
        if 'recommendations' in kwargs:
            recommendations = kwargs['recommendations']
            diversities = []
            
            for user_id, rec_items in recommendations.items():
                if len(rec_items) >= 2:
                    rec_items = rec_items[:self.k]
                    diversity = self._calculate_list_diversity(rec_items, item_features)
                    diversities.append(diversity)
            
            return np.mean(diversities) if diversities else 0.0
        else:
            # 矩阵格式
            diversities = []
            n_users = y_pred.shape[0]
            
            for user_idx in range(n_users):
                top_k_items = np.argsort(y_pred[user_idx])[::-1][:self.k]
                if len(top_k_items) >= 2:
                    diversity = self._calculate_list_diversity(top_k_items, item_features)
                    diversities.append(diversity)
            
            return np.mean(diversities) if diversities else 0.0
    
    def _calculate_list_diversity(self, items: List[int], 
                                 item_features: np.ndarray) -> float:
        """计算推荐列表的多样性"""
        n_items = len(items)
        if n_items < 2:
            return 0.0
        
        # 计算所有物品对之间的距离
        distances = []
        for i in range(n_items):
            for j in range(i + 1, n_items):
                if items[i] < len(item_features) and items[j] < len(item_features):
                    # 使用余弦距离
                    feat_i = item_features[items[i]]
                    feat_j = item_features[items[j]]
                    
                    cosine_sim = np.dot(feat_i, feat_j) / (
                        np.linalg.norm(feat_i) * np.linalg.norm(feat_j) + 1e-8
                    )
                    distance = 1 - cosine_sim
                    distances.append(distance)
        
        return np.mean(distances) if distances else 0.0


class Novelty(Metric):
    """推荐新颖度"""
    
    def __init__(self, k: int = 10):
        super().__init__(f"Novelty@{k}")
        self.k = k
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        """
        计算推荐的新颖度（基于物品流行度）
        
        Args:
            y_true: 历史交互
            y_pred: 预测
            
        Returns:
            新颖度分数
        """
        if 'item_popularity' not in kwargs:
            # 从训练数据计算物品流行度
            if 'train_data' in kwargs:
                item_popularity = self._calculate_popularity(kwargs['train_data'])
            else:
                warnings.warn("未提供物品流行度信息")
                return 0.0
        else:
            item_popularity = kwargs['item_popularity']
        
        if 'recommendations' in kwargs:
            recommendations = kwargs['recommendations']
            novelties = []
            
            for user_id, rec_items in recommendations.items():
                rec_items = rec_items[:self.k]
                novelty = self._calculate_list_novelty(rec_items, item_popularity)
                novelties.append(novelty)
            
            return np.mean(novelties) if novelties else 0.0
        else:
            # 矩阵格式
            novelties = []
            n_users = y_pred.shape[0]
            
            for user_idx in range(n_users):
               top_k_items = np.argsort(y_pred[user_idx])[::-1][:self.k]
               novelty = self._calculate_list_novelty(top_k_items, item_popularity)
               novelties.append(novelty)
           
            return np.mean(novelties) if novelties else 0.0
   
    def _calculate_popularity(self, train_data: np.ndarray) -> Dict[int, float]:
        """计算物品流行度"""
        item_counts = defaultdict(int)
        
        if len(train_data.shape) == 2:
            # 矩阵格式
            for item_idx in range(train_data.shape[1]):
                item_counts[item_idx] = np.sum(train_data[:, item_idx] > 0)
        else:
            # 列表格式 [user, item, rating]
            for row in train_data:
                item_counts[int(row[1])] += 1
        
        total_interactions = sum(item_counts.values())
        item_popularity = {
            item: count / total_interactions 
            for item, count in item_counts.items()
        }
        
        return item_popularity

    def _calculate_list_novelty(self, items: List[int], 
                                item_popularity: Dict[int, float]) -> float:
        """计算推荐列表的新颖度"""
        if not items:
            return 0.0
        
        # 使用负对数流行度作为新颖度
        novelties = []
        for item in items:
            pop = item_popularity.get(item, 0.0001)  # 避免log(0)
            novelty = -np.log2(pop)
            novelties.append(novelty)
        
        return np.mean(novelties)


class Serendipity(Metric):
   """推荐惊喜度"""
   
   def __init__(self, k: int = 10):
       super().__init__(f"Serendipity@{k}")
       self.k = k
   
   def calculate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
       """
       计算推荐的惊喜度
       
       Args:
           y_true: 测试集真实交互
           y_pred: 预测
           
       Returns:
           惊喜度分数
       """
       if 'baseline_recommendations' not in kwargs:
           warnings.warn("未提供基线推荐，无法计算惊喜度")
           return 0.0
       
       baseline_recs = kwargs['baseline_recommendations']
       
       if 'recommendations' in kwargs:
           recommendations = kwargs['recommendations']
           user_items = kwargs.get('user_items', {})
           serendipities = []
           
           for user_id in recommendations:
               if user_id in baseline_recs and user_id in user_items:
                   rec_items = recommendations[user_id][:self.k]
                   baseline_items = baseline_recs[user_id][:self.k]
                   true_items = user_items[user_id]
                   
                   serendipity = self._calculate_user_serendipity(
                       rec_items, baseline_items, true_items
                   )
                   serendipities.append(serendipity)
           
           return np.mean(serendipities) if serendipities else 0.0
       else:
           # 矩阵格式实现略
           return 0.0
   
   def _calculate_user_serendipity(self, rec_items: List[int],
                                  baseline_items: List[int],
                                  true_items: List[int]) -> float:
       """计算单个用户的惊喜度"""
       # 找出既相关又不在基线推荐中的物品
       relevant_items = set(rec_items) & set(true_items)
       unexpected_relevant = relevant_items - set(baseline_items)
       
       if not rec_items:
           return 0.0
       
       return len(unexpected_relevant) / len(rec_items)


# ============= 综合评估指标工厂 =============

class MetricFactory:
   """评估指标工厂"""
   
   _metrics = {
       # 预测准确性
       'mae': MAE,
       'rmse': RMSE,
       'mse': MSE,
       'r2': R2Score,
       # 排序质量
       'hr': HitRate,
       'hitrate': HitRate,
       'precision': Precision,
       'recall': Recall,
       'map': MAP,
       'ndcg': NDCG,
       'mrr': MRR,
       # 覆盖度和多样性
       'catalog_coverage': CatalogCoverage,
       'user_coverage': UserCoverage,
       'diversity': Diversity,
       'novelty': Novelty,
       'serendipity': Serendipity
   }
   
   @classmethod
   def create(cls, metric_name: str, **kwargs) -> Metric:
       """
       创建评估指标实例
       
       Args:
           metric_name: 指标名称
           **kwargs: 指标参数
           
       Returns:
           指标实例
       """
       metric_name_lower = metric_name.lower().replace('@', '')
       
       # 处理带K值的指标
       k_value = None
       for k in [5, 10, 20, 50, 100]:
           if str(k) in metric_name_lower:
               k_value = k
               metric_name_lower = metric_name_lower.replace(str(k), '')
               break
       
       if metric_name_lower not in cls._metrics:
           raise ValueError(f"未知的评估指标: {metric_name}")
       
       metric_class = cls._metrics[metric_name_lower]
       
       # 如果是排序指标且提供了K值
       if k_value and hasattr(metric_class, '__init__') and 'k' in metric_class.__init__.__code__.co_varnames:
           kwargs['k'] = k_value
       
       return metric_class(**kwargs)
   
   @classmethod
   def create_multiple(cls, metric_names: List[str], **kwargs) -> Dict[str, Metric]:
       """
       创建多个评估指标
       
       Args:
           metric_names: 指标名称列表
           **kwargs: 共享参数
           
       Returns:
           指标字典
       """
       metrics = {}
       for name in metric_names:
           metrics[name] = cls.create(name, **kwargs)
       return metrics