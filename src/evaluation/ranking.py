# src/evaluation/ranking.py
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
import heapq
from collections import defaultdict
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TopKGenerator:
    """高效的Top-K推荐生成器"""
    
    def __init__(self, model, use_multithread: bool = True, n_threads: int = 4):
        """
        初始化Top-K生成器
        
        Args:
            model: 推荐模型
            use_multithread: 是否使用多线程
            n_threads: 线程数
        """
        self.model = model
        self.use_multithread = use_multithread
        self.n_threads = n_threads
    
    def generate_top_k_for_user(self, user_id: int, k: int = 10,
                               candidate_items: Optional[List[int]] = None,
                               exclude_items: Optional[Set[int]] = None) -> List[int]:
        """
        为单个用户生成Top-K推荐
        
        Args:
            user_id: 用户ID
            k: 推荐数量
            candidate_items: 候选物品列表（None表示所有物品）
            exclude_items: 需要排除的物品集合
            
        Returns:
            Top-K物品ID列表
        """
        # 确定候选物品
        if candidate_items is None:
            candidate_items = list(range(self.model.n_items))
        
        if exclude_items:
            candidate_items = [item for item in candidate_items if item not in exclude_items]
        
        if not candidate_items:
            return []
        
        # 批量预测评分
        n_candidates = len(candidate_items)
        if n_candidates <= k:
            # 候选物品数量小于k，直接返回所有
            scores = self.model.predict(
                [user_id] * n_candidates,
                candidate_items
            )
            sorted_indices = np.argsort(scores)[::-1]
            return [candidate_items[i] for i in sorted_indices]
        
        # 使用堆优化Top-K选择
        if n_candidates > 1000:  # 大规模候选集
            return self._generate_top_k_with_heap(user_id, candidate_items, k)
        else:  # 小规模候选集
            scores = self.model.predict(
                [user_id] * n_candidates,
                candidate_items
            )
            top_k_indices = np.argpartition(scores, -k)[-k:]
            top_k_indices = top_k_indices[np.argsort(scores[top_k_indices])[::-1]]
            return [candidate_items[i] for i in top_k_indices]
    
    def _generate_top_k_with_heap(self, user_id: int, 
                                 candidate_items: List[int], 
                                 k: int) -> List[int]:
        """使用堆结构生成Top-K（适用于大规模候选集）"""
        # 使用最小堆维护Top-K
        heap = []
        batch_size = 1000
        
        for i in range(0, len(candidate_items), batch_size):
            batch_items = candidate_items[i:i + batch_size]
            batch_scores = self.model.predict(
                [user_id] * len(batch_items),
                batch_items
            )
            
            for item, score in zip(batch_items, batch_scores):
                if len(heap) < k:
                    heapq.heappush(heap, (score, item))
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, (score, item))
        
        # 按分数降序排序
        top_k = sorted(heap, key=lambda x: x[0], reverse=True)
        return [item for score, item in top_k]
    
    def generate_top_k_for_users(self, user_ids: List[int], k: int = 10,
                                exclude_known: bool = True,
                                known_items: Optional[Dict[int, List[int]]] = None,
                                show_progress: bool = True) -> Dict[int, List[int]]:
        """
        为多个用户生成Top-K推荐
        
        Args:
            user_ids: 用户ID列表
            k: 推荐数量
            exclude_known: 是否排除已知物品
            known_items: 用户已知物品字典
            show_progress: 是否显示进度条
            
        Returns:
            用户ID到推荐列表的字典
        """
        recommendations = {}
        
        if self.use_multithread and len(user_ids) > 10:
            # 多线程处理
            with ThreadPoolExecutor(max_workers=self.n_threads) as executor:
                # 提交任务
                future_to_user = {}
                for user_id in user_ids:
                    exclude_items = None
                    if exclude_known and known_items and user_id in known_items:
                        exclude_items = set(known_items[user_id])
                    
                    future = executor.submit(
                        self.generate_top_k_for_user,
                        user_id, k, None, exclude_items
                    )
                    future_to_user[future] = user_id
                
                # 收集结果
                if show_progress:
                    pbar = tqdm(total=len(user_ids), desc="生成推荐")
                
                for future in as_completed(future_to_user):
                    user_id = future_to_user[future]
                    try:
                        rec_items = future.result()
                        recommendations[user_id] = rec_items
                    except Exception as e:
                        logger.error(f"为用户 {user_id} 生成推荐时出错: {str(e)}")
                        recommendations[user_id] = []
                    
                    if show_progress:
                        pbar.update(1)
                
                if show_progress:
                    pbar.close()
        else:
            # 单线程处理
            if show_progress:
                user_ids = tqdm(user_ids, desc="生成推荐")
            
            for user_id in user_ids:
                exclude_items = None
                if exclude_known and known_items and user_id in known_items:
                    exclude_items = set(known_items[user_id])
                
                rec_items = self.generate_top_k_for_user(
                    user_id, k, None, exclude_items
                )
                recommendations[user_id] = rec_items
        
        return recommendations


class RankingEvaluator:
    """排序评估器"""
    
    def __init__(self, model, k_values: List[int] = [5, 10, 20]):
        """
        初始化排序评估器
        
        Args:
            model: 推荐模型
            k_values: 评估的K值列表
        """
        self.model = model
        self.k_values = k_values
        self.topk_generator = TopKGenerator(model)
    
    def evaluate_user_ranking(self, user_id: int, true_items: List[int],
                            k: int = 10, 
                            candidate_items: Optional[List[int]] = None) -> Dict[str, float]:
        """
        评估单个用户的排序质量
        
        Args:
            user_id: 用户ID
            true_items: 真实相关物品列表
            k: Top-K值
            candidate_items: 候选物品列表
            
        Returns:
            评估指标字典
        """
        # 生成推荐
        rec_items = self.topk_generator.generate_top_k_for_user(
            user_id, k, candidate_items
        )
        
        if not rec_items or not true_items:
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'hit': 0.0
            }
        
        # 计算指标
        rec_set = set(rec_items)
        true_set = set(true_items)
        n_relevant = len(rec_set & true_set)
        
        precision = n_relevant / len(rec_items)
        recall = n_relevant / len(true_items)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        hit = 1.0 if n_relevant > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'hit': hit
        }
    
    def evaluate_ranking_efficiency(self, n_users: int = 100,
                                  k: int = 10) -> Dict[str, float]:
        """
        评估排序效率
        
        Args:
            n_users: 测试用户数
            k: Top-K值
            
        Returns:
            效率指标
        """
        import time
        
        # 随机选择用户
        test_users = np.random.choice(self.model.n_users, n_users, replace=False)
        
        # 测试单线程性能
        start_time = time.time()
        single_thread_gen = TopKGenerator(self.model, use_multithread=False)
        single_thread_gen.generate_top_k_for_users(test_users.tolist(), k, show_progress=False)
        single_thread_time = time.time() - start_time
        
        # 测试多线程性能
        start_time = time.time()
        multi_thread_gen = TopKGenerator(self.model, use_multithread=True, n_threads=4)
        multi_thread_gen.generate_top_k_for_users(test_users.tolist(), k, show_progress=False)
        multi_thread_time = time.time() - start_time
        
        return {
            'single_thread_time': single_thread_time,
            'multi_thread_time': multi_thread_time,
            'speedup': single_thread_time / multi_thread_time,
            'users_per_second_single': n_users / single_thread_time,
            'users_per_second_multi': n_users / multi_thread_time
        }


class DiversityOptimizer:
    """多样性优化器"""
    
    def __init__(self, item_features: Optional[np.ndarray] = None,
                 similarity_threshold: float = 0.5):
        """
        初始化多样性优化器
        Args:
           item_features: 物品特征矩阵
           similarity_threshold: 相似度阈值
       """
        self.item_features = item_features
        self.similarity_threshold = similarity_threshold

        # 预计算物品相似度（如果提供了特征）
        if item_features is not None:
            self._precompute_similarities()
   
    def _precompute_similarities(self):
        """预计算物品间的相似度"""
        # 使用余弦相似度
        norms = np.linalg.norm(self.item_features, axis=1)
        self.item_features_normalized = self.item_features / (norms[:, np.newaxis] + 1e-8)
        
        # 为了内存效率，不预计算完整的相似度矩阵
        # 而是在需要时动态计算

    def diversify_recommendations(self, original_recs: List[int],
                                scores: List[float],
                                k: int,
                                lambda_diversity: float = 0.5) -> List[int]:
        """
        使用MMR（Maximal Marginal Relevance）算法增加推荐多样性
        
        Args:
            original_recs: 原始推荐列表
            scores: 对应的分数
            k: 最终推荐数量
            lambda_diversity: 多样性权重（0-1之间）
            
        Returns:
            多样化后的推荐列表
        """
        if self.item_features is None or len(original_recs) <= k:
            return original_recs[:k]
        
        # 归一化分数
        scores = np.array(scores)
        if np.max(scores) > np.min(scores):
            scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        
        # MMR算法
        selected = []
        candidates = list(zip(original_recs, scores))
        
        # 选择第一个物品（最高分）
        first_item = max(candidates, key=lambda x: x[1])
        selected.append(first_item[0])
        candidates.remove(first_item)
        
        # 迭代选择剩余物品
        while len(selected) < k and candidates:
            mmr_scores = []
            
            for item, score in candidates:
                # 相关性分数
                relevance = score
                
                # 多样性分数（与已选择物品的最大相似度）
                max_sim = 0
                for selected_item in selected:
                    sim = self._compute_similarity(item, selected_item)
                    max_sim = max(max_sim, sim)
                
                # MMR分数
                mmr = lambda_diversity * relevance - (1 - lambda_diversity) * max_sim
                mmr_scores.append((item, mmr))
            
            # 选择MMR分数最高的物品
            best_item = max(mmr_scores, key=lambda x: x[1])[0]
            selected.append(best_item)
            
            # 从候选中移除
            candidates = [(item, score) for item, score in candidates if item != best_item]
        
        return selected
   
    def _compute_similarity(self, item1: int, item2: int) -> float:
        """计算两个物品的相似度"""
        if self.item_features is None:
            return 0.0
        
        if item1 >= len(self.item_features) or item2 >= len(self.item_features):
            return 0.0
        
        # 余弦相似度
        feat1 = self.item_features_normalized[item1]
        feat2 = self.item_features_normalized[item2]
        
        return float(np.dot(feat1, feat2))