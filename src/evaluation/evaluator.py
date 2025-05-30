# src/evaluation/evaluator.py
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import time
from pathlib import Path
import json
from collections import defaultdict

from .metrics import Metric, MetricFactory
from .ranking import TopKGenerator

logger = logging.getLogger(__name__)


class Evaluator:
    """通用评估器"""
    
    def __init__(self, metrics: Union[List[str], Dict[str, Metric]],
                 k_values: List[int] = [5, 10, 20],
                 verbose: bool = True):
        """
        初始化评估器
        
        Args:
            metrics: 评估指标列表或字典
            k_values: K值列表（用于Top-K指标）
            verbose: 是否打印详细信息
        """
        self.k_values = k_values
        self.verbose = verbose
        
        # 初始化指标
        if isinstance(metrics, list):
            self.metrics = self._create_metrics_from_list(metrics)
        else:
            self.metrics = metrics
    
    def _create_metrics_from_list(self, metric_names: List[str]) -> Dict[str, Metric]:
        """从指标名称列表创建指标实例"""
        metrics = {}
        
        for metric_name in metric_names:
            # 处理需要多个K值的指标
            if any(x in metric_name.lower() for x in ['hr', 'precision', 'recall', 'map', 'ndcg']):
                for k in self.k_values:
                    full_name = f"{metric_name}@{k}"
                    metrics[full_name] = MetricFactory.create(metric_name, k=k)
            else:
                metrics[metric_name] = MetricFactory.create(metric_name)
        
        return metrics
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray,
                 additional_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        执行评估
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            additional_data: 额外数据（如用户物品字典、推荐列表等）
            
        Returns:
            评估结果字典
        """
        results = {}
        additional_data = additional_data or {}
        
        if self.verbose:
            logger.info("开始评估...")
            start_time = time.time()
        
        # 计算每个指标
        for metric_name, metric in self.metrics.items():
            try:
                result = metric.calculate(y_true, y_pred, **additional_data)
                results[metric_name] = result
                
                if self.verbose:
                    logger.info(f"{metric_name}: {result:.4f}")
                    
            except Exception as e:
                logger.error(f"计算指标 {metric_name} 时出错: {str(e)}")
                results[metric_name] = None
        
        if self.verbose:
            elapsed_time = time.time() - start_time
            logger.info(f"评估完成，用时: {elapsed_time:.2f}秒")
        
        return results
    
    def evaluate_multiple_models(self, models: Dict[str, Any],
                               test_data: np.ndarray,
                               **kwargs) -> Dict[str, Dict[str, float]]:
        """
        评估多个模型
        
        Args:
            models: 模型字典
            test_data: 测试数据
            **kwargs: 额外参数
            
        Returns:
            所有模型的评估结果
        """
        all_results = {}
        
        for model_name, model in models.items():
            if self.verbose:
                logger.info(f"\n评估模型: {model_name}")
            
            # 生成预测
            if hasattr(model, 'predict_all'):
                y_pred = model.predict_all()
            else:
                # 需要自定义预测逻辑
                raise NotImplementedError(f"模型 {model_name} 需要实现 predict_all 方法")
            
            # 评估
            results = self.evaluate(test_data, y_pred, **kwargs)
            all_results[model_name] = results
        
        return all_results


class ModelEvaluator:
    """模型评估器（专门用于推荐系统模型）"""
    
    def __init__(self, model, test_data: np.ndarray,
                 k_values: List[int] = [5, 10, 20],
                 metrics: Optional[List[str]] = None,
                 n_users_sample: Optional[int] = None):
        """
        初始化模型评估器
        
        Args:
            model: 推荐模型
            test_data: 测试数据 [user_id, item_id, rating]
            k_values: K值列表
            metrics: 评估指标列表
            n_users_sample: 采样用户数（用于加速评估）
        """
        self.model = model
        self.test_data = test_data
        self.k_values = k_values
        self.n_users_sample = n_users_sample
        
        # 默认指标
        if metrics is None:
            metrics = ['mae', 'rmse', 'hr', 'precision', 'recall', 'map', 'ndcg']
        
        self.evaluator = Evaluator(metrics, k_values)
        
        # 准备测试数据
        self._prepare_test_data()
    
    def _prepare_test_data(self):
        """准备测试数据"""
        # 构建用户-物品字典
        self.user_items = defaultdict(list)
        self.user_ratings = defaultdict(dict)
        
        for row in self.test_data:
            user_id = int(row[0])
            item_id = int(row[1])
            rating = float(row[2])
            
            self.user_items[user_id].append(item_id)
            self.user_ratings[user_id][item_id] = rating
        
        # 获取所有用户和物品
        self.all_users = list(self.user_items.keys())
        self.all_items = list(set(int(row[1]) for row in self.test_data))
        
        # 采样用户（如果需要）
        if self.n_users_sample and self.n_users_sample < len(self.all_users):
            np.random.seed(42)
            self.eval_users = np.random.choice(
                self.all_users, 
                self.n_users_sample, 
                replace=False
            ).tolist()
        else:
            self.eval_users = self.all_users
    
    def evaluate_rating_prediction(self) -> Dict[str, float]:
        """评估评分预测性能"""
        predictions = []
        ground_truth = []
        
        for row in self.test_data:
            user_id = int(row[0])
            item_id = int(row[1])
            true_rating = float(row[2])
            
            # 只评估采样用户
            if user_id in self.eval_users:
                pred_rating = self.model.predict(user_id, item_id)[0]
                predictions.append(pred_rating)
                ground_truth.append(true_rating)
        
        predictions = np.array(predictions)
        ground_truth = np.array(ground_truth)
        
        # 计算预测准确性指标
        results = {}
        for metric_name in ['mae', 'rmse', 'mse', 'r2']:
            if metric_name in self.evaluator.metrics:
                metric = self.evaluator.metrics[metric_name]
                results[metric_name] = metric.calculate(ground_truth, predictions)
        
        return results
    
    def evaluate_ranking(self, n_recommendations: Optional[int] = None) -> Dict[str, float]:
        """评估排序性能"""
        if n_recommendations is None:
            n_recommendations = max(self.k_values)
        
        # 生成Top-K推荐
        topk_gen = TopKGenerator(self.model)
        recommendations = topk_gen.generate_top_k_for_users(
            self.eval_users,
            k=n_recommendations,
            exclude_known=True,
            known_items=self.user_items
        )
        
        # 准备评估数据
        additional_data = {
            'user_items': {uid: self.user_items[uid] for uid in self.eval_users},
            'recommendations': recommendations,
            'ratings': self.user_ratings,
            'n_users': len(self.eval_users),
            'n_items': len(self.all_items)
        }
        
        # 计算排序指标
        results = {}
        ranking_metrics = ['hr', 'precision', 'recall', 'map', 'ndcg', 'mrr']
        
        for metric_name, metric in self.evaluator.metrics.items():
            if any(m in metric_name.lower() for m in ranking_metrics):
                result = metric.calculate(None, None, **additional_data)
                results[metric_name] = result
        
        return results
    
    def evaluate_all(self) -> Dict[str, float]:
        """执行完整评估"""
        logger.info("评估评分预测性能...")
        rating_results = self.evaluate_rating_prediction()
        
        logger.info("评估排序性能...")
        ranking_results = self.evaluate_ranking()
        
        # 合并结果
        all_results = {**rating_results, **ranking_results}
        
        # 计算额外指标（如果有）
        if 'catalog_coverage' in self.evaluator.metrics:
            # 需要所有推荐
            topk_gen = TopKGenerator(self.model)
            all_recommendations = topk_gen.generate_top_k_for_users(
                self.all_users,
                k=10,
                exclude_known=True,
                known_items=self.user_items
            )
            
            coverage = self.evaluator.metrics['catalog_coverage'].calculate(
                None, None,
                recommendations=all_recommendations,
                n_items=len(self.all_items)
            )
            all_results['catalog_coverage'] = coverage
        
        return all_results
    
    def cross_validate(self, n_folds: int = 5) -> Dict[str, Tuple[float, float]]:
        """
        交叉验证评估
        
        Args:
            n_folds: 折数
            
        Returns:
            每个指标的均值和标准差
        """
        # 分割数据
        np.random.seed(42)
        n_samples = len(self.test_data)
        indices = np.random.permutation(n_samples)
        fold_size = n_samples // n_folds
        
        fold_results = []
        
        for fold in range(n_folds):
            logger.info(f"\n交叉验证 - 折 {fold + 1}/{n_folds}")
            
            # 获取当前折的测试集
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < n_folds - 1 else n_samples
            
            fold_test_indices = indices[start_idx:end_idx]
            fold_test_data = self.test_data[fold_test_indices]
            
            # 更新测试数据
            original_test_data = self.test_data
            self.test_data = fold_test_data
            self._prepare_test_data()
            
            # 评估
            fold_result = self.evaluate_all()
            fold_results.append(fold_result)
            
            # 恢复原始数据
            self.test_data = original_test_data
        
        # 计算统计信息
        cv_results = {}
        
        for metric_name in fold_results[0].keys():
            values = [r[metric_name] for r in fold_results if r[metric_name] is not None]
            if values:
                mean_value = np.mean(values)
                std_value = np.std(values)
                cv_results[metric_name] = (mean_value, std_value)
        
        return cv_results
    
    def save_results(self, results: Dict[str, float], save_path: str):
        """保存评估结果"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加元信息
        full_results = {
            'model': self.model.__class__.__name__,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_test_samples': len(self.test_data),
            'n_eval_users': len(self.eval_users),
            'k_values': self.k_values,
            'metrics': results
        }
        
        with open(save_path, 'w') as f:
            json.dump(full_results, f, indent=2)
        
        logger.info(f"评估结果已保存至: {save_path}")