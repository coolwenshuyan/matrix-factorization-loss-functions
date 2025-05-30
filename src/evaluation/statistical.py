# src/evaluation/statistical.py
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from collections import defaultdict
import logging
import warnings

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """统计分析器"""
    
    def __init__(self, confidence_level: float = 0.95):
        """
        初始化统计分析器
        
        Args:
            confidence_level: 置信水平
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze_results(self, results: Dict[str, float]) -> Dict[str, Any]:
        """
        分析评估结果的统计特性
        
        Args:
            results: 评估结果字典
            
        Returns:
            统计分析结果
        """
        analysis = {
            'summary_statistics': self._compute_summary_statistics(results),
            'metric_correlations': self._compute_metric_correlations(results)
        }
        
        return analysis
    
    def _compute_summary_statistics(self, results: Dict[str, float]) -> Dict[str, Any]:
        """计算汇总统计"""
        values = list(results.values())
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q1': np.percentile(values, 25),
            'q3': np.percentile(values, 75)
        }
    
    def _compute_metric_correlations(self, results: Dict[str, float]) -> Dict[str, float]:
        """计算指标间的相关性"""
        # 这里需要多个模型的结果来计算相关性
        # 简化版本：返回空字典
        return {}
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]],
                      metric: str) -> Dict[str, Any]:
        """
        比较多个模型在特定指标上的表现
        
        Args:
            model_results: 模型结果字典 {model_name: {metric_name: value}}
            metric: 要比较的指标
            
        Returns:
            比较结果
        """
        # 提取指标值
        model_names = []
        values = []
        
        for model_name, results in model_results.items():
            if metric in results:
                model_names.append(model_name)
                values.append(results[metric])
        
        if len(values) < 2:
            return {'error': '至少需要两个模型进行比较'}
        
        # 执行统计检验
        comparison = {
            'models': model_names,
            'values': values,
            'best_model': model_names[np.argmax(values)],
            'worst_model': model_names[np.argmin(values)]
        }
        
        # 如果有多个模型，执行方差分析
        if len(values) > 2:
            comparison['anova'] = self._perform_anova(values)
        
        # 成对比较
        comparison['pairwise'] = self._pairwise_comparison(model_names, values)
        
        return comparison
    
    def _perform_anova(self, groups: List[List[float]]) -> Dict[str, float]:
        """执行单因素方差分析"""
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            return {
                'f_statistic': f_stat,
                'p_value': p_value,
                'significant': p_value < self.alpha
            }
        except:
            return {'error': '方差分析失败'}
    
    def _pairwise_comparison(self, names: List[str], 
                           values: List[float]) -> List[Dict[str, Any]]:
        """成对比较"""
        comparisons = []
        
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                comparison = {
                    'model1': names[i],
                    'model2': names[j],
                    'value1': values[i],
                    'value2': values[j],
                    'difference': values[i] - values[j],
                    'relative_improvement': (values[i] - values[j]) / values[j] * 100
                }
                comparisons.append(comparison)
        
        return comparisons
    
    def bootstrap_confidence_interval(self, data: np.ndarray,
                                    statistic_func: callable,
                                    n_bootstrap: int = 1000) -> Tuple[float, float]:
        """
        使用Bootstrap方法计算置信区间
        
        Args:
            data: 数据数组
            statistic_func: 统计函数
            n_bootstrap: Bootstrap样本数
            
        Returns:
            置信区间 (lower, upper)
        """
        bootstrap_statistics = []
        n_samples = len(data)
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            bootstrap_sample = np.random.choice(data, size=n_samples, replace=True)
            statistic = statistic_func(bootstrap_sample)
            bootstrap_statistics.append(statistic)
        
        # 计算置信区间
        lower = np.percentile(bootstrap_statistics, (1 - self.confidence_level) / 2 * 100)
        upper = np.percentile(bootstrap_statistics, (1 + self.confidence_level) / 2 * 100)
        
        return lower, upper
    
    def effect_size(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, float]:
        """
        计算效应量
        
        Args:
            group1: 第一组数据
            group2: 第二组数据
            
        Returns:
            效应量指标
        """
        # Cohen's d
        mean_diff = np.mean(group1) - np.mean(group2)
        pooled_std = np.sqrt((np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
        
        # 相对改善
        relative_improvement = mean_diff / np.mean(group2) * 100 if np.mean(group2) != 0 else 0
        
        return {
            'cohens_d': cohens_d,
            'relative_improvement': relative_improvement,
            'mean_difference': mean_diff
        }


class SignificanceTest:
    """显著性检验"""
    
    def __init__(self, alpha: float = 0.05):
        """
        初始化显著性检验
        
        Args:
            alpha: 显著性水平
        """
        self.alpha = alpha
    
    def paired_t_test(self, scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, Any]:
        """
        配对t检验
        
        Args:
            scores1: 第一组分数
            scores2: 第二组分数
            
        Returns:
            检验结果
        """
        if len(scores1) != len(scores2):
            raise ValueError("两组数据长度必须相同")
        
        differences = scores1 - scores2
        
        # 执行t检验
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        
        # 计算置信区间
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        n = len(differences)
        
        t_critical = stats.t.ppf(1 - self.alpha/2, n-1)
        margin_error = t_critical * std_diff / np.sqrt(n)
        
        ci_lower = mean_diff - margin_error
        ci_upper = mean_diff + margin_error
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_difference': mean_diff,
            'confidence_interval': (ci_lower, ci_upper),
            'effect_size': self._calculate_effect_size(scores1, scores2)
        }
    
    def wilcoxon_test(self, scores1: np.ndarray, scores2: np.ndarray) -> Dict[str, Any]:
        """
        Wilcoxon符号秩检验（非参数检验）
        
        Args:
            scores1: 第一组分数
            scores2: 第二组分数
            
        Returns:
            检验结果
        """
        try:
            statistic, p_value = stats.wilcoxon(scores1, scores2)
            
            return {
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < self.alpha,
                'test_type': 'wilcoxon'
            }
        except:
            return {'error': 'Wilcoxon检验失败'}
    
    def multiple_comparison_correction(self, p_values: List[float],
                                     method: str = 'bonferroni') -> List[float]:
        """
        多重比较校正
        
        Args:
            p_values: p值列表
            method: 校正方法 ('bonferroni', 'holm', 'fdr')
            
        Returns:
            校正后的p值
        """
        n = len(p_values)
        p_values = np.array(p_values)
        
        if method == 'bonferroni':
            return np.minimum(p_values * n, 1.0)
        
        elif method == 'holm':
            # Holm-Bonferroni方法
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            adjusted_p = np.zeros(n)
            for i in range(n):
                adjusted_p[sorted_indices[i]] = min(sorted_p[i] * (n - i), 1.0)
            
            return adjusted_p
        
        elif method == 'fdr':
            # Benjamini-Hochberg FDR
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]
            
            adjusted_p = np.zeros(n)
            for i in range(n-1, -1, -1):
                if i == n-1:
                    adjusted_p[sorted_indices[i]] = sorted_p[i]
                else:
                    adjusted_p[sorted_indices[i]] = min(
                        adjusted_p[sorted_indices[i+1]],
                        sorted_p[i] * n / (i + 1)
                    )
            
            return adjusted_p
        
        else:
            raise ValueError(f"未知的校正方法: {method}")
    
    def _calculate_effect_size(self, scores1: np.ndarray, scores2: np.ndarray) -> float:
        """计算效应量（Cohen's d）"""
        mean_diff = np.mean(scores1) - np.mean(scores2)
        
        # 合并标准差
        n1, n2 = len(scores1), len(scores2)
        var1, var2 = np.var(scores1, ddof=1), np.var(scores2, ddof=1)
        
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        pooled_std = np.sqrt(pooled_var)
        
        return mean_diff / pooled_std if pooled_std > 0 else 0.0


class ExperimentAnalyzer:
    """实验分析器"""
    
    def __init__(self):
        self.stat_analyzer = StatisticalAnalyzer()
        self.sig_test = SignificanceTest()
    
    def analyze_ab_test(self, control_results: Dict[str, List[float]],
                       treatment_results: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        分析A/B测试结果
        
        Args:
            control_results: 控制组结果
            treatment_results: 实验组结果
            
        Returns:
            分析结果
        """
        analysis = {}
        
        for metric in control_results.keys():
            if metric not in treatment_results:
                continue
            
            control_scores = np.array(control_results[metric])
            treatment_scores = np.array(treatment_results[metric])
            
            # 基础统计
            analysis[metric] = {
                'control_mean': np.mean(control_scores),
                'control_std': np.std(control_scores),
                'treatment_mean': np.mean(treatment_scores),
                'treatment_std': np.std(treatment_scores),
                'improvement': (np.mean(treatment_scores) - np.mean(control_scores)) / 
                              np.mean(control_scores) * 100
            }
            
            # 显著性检验
            if len(control_scores) == len(treatment_scores):
                # 配对检验
                test_result = self.sig_test.paired_t_test(treatment_scores, control_scores)
            else:
                # 独立样本检验
                test_result = self._independent_t_test(treatment_scores, control_scores)
            
            analysis[metric].update(test_result)
        
        return analysis
    
    def _independent_t_test(self, group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """独立样本t检验"""
        t_stat, p_value = stats.ttest_ind(group1, group2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'test_type': 'independent_t_test'
        }