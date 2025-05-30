# src/evaluation/__init__.py
from .metrics import (
    MAE, RMSE, MSE, R2Score,
    HitRate, Precision, Recall, MAP, NDCG, MRR,
    CatalogCoverage, UserCoverage, Diversity,
    Novelty, Serendipity
)
from .evaluator import Evaluator, ModelEvaluator
from .ranking import RankingEvaluator, TopKGenerator
from .statistical import StatisticalAnalyzer, SignificanceTest
from .utils import (
    create_evaluation_report, plot_metrics_comparison,
    save_evaluation_results, load_evaluation_results
)

__all__ = [
    # 预测准确性指标
    'MAE', 'RMSE', 'MSE', 'R2Score',
    # 排序质量指标
    'HitRate', 'Precision', 'Recall', 'MAP', 'NDCG', 'MRR',
    # 覆盖度和多样性指标
    'CatalogCoverage', 'UserCoverage', 'Diversity',
    'Novelty', 'Serendipity',
    # 评估器
    'Evaluator', 'ModelEvaluator',
    'RankingEvaluator', 'TopKGenerator',
    # 统计分析
    'StatisticalAnalyzer', 'SignificanceTest',
    # 工具函数
    'create_evaluation_report', 'plot_metrics_comparison',
    'save_evaluation_results', 'load_evaluation_results'
]