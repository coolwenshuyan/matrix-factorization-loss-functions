"""
实验管理模块

提供完整的实验管理功能，包括：
- 配置管理
- 实验运行
- 基线对比
- 结果分析
- 统计检验
- 报告生成
"""

from .config_manager import (
    ConfigManager,
    ExperimentConfig,
    ConfigValidator,
    ConfigTemplate
)
from .experiment_runner import (
    ExperimentRunner,
    BatchRunner,
    ExperimentStatus,
    RunResult
)
from .baseline_manager import (
    BaselineManager,
    BaselineConfig,
    BaselineResult,
    BaselineComparison
)
from .results_analyzer import (
    ResultsAnalyzer,
    ResultsAggregator,
    PerformanceAnalyzer,
    ConvergenceAnalyzer
)
from .significance_test import (
    SignificanceTest,
    TestResult,
    MultipleComparison,
    EffectSize
)
from .reproducibility import (
    ReproducibilityManager,
    EnvironmentSnapshot,
    RandomStateManager,
    CheckpointManager
)
from .report_generator import (
    ReportGenerator,
    TableGenerator,
    FigureGenerator,
    LatexExporter
)
from .workflow import (
    ExperimentWorkflow,
    WorkflowStep,
    WorkflowExecutor,
    WorkflowMonitor
)

__all__ = [
    # 配置管理
    'ConfigManager',
    'ExperimentConfig',
    'ConfigValidator',
    'ConfigTemplate',
    
    # 实验运行
    'ExperimentRunner',
    'BatchRunner',
    'ExperimentStatus',
    'RunResult',
    
    # 基线管理
    'BaselineManager',
    'BaselineConfig',
    'BaselineResult',
    'BaselineComparison',
    
    # 结果分析
    'ResultsAnalyzer',
    'ResultsAggregator',
    'PerformanceAnalyzer',
    'ConvergenceAnalyzer',
    
    # 统计检验
    'SignificanceTest',
    'TestResult',
    'MultipleComparison',
    'EffectSize',
    
    # 复现性
    'ReproducibilityManager',
    'EnvironmentSnapshot',
    'RandomStateManager',
    'CheckpointManager',
    
    # 报告生成
    'ReportGenerator',
    'TableGenerator',
    'FigureGenerator',
    'LatexExporter',
    
    # 工作流
    'ExperimentWorkflow',
    'WorkflowStep',
    'WorkflowExecutor',
    'WorkflowMonitor'
]

__version__ = '1.0.0'