"""
分析和可视化模块

提供论文级别的图表生成功能
"""

# 首先导入基础模块
from .plot_config import (
    PlotConfig,
    set_plot_style,
    get_color_palette,
    get_marker_styles,
    save_figure,
    apply_axis_style,
    create_figure_with_subplots,
    add_subplot_labels
)

# 然后导入依赖于基础模块的其他模块
from .performance_plots import (
    PerformancePlotter,
    plot_performance_comparison,
    plot_performance_heatmap,
    plot_performance_radar,
    plot_method_ranking
)

from .distribution_plots import (
    DistributionPlotter,
    plot_error_distribution,
    plot_error_boxplot,
    plot_qq_plot,
    plot_error_heatmap
)

from .convergence_plots import (
    ConvergencePlotter,
    plot_training_curves,
    plot_validation_curves,
    plot_learning_rate_schedule,
    plot_gradient_statistics
)

from .parameter_plots import (
    ParameterPlotter,
    plot_parameter_sensitivity,
    plot_parameter_heatmap,
    plot_parameter_parallel_coordinates,
    plot_parameter_3d_surface
)

from .robustness_plots import (
    RobustnessPlotter,
    plot_noise_robustness,
    plot_sparsity_analysis,
    plot_stability_analysis,
    plot_confidence_intervals
)

from .table_formatter import (
    TableFormatter,
    create_performance_table,
    create_comparison_table,
    create_ablation_table,
    create_significance_table
)

from .export_manager import (
    ExportManager,
    export_figure,
    export_table,
    export_all_figures,
    create_figure_package
)

from .paper_figures import (
    PaperFigureGenerator,
    generate_main_results_figure,
    generate_method_comparison_figure,
    generate_hpl_illustration,
    generate_all_paper_figures
)

__version__ = '1.0.0'
