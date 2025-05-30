from visualization import *
# 生成模拟的 experiment_results
import numpy as np

np.random.seed(42)

datasets = ['ML-100K', 'ML-1M', 'Netflix', 'Yahoo Music']
methods = ['HPL', 'MF-L2', 'MF-L1', 'MF-Huber', 'MF-Logcosh']

experiment_results = []

for dataset in datasets:
    for method in methods:
        # 生成模拟数据
        base_mae = 0.85 if method == 'HPL' else np.random.uniform(0.87, 0.92)
        base_rmse = base_mae * 1.25
        
        experiment_results.append({
            'method': method,
            'dataset': dataset,
            'mae': base_mae + np.random.normal(0, 0.01),
            'mae_std': np.random.uniform(0.002, 0.005),
            'rmse': base_rmse + np.random.normal(0, 0.015),
            'rmse_std': np.random.uniform(0.003, 0.008),
            'hr@10': 0.68 + np.random.normal(0, 0.02) if method == 'HPL' else 0.65 + np.random.normal(0, 0.02),
            'hr@10_std': np.random.uniform(0.005, 0.01),
            'ndcg@10': 0.52 + np.random.normal(0, 0.015) if method == 'HPL' else 0.49 + np.random.normal(0, 0.015),
            'ndcg@10_std': np.random.uniform(0.004, 0.008)
        })

# 转换为DataFrame用于可视化
import pandas as pd
results_df = pd.DataFrame(experiment_results)
# 设置全局样式
set_plot_style('academic', use_latex=True)

# 1. 生成性能对比图
results_df = pd.DataFrame(experiment_results)
fig1 = plot_performance_comparison(results_df, 
                                  metrics=['mae', 'rmse', 'hr@10', 'ndcg@10'],
                                  save_path='figures/performance_comparison')

# 2. 生成误差分布图
errors = {
    'HPL': hpl_errors,
    'L2': l2_errors,
    'L1': l1_errors,
    'Huber': huber_errors
}
fig2 = plot_error_distribution(errors, 
                              plot_types=['histogram', 'boxplot', 'qq'],
                              save_path='figures/error_dist')

# 3. 生成收敛曲线
histories = {
    'HPL': hpl_history,
    'L2': l2_history,
    'L1': l1_history,
    'Huber': huber_history
}
fig3 = plot_training_curves(histories,
                           save_path='figures/convergence')

# 4. 生成参数分析图
sensitivities = {
    'learning_rate': 0.85,
    'latent_factors': 0.72,
    'delta1': 0.68,
    'delta2': 0.65,
    'lambda': 0.45
}
fig4 = plot_parameter_sensitivity(sensitivities,
                                save_path='figures/param_sensitivity')

# 5. 生成鲁棒性分析
noise_levels = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5])
noise_performances = {
    'HPL': np.array([0.862, 0.871, 0.885, 0.903, 0.925, 0.951]),
    'L2': np.array([0.875, 0.892, 0.918, 0.948, 0.982, 1.021])
}
fig5 = plot_noise_robustness(noise_levels, noise_performances,
                            save_path='figures/robustness')

# 6. 生成论文所需的所有图表
generator = PaperFigureGenerator(export_dir='./paper_figures')
all_figures = generator.generate_all_paper_figures(
    results_data=results_df,
    histories=histories,
    param_results=param_analysis_results,
    robustness_results=robustness_results
)

# 7. 创建性能表格
table = create_performance_table(
    results_df,
    metrics=['mae', 'rmse', 'hr@10', 'ndcg@10'],
    format_type='latex',
    highlight_best=True,
    show_std=True,
    save_path='tables/main_results.tex'
)

# 8. 批量导出
export_manager = ExportManager('./exports')
export_manager.export_all_figures(all_figures, 
                                formats=['pdf', 'eps', 'png'],
                                create_package=True)