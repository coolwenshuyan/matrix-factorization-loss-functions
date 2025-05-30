#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块使用示例
展示如何使用src/visualization模块生成各种图表和表格
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入visualization模块
from src.visualization import (
    # 绘图器类
    PerformancePlotter,
    DistributionPlotter,
    ConvergencePlotter,
    ParameterPlotter,
    RobustnessPlotter,
    
    # 便捷函数
    plot_performance_comparison,
    plot_error_distribution,
    plot_training_curves,
    plot_parameter_sensitivity,
    plot_noise_robustness,
    
    # 表格生成
    create_performance_table,
    create_comparison_table,
    create_ablation_table,
    
    # 论文图表
    generate_main_results_figure,
    
    # 导出管理
    ExportManager,
    export_figure,
    create_figure_package
)

# 创建输出目录
output_dir = Path("./visualization_output")
output_dir.mkdir(exist_ok=True)


def generate_sample_data():
    """生成示例数据"""
    # 模拟实验结果数据
    methods = ['HPL', 'L2', 'L1', 'Huber', 'LogCosh']
    datasets = ['MovieLens', 'Amazon', 'Yelp']
    
    np.random.seed(42)  # 固定随机种子以确保可重复性
    
    # 生成性能结果
    results = []
    for method in methods:
        for dataset in datasets:
            # 为不同方法设置不同的基准性能
            base_mae = 0.8 if method == 'HPL' else 0.9
            base_rmse = 1.1 if method == 'HPL' else 1.2
            
            # 添加一些随机变化
            result = {
                'method': method,
                'dataset': dataset,
                'mae': base_mae + np.random.normal(0, 0.05),
                'rmse': base_rmse + np.random.normal(0, 0.07),
                'hr@10': 0.6 + np.random.normal(0, 0.05),
                'ndcg@10': 0.5 + np.random.normal(0, 0.05)
            }
            results.append(result)
    
    # 生成训练历史数据
    histories = {}
    epochs = 50
    
    for method in methods:
        # 为不同方法设置不同的收敛曲线
        if method == 'HPL':
            base_loss = np.linspace(1.0, 0.3, epochs) + np.random.normal(0, 0.02, epochs)
        else:
            base_loss = np.linspace(1.0, 0.4, epochs) + np.random.normal(0, 0.03, epochs)
            
        val_loss = base_loss + 0.1 + np.random.normal(0, 0.02, epochs)
        val_mae = base_loss * 0.8 + np.random.normal(0, 0.01, epochs)
        
        histories[method] = {
            'train_loss': base_loss,
            'val_loss': val_loss,
            'val_mae': val_mae
        }
    
    # 生成参数扫描结果
    param_results = []
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
    reg_lambdas = [0.0001, 0.001, 0.01, 0.1, 1.0]
    
    for lr in learning_rates:
        for reg in reg_lambdas:
            # 模拟参数影响
            mae = 0.8 + 0.2 * (np.abs(np.log10(lr) + 3) / 4) + 0.1 * (np.abs(np.log10(reg) + 3) / 4)
            mae += np.random.normal(0, 0.02)
            
            param_results.append({
                'learning_rate': lr,
                'reg_lambda': reg,
                'mae': mae
            })
    
    # 生成误差数据
    errors = {}
    for method in methods:
        if method == 'HPL':
            # 正态分布误差，均值接近0
            errors[method] = np.random.normal(0, 0.5, 1000)
        else:
            # 偏斜分布误差
            errors[method] = np.random.normal(0.2, 0.7, 1000)
    
    # 生成鲁棒性测试数据
    noise_levels = np.array([0, 0.05, 0.1, 0.15, 0.2])
    noise_performances = {}
    
    for method in methods:
        base = 0.8 if method == 'HPL' else 0.9
        slope = 1.0 if method == 'HPL' else 1.5
        noise_performances[method] = base + slope * noise_levels + np.random.normal(0, 0.02, len(noise_levels))
    
    return {
        'results': results,
        'histories': histories,
        'param_results': pd.DataFrame(param_results),
        'errors': errors,
        'noise_levels': noise_levels,
        'noise_performances': noise_performances
    }


def demo_performance_plots(data, output_dir):
    """演示性能图表生成"""
    print("生成性能对比图表...")
    
    results = data['results']
    
    # 使用便捷函数生成性能对比图
    fig = plot_performance_comparison(
        results, 
        metrics=['mae', 'rmse'],
        save_path=output_dir / "performance_comparison.pdf"
    )
    
    # 使用PerformancePlotter类生成更多自定义图表
    plotter = PerformancePlotter()
    
    # 生成性能热力图
    fig = plotter.plot_heatmap(
        results, 
        metric='mae',
        save_path=output_dir / "performance_heatmap.pdf"
    )
    
    # 生成雷达图
    fig = plotter.plot_radar(
        results, 
        metrics=['mae', 'rmse', 'hr@10', 'ndcg@10'],
        dataset='MovieLens',
        save_path=output_dir / "performance_radar.pdf"
    )
    
    # 生成排名图
    fig = plotter.plot_ranking(
        results, 
        metric='mae',
        save_path=output_dir / "performance_ranking.pdf"
    )
    
    print("性能图表已保存到:", output_dir)


def demo_convergence_plots(data, output_dir):
    """演示收敛曲线图表生成"""
    print("生成收敛曲线图表...")
    
    histories = data['histories']
    
    # 使用便捷函数生成训练曲线
    fig = plot_training_curves(
        histories,
        metric='train_loss',
        save_path=output_dir / "training_loss.pdf"
    )
    
    # 使用ConvergencePlotter类生成更多自定义图表
    plotter = ConvergencePlotter()
    
    # 生成训练vs验证损失对比图
    fig = plotter.plot_train_val_comparison(
        histories,
        train_metric='train_loss',
        val_metric='val_loss',
        methods=['HPL', 'L2', 'Huber'],
        save_path=output_dir / "train_val_comparison.pdf"
    )
    
    # 生成多指标收敛图
    fig = plotter.plot_multi_metric(
        histories,
        metrics=['val_loss', 'val_mae'],
        method='HPL',
        save_path=output_dir / "multi_metric_convergence.pdf"
    )
    
    print("收敛曲线图表已保存到:", output_dir)


def demo_parameter_plots(data, output_dir):
    """演示参数分析图表生成"""
    print("生成参数分析图表...")
    
    param_results = data['param_results']
    
    # 使用便捷函数生成参数敏感性图
    fig = plot_parameter_sensitivity(
        param_results,
        parameter='learning_rate',
        metric='mae',
        save_path=output_dir / "lr_sensitivity.pdf"
    )
    
    # 使用ParameterPlotter类生成更多自定义图表
    plotter = ParameterPlotter()
    
    # 生成参数热力图
    fig = plotter.plot_heatmap(
        param_results,
        x_param='learning_rate',
        y_param='reg_lambda',
        metric='mae',
        save_path=output_dir / "parameter_heatmap.pdf"
    )
    
    # 生成参数交互图
    fig = plotter.plot_interaction(
        param_results,
        x_param='learning_rate',
        y_param='reg_lambda',
        metric='mae',
        save_path=output_dir / "parameter_interaction.pdf"
    )
    
    print("参数分析图表已保存到:", output_dir)


def demo_distribution_plots(data, output_dir):
    """演示分布图表生成"""
    print("生成误差分布图表...")
    
    errors = data['errors']
    
    # 使用便捷函数生成误差分布图
    fig = plot_error_distribution(
        errors,
        methods=['HPL', 'L2', 'Huber'],
        save_path=output_dir / "error_distribution.pdf"
    )
    
    # 使用DistributionPlotter类生成更多自定义图表
    plotter = DistributionPlotter()
    
    # 生成箱线图
    fig = plotter.plot_boxplot(
        errors,
        save_path=output_dir / "error_boxplot.pdf"
    )
    
    # 生成Q-Q图
    fig = plotter.plot_qq(
        errors,
        method='HPL',
        save_path=output_dir / "qq_plot.pdf"
    )
    
    print("误差分布图表已保存到:", output_dir)


def demo_robustness_plots(data, output_dir):
    """演示鲁棒性图表生成"""
    print("生成鲁棒性图表...")
    
    noise_levels = data['noise_levels']
    noise_performances = data['noise_performances']
    
    # 使用便捷函数生成噪声鲁棒性图
    fig = plot_noise_robustness(
        noise_levels,
        noise_performances,
        methods=['HPL', 'L2', 'Huber'],
        metric_name='MAE',
        save_path=output_dir / "noise_robustness.pdf"
    )
    
    # 使用RobustnessPlotter类生成更多自定义图表
    plotter = RobustnessPlotter()
    
    # 生成置信区间图
    # 模拟置信区间数据
    ci_data = {}
    for method in ['HPL', 'L2', 'Huber']:
        means = noise_performances[method]
        lower = means - 0.05
        upper = means + 0.05
        ci_data[method] = {'mean': means, 'lower': lower, 'upper': upper}
    
    fig = plotter.plot_confidence_intervals(
        noise_levels,
        ci_data,
        x_label='噪声水平',
        y_label='MAE',
        save_path=output_dir / "confidence_intervals.pdf"
    )
    
    print("鲁棒性图表已保存到:", output_dir)


def demo_tables(data, output_dir):
    """演示表格生成"""
    print("生成表格...")
    
    results = data['results']
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results)
    
    # 创建性能对比表格
    table_path = create_performance_table(
        df,
        metrics=['mae', 'rmse'],
        datasets=['MovieLens', 'Amazon', 'Yelp'],
        methods=['HPL', 'L2', 'L1', 'Huber', 'LogCosh'],
        save_path=output_dir / "performance_table.tex"
    )
    
    # 创建消融实验表格
    # 模拟消融实验数据
    ablation_data = pd.DataFrame({
        'components': ['Base', 'Base+A', 'Base+A+B', 'Full'],
        'mae': [0.95, 0.88, 0.82, 0.78],
        'rmse': [1.40, 1.30, 1.22, 1.15],
        'improvement': ['0%', '7.4%', '13.7%', '17.9%']
    })
    
    table_path = create_ablation_table(
        ablation_data,
        save_path=output_dir / "ablation_table.tex"
    )
    
    print("表格已保存到:", output_dir)


def demo_paper_figures(data, output_dir):
    """演示论文图表生成"""
    print("生成论文图表...")
    
    # 生成主要结果图
    fig_path = generate_main_results_figure(
        data['results'],
        save_path=output_dir / "main_results.pdf"
    )
    
    print("论文图表已保存到:", output_dir)


def demo_export_package(output_dir):
    """演示导出图表包"""
    print("创建图表导出包...")
    
    # 创建导出管理器
    export_manager = ExportManager()
    
    # 添加所有生成的图表
    for file_path in output_dir.glob("*.pdf"):
        export_manager.add_file(file_path, file_path.stem)
    
    # 添加所有生成的表格
    for file_path in output_dir.glob("*.tex"):
        export_manager.add_file(file_path, file_path.stem)
    
    # 创建图表包
    package_path = export_manager.create_figure_package(
        "visualization_demo",
        output_dir=output_dir
    )
    
    print(f"图表包已创建: {package_path}")


def main():
    """主函数"""
    print("开始可视化模块演示...")
    
    # 生成示例数据
    data = generate_sample_data()
    
    # 演示各种图表生成
    demo_performance_plots(data, output_dir)
    demo_convergence_plots(data, output_dir)
    demo_parameter_plots(data, output_dir)
    demo_distribution_plots(data, output_dir)
    demo_robustness_plots(data, output_dir)
    demo_tables(data, output_dir)
    demo_paper_figures(data, output_dir)
    
    # 创建导出包
    demo_export_package(output_dir)
    
    print("演示完成! 所有输出文件保存在:", output_dir)


if __name__ == "__main__":
    main()