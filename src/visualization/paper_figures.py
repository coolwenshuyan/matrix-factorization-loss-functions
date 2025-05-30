"""
论文图表生成模块

生成专门用于论文的高质量图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

from .plot_config import (
    PlotConfig,
    set_plot_style,
    apply_axis_style,
    get_color_palette,
    save_figure,
    add_subplot_labels
)
from .performance_plots import PerformancePlotter
from .convergence_plots import ConvergencePlotter
from .parameter_plots import ParameterPlotter
from .distribution_plots import DistributionPlotter
from .export_manager import ExportManager


class PaperFigureGenerator:
    """论文图表生成器"""
    
    def __init__(self, 
                 style: str = 'academic',
                 export_dir: str = "./paper_figures"):
        self.style = style
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置全局样式
        set_plot_style(style, use_latex=True)
        
        # 初始化绘图器
        self.perf_plotter = PerformancePlotter(style)
        self.conv_plotter = ConvergencePlotter(style)
        self.param_plotter = ParameterPlotter(style)
        self.dist_plotter = DistributionPlotter(style)
        
        # 导出管理器
        self.export_manager = ExportManager(export_dir)
        
    def generate_main_results_figure(self,
                                   results_data: pd.DataFrame,
                                   datasets: List[str],
                                   save_name: str = "main_results") -> plt.Figure:
        """生成主要结果图（图1）
        
        Args:
            results_data: 结果数据
            datasets: 数据集列表
            save_name: 保存名称
        """
        # 创建4子图布局
        fig, axes = plt.subplots(2, 2, 
                               figsize=PlotConfig.get_figure_size('double', 0.8))
        axes = axes.flatten()
        
        # 定义指标和对应的子图
        metrics = ['mae', 'rmse', 'hr@10', 'ndcg@10']
        metric_labels = ['MAE', 'RMSE', 'HR@10', 'NDCG@10']
        
        # 为每个指标创建条形图
        for ax, metric, label in zip(axes, metrics, metric_labels):
            # 准备数据
            pivot_data = results_data.pivot(
                index='dataset', 
                columns='method', 
                values=metric
            )
            
            # 确保数据集顺序一致
            pivot_data = pivot_data.reindex(datasets)
            
            # 创建条形图
            pivot_data.plot(kind='bar', ax=ax, 
                          color=get_color_palette('academic'),
                          width=0.8, edgecolor='black', linewidth=0.5)
            
            # 设置标签
            apply_axis_style(ax, 
                           ylabel=label,
                           legend=(ax == axes[0]),  # 只在第一个子图显示图例
                           tight_layout=False)
            
            # 旋转x轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 对于误差指标，添加向下的箭头表示越小越好
            if metric in ['mae', 'rmse']:
                ax.annotate('', xy=(0.95, 0.05), xytext=(0.95, 0.15),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->',
                                         connectionstyle='arc3',
                                         color='gray', lw=1.5))
                ax.text(0.97, 0.1, 'Better', rotation=270, 
                       transform=ax.transAxes, va='center', fontsize=8)
                       
        # 添加子图标签
        add_subplot_labels(axes)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图形
        self.export_manager.export_figure(
            fig, save_name, 
            formats=['pdf', 'eps', 'png'],
            category='main'
        )
        
        return fig
        
    def generate_hpl_illustration(self,
                                save_name: str = "hpl_loss_function") -> plt.Figure:
        """生成HPL损失函数示意图（图2）
        
        Args:
            save_name: 保存名称
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, 
                                      figsize=PlotConfig.get_figure_size('double', 0.5))
        
        # 定义参数
        delta1, delta2 = 0.5, 1.5
        lambda_val = 0.5
        
        # 生成误差范围
        e = np.linspace(-3, 3, 1000)
        
        # 计算HPL损失
        hpl_loss = np.zeros_like(e)
        mask1 = np.abs(e) <= delta1
        mask2 = (np.abs(e) > delta1) & (np.abs(e) <= delta2)
        mask3 = np.abs(e) > delta2
        
        hpl_loss[mask1] = e[mask1]**2
        hpl_loss[mask2] = 2*delta1*np.abs(e[mask2]) - delta1**2
        hpl_loss[mask3] = 2*delta1*delta2 - delta1**2 + lambda_val*(np.abs(e[mask3]) - delta2)**2
        
        # 计算其他损失函数进行对比
        l2_loss = e**2
        l1_loss = np.abs(e)
        huber_loss = np.where(np.abs(e) <= 1, 0.5*e**2, np.abs(e) - 0.5)
        
        # 图1：HPL损失函数
        ax1.plot(e, hpl_loss, 'b-', linewidth=2.5, label='HPL')
        ax1.plot(e, l2_loss, 'r--', linewidth=2, alpha=0.7, label='L2')
        ax1.plot(e, l1_loss, 'g-.', linewidth=2, alpha=0.7, label='L1')
        ax1.plot(e, huber_loss, 'm:', linewidth=2, alpha=0.7, label='Huber')
        
        # 标记关键点
        ax1.axvline(x=delta1, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=-delta1, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=delta2, color='gray', linestyle=':', alpha=0.5)
        ax1.axvline(x=-delta2, color='gray', linestyle=':', alpha=0.5)
        
        # 添加区域标注
        ax1.text(0, -0.5, 'I', ha='center', fontsize=10, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.5))
        ax1.text(1, -0.5, 'II', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.5))
        ax1.text(2.2, -0.5, 'III', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.5))
        
        apply_axis_style(ax1,
                        xlabel='Prediction Error (e)',
                        ylabel='Loss',
                        title='Loss Functions Comparison',
                        legend=True)
        
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-1, 8)
        
        # 图2：梯度对比
        # 计算梯度
        hpl_grad = np.zeros_like(e)
        hpl_grad[mask1] = 2*e[mask1]
        hpl_grad[mask2] = 2*delta1*np.sign(e[mask2])
        hpl_grad[mask3] = 2*delta1*np.sign(e[mask3]) + 2*lambda_val*(e[mask3] - delta2*np.sign(e[mask3]))
        
        l2_grad = 2*e
        l1_grad = np.sign(e)
        huber_grad = np.where(np.abs(e) <= 1, e, np.sign(e))
        
        ax2.plot(e, hpl_grad, 'b-', linewidth=2.5, label='HPL')
        ax2.plot(e, l2_grad, 'r--', linewidth=2, alpha=0.7, label='L2')
        ax2.plot(e, l1_grad, 'g-.', linewidth=2, alpha=0.7, label='L1')
        ax2.plot(e, huber_grad, 'm:', linewidth=2, alpha=0.7, label='Huber')
        
        # 标记关键点
        ax2.axvline(x=delta1, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=-delta1, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=delta2, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(x=-delta2, color='gray', linestyle=':', alpha=0.5)
        
        apply_axis_style(ax2,
                        xlabel='Prediction Error (e)',
                        ylabel='Gradient',
                        title='Gradient Comparison',
                        legend=False)
        
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-6, 6)
        
        # 添加子图标签
        add_subplot_labels([ax1, ax2])
        
        plt.tight_layout()
        
        # 保存图形
        self.export_manager.export_figure(
            fig, save_name,
            formats=['pdf', 'eps', 'png'],
            category='methodology'
        )
        
        return fig
        
    def generate_convergence_comparison(self,
                                      histories: Dict[str, Dict],
                                      save_name: str = "convergence_comparison") -> plt.Figure:
        """生成收敛对比图（图3）
        
        Args:
            histories: 训练历史数据
            save_name: 保存名称
        """
        # 使用双栏宽度，两行布局
        fig, axes = plt.subplots(2, 2,
                               figsize=PlotConfig.get_figure_size('double', 0.8))
        
        # 选择代表性数据集
        datasets = ['ML-100K', 'ML-1M', 'Netflix', 'Yahoo Music']
        
        for ax, dataset in zip(axes.flatten(), datasets):
            # 获取该数据集的历史
            dataset_histories = {
                method: hist for method, hist in histories.items()
                if dataset in method
            }
            
            # 绘制验证损失曲线
            for method, history in dataset_histories.items():
                if 'val_loss' in history:
                    epochs = np.arange(1, len(history['val_loss']) + 1)
                    
                    # 提取方法名
                    method_name = method.split('_')[0]
                    
                    # 选择合适的颜色和线型
                    if 'HPL' in method_name:
                        style = {'color': 'blue', 'linewidth': 2.5, 'linestyle': '-'}
                    elif 'L2' in method_name:
                        style = {'color': 'red', 'linewidth': 2, 'linestyle': '--'}
                    elif 'L1' in method_name:
                        style = {'color': 'green', 'linewidth': 2, 'linestyle': '-.'}
                    elif 'Huber' in method_name:
                        style = {'color': 'magenta', 'linewidth': 2, 'linestyle': ':'}
                    else:
                        style = {'linewidth': 2}
                        
                    ax.plot(epochs, history['val_loss'], 
                           label=method_name, **style)
                           
            apply_axis_style(ax,
                           xlabel='Epoch',
                           ylabel='Validation Loss',
                           title=dataset,
                           legend=True)
            
            # 设置y轴范围以便更好地比较
            ax.set_ylim(bottom=0)
            
        # 添加子图标签
        add_subplot_labels(axes.flatten())
        
        plt.tight_layout()
        
        # 保存图形
        self.export_manager.export_figure(
            fig, save_name,
            formats=['pdf', 'eps', 'png'],
            category='results'
        )
        
        return fig
        
    def generate_parameter_analysis(self,
                                  param_results: Dict,
                                  save_name: str = "parameter_analysis") -> plt.Figure:
        """生成参数分析图（图4）
        
        Args:
            param_results: 参数分析结果
            save_name: 保存名称
        """
        # 创建2x2子图
        fig, axes = plt.subplots(2, 2,
                               figsize=PlotConfig.get_figure_size('double', 0.8))
        
        # 1. δ1的影响
        ax = axes[0, 0]
        delta1_values = param_results['delta1']['values']
        delta1_mae = param_results['delta1']['mae']
        
        ax.plot(delta1_values, delta1_mae, 'o-', 
               color='blue', linewidth=2, markersize=8)
        
        # 标记最优值
        best_idx = np.argmin(delta1_mae)
        ax.plot(delta1_values[best_idx], delta1_mae[best_idx],
               'r*', markersize=15, markeredgecolor='black')
        
        apply_axis_style(ax,
                        xlabel=r'$\delta_1$',
                        ylabel='MAE',
                        title=r'Effect of $\delta_1$')
        
        # 2. δ2的影响
        ax = axes[0, 1]
        delta2_values = param_results['delta2']['values']
        delta2_mae = param_results['delta2']['mae']
        
        ax.plot(delta2_values, delta2_mae, 'o-',
               color='green', linewidth=2, markersize=8)
        
        best_idx = np.argmin(delta2_mae)
        ax.plot(delta2_values[best_idx], delta2_mae[best_idx],
               'r*', markersize=15, markeredgecolor='black')
        
        apply_axis_style(ax,
                        xlabel=r'$\delta_2$',
                        ylabel='MAE',
                        title=r'Effect of $\delta_2$')
        
        # 3. λ的影响
        ax = axes[1, 0]
        lambda_values = param_results['lambda']['values']
        lambda_mae = param_results['lambda']['mae']
        
        ax.semilogx(lambda_values, lambda_mae, 'o-',
                   color='orange', linewidth=2, markersize=8)
        
        best_idx = np.argmin(lambda_mae)
        ax.plot(lambda_values[best_idx], lambda_mae[best_idx],
               'r*', markersize=15, markeredgecolor='black')
        
        apply_axis_style(ax,
                        xlabel=r'$\lambda$',
                        ylabel='MAE',
                        title=r'Effect of $\lambda$')
        
        # 4. δ1-δ2交互热力图
        ax = axes[1, 1]
        
        # 创建网格数据
        delta1_grid = param_results['interaction']['delta1_grid']
        delta2_grid = param_results['interaction']['delta2_grid']
        mae_grid = param_results['interaction']['mae_grid']
        
        # 绘制热力图
        im = ax.contourf(delta1_grid, delta2_grid, mae_grid,
                        levels=20, cmap='RdYlGn_r')
        
        # 添加等高线
        contours = ax.contour(delta1_grid, delta2_grid, mae_grid,
                            levels=10, colors='black', alpha=0.4, linewidths=0.5)
        
        # 添加约束线 δ1 < δ2
        ax.plot([0, max(delta2_grid.flatten())], 
               [0, max(delta2_grid.flatten())], 
               'k--', linewidth=2, label=r'$\delta_1 = \delta_2$')
        
        # 标记最优点
        best_idx = np.unravel_index(np.argmin(mae_grid), mae_grid.shape)
        ax.plot(delta1_grid[best_idx], delta2_grid[best_idx],
               'w*', markersize=15, markeredgecolor='black')
        
        # 颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('MAE')
        
        apply_axis_style(ax,
                        xlabel=r'$\delta_1$',
                        ylabel=r'$\delta_2$',
                        title=r'$\delta_1$-$\delta_2$ Interaction',
                        grid=False)
        
        # 添加子图标签
        add_subplot_labels(axes.flatten())
        
        plt.tight_layout()
        
        # 保存图形
        self.export_manager.export_figure(
            fig, save_name,
            formats=['pdf', 'eps', 'png'],
            category='analysis'
        )
        
        return fig
        
    def generate_robustness_analysis(self,
                                   robustness_results: Dict,
                                   save_name: str = "robustness_analysis") -> plt.Figure:
        """生成鲁棒性分析图（图5）
        
        Args:
            robustness_results: 鲁棒性测试结果
            save_name: 保存名称
        """
        # 创建1x2子图
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                      figsize=PlotConfig.get_figure_size('double', 0.5))
        
        # 1. 噪声鲁棒性
        noise_levels = robustness_results['noise']['levels']
        methods = ['HPL', 'L2', 'L1', 'Huber']
        colors = ['blue', 'red', 'green', 'magenta']
        markers = ['o', 's', '^', 'D']
        
        for method, color, marker in zip(methods, colors, markers):
            performances = robustness_results['noise'][method]
            # 计算相对性能退化
            relative_perf = (performances - performances[0]) / performances[0] * 100
            
            ax1.plot(noise_levels, relative_perf,
                    color=color, marker=marker, markersize=8,
                    linewidth=2, label=method)
                    
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        apply_axis_style(ax1,
                        xlabel='Noise Level',
                        ylabel='Performance Degradation (%)',
                        title='Robustness to Noise',
                        legend=True)
        
        ax1.grid(True, alpha=0.3)
        
        # 2. 数据稀疏性分析
        sparsity_levels = robustness_results['sparsity']['levels']
        
        for method, color, marker in zip(methods, colors, markers):
            performances = robustness_results['sparsity'][method]
            
            ax2.plot(sparsity_levels * 100, performances,
                    color=color, marker=marker, markersize=8,
                    linewidth=2, label=method)
                    
        ax2.invert_xaxis()  # 反转x轴
        
        apply_axis_style(ax2,
                        xlabel='Data Density (%)',
                        ylabel='MAE',
                        title='Performance vs Data Sparsity',
                        legend=False)
        
        ax2.grid(True, alpha=0.3)
        
        # 添加子图标签
        add_subplot_labels([ax1, ax2])
        
        plt.tight_layout()
        
        # 保存图形
        self.export_manager.export_figure(
            fig, save_name,
            formats=['pdf', 'eps', 'png'],
            category='analysis'
        )
        
        return fig
        
    def generate_all_paper_figures(self,
                                 results_data: pd.DataFrame,
                                 histories: Dict,
                                 param_results: Dict,
                                 robustness_results: Dict) -> Dict[str, plt.Figure]:
        """生成所有论文图表
        
        Args:
            results_data: 性能结果数据
            histories: 训练历史
            param_results: 参数分析结果
            robustness_results: 鲁棒性分析结果
            
        Returns:
            图表字典
        """
        figures = {}
        
        # 图1：主要结果
        print("生成图1：主要结果对比...")
        figures['fig1_main_results'] = self.generate_main_results_figure(
            results_data,
            datasets=['ML-100K', 'ML-1M', 'Netflix', 'Yahoo Music']
        )
        
        # 图2：HPL损失函数示意
        print("生成图2：HPL损失函数...")
        figures['fig2_hpl_loss'] = self.generate_hpl_illustration()
        
        # 图3：收敛对比
        print("生成图3：收敛曲线对比...")
        figures['fig3_convergence'] = self.generate_convergence_comparison(histories)
        
        # 图4：参数分析
        print("生成图4：参数影响分析...")
        figures['fig4_parameters'] = self.generate_parameter_analysis(param_results)
        
        # 图5：鲁棒性分析
        print("生成图5：鲁棒性分析...")
        figures['fig5_robustness'] = self.generate_robustness_analysis(robustness_results)
        
        # 创建图表包
        figure_list = list(figures.items())
        package_path = self.export_manager.create_figure_package(
            figure_list,
            "paper_figures_all"
        )
        
        # 生成LaTeX包含文件
        latex_file = self.export_manager.generate_latex_figures_file(
            list(figures.keys()),
            "paper_figures.tex"
        )
        
        print(f"\n所有图表已生成完毕！")
        print(f"图表目录: {self.export_dir}")
        print(f"图表包: {package_path}")
        print(f"LaTeX文件: {latex_file}")
        
        return figures


# 便捷函数
def generate_main_results_figure(results_data: pd.DataFrame,
                               datasets: List[str],
                               save_path: Optional[str] = None) -> plt.Figure:
    """快速生成主要结果图"""
    generator = PaperFigureGenerator()
    return generator.generate_main_results_figure(results_data, datasets)


def generate_method_comparison_figure(results_data: pd.DataFrame,
                                    save_path: Optional[str] = None) -> plt.Figure:
    """快速生成方法对比图"""
    generator = PaperFigureGenerator()
    datasets = results_data['dataset'].unique().tolist()
    return generator.generate_main_results_figure(results_data, datasets)


def generate_hpl_illustration(save_path: Optional[str] = None) -> plt.Figure:
    """快速生成HPL损失函数示意图"""
    generator = PaperFigureGenerator()
    return generator.generate_hpl_illustration()


def generate_all_paper_figures(results_data: pd.DataFrame,
                             histories: Dict,
                             param_results: Dict,
                             robustness_results: Dict,
                             export_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
    """快速生成所有论文图表"""
    if export_dir:
        generator = PaperFigureGenerator(export_dir=export_dir)
    else:
        generator = PaperFigureGenerator()
        
    return generator.generate_all_paper_figures(
        results_data, histories, param_results, robustness_results
    )