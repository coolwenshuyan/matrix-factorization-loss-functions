"""
鲁棒性分析图表模块

生成鲁棒性测试相关的可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats

from .plot_config import (
    PlotConfig,
    apply_axis_style,
    get_color_palette,
    get_marker_styles,
    create_figure_with_subplots
)


class RobustnessPlotter:
    """鲁棒性图表绘制器"""
    
    def __init__(self, style: str = 'academic'):
        self.style = style
        self.colors = get_color_palette('academic')
        self.markers = get_marker_styles(10)
        
    def plot_noise_robustness(self,
                            noise_levels: np.ndarray,
                            performances: Dict[str, np.ndarray],
                            metric_name: str = 'MAE',
                            figsize: Optional[Tuple] = None,
                            show_relative: bool = True) -> plt.Figure:
        """绘制噪声鲁棒性曲线
        
        Args:
            noise_levels: 噪声水平数组
            performances: 方法名到性能数组的字典
            metric_name: 指标名称
            figsize: 图形大小
            show_relative: 是否显示相对性能
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 基准性能（无噪声）
        baseline_perfs = {method: perfs[0] for method, perfs in performances.items()}
        
        for i, (method, perfs) in enumerate(performances.items()):
            if show_relative:
                # 相对性能退化
                if metric_name.lower() in ['mae', 'rmse', 'mse']:
                    # 误差指标：增加百分比
                    relative_perfs = (perfs - baseline_perfs[method]) / baseline_perfs[method] * 100
                else:
                    # 其他指标：减少百分比
                    relative_perfs = (baseline_perfs[method] - perfs) / baseline_perfs[method] * 100
                    
                ax.plot(noise_levels, relative_perfs,
                       label=method, **self.markers[i])
            else:
                # 绝对性能
                ax.plot(noise_levels, perfs,
                       label=method, **self.markers[i])
                       
        # 添加参考线
        if show_relative:
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ylabel = f'{metric_name} Degradation (%)'
        else:
            ylabel = metric_name
            
        apply_axis_style(ax,
                        xlabel='Noise Level',
                        ylabel=ylabel,
                        title=f'Robustness to Noise',
                        legend=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_sparsity_performance(self,
                                sparsity_levels: np.ndarray,
                                performances: Dict[str, np.ndarray],
                                metric_name: str = 'MAE',
                                figsize: Optional[Tuple] = None,
                                show_data_points: bool = True) -> plt.Figure:
        """绘制稀疏性-性能曲线
        
        Args:
            sparsity_levels: 稀疏度水平
            performances: 性能数据
            metric_name: 指标名称
            figsize: 图形大小
            show_data_points: 是否显示数据点
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (method, perfs) in enumerate(performances.items()):
            # 绘制曲线
            ax.plot(sparsity_levels * 100, perfs,
                   label=method, **self.markers[i])
                   
            if show_data_points:
                # 添加数据点标记
                ax.scatter(sparsity_levels * 100, perfs,
                          s=50, color=self.markers[i]['color'],
                          edgecolor='black', linewidth=0.5, zorder=5)
                          
        # 反转x轴（稀疏度从高到低）
        ax.invert_xaxis()
        
        apply_axis_style(ax,
                        xlabel='Data Density (%)',
                        ylabel=metric_name,
                        title='Performance vs Data Sparsity',
                        legend=True)
        
        # 添加网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
        
    def plot_stability_analysis(self,
                              results: Dict[str, List[np.ndarray]],
                              metric_name: str = 'MAE',
                              figsize: Optional[Tuple] = None,
                              show_individual: bool = False) -> plt.Figure:
        """绘制稳定性分析图
        
        Args:
            results: 多次运行的结果
            metric_name: 指标名称
            figsize: 图形大小
            show_individual: 是否显示单次运行
        """
        n_methods = len(results)
        
        if figsize is None:
            figsize = PlotConfig.get_subplot_size(2, 2)
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        # 1. 箱线图
        ax = axes[0]
        data = []
        labels = []
        
        for method, runs in results.items():
            # 每次运行的最终值
            final_values = [run[-1] if len(run) > 0 else np.nan for run in runs]
            data.append(final_values)
            labels.append(method)
            
        bp = ax.boxplot(data, labels=labels, notch=True, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], self.colors[:n_methods]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        apply_axis_style(ax,
                        ylabel=metric_name,
                        title='Performance Variability',
                        legend=False)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 2. 方差系数
        ax = axes[1]
        cv_values = []
        methods = []
        
        for method, runs in results.items():
            final_values = [run[-1] for run in runs if len(run) > 0]
            if final_values:
                mean_val = np.mean(final_values)
                std_val = np.std(final_values)
                cv = (std_val / mean_val * 100) if mean_val != 0 else 0
                cv_values.append(cv)
                methods.append(method)
                
        bars = ax.bar(range(len(methods)), cv_values, 
                      color=self.colors[:len(methods)], alpha=0.7)
        
        # 添加数值标签
        for bar, cv in zip(bars, cv_values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{cv:.1f}%', ha='center', va='bottom', fontsize=9)
                   
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        
        apply_axis_style(ax,
                        ylabel='Coefficient of Variation (%)',
                        title='Relative Stability',
                        legend=False)
        
        # 3. 收敛轨迹
        ax = axes[2]
        
        for i, (method, runs) in enumerate(results.items()):
            if show_individual:
                # 显示所有运行
                for run in runs:
                    epochs = np.arange(1, len(run) + 1)
                    ax.plot(epochs, run, color=self.colors[i], 
                           alpha=0.2, linewidth=0.5)
                           
            # 平均轨迹
            max_len = max(len(run) for run in runs)
            avg_trajectory = []
            
            for epoch in range(max_len):
                values = [run[epoch] for run in runs if len(run) > epoch]
                if values:
                    avg_trajectory.append(np.mean(values))
                    
            epochs = np.arange(1, len(avg_trajectory) + 1)
            ax.plot(epochs, avg_trajectory, 
                   label=method, color=self.colors[i], linewidth=2)
                   
        apply_axis_style(ax,
                        xlabel='Epoch',
                        ylabel=metric_name,
                        title='Average Convergence',
                        legend=True)
        
        # 4. 最终性能分布
        ax = axes[3]
        
        for i, (method, runs) in enumerate(results.items()):
            final_values = [run[-1] for run in runs if len(run) > 0]
            
            if final_values:
                # 核密度估计
                kde_x = np.linspace(min(final_values), max(final_values), 100)
                kde_y = stats.gaussian_kde(final_values)(kde_x)
                
                ax.plot(kde_x, kde_y, label=method, 
                       color=self.colors[i], linewidth=2)
                
                # 添加地毯图
                ax.plot(final_values, np.zeros_like(final_values) - 0.05 * (i+1),
                       '|', color=self.colors[i], markersize=10, alpha=0.5)
                       
        apply_axis_style(ax,
                        xlabel=metric_name,
                        ylabel='Density',
                        title='Final Performance Distribution',
                        legend=True)
        
        ax.set_ylim(bottom=-0.05 * (len(results) + 1))
        
        plt.tight_layout()
        return fig
        
    def plot_confidence_intervals(self,
                                methods: List[str],
                                means: np.ndarray,
                                confidence_intervals: np.ndarray,
                                metric_name: str = 'MAE',
                                figsize: Optional[Tuple] = None,
                                horizontal: bool = True) -> plt.Figure:
        """绘制置信区间图
        
        Args:
            methods: 方法名列表
            means: 均值数组
            confidence_intervals: 置信区间 (lower, upper)
            metric_name: 指标名称
            figsize: 图形大小
            horizontal: 是否水平绘制
        """
        if figsize is None:
            aspect = 0.5 if horizontal else 1.2
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=aspect)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        positions = np.arange(len(methods))
        
        if horizontal:
            # 水平误差条
            ax.barh(positions, means, xerr=confidence_intervals,
                   color=self.colors[:len(methods)], alpha=0.7,
                   error_kw={'linewidth': 2, 'capsize': 5, 'capthick': 2})
                   
            # 添加数值标签
            for i, (mean, ci) in enumerate(zip(means, confidence_intervals.T)):
                ax.text(mean + ci[1] + 0.01, i,
                       f'{mean:.3f} ± {ci[1]-mean:.3f}',
                       va='center', fontsize=9)
                       
            ax.set_yticks(positions)
            ax.set_yticklabels(methods)
            ax.invert_yaxis()
            
            apply_axis_style(ax,
                           xlabel=metric_name,
                           title=f'{metric_name} with 95% Confidence Intervals',
                           legend=False)
        else:
            # 垂直误差条
            ax.bar(positions, means, yerr=confidence_intervals,
                  color=self.colors[:len(methods)], alpha=0.7,
                  error_kw={'linewidth': 2, 'capsize': 5, 'capthick': 2})
                  
            ax.set_xticks(positions)
            ax.set_xticklabels(methods, rotation=45, ha='right')
            
            apply_axis_style(ax,
                           ylabel=metric_name,
                           title=f'{metric_name} with 95% Confidence Intervals',
                           legend=False)
        
        plt.tight_layout()
        return fig
        
    def plot_group_robustness(self,
                            group_performances: Dict[str, Dict[str, float]],
                            group_sizes: Dict[str, int],
                            metric_name: str = 'MAE',
                            figsize: Optional[Tuple] = None) -> plt.Figure:
        """绘制分组鲁棒性分析
        
        Args:
            group_performances: 组名到方法性能的嵌套字典
            group_sizes: 组大小
            metric_name: 指标名称
            figsize: 图形大小
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                      gridspec_kw={'height_ratios': [3, 1]})
        
        # 准备数据
        groups = list(group_performances.keys())
        methods = list(next(iter(group_performances.values())).keys())
        
        x = np.arange(len(groups))
        width = 0.8 / len(methods)
        
        # 绘制分组条形图
        for i, method in enumerate(methods):
            values = [group_performances[g][method] for g in groups]
            offset = (i - len(methods)/2 + 0.5) * width
            
            bars = ax1.bar(x + offset, values, width, 
                          label=method, color=self.colors[i], alpha=0.8)
                          
            # 添加数值标签
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)
                        
        ax1.set_xticks(x)
        ax1.set_xticklabels(groups, rotation=45, ha='right')
        
        apply_axis_style(ax1,
                        ylabel=metric_name,
                        title='Performance Across User/Item Groups',
                        legend=True)
        
        # 绘制组大小
        ax2.bar(x, [group_sizes[g] for g in groups], 
               color='gray', alpha=0.5)
               
        ax2.set_xticks(x)
        ax2.set_xticklabels(groups, rotation=45, ha='right')
        
        apply_axis_style(ax2,
                        ylabel='Group Size',
                        legend=False)
        
        plt.tight_layout()
        return fig


# 便捷函数
def plot_noise_robustness(noise_levels: np.ndarray,
                        performances: Dict[str, np.ndarray],
                        save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制噪声鲁棒性图"""
    plotter = RobustnessPlotter()
    fig = plotter.plot_noise_robustness(noise_levels, performances)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_sparsity_analysis(sparsity_levels: np.ndarray,
                         performances: Dict[str, np.ndarray],
                         save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制稀疏性分析图"""
    plotter = RobustnessPlotter()
    fig = plotter.plot_sparsity_performance(sparsity_levels, performances)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_stability_analysis(results: Dict[str, List[np.ndarray]],
                          save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制稳定性分析图"""
    plotter = RobustnessPlotter()
    fig = plotter.plot_stability_analysis(results)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_confidence_intervals(methods: List[str],
                            means: np.ndarray,
                            confidence_intervals: np.ndarray,
                            save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制置信区间图"""
    plotter = RobustnessPlotter()
    fig = plotter.plot_confidence_intervals(methods, means, confidence_intervals)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig