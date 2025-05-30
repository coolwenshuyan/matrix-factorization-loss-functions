"""
分布分析图表模块

生成误差分布相关的可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats
from matplotlib.patches import Rectangle

from .plot_config import (
    PlotConfig,
    apply_axis_style,
    get_color_palette,
    create_figure_with_subplots
)


class DistributionPlotter:
    """分布图表绘制器"""
    
    def __init__(self, style: str = 'academic'):
        self.style = style
        self.colors = get_color_palette('academic')
        
    def plot_error_histogram(self,
                           errors: Dict[str, np.ndarray],
                           bins: int = 50,
                           figsize: Optional[Tuple] = None,
                           show_stats: bool = True,
                           kde: bool = True) -> plt.Figure:
        """绘制误差直方图
        
        Args:
            errors: 方法名到误差数组的字典
            bins: 直方图箱数
            figsize: 图形大小
            show_stats: 是否显示统计信息
            kde: 是否显示核密度估计
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制每个方法的误差分布
        for i, (method, err) in enumerate(errors.items()):
            # 直方图
            n, bins_edges, patches = ax.hist(err, bins=bins, alpha=0.5, 
                                            label=method, color=self.colors[i],
                                            density=True, edgecolor='black', 
                                            linewidth=0.5)
            
            # 核密度估计
            if kde:
                kde_x = np.linspace(err.min(), err.max(), 200)
                kde_y = stats.gaussian_kde(err)(kde_x)
                ax.plot(kde_x, kde_y, color=self.colors[i], linewidth=2, 
                       linestyle='-', alpha=0.8)
                
            # 添加统计信息
            if show_stats:
                mean_val = np.mean(err)
                median_val = np.median(err)
                std_val = np.std(err)
                
                # 添加均值线
                ax.axvline(mean_val, color=self.colors[i], linestyle='--', 
                         linewidth=1.5, alpha=0.7)
                
                # 添加文本标注
                ax.text(0.02, 0.98 - i*0.1, 
                       f'{method}: μ={mean_val:.3f}, σ={std_val:.3f}',
                       transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                       
        apply_axis_style(ax,
                        xlabel='Error',
                        ylabel='Density',
                        title='Error Distribution Comparison',
                        legend=True)
        
        plt.tight_layout()
        return fig
        
    def plot_error_boxplot(self,
                         errors: Dict[str, np.ndarray],
                         figsize: Optional[Tuple] = None,
                         show_outliers: bool = True,
                         notch: bool = True) -> plt.Figure:
        """绘制误差箱线图
        
        Args:
            errors: 方法名到误差数组的字典
            figsize: 图形大小
            show_outliers: 是否显示异常值
            notch: 是否显示缺口（置信区间）
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.5)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备数据
        data = []
        labels = []
        for method, err in errors.items():
            data.append(err)
            labels.append(method)
            
        # 绘制箱线图
        bp = ax.boxplot(data, labels=labels, notch=notch,
                       showfliers=show_outliers,
                       patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='red', linewidth=2),
                       flierprops=dict(marker='o', markersize=4, alpha=0.5))
        
        # 设置颜色
        for patch, color in zip(bp['boxes'], self.colors[:len(data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            
        # 添加均值点
        means = [np.mean(d) for d in data]
        ax.scatter(range(1, len(data)+1), means, 
                  marker='D', s=50, color='black', zorder=3, label='Mean')
        
        # 添加水平参考线
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        
        apply_axis_style(ax,
                        ylabel='Error',
                        title='Error Distribution Box Plot',
                        legend=True)
        
        # 旋转x轴标签
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
        
    def plot_qq_plots(self,
                     errors: Dict[str, np.ndarray],
                     figsize: Optional[Tuple] = None,
                     dist: str = 'norm') -> plt.Figure:
        """绘制Q-Q图
        
        Args:
            errors: 方法名到误差数组的字典
            figsize: 图形大小
            dist: 参考分布 ('norm', 't', 'uniform')
        """
        n_methods = len(errors)
        
        if figsize is None:
            figsize = PlotConfig.get_subplot_size(n_methods, min(n_methods, 3))
            
        fig, axes = create_figure_with_subplots(n_methods, min(n_methods, 3))
        
        for ax, (method, err) in zip(axes, errors.items()):
            # Q-Q图
            stats.probplot(err, dist=dist, plot=ax)
            
            # 设置标题和标签
            apply_axis_style(ax,
                           xlabel='Theoretical Quantiles',
                           ylabel='Sample Quantiles',
                           title=f'Q-Q Plot: {method}',
                           legend=False)
            
            # 添加45度参考线
            ax.get_lines()[0].set_color(self.colors[list(errors.keys()).index(method)])
            ax.get_lines()[0].set_markersize(4)
            ax.get_lines()[1].set_color('red')
            ax.get_lines()[1].set_linewidth(2)
            
            # 添加正态性检验结果
            statistic, p_value = stats.shapiro(err)
            ax.text(0.05, 0.95, f'Shapiro-Wilk\np={p_value:.4f}',
                   transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   fontsize=8, verticalalignment='top')
                   
        plt.tight_layout()
        return fig
        
    def plot_error_heatmap_matrix(self,
                                errors: np.ndarray,
                                user_ids: Optional[List] = None,
                                item_ids: Optional[List] = None,
                                figsize: Optional[Tuple] = None,
                                cmap: str = 'RdBu_r',
                                center: float = 0) -> plt.Figure:
        """绘制用户-物品误差热力图
        
        Args:
            errors: 误差矩阵 (n_users, n_items)
            user_ids: 用户ID列表
            item_ids: 物品ID列表
            figsize: 图形大小
            cmap: 颜色映射
            center: 颜色中心值
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 采样显示（如果矩阵太大）
        max_display = 50
        if errors.shape[0] > max_display or errors.shape[1] > max_display:
            # 随机采样
            user_sample = np.random.choice(errors.shape[0], 
                                         min(max_display, errors.shape[0]), 
                                         replace=False)
            item_sample = np.random.choice(errors.shape[1], 
                                         min(max_display, errors.shape[1]), 
                                         replace=False)
            errors = errors[user_sample][:, item_sample]
            
            if user_ids:
                user_ids = [user_ids[i] for i in user_sample]
            if item_ids:
                item_ids = [item_ids[i] for i in item_sample]
                
        # 绘制热力图
        im = ax.imshow(errors, cmap=cmap, aspect='auto', 
                      interpolation='nearest', center=center)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Prediction Error')
        
        # 设置标签
        if user_ids:
            ax.set_yticks(range(len(user_ids)))
            ax.set_yticklabels(user_ids)
        if item_ids:
            ax.set_xticks(range(len(item_ids)))
            ax.set_xticklabels(item_ids, rotation=90)
            
        apply_axis_style(ax,
                        xlabel='Item ID',
                        ylabel='User ID',
                        title='User-Item Error Heatmap',
                        grid=False,
                        legend=False)
        
        plt.tight_layout()
        return fig
        
    def plot_error_by_group(self,
                          errors: pd.DataFrame,
                          group_by: str,
                          error_col: str = 'error',
                          figsize: Optional[Tuple] = None,
                          plot_type: str = 'violin') -> plt.Figure:
        """按组绘制误差分布
        
        Args:
            errors: 包含误差和分组信息的DataFrame
            group_by: 分组字段
            error_col: 误差列名
            figsize: 图形大小
            plot_type: 图表类型 ('violin', 'box', 'strip')
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.6)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 根据类型绘图
        if plot_type == 'violin':
            sns.violinplot(data=errors, x=group_by, y=error_col, ax=ax,
                          palette=self.colors, inner='box')
        elif plot_type == 'box':
            sns.boxplot(data=errors, x=group_by, y=error_col, ax=ax,
                       palette=self.colors, showmeans=True)
        elif plot_type == 'strip':
            sns.stripplot(data=errors, x=group_by, y=error_col, ax=ax,
                         palette=self.colors, alpha=0.6, size=3)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")
            
        # 添加均值线
        group_means = errors.groupby(group_by)[error_col].mean()
        for i, (group, mean) in enumerate(group_means.items()):
            ax.hlines(mean, i-0.4, i+0.4, colors='red', linestyles='--', linewidth=2)
            
        apply_axis_style(ax,
                        xlabel=group_by.capitalize(),
                        ylabel='Error',
                        title=f'Error Distribution by {group_by}',
                        legend=False)
        
        # 旋转x轴标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        return fig
        
    def plot_cumulative_error(self,
                            errors: Dict[str, np.ndarray],
                            figsize: Optional[Tuple] = None,
                            percentiles: List[float] = [50, 80, 90, 95, 99]) -> plt.Figure:
        """绘制累积误差分布
        
        Args:
            errors: 方法名到误差数组的字典
            figsize: 图形大小
            percentiles: 要标注的百分位数
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, (method, err) in enumerate(errors.items()):
            # 计算累积分布
            sorted_err = np.sort(np.abs(err))
            cumulative = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
            
            # 绘制曲线
            ax.plot(sorted_err, cumulative * 100, 
                   label=method, color=self.colors[i], linewidth=2)
            
            # 标注百分位数
            for p in percentiles:
                idx = int(p / 100 * len(sorted_err))
                if idx < len(sorted_err):
                    ax.plot(sorted_err[idx], p, 'o', color=self.colors[i], 
                           markersize=4)
                    
        # 添加参考线
        for p in percentiles:
            ax.axhline(y=p, color='gray', linestyle=':', alpha=0.5)
            ax.text(ax.get_xlim()[1]*0.95, p+1, f'{p}%', 
                   ha='right', fontsize=8)
                   
        apply_axis_style(ax,
                        xlabel='Absolute Error',
                        ylabel='Cumulative Percentage (%)',
                        title='Cumulative Error Distribution',
                        legend=True)
        
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


# 便捷函数
def plot_error_distribution(errors: Dict[str, np.ndarray],
                          plot_types: List[str] = ['histogram', 'boxplot'],
                          save_path: Optional[str] = None) -> Dict[str, plt.Figure]:
    """快速绘制误差分布图
    
    Args:
        errors: 误差数据
        plot_types: 要绘制的图表类型
        save_path: 保存路径前缀
    """
    plotter = DistributionPlotter()
    figures = {}
    
    if 'histogram' in plot_types:
        fig = plotter.plot_error_histogram(errors)
        figures['histogram'] = fig
        if save_path:
            from .plot_config import save_figure
            save_figure(fig, f"{save_path}_histogram")
            
    if 'boxplot' in plot_types:
        fig = plotter.plot_error_boxplot(errors)
        figures['boxplot'] = fig
        if save_path:
            from .plot_config import save_figure
            save_figure(fig, f"{save_path}_boxplot")
            
    if 'qq' in plot_types:
        fig = plotter.plot_qq_plots(errors)
        figures['qq'] = fig
        if save_path:
            from .plot_config import save_figure
            save_figure(fig, f"{save_path}_qq")
            
    if 'cumulative' in plot_types:
        fig = plotter.plot_cumulative_error(errors)
        figures['cumulative'] = fig
        if save_path:
            from .plot_config import save_figure
            save_figure(fig, f"{save_path}_cumulative")
            
    return figures


def plot_error_boxplot(errors: Dict[str, np.ndarray],
                      save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制误差箱线图"""
    plotter = DistributionPlotter()
    fig = plotter.plot_error_boxplot(errors)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_qq_plot(errors: Dict[str, np.ndarray],
                save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制Q-Q图"""
    plotter = DistributionPlotter()
    fig = plotter.plot_qq_plots(errors)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_error_heatmap(error_matrix: np.ndarray,
                      save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制误差热力图"""
    plotter = DistributionPlotter()
    fig = plotter.plot_error_heatmap_matrix(error_matrix)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig