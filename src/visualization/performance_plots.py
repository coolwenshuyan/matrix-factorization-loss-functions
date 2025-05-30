"""
性能对比图表模块

生成各种性能对比可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from .plot_config import (
    PlotConfig, 
    apply_axis_style, 
    get_color_palette,
    get_marker_styles,
    create_figure_with_subplots
)


class PerformancePlotter:
    """性能图表绘制器"""
    
    def __init__(self, style: str = 'academic'):
        self.style = style
        self.colors = get_color_palette('academic')
        
    def plot_comparison_bars(self,
                           data: pd.DataFrame,
                           metrics: List[str],
                           group_by: str = 'method',
                           split_by: str = 'dataset',
                           figsize: Optional[Tuple] = None,
                           show_values: bool = True,
                           ylabel_map: Optional[Dict] = None) -> plt.Figure:
        """绘制分组条形图
        
        Args:
            data: 包含性能数据的DataFrame
            metrics: 要显示的指标列表
            group_by: 分组字段
            split_by: 分割字段
            figsize: 图形大小
            show_values: 是否显示数值
            ylabel_map: Y轴标签映射
        """
        n_metrics = len(metrics)
        
        if figsize is None:
            figsize = PlotConfig.get_subplot_size(n_metrics, n_metrics)
            
        fig, axes = create_figure_with_subplots(n_metrics, n_metrics, width='double')
        
        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            # 准备数据
            pivot_data = data.pivot(index=split_by, columns=group_by, values=metric)
            
            # 绘制条形图
            pivot_data.plot(kind='bar', ax=ax, 
                          color=self.colors[:len(pivot_data.columns)],
                          width=0.8, edgecolor='black', linewidth=0.5)
            
            # 设置标签
            ylabel = ylabel_map.get(metric, metric) if ylabel_map else metric
            apply_axis_style(ax, 
                           xlabel=split_by.capitalize(),
                           ylabel=ylabel,
                           title=f'{metric.upper()} Comparison',
                           legend=(idx == 0))
            
            # 旋转x轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
            # 添加数值标签
            if show_values:
                for container in ax.containers:
                    ax.bar_label(container, fmt='%.3f', padding=3, fontsize=8)
                    
            # 添加基准线（可选）
            if metric in ['mae', 'rmse']:
                # 为误差指标添加零基准线
                ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                
        plt.tight_layout()
        return fig
        
    def plot_performance_heatmap(self,
                               data: pd.DataFrame,
                               metric: str,
                               row_field: str = 'dataset',
                               col_field: str = 'method',
                               figsize: Optional[Tuple] = None,
                               cmap: str = 'RdYlGn_r',
                               annot_fmt: str = '.3f') -> plt.Figure:
        """绘制性能热力图
        
        Args:
            data: 性能数据
            metric: 指标名称
            row_field: 行字段
            col_field: 列字段
            figsize: 图形大小
            cmap: 颜色映射
            annot_fmt: 标注格式
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 准备数据
        pivot_data = data.pivot(index=row_field, columns=col_field, values=metric)
        
        # 确定颜色映射方向
        if metric.lower() in ['mae', 'rmse', 'mse']:
            # 误差指标：越小越好，使用反向颜色
            cmap = cmap if cmap.endswith('_r') else cmap + '_r'
        else:
            # 其他指标：越大越好
            cmap = cmap.replace('_r', '') if cmap.endswith('_r') else cmap
            
        # 绘制热力图
        sns.heatmap(pivot_data, 
                   annot=True, 
                   fmt=annot_fmt,
                   cmap=cmap,
                   cbar_kws={'label': metric},
                   linewidths=0.5,
                   linecolor='gray',
                   square=True,
                   ax=ax)
        
        # 高亮最佳值
        self._highlight_best_values(ax, pivot_data, metric)
        
        apply_axis_style(ax,
                        xlabel=col_field.capitalize(),
                        ylabel=row_field.capitalize(),
                        title=f'{metric.upper()} Performance Heatmap',
                        grid=False,
                        legend=False)
        
        # 调整标签
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        plt.tight_layout()
        return fig
        
    def plot_performance_radar(self,
                             data: pd.DataFrame,
                             methods: List[str],
                             metrics: List[str],
                             figsize: Optional[Tuple] = None,
                             fill_alpha: float = 0.2) -> plt.Figure:
        """绘制雷达图（蜘蛛图）
        
        Args:
            data: 性能数据
            methods: 方法列表
            metrics: 指标列表
            figsize: 图形大小
            fill_alpha: 填充透明度
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=1.0)
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='polar')
        
        # 准备数据
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形
        
        # 归一化数据（0-1范围）
        normalized_data = {}
        for metric in metrics:
            values = data[data['method'].isin(methods)][metric].values
            min_val, max_val = values.min(), values.max()
            
            # 根据指标类型归一化
            if metric.lower() in ['mae', 'rmse', 'mse']:
                # 误差指标：反向归一化（越小越好）
                normalized_data[metric] = 1 - (values - min_val) / (max_val - min_val)
            else:
                # 其他指标：正向归一化（越大越好）
                normalized_data[metric] = (values - min_val) / (max_val - min_val)
                
        # 绘制每个方法
        for i, method in enumerate(methods):
            method_data = data[data['method'] == method]
            values = []
            
            for metric in metrics:
                idx = method_data.index[0]
                norm_val = normalized_data[metric][list(data.index).index(idx)]
                values.append(norm_val)
                
            values += values[:1]  # 闭合
            
            # 绘制线条和填充
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=method, color=self.colors[i])
            ax.fill(angles, values, alpha=fill_alpha, color=self.colors[i])
            
        # 设置轴标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        
        # 设置范围
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # 添加网格
        ax.grid(True, linestyle=':', alpha=0.5)
        
        # 图例
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        plt.title('Multi-Metric Performance Comparison', pad=20)
        plt.tight_layout()
        
        return fig
        
    def plot_method_ranking(self,
                          data: pd.DataFrame,
                          metrics: List[str],
                          datasets: List[str],
                          figsize: Optional[Tuple] = None) -> plt.Figure:
        """绘制方法排名图
        
        Args:
            data: 性能数据
            metrics: 指标列表
            datasets: 数据集列表
            figsize: 图形大小
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算平均排名
        rankings = {}
        methods = data['method'].unique()
        
        for method in methods:
            method_ranks = []
            
            for dataset in datasets:
                for metric in metrics:
                    subset = data[data['dataset'] == dataset]
                    
                    # 根据指标类型排序
                    if metric.lower() in ['mae', 'rmse', 'mse']:
                        # 误差指标：升序排序
                        subset = subset.sort_values(metric)
                    else:
                        # 其他指标：降序排序
                        subset = subset.sort_values(metric, ascending=False)
                        
                    # 获取排名
                    rank = subset['method'].tolist().index(method) + 1
                    method_ranks.append(rank)
                    
            rankings[method] = np.mean(method_ranks)
            
        # 排序方法
        sorted_methods = sorted(rankings.keys(), key=lambda x: rankings[x])
        sorted_ranks = [rankings[m] for m in sorted_methods]
        
        # 绘制条形图
        y_pos = np.arange(len(sorted_methods))
        bars = ax.barh(y_pos, sorted_ranks, color=self.colors[:len(sorted_methods)])
        
        # 添加数值标签
        for i, (bar, rank) in enumerate(zip(bars, sorted_ranks)):
            ax.text(rank + 0.05, bar.get_y() + bar.get_height()/2,
                   f'{rank:.2f}', va='center', fontsize=9)
                   
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_methods)
        ax.invert_yaxis()
        
        apply_axis_style(ax,
                        xlabel='Average Rank',
                        title='Method Ranking Across All Metrics and Datasets',
                        grid=True,
                        legend=False)
        
        # 添加垂直参考线
        ax.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Best')
        ax.axvline(x=len(methods), color='red', linestyle='--', alpha=0.5, label='Worst')
        
        plt.tight_layout()
        return fig
        
    def plot_improvement_bars(self,
                            baseline_data: pd.DataFrame,
                            improved_data: pd.DataFrame,
                            metrics: List[str],
                            figsize: Optional[Tuple] = None) -> plt.Figure:
        """绘制改进幅度条形图
        
        Args:
            baseline_data: 基线数据
            improved_data: 改进后数据
            metrics: 指标列表
            figsize: 图形大小
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.6)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算改进百分比
        improvements = []
        labels = []
        
        for _, baseline_row in baseline_data.iterrows():
            dataset = baseline_row['dataset']
            method = baseline_row['method']
            
            # 找到对应的改进数据
            improved_row = improved_data[
                (improved_data['dataset'] == dataset) &
                (improved_data['method'] == 'HPL')
            ].iloc[0]
            
            for metric in metrics:
                baseline_val = baseline_row[metric]
                improved_val = improved_row[metric]
                
                # 计算改进百分比
                if metric.lower() in ['mae', 'rmse', 'mse']:
                    # 误差指标：减少百分比
                    improvement = (baseline_val - improved_val) / baseline_val * 100
                else:
                    # 其他指标：增加百分比
                    improvement = (improved_val - baseline_val) / baseline_val * 100
                    
                improvements.append(improvement)
                labels.append(f'{dataset}\n{method}\n{metric}')
                
        # 绘制条形图
        x_pos = np.arange(len(improvements))
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        
        bars = ax.bar(x_pos, improvements, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, 
                   height + (1 if height > 0 else -3),
                   f'{imp:.1f}%', ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
                   
        # 设置标签
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        
        # 添加基准线
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        apply_axis_style(ax,
                        ylabel='Improvement (%)',
                        title='Performance Improvement over Baselines',
                        grid=True,
                        legend=False)
        
        plt.tight_layout()
        return fig
        
    def _highlight_best_values(self, ax: plt.Axes, data: pd.DataFrame, metric: str):
        """在热力图中高亮最佳值"""
        # 确定是否越小越好
        minimize = metric.lower() in ['mae', 'rmse', 'mse']
        
        # 找到每行的最佳值
        for i, row in enumerate(data.values):
            if minimize:
                best_idx = np.argmin(row)
            else:
                best_idx = np.argmax(row)
                
            # 添加边框
            rect = Rectangle((best_idx, i), 1, 1, linewidth=2, 
                           edgecolor='black', facecolor='none')
            ax.add_patch(rect)
            
        # 找到每列的最佳值
        for j in range(data.shape[1]):
            col = data.iloc[:, j].values
            if minimize:
                best_idx = np.argmin(col)
            else:
                best_idx = np.argmax(col)
                
            # 添加星号标记
            ax.text(j + 0.5, best_idx + 0.5, '*', 
                   ha='center', va='center', 
                   fontsize=16, fontweight='bold')


def plot_performance_comparison(results: List[Dict],
                              metrics: List[str] = ['mae', 'rmse', 'hr@10', 'ndcg@10'],
                              save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制性能对比图
    
    Args:
        results: 结果列表
        metrics: 指标列表
        save_path: 保存路径
    """
    plotter = PerformancePlotter()
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 绘制对比条形图
    fig = plotter.plot_comparison_bars(df, metrics)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_performance_heatmap(results: List[Dict],
                           metric: str = 'mae',
                           save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制性能热力图
    
    Args:
        results: 结果列表
        metric: 指标名称
        save_path: 保存路径
    """
    plotter = PerformancePlotter()
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 绘制热力图
    fig = plotter.plot_performance_heatmap(df, metric)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_performance_radar(results: List[Dict],
                         methods: List[str],
                         metrics: List[str] = ['mae', 'rmse', 'hr@10', 'ndcg@10'],
                         save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制雷达图
    
    Args:
        results: 结果列表
        methods: 要比较的方法
        metrics: 指标列表
        save_path: 保存路径
    """
    plotter = PerformancePlotter()
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 绘制雷达图
    fig = plotter.plot_performance_radar(df, methods, metrics)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_method_ranking(results: List[Dict],
                      metrics: List[str] = ['mae', 'rmse', 'hr@10', 'ndcg@10'],
                      save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制方法排名图
    
    Args:
        results: 结果列表
        metrics: 指标列表
        save_path: 保存路径
    """
    plotter = PerformancePlotter()
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    datasets = df['dataset'].unique().tolist()
    
    # 绘制排名图
    fig = plotter.plot_method_ranking(df, metrics, datasets)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig