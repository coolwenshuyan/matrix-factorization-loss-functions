"""
收敛曲线图表模块

生成训练过程相关的可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from scipy.signal import savgol_filter

from .plot_config import (
    PlotConfig,
    apply_axis_style,
    get_color_palette,
    get_marker_styles,
    create_figure_with_subplots,
    add_subplot_labels
)


class ConvergencePlotter:
    """收敛图表绘制器"""

    def __init__(self, style: str = 'academic'):
        self.style = style
        self.colors = get_color_palette('academic')
        self.markers = get_marker_styles(10)

    def plot_loss_curves(self,
                        histories: Dict[str, Dict[str, List[float]]],
                        metrics: List[str] = ['train_loss', 'val_loss'],
                        figsize: Optional[Tuple] = None,
                        smooth: bool = True,
                        smooth_window: int = 5,
                        mark_best: bool = True) -> plt.Figure:
        """绘制损失曲线

        Args:
            histories: 方法名到训练历史的字典
            metrics: 要绘制的指标
            figsize: 图形大小
            smooth: 是否平滑曲线
            smooth_window: 平滑窗口大小
            mark_best: 是否标记最佳点
        """
        n_metrics = len(metrics)

        if figsize is None:
            figsize = PlotConfig.get_subplot_size(n_metrics, n_metrics)

        fig, axes = create_figure_with_subplots(n_metrics, n_metrics)
        if n_metrics == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics):
            for i, (method, history) in enumerate(histories.items()):
                if metric in history:
                    values = history[metric]
                    epochs = np.arange(1, len(values) + 1)

                    # 原始曲线
                    ax.plot(epochs, values,
                           alpha=0.3, color=self.colors[i])

                    # 平滑曲线
                    if smooth and len(values) > smooth_window:
                        smoothed = savgol_filter(values, smooth_window, 3)
                        ax.plot(epochs, smoothed,
                               label=method, color=self.colors[i],
                               linewidth=2)
                    else:
                        ax.plot(epochs, values,
                               label=method, color=self.colors[i],
                               linewidth=2)

                    # 标记最佳点
                    if mark_best:
                        if 'loss' in metric.lower():
                            best_idx = np.argmin(values)
                        else:
                            best_idx = np.argmax(values)

                        ax.plot(epochs[best_idx], values[best_idx],
                               'o', color=self.colors[i], markersize=8,
                               markeredgecolor='white', markeredgewidth=2)

                        # 添加文本标注
                        ax.annotate(f'{values[best_idx]:.4f}',
                                   xy=(epochs[best_idx], values[best_idx]),
                                   xytext=(5, 5), textcoords='offset points',
                                   fontsize=8, ha='left')

            apply_axis_style(ax,
                           xlabel='Epoch',
                           ylabel=metric.replace('_', ' ').title(),
                           title=f'{metric.replace("_", " ").title()} During Training',
                           legend=True)

        # 添加子图标签
        add_subplot_labels(axes)

        plt.tight_layout()
        return fig

    def plot_metric_comparison(self,
                             histories: Dict[str, Dict[str, List[float]]],
                             train_metric: str = 'train_loss',
                             val_metric: str = 'val_loss',
                             figsize: Optional[Tuple] = None) -> plt.Figure:
        """绘制训练/验证指标对比

        Args:
            histories: 训练历史
            train_metric: 训练指标名
            val_metric: 验证指标名
            figsize: 图形大小
        """
        n_methods = len(histories)

        if figsize is None:
            figsize = PlotConfig.get_subplot_size(n_methods, min(n_methods, 3))

        fig, axes = create_figure_with_subplots(n_methods, min(n_methods, 3))

        for ax, (method, history) in zip(axes, histories.items()):
            if train_metric in history and val_metric in history:
                train_values = history[train_metric]
                val_values = history[val_metric]
                epochs = np.arange(1, len(train_values) + 1)

                # 绘制训练和验证曲线
                ax.plot(epochs, train_values,
                       label='Training', color='blue', linewidth=2)
                ax.plot(epochs, val_values,
                       label='Validation', color='red', linewidth=2)

                # 填充区域表示过拟合
                ax.fill_between(epochs, train_values, val_values,
                               where=(np.array(val_values) > np.array(train_values)),
                               alpha=0.3, color='red', label='Overfitting')

                # 标记早停点
                if len(val_values) < len(train_values):
                    early_stop_epoch = len(val_values)
                    ax.axvline(x=early_stop_epoch, color='green',
                              linestyle='--', alpha=0.7, label='Early Stop')

            apply_axis_style(ax,
                           xlabel='Epoch',
                           ylabel='Loss',
                           title=f'{method}',
                           legend=True)

        plt.tight_layout()
        return fig

    def plot_learning_rate_schedule(self,
                                  lr_schedules: Dict[str, List[float]],
                                  figsize: Optional[Tuple] = None,
                                  log_scale: bool = True) -> plt.Figure:
        """绘制学习率调度曲线

        Args:
            lr_schedules: 方法名到学习率列表的字典
            figsize: 图形大小
            log_scale: 是否使用对数尺度
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.5)

        fig, ax = plt.subplots(figsize=figsize)

        for i, (method, lr_values) in enumerate(lr_schedules.items()):
            epochs = np.arange(1, len(lr_values) + 1)

            ax.plot(epochs, lr_values,
                   label=method, color=self.colors[i],
                   linewidth=2, marker='o', markersize=4,
                   markevery=max(1, len(epochs)//20))

            # 标记关键点（学习率变化点）
            changes = []
            for j in range(1, len(lr_values)):
                if lr_values[j] != lr_values[j-1]:
                    changes.append(j)

            for change_epoch in changes:
                ax.axvline(x=change_epoch, color=self.colors[i],
                          linestyle=':', alpha=0.5)

        if log_scale:
            ax.set_yscale('log')

        apply_axis_style(ax,
                        xlabel='Epoch',
                        ylabel='Learning Rate',
                        title='Learning Rate Schedule',
                        legend=True)

        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_gradient_statistics(self,
                               grad_stats: Dict[str, Dict[str, List[float]]],
                               figsize: Optional[Tuple] = None) -> plt.Figure:
        """绘制梯度统计信息

        Args:
            grad_stats: 梯度统计数据
            figsize: 图形大小
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=1.2)

        fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

        # 梯度范数
        ax = axes[0]
        for i, (method, stats) in enumerate(grad_stats.items()):
            if 'grad_norm' in stats:
                epochs = np.arange(1, len(stats['grad_norm']) + 1)
                ax.plot(epochs, stats['grad_norm'],
                       label=method, color=self.colors[i], linewidth=2)

        ax.set_yscale('log')
        apply_axis_style(ax,
                        ylabel='Gradient Norm',
                        title='Gradient Norm Evolution',
                        legend=True,
                        tight_layout=False)

        # 梯度裁剪比例
        ax = axes[1]
        for i, (method, stats) in enumerate(grad_stats.items()):
            if 'clip_ratio' in stats:
                epochs = np.arange(1, len(stats['clip_ratio']) + 1)
                ax.plot(epochs, stats['clip_ratio'],
                       label=method, color=self.colors[i], linewidth=2)

        apply_axis_style(ax,
                        ylabel='Clip Ratio',
                        title='Gradient Clipping Ratio',
                        legend=False,
                        tight_layout=False)

        # 更新比例（参数更新量/参数值）
        ax = axes[2]
        for i, (method, stats) in enumerate(grad_stats.items()):
            if 'update_ratio' in stats:
                epochs = np.arange(1, len(stats['update_ratio']) + 1)
                ax.plot(epochs, stats['update_ratio'],
                       label=method, color=self.colors[i], linewidth=2)

        ax.set_yscale('log')
        apply_axis_style(ax,
                        xlabel='Epoch',
                        ylabel='Update Ratio',
                        title='Parameter Update Ratio',
                        legend=False,
                        tight_layout=False)

        plt.tight_layout()
        return fig

    def plot_convergence_comparison(self,
                                  histories: Dict[str, Dict[str, List[float]]],
                                  metric: str = 'val_loss',
                                  figsize: Optional[Tuple] = None,
                                  show_variance: bool = True) -> plt.Figure:
        """绘制收敛性对比（多次运行）

        Args:
            histories: 训练历史（可包含多次运行）
            metric: 要比较的指标
            figsize: 图形大小
            show_variance: 是否显示方差带
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)

        fig, ax = plt.subplots(figsize=figsize)

        for i, (method, history) in enumerate(histories.items()):
            if metric in history:
                values = history[metric]

                # 如果是多次运行的结果（二维数组）
                if isinstance(values[0], list):
                    values_array = np.array(values)
                    mean_values = np.mean(values_array, axis=0)
                    std_values = np.std(values_array, axis=0)
                    epochs = np.arange(1, len(mean_values) + 1)

                    # 绘制均值曲线
                    ax.plot(epochs, mean_values,
                           label=method, color=self.colors[i], linewidth=2)

                    # 绘制方差带
                    if show_variance:
                        ax.fill_between(epochs,
                                      mean_values - std_values,
                                      mean_values + std_values,
                                      alpha=0.3, color=self.colors[i])
                else:
                    # 单次运行
                    epochs = np.arange(1, len(values) + 1)
                    ax.plot(epochs, values,
                           label=method, color=self.colors[i], linewidth=2)

        apply_axis_style(ax,
                        xlabel='Epoch',
                        ylabel=metric.replace('_', ' ').title(),
                        title='Convergence Comparison',
                        legend=True)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_early_stopping_analysis(self,
                                   histories: Dict[str, Dict[str, List[float]]],
                                   patience: int = 10,
                                   figsize: Optional[Tuple] = None) -> plt.Figure:
        """分析早停行为

        Args:
            histories: 训练历史
            patience: 早停耐心值
            figsize: 图形大小
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

        early_stop_epochs = {}

        for i, (method, history) in enumerate(histories.items()):
            if 'val_loss' in history:
                val_losses = history['val_loss']
                epochs = np.arange(1, len(val_losses) + 1)

                # 绘制验证损失
                ax1.plot(epochs, val_losses,
                        label=method, color=self.colors[i], linewidth=2)

                # 计算早停点
                best_loss = float('inf')
                best_epoch = 0
                wait = 0

                for epoch, loss in enumerate(val_losses):
                    if loss < best_loss:
                        best_loss = loss
                        best_epoch = epoch
                        wait = 0
                    else:
                        wait += 1

                    if wait >= patience:
                        early_stop_epochs[method] = epoch + 1
                        break

                # 标记早停点
                if method in early_stop_epochs:
                    stop_epoch = early_stop_epochs[method]
                    ax1.axvline(x=stop_epoch, color=self.colors[i],
                               linestyle='--', alpha=0.7)
                    ax1.plot(stop_epoch, val_losses[stop_epoch-1],
                            'o', color=self.colors[i], markersize=8)

        # 绘制相对改进
        for i, (method, history) in enumerate(histories.items()):
            if 'val_loss' in history:
                val_losses = history['val_loss']
                epochs = np.arange(1, len(val_losses) + 1)

                # 计算相对改进
                improvements = []
                for j in range(1, len(val_losses)):
                    imp = (val_losses[j-1] - val_losses[j]) / val_losses[j-1] * 100
                    improvements.append(imp)

                ax2.plot(epochs[1:], improvements,
                        label=method, color=self.colors[i], linewidth=2)

        # 添加零线
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

        apply_axis_style(ax1,
                        ylabel='Validation Loss',
                        title='Early Stopping Analysis',
                        legend=True,
                        tight_layout=False)

        apply_axis_style(ax2,
                        xlabel='Epoch',
                        ylabel='Relative Improvement (%)',
                        legend=False,
                        tight_layout=False)

        plt.tight_layout()
        return fig


# 便捷函数
def plot_training_curves(histories: Dict[str, Dict[str, List[float]]],
                       save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制训练曲线"""
    plotter = ConvergencePlotter()
    fig = plotter.plot_loss_curves(histories)

    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)

    return fig


def plot_validation_curves(histories: Dict[str, Dict[str, List[float]]],
                         save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制验证曲线对比"""
    plotter = ConvergencePlotter()
    fig = plotter.plot_metric_comparison(histories)

    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)

    return fig


def plot_learning_rate_schedule(lr_schedules: Dict[str, List[float]],
                              save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制学习率调度"""
    plotter = ConvergencePlotter()
    fig = plotter.plot_learning_rate_schedule(lr_schedules)

    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)

    return fig


def plot_gradient_statistics(grad_stats: Dict[str, Dict[str, List[float]]],
                           save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制梯度统计"""
    plotter = ConvergencePlotter()
    fig = plotter.plot_gradient_statistics(grad_stats)

    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)

    return fig
