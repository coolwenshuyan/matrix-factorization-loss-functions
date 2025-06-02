"""
绘图配置模块

提供统一的绘图样式和配置
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os

class PlotConfig:
    """绘图配置类"""

    # 默认图形大小
    FIGURE_SIZES = {
        'small': (4, 3),
        'medium': (6, 4),
        'large': (8, 6),
        'double': (10, 6),
        'wide': (12, 6),
        'poster': (16, 9)
    }

    # 默认颜色方案
    COLOR_PALETTES = {
        'academic': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        'pastel': sns.color_palette("pastel"),
        'muted': sns.color_palette("muted"),
        'bright': sns.color_palette("bright"),
        'dark': sns.color_palette("dark"),
        'colorblind': sns.color_palette("colorblind")
    }

    # 默认标记样式
    MARKER_STYLES = [
        {'marker': 'o', 'markersize': 8, 'linewidth': 2, 'color': '#1f77b4'},
        {'marker': 's', 'markersize': 8, 'linewidth': 2, 'color': '#ff7f0e'},
        {'marker': '^', 'markersize': 8, 'linewidth': 2, 'color': '#2ca02c'},
        {'marker': 'D', 'markersize': 8, 'linewidth': 2, 'color': '#d62728'},
        {'marker': 'v', 'markersize': 8, 'linewidth': 2, 'color': '#9467bd'},
        {'marker': 'p', 'markersize': 8, 'linewidth': 2, 'color': '#8c564b'},
        {'marker': '*', 'markersize': 10, 'linewidth': 2, 'color': '#e377c2'},
        {'marker': 'h', 'markersize': 8, 'linewidth': 2, 'color': '#7f7f7f'},
        {'marker': 'X', 'markersize': 8, 'linewidth': 2, 'color': '#bcbd22'},
        {'marker': 'P', 'markersize': 8, 'linewidth': 2, 'color': '#17becf'}
    ]

    @staticmethod
    def get_figure_size(size: str = 'medium', aspect_ratio: float = None) -> Tuple[float, float]:
        """获取图形大小

        Args:
            size: 预定义大小名称
            aspect_ratio: 宽高比

        Returns:
            (宽, 高)元组
        """
        if size in PlotConfig.FIGURE_SIZES:
            width, height = PlotConfig.FIGURE_SIZES[size]
        else:
            width, height = PlotConfig.FIGURE_SIZES['medium']

        if aspect_ratio is not None:
            height = width / aspect_ratio

        return (width, height)

    @staticmethod
    def get_subplot_size(n_rows: int, n_cols: int, size: str = 'medium') -> Tuple[float, float]:
        """获取子图大小

        Args:
            n_rows: 行数
            n_cols: 列数
            size: 预定义大小名称

        Returns:
            (宽, 高)元组
        """
        base_width, base_height = PlotConfig.get_figure_size(size)
        return (base_width * n_cols, base_height * n_rows)


def set_plot_style(style: str = 'academic', use_latex: bool = False):
    """设置全局绘图样式

    Args:
        style: 样式名称
        use_latex: 是否使用LaTeX渲染
    """
    if style == 'academic':
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("paper", font_scale=1.5)
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-talk')
        sns.set_context("talk", font_scale=1.2)
    elif style == 'poster':
        plt.style.use('seaborn-v0_8-poster')
        sns.set_context("poster", font_scale=1.0)
    else:
        plt.style.use('seaborn-v0_8-whitegrid')

    if use_latex:
        # 设置LaTeX渲染
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
        })


def apply_axis_style(ax,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    title: Optional[str] = None,
                    legend: bool = False,
                    grid: bool = False,
                    tight_layout: bool = True):
    """应用坐标轴样式

    Args:
        ax: 坐标轴对象
        xlabel: x轴标签
        ylabel: y轴标签
        title: 标题
        legend: 是否显示图例
        grid: 是否显示网格
        tight_layout: 是否应用紧凑布局
    """
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(frameon=True, fancybox=True, framealpha=0.8,
                 loc='best', fontsize='small')
    if grid:
        ax.grid(True, alpha=0.3)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if tight_layout:
        plt.tight_layout()


def get_color_palette(palette_name: str = 'academic', n_colors: int = 10) -> List:
    """获取颜色方案

    Args:
        palette_name: 方案名称
        n_colors: 颜色数量

    Returns:
        颜色列表
    """
    if palette_name in PlotConfig.COLOR_PALETTES:
        palette = PlotConfig.COLOR_PALETTES[palette_name]
    else:
        palette = sns.color_palette("tab10", n_colors)

    return palette[:n_colors]


def get_marker_styles(n_markers: int = 10) -> List[Dict]:
    """获取标记样式

    Args:
        n_markers: 标记数量

    Returns:
        标记样式列表
    """
    markers = PlotConfig.MARKER_STYLES[:n_markers]
    return markers


def create_figure_with_subplots(n_rows: int, n_cols: int,
                              figsize: Optional[Tuple] = None) -> Tuple[plt.Figure, Union[plt.Axes, np.ndarray]]:
    """创建带子图的图形

    Args:
        n_rows: 行数
        n_cols: 列数
        figsize: 图形大小

    Returns:
        (图形, 坐标轴)元组
    """
    if figsize is None:
        figsize = PlotConfig.get_subplot_size(n_rows, n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    return fig, axes


def add_subplot_labels(axes, labels: Optional[List[str]] = None,
                     fontsize: int = 12, loc: str = 'upper left'):
    """添加子图标签

    Args:
        axes: 坐标轴数组
        labels: 标签列表
        fontsize: 字体大小
        loc: 位置
    """
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    flat_axes = axes.flatten()

    if labels is None:
        labels = [f'({chr(97+i)})' for i in range(len(flat_axes))]

    for ax, label in zip(flat_axes, labels):
        ax.text(0.05, 0.95, label, transform=ax.transAxes,
               fontsize=fontsize, fontweight='bold', va='top')


def save_figure(fig: plt.Figure,
              filename: str,
              dpi: int = 300,
              formats: List[str] = ['png', 'pdf']):
    """保存图形

    Args:
        fig: 图形对象
        filename: 文件名
        dpi: 分辨率
        formats: 格式列表
    """
    # 确保目录存在
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    # 保存为多种格式
    for fmt in formats:
        if '.' not in filename:
            save_path = f"{filename}.{fmt}"
        else:
            base, _ = os.path.splitext(filename)
            save_path = f"{base}.{fmt}"

        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图形已保存至: {save_path}")

