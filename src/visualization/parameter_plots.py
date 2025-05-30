"""
参数分析图表模块

生成超参数影响分析的可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata

from .plot_config import (
    PlotConfig,
    apply_axis_style,
    get_color_palette,
    create_figure_with_subplots
)


class ParameterPlotter:
    """参数图表绘制器"""
    
    def __init__(self, style: str = 'academic'):
        self.style = style
        self.colors = get_color_palette('academic')
        
    def plot_single_parameter_effect(self,
                                   param_values: np.ndarray,
                                   metric_values: np.ndarray,
                                   param_name: str,
                                   metric_name: str,
                                   figsize: Optional[Tuple] = None,
                                   show_trend: bool = True,
                                   confidence_interval: bool = True) -> plt.Figure:
        """绘制单参数影响图
        
        Args:
            param_values: 参数值数组
            metric_values: 指标值数组
            param_name: 参数名称
            metric_name: 指标名称
            figsize: 图形大小
            show_trend: 是否显示趋势线
            confidence_interval: 是否显示置信区间
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('single', aspect_ratio=0.8)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 散点图
        ax.scatter(param_values, metric_values, alpha=0.6, s=50, 
                  color=self.colors[0], edgecolor='black', linewidth=0.5)
        
        # 趋势线
        if show_trend:
            # 多项式拟合
            z = np.polyfit(param_values, metric_values, 3)
            p = np.poly1d(z)
            x_smooth = np.linspace(param_values.min(), param_values.max(), 100)
            y_smooth = p(x_smooth)
            
            ax.plot(x_smooth, y_smooth, color='red', linewidth=2, 
                   label='Trend (cubic)')
            
            # 置信区间
            if confidence_interval:
                # 简单的置信带（基于残差标准差）
                residuals = metric_values - p(param_values)
                std_residuals = np.std(residuals)
                
                ax.fill_between(x_smooth,
                              y_smooth - 1.96 * std_residuals,
                              y_smooth + 1.96 * std_residuals,
                              alpha=0.2, color='red', label='95% CI')
                              
        # 标记最优点
        if metric_name.lower() in ['mae', 'rmse', 'mse']:
            best_idx = np.argmin(metric_values)
        else:
            best_idx = np.argmax(metric_values)
            
        ax.plot(param_values[best_idx], metric_values[best_idx],
               'o', color='green', markersize=12, markeredgecolor='black',
               markeredgewidth=2, label='Best')
               
        # 添加文本标注
        ax.annotate(f'Best: {param_values[best_idx]:.3f}',
                   xy=(param_values[best_idx], metric_values[best_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, ha='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
                   
        apply_axis_style(ax,
                        xlabel=param_name,
                        ylabel=metric_name,
                        title=f'Effect of {param_name} on {metric_name}',
                        legend=show_trend)
        
        plt.tight_layout()
        return fig
        
    def plot_parameter_sensitivity(self,
                                 sensitivities: Dict[str, float],
                                 figsize: Optional[Tuple] = None,
                                 top_k: Optional[int] = None) -> plt.Figure:
        """绘制参数敏感性分析
        
        Args:
            sensitivities: 参数名到敏感性值的字典
            figsize: 图形大小
            top_k: 只显示前k个最敏感的参数
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.6)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 排序参数
        sorted_params = sorted(sensitivities.items(), 
                             key=lambda x: abs(x[1]), reverse=True)
        
        if top_k:
            sorted_params = sorted_params[:top_k]
            
        params = [p[0] for p in sorted_params]
        values = [p[1] for p in sorted_params]
        
        # 归一化敏感性值
        max_sensitivity = max(abs(v) for v in values)
        normalized_values = [v / max_sensitivity * 100 for v in values]
        
        # 创建条形图
        y_pos = np.arange(len(params))
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = ax.barh(y_pos, normalized_values, color=colors, alpha=0.7,
                       edgecolor='black', linewidth=0.5)
        
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, normalized_values)):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                   f'{val:.1f}%', va='center', fontsize=9)
                   
        # 设置标签
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        
        # 添加垂直参考线
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        
        apply_axis_style(ax,
                        xlabel='Relative Sensitivity (%)',
                        title='Parameter Sensitivity Analysis',
                        legend=False)
        
        plt.tight_layout()
        return fig
        
    def plot_parameter_heatmap(self,
                             param1_values: np.ndarray,
                             param2_values: np.ndarray,
                             metric_values: np.ndarray,
                             param1_name: str,
                             param2_name: str,
                             metric_name: str,
                             figsize: Optional[Tuple] = None,
                             interpolation: str = 'nearest',
                             show_optimal: bool = True) -> plt.Figure:
        """绘制双参数热力图
        
        Args:
            param1_values: 第一个参数的值
            param2_values: 第二个参数的值
            metric_values: 对应的指标值
            param1_name: 第一个参数名
            param2_name: 第二个参数名
            metric_name: 指标名
            figsize: 图形大小
            interpolation: 插值方法
            show_optimal: 是否标记最优点
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 创建网格
        param1_unique = np.unique(param1_values)
        param2_unique = np.unique(param2_values)
        
        # 如果数据不是规则网格，进行插值
        if len(param1_unique) * len(param2_unique) != len(metric_values):
            # 插值到规则网格
            grid_x, grid_y = np.meshgrid(param1_unique, param2_unique)
            grid_z = griddata((param1_values, param2_values), metric_values,
                            (grid_x, grid_y), method='cubic')
        else:
            # 重塑为矩阵
            grid_z = metric_values.reshape(len(param2_unique), len(param1_unique))
            
        # 确定颜色映射
        if metric_name.lower() in ['mae', 'rmse', 'mse']:
            cmap = 'RdYlGn_r'  # 误差指标：红色表示高（差）
        else:
            cmap = 'RdYlGn'    # 其他指标：绿色表示高（好）
            
        # 绘制热力图
        im = ax.imshow(grid_z, cmap=cmap, aspect='auto',
                      interpolation=interpolation,
                      extent=[param1_unique.min(), param1_unique.max(),
                             param2_unique.min(), param2_unique.max()],
                      origin='lower')
                      
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric_name)
        
        # 添加等高线
        contours = ax.contour(param1_unique, param2_unique, grid_z,
                            colors='black', alpha=0.4, linewidths=0.5)
        ax.clabel(contours, inline=True, fontsize=8)
        
        # 标记最优点
        if show_optimal:
            if metric_name.lower() in ['mae', 'rmse', 'mse']:
                best_idx = np.argmin(metric_values)
            else:
                best_idx = np.argmax(metric_values)
                
            ax.plot(param1_values[best_idx], param2_values[best_idx],
                   'w*', markersize=15, markeredgecolor='black',
                   markeredgewidth=2)
                   
            # 添加文本标注
            ax.annotate(f'Optimal\n{metric_values[best_idx]:.3f}',
                       xy=(param1_values[best_idx], param2_values[best_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=9, ha='left',
                       bbox=dict(boxstyle='round,pad=0.3', 
                               facecolor='white', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
                       
        apply_axis_style(ax,
                        xlabel=param1_name,
                        ylabel=param2_name,
                        title=f'{metric_name} vs {param1_name} and {param2_name}',
                        grid=False,
                        legend=False)
        
        plt.tight_layout()
        return fig
        
    def plot_parallel_coordinates(self,
                                data: pd.DataFrame,
                                param_columns: List[str],
                                metric_column: str,
                                figsize: Optional[Tuple] = None,
                                color_scale: bool = True,
                                highlight_best: int = 5) -> plt.Figure:
        """绘制平行坐标图
        
        Args:
            data: 包含参数和指标的DataFrame
            param_columns: 参数列名列表
            metric_column: 指标列名
            figsize: 图形大小
            color_scale: 是否根据指标值着色
            highlight_best: 高亮前几个最佳配置
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.6)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 归一化数据
        normalized_data = data.copy()
        for col in param_columns:
            min_val = data[col].min()
            max_val = data[col].max()
            if max_val > min_val:
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val)
            else:
                normalized_data[col] = 0.5
                
        # 设置x轴位置
        x = np.arange(len(param_columns))
        
        # 获取颜色映射
        if color_scale:
            if metric_column.lower() in ['mae', 'rmse', 'mse']:
                # 误差指标：反向映射
                norm = plt.Normalize(vmin=data[metric_column].max(), 
                                   vmax=data[metric_column].min())
            else:
                norm = plt.Normalize(vmin=data[metric_column].min(), 
                                   vmax=data[metric_column].max())
            cmap = plt.cm.RdYlGn
            
        # 绘制所有线条（透明度较低）
        for idx, row in normalized_data.iterrows():
            values = [row[col] for col in param_columns]
            
            if color_scale:
                color = cmap(norm(data.loc[idx, metric_column]))
                ax.plot(x, values, color=color, alpha=0.3, linewidth=1)
            else:
                ax.plot(x, values, color='gray', alpha=0.3, linewidth=1)
                
        # 高亮最佳配置
        if highlight_best > 0:
            if metric_column.lower() in ['mae', 'rmse', 'mse']:
                best_indices = data.nsmallest(highlight_best, metric_column).index
            else:
                best_indices = data.nlargest(highlight_best, metric_column).index
                
            for i, idx in enumerate(best_indices):
                values = [normalized_data.loc[idx, col] for col in param_columns]
                ax.plot(x, values, color=self.colors[i], alpha=0.8, 
                       linewidth=2, label=f'Top {i+1}')
                
        # 设置轴标签
        ax.set_xticks(x)
        ax.set_xticklabels(param_columns, rotation=45, ha='right')
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('Normalized Value')
        
        # 添加原始值范围标注
        for i, col in enumerate(param_columns):
            min_val = data[col].min()
            max_val = data[col].max()
            ax.text(i, -0.1, f'{min_val:.2g}', ha='center', va='top', fontsize=8)
            ax.text(i, 1.1, f'{max_val:.2g}', ha='center', va='bottom', fontsize=8)
            
        # 添加颜色条
        if color_scale:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label(metric_column)
            
        apply_axis_style(ax,
                        title='Parallel Coordinates Plot',
                        legend=(highlight_best > 0),
                        grid=True)
        
        plt.tight_layout()
        return fig
        
    def plot_3d_surface(self,
                       param1_values: np.ndarray,
                       param2_values: np.ndarray,
                       metric_values: np.ndarray,
                       param1_name: str,
                       param2_name: str,
                       metric_name: str,
                       figsize: Optional[Tuple] = None,
                       azim: int = 45,
                       elev: int = 30) -> plt.Figure:
        """绘制3D曲面图
        
        Args:
            param1_values: 第一个参数的值
            param2_values: 第二个参数的值
            metric_values: 对应的指标值
            param1_name: 第一个参数名
            param2_name: 第二个参数名
            metric_name: 指标名
            figsize: 图形大小
            azim: 方位角
            elev: 仰角
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.8)
            
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # 创建网格
        param1_unique = np.unique(param1_values)
        param2_unique = np.unique(param2_values)
        
        # 插值到规则网格
        grid_x, grid_y = np.meshgrid(param1_unique, param2_unique)
        grid_z = griddata((param1_values, param2_values), metric_values,
                         (grid_x, grid_y), method='cubic')
        
        # 确定颜色映射
        if metric_name.lower() in ['mae', 'rmse', 'mse']:
            cmap = cm.RdYlGn_r
        else:
            cmap = cm.RdYlGn
            
        # 绘制曲面
        surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap=cmap,
                              linewidth=0, antialiased=True, alpha=0.8)
                              
        # 添加等高线投影
        contours = ax.contour(grid_x, grid_y, grid_z, zdir='z',
                            offset=grid_z.min(), cmap=cmap, alpha=0.5)
                            
        # 标记最优点
        if metric_name.lower() in ['mae', 'rmse', 'mse']:
            best_idx = np.argmin(metric_values)
        else:
            best_idx = np.argmax(metric_values)
            
        ax.scatter(param1_values[best_idx], param2_values[best_idx],
                  metric_values[best_idx], color='red', s=100,
                  marker='*', edgecolor='black', linewidth=2)
                  
        # 设置标签
        ax.set_xlabel(param1_name, labelpad=10)
        ax.set_ylabel(param2_name, labelpad=10)
        ax.set_zlabel(metric_name, labelpad=10)
        ax.set_title(f'3D Surface: {metric_name}', pad=20)
        
        # 设置视角
        ax.view_init(elev=elev, azim=azim)
        
        # 添加颜色条
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        return fig
        
    def plot_interaction_effects(self,
                               data: pd.DataFrame,
                               param1: str,
                               param2: str,
                               metric: str,
                               figsize: Optional[Tuple] = None,
                               n_levels: int = 5) -> plt.Figure:
        """绘制参数交互效应图
        
        Args:
            data: 数据
            param1: 第一个参数名
            param2: 第二个参数名
            metric: 指标名
            figsize: 图形大小
            n_levels: 每个参数的水平数
        """
        if figsize is None:
            figsize = PlotConfig.get_figure_size('double', aspect_ratio=0.618)
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # 将连续参数离散化为水平
        data_copy = data.copy()
        data_copy[f'{param1}_level'] = pd.qcut(data[param1], n_levels, 
                                              labels=range(n_levels))
        data_copy[f'{param2}_level'] = pd.qcut(data[param2], n_levels, 
                                              labels=range(n_levels))
        
        # 计算每个组合的平均值
        grouped = data_copy.groupby([f'{param1}_level', f'{param2}_level'])[metric].mean()
        
        # 绘制交互图
        for i in range(n_levels):
            values = []
            x_values = []
            
            for j in range(n_levels):
                if (i, j) in grouped.index:
                    values.append(grouped[(i, j)])
                    x_values.append(j)
                    
            if values:
                param1_val = data_copy[data_copy[f'{param1}_level'] == i][param1].mean()
                ax.plot(x_values, values, 'o-', 
                       label=f'{param1}={param1_val:.2f}',
                       color=self.colors[i], linewidth=2, markersize=8)
                       
        ax.set_xticks(range(n_levels))
        ax.set_xticklabels([f'L{i+1}' for i in range(n_levels)])
        
        apply_axis_style(ax,
                        xlabel=f'{param2} Level',
                        ylabel=metric,
                        title=f'Interaction Effect: {param1} × {param2}',
                        legend=True)
        
        plt.tight_layout()
        return fig


# 便捷函数
def plot_parameter_sensitivity(sensitivities: Dict[str, float],
                             save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制参数敏感性分析"""
    plotter = ParameterPlotter()
    fig = plotter.plot_parameter_sensitivity(sensitivities)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_parameter_heatmap(param1_values: np.ndarray,
                         param2_values: np.ndarray,
                         metric_values: np.ndarray,
                         param1_name: str,
                         param2_name: str,
                         metric_name: str,
                         save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制参数热力图"""
    plotter = ParameterPlotter()
    fig = plotter.plot_parameter_heatmap(
        param1_values, param2_values, metric_values,
        param1_name, param2_name, metric_name
    )
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_parameter_parallel_coordinates(data: pd.DataFrame,
                                      param_columns: List[str],
                                      metric_column: str,
                                      save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制平行坐标图"""
    plotter = ParameterPlotter()
    fig = plotter.plot_parallel_coordinates(data, param_columns, metric_column)
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig


def plot_parameter_3d_surface(param1_values: np.ndarray,
                            param2_values: np.ndarray,
                            metric_values: np.ndarray,
                            param1_name: str,
                            param2_name: str,
                            metric_name: str,
                            save_path: Optional[str] = None) -> plt.Figure:
    """快速绘制3D曲面图"""
    plotter = ParameterPlotter()
    fig = plotter.plot_3d_surface(
        param1_values, param2_values, metric_values,
        param1_name, param2_name, metric_name
    )
    
    if save_path:
        from .plot_config import save_figure
        save_figure(fig, save_path)
        
    return fig