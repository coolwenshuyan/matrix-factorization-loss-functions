"""
表格格式化模块

生成论文级别的表格
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path


class TableFormatter:
    """表格格式化器"""
    
    def __init__(self, style: str = 'academic'):
        self.style = style
        self.decimal_places = {
            'mae': 4,
            'rmse': 4,
            'mse': 4,
            'hr': 3,
            'ndcg': 3,
            'map': 3,
            'auc': 3,
            'default': 3
        }
        
    def format_performance_table(self,
                               data: pd.DataFrame,
                               metrics: List[str],
                               methods_order: Optional[List[str]] = None,
                               datasets_order: Optional[List[str]] = None,
                               highlight_best: bool = True,
                               show_std: bool = False,
                               format_type: str = 'latex') -> str:
        """格式化性能表格
        
        Args:
            data: 性能数据DataFrame
            metrics: 指标列表
            methods_order: 方法排序
            datasets_order: 数据集排序
            highlight_best: 是否高亮最佳值
            show_std: 是否显示标准差
            format_type: 输出格式 ('latex', 'markdown', 'html')
        """
        # 复制数据避免修改原始数据
        df = data.copy()
        
        # 排序
        if methods_order:
            df['method'] = pd.Categorical(df['method'], 
                                        categories=methods_order, 
                                        ordered=True)
        if datasets_order:
            df['dataset'] = pd.Categorical(df['dataset'], 
                                         categories=datasets_order, 
                                         ordered=True)
            
        # 数据透视
        formatted_dfs = []
        
        for metric in metrics:
            if show_std and f'{metric}_std' in df.columns:
                # 格式化为 mean±std
                pivot_mean = df.pivot(index='method', 
                                    columns='dataset', 
                                    values=metric)
                pivot_std = df.pivot(index='method', 
                                   columns='dataset', 
                                   values=f'{metric}_std')
                
                # 合并均值和标准差
                decimal = self.decimal_places.get(metric.lower(), 
                                                self.decimal_places['default'])
                
                pivot_formatted = pivot_mean.copy()
                for col in pivot_mean.columns:
                    for row in pivot_mean.index:
                        mean_val = pivot_mean.loc[row, col]
                        std_val = pivot_std.loc[row, col]
                        
                        if pd.notna(mean_val):
                            if pd.notna(std_val):
                                pivot_formatted.loc[row, col] = \
                                    f"{mean_val:.{decimal}f}±{std_val:.{decimal}f}"
                            else:
                                pivot_formatted.loc[row, col] = \
                                    f"{mean_val:.{decimal}f}"
                                    
            else:
                # 只有均值
                pivot_formatted = df.pivot(index='method', 
                                         columns='dataset', 
                                         values=metric)
                                         
                # 格式化数值
                decimal = self.decimal_places.get(metric.lower(), 
                                                self.decimal_places['default'])
                pivot_formatted = pivot_formatted.applymap(
                    lambda x: f"{x:.{decimal}f}" if pd.notna(x) else "-"
                )
                
            # 高亮最佳值
            if highlight_best:
                pivot_formatted = self._highlight_best_values(
                    pivot_formatted, metric, format_type
                )
                
            # 添加指标名作为多级索引
            pivot_formatted = pd.concat({metric.upper(): pivot_formatted}, 
                                      axis=0)
            formatted_dfs.append(pivot_formatted)
            
        # 合并所有指标
        result_df = pd.concat(formatted_dfs)
        
        # 转换为指定格式
        if format_type == 'latex':
            return self._to_latex(result_df)
        elif format_type == 'markdown':
            return result_df.to_markdown()
        elif format_type == 'html':
            return self._to_html(result_df)
        else:
            return str(result_df)
            
    def format_comparison_table(self,
                              baseline_results: Dict[str, float],
                              method_results: Dict[str, float],
                              method_name: str = "Our Method",
                              format_type: str = 'latex') -> str:
        """格式化对比表格
        
        Args:
            baseline_results: 基线结果
            method_results: 方法结果
            method_name: 方法名称
            format_type: 输出格式
        """
        rows = []
        
        for metric in sorted(set(baseline_results.keys()) & set(method_results.keys())):
            baseline_val = baseline_results[metric]
            method_val = method_results[metric]
            
            # 计算改进
            if metric.lower() in ['mae', 'rmse', 'mse']:
                # 误差指标：减少百分比
                improvement = (baseline_val - method_val) / baseline_val * 100
                better = method_val < baseline_val
            else:
                # 其他指标：增加百分比
                improvement = (method_val - baseline_val) / baseline_val * 100
                better = method_val > baseline_val
                
            # 格式化数值
            decimal = self.decimal_places.get(metric.lower(), 
                                            self.decimal_places['default'])
            
            row = {
                'Metric': metric.upper(),
                'Baseline': f"{baseline_val:.{decimal}f}",
                method_name: f"{method_val:.{decimal}f}",
                'Improvement': f"{improvement:+.1f}%"
            }
            
            # 高亮更好的值
            if format_type == 'latex' and better:
                row[method_name] = f"\\textbf{{{row[method_name]}}}"
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # 转换格式
        if format_type == 'latex':
            return self._to_latex(df, index=False)
        elif format_type == 'markdown':
            return df.to_markdown(index=False)
        elif format_type == 'html':
            return self._to_html(df)
        else:
            return str(df)
            
    def format_ablation_table(self,
                            ablation_results: List[Dict],
                            baseline_name: str = "Full Model",
                            format_type: str = 'latex') -> str:
        """格式化消融实验表格
        
        Args:
            ablation_results: 消融实验结果
            baseline_name: 基线模型名称
            format_type: 输出格式
        """
        df = pd.DataFrame(ablation_results)
        
        # 找到基线行
        baseline_row = df[df['variant'] == baseline_name].iloc[0]
        
        # 计算相对变化
        metrics = [col for col in df.columns 
                  if col not in ['variant', 'description']]
        
        for metric in metrics:
            baseline_val = baseline_row[metric]
            
            # 计算变化百分比
            changes = []
            for _, row in df.iterrows():
                if row['variant'] == baseline_name:
                    changes.append("-")
                else:
                    val = row[metric]
                    if metric.lower() in ['mae', 'rmse', 'mse']:
                        # 误差指标
                        change = (val - baseline_val) / baseline_val * 100
                    else:
                        # 其他指标
                        change = (baseline_val - val) / baseline_val * 100
                        
                    changes.append(f"{change:+.1f}%")
                    
            df[f'{metric}_change'] = changes
            
            # 格式化原始值
            decimal = self.decimal_places.get(metric.lower(), 
                                            self.decimal_places['default'])
            df[metric] = df[metric].apply(lambda x: f"{x:.{decimal}f}")
            
        # 重新排列列
        cols = ['variant', 'description']
        for metric in metrics:
            cols.extend([metric, f'{metric}_change'])
        df = df[cols]
        
        # 转换格式
        if format_type == 'latex':
            return self._to_latex(df, index=False)
        elif format_type == 'markdown':
            return df.to_markdown(index=False)
        elif format_type == 'html':
            return self._to_html(df)
        else:
            return str(df)
            
    def format_significance_table(self,
                                test_results: Dict[Tuple[str, str], Dict],
                                format_type: str = 'latex') -> str:
        """格式化统计显著性表格
        
        Args:
            test_results: 检验结果字典
            format_type: 输出格式
        """
        rows = []
        
        for (method1, method2), result in test_results.items():
            row = {
                'Method 1': method1,
                'Method 2': method2,
                'Test': result.get('test_name', 'Unknown'),
                'Statistic': f"{result.get('statistic', 0):.3f}",
                'p-value': f"{result.get('p_value', 1):.4f}",
                'Significant': '✓' if result.get('significant', False) else '✗'
            }
            
            # 添加效应量
            if 'effect_size' in result:
                row['Effect Size'] = f"{result['effect_size']:.3f}"
                
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # 根据p值排序
        df['p_sort'] = df['p-value'].apply(lambda x: float(x))
        df = df.sort_values('p_sort').drop('p_sort', axis=1)
        
        # 转换格式
        if format_type == 'latex':
            # LaTeX特殊处理
            df['Significant'] = df['Significant'].replace({
                '✓': '$\\checkmark$',
                '✗': '$\\times$'
            })
            return self._to_latex(df, index=False)
        elif format_type == 'markdown':
            return df.to_markdown(index=False)
        elif format_type == 'html':
            return self._to_html(df)
        else:
            return str(df)
            
    def _highlight_best_values(self, 
                             df: pd.DataFrame, 
                             metric: str,
                             format_type: str) -> pd.DataFrame:
        """高亮最佳值"""
        # 确定是否越小越好
        minimize = metric.lower() in ['mae', 'rmse', 'mse']
        
        # 对每列找最佳值
        for col in df.columns:
            # 转换回数值进行比较
            numeric_vals = []
            for val in df[col]:
                if isinstance(val, str) and '±' in val:
                    # 提取均值部分
                    numeric_val = float(val.split('±')[0])
                elif isinstance(val, str) and val != '-':
                    try:
                        numeric_val = float(val)
                    except:
                        numeric_val = np.inf if minimize else -np.inf
                else:
                    numeric_val = np.inf if minimize else -np.inf
                    
                numeric_vals.append(numeric_val)
                
            # 找到最佳值的索引
            if minimize:
                best_idx = np.argmin(numeric_vals)
            else:
                best_idx = np.argmax(numeric_vals)
                
            # 高亮
            if format_type == 'latex':
                old_val = df.iloc[best_idx, df.columns.get_loc(col)]
                if old_val != '-':
                    df.iloc[best_idx, df.columns.get_loc(col)] = \
                        f"\\textbf{{{old_val}}}"
            elif format_type == 'html':
                old_val = df.iloc[best_idx, df.columns.get_loc(col)]
                if old_val != '-':
                    df.iloc[best_idx, df.columns.get_loc(col)] = \
                        f"<b>{old_val}</b>"
                        
        return df
        
    def _to_latex(self, df: pd.DataFrame, index: bool = True) -> str:
        """转换为LaTeX格式"""
        latex_str = df.to_latex(index=index, escape=False)
        
        # 学术风格调整
        if self.style == 'academic':
            # 使用booktabs
            latex_str = latex_str.replace('\\toprule', '\\toprule\n\\midrule')
            latex_str = latex_str.replace('\\bottomrule', '\\midrule\n\\bottomrule')
            
            # 添加包引用注释
            header = "% Requires \\usepackage{booktabs}\n"
            latex_str = header + latex_str
            
        return latex_str
        
    def _to_html(self, df: pd.DataFrame) -> str:
        """转换为HTML格式"""
        html_str = df.to_html(index=True, escape=False, 
                            classes='academic-table')
        
        # 添加CSS样式
        style = """
<style>
.academic-table {
    border-collapse: collapse;
    font-family: 'Times New Roman', serif;
    margin: 20px auto;
}
.academic-table th, .academic-table td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: center;
}
.academic-table th {
    background-color: #f8f9fa;
    font-weight: bold;
}
.academic-table tr:nth-child(even) {
    background-color: #f8f9fa;
}
.academic-table b {
    color: #0066cc;
}
</style>
"""
        
        return style + html_str
        
    def save_table(self, 
                  table_str: str,
                  filename: str,
                  format_type: str = 'latex'):
        """保存表格到文件
        
        Args:
            table_str: 表格字符串
            filename: 文件名
            format_type: 格式类型
        """
        # 确保目录存在
        filepath = Path(filename)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # 添加适当的扩展名
        if not filepath.suffix:
            if format_type == 'latex':
                filepath = filepath.with_suffix('.tex')
            elif format_type == 'html':
                filepath = filepath.with_suffix('.html')
            elif format_type == 'markdown':
                filepath = filepath.with_suffix('.md')
            else:
                filepath = filepath.with_suffix('.txt')
                
        # 保存文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_str)
            
        print(f"表格已保存: {filepath}")


# 便捷函数
def create_performance_table(data: pd.DataFrame,
                           metrics: List[str],
                           save_path: Optional[str] = None,
                           **kwargs) -> str:
    """快速创建性能表格"""
    formatter = TableFormatter()
    table = formatter.format_performance_table(data, metrics, **kwargs)
    
    if save_path:
        formatter.save_table(table, save_path, kwargs.get('format_type', 'latex'))
        
    return table


def create_comparison_table(baseline_results: Dict[str, float],
                          method_results: Dict[str, float],
                          save_path: Optional[str] = None,
                          **kwargs) -> str:
    """快速创建对比表格"""
    formatter = TableFormatter()
    table = formatter.format_comparison_table(
        baseline_results, method_results, **kwargs
    )
    
    if save_path:
        formatter.save_table(table, save_path, kwargs.get('format_type', 'latex'))
        
    return table


def create_ablation_table(ablation_results: List[Dict],
                        save_path: Optional[str] = None,
                        **kwargs) -> str:
    """快速创建消融实验表格"""
    formatter = TableFormatter()
    table = formatter.format_ablation_table(ablation_results, **kwargs)
    
    if save_path:
        formatter.save_table(table, save_path, kwargs.get('format_type', 'latex'))
        
    return table


def create_significance_table(test_results: Dict[Tuple[str, str], Dict],
                            save_path: Optional[str] = None,
                            **kwargs) -> str:
    """快速创建显著性检验表格"""
    formatter = TableFormatter()
    table = formatter.format_significance_table(test_results, **kwargs)
    
    if save_path:
        formatter.save_table(table, save_path, kwargs.get('format_type', 'latex'))
        
    return table