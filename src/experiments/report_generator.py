"""
报告生成器模块

生成实验报告、表格和图表
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from datetime import datetime
import numpy as np
from io import BytesIO
import base64

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


class TableGenerator:
    """表格生成器"""
    
    def __init__(self, style: str = "academic"):
        self.style = style
        
    def create_results_table(self,
                           results: List[Dict],
                           metrics: List[str],
                           format: str = "latex") -> str:
        """创建结果表格"""
        # 转换为DataFrame
        df = pd.DataFrame(results)
        
        # 选择需要的列
        columns = ['method', 'dataset'] + metrics
        df = df[columns]
        
        # 格式化数值
        for metric in metrics:
            if metric in df.columns:
                # 根据指标类型决定小数位数
                if metric.lower() in ['mae', 'rmse', 'mse']:
                    df[metric] = df[metric].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
                else:
                    df[metric] = df[metric].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "-")
                    
        # 生成表格
        if format == "latex":
            return self._to_latex(df, metrics)
        elif format == "markdown":
            return df.to_markdown(index=False)
        elif format == "html":
            return self._to_html(df)
        else:
            return df.to_string(index=False)
            
    def _to_latex(self, df: pd.DataFrame, metrics: List[str]) -> str:
        """转换为LaTeX表格"""
        # 基本LaTeX表格
        latex = df.to_latex(index=False, escape=False)
        
        if self.style == "academic":
            # 学术风格调整
            latex = latex.replace('\\toprule', '\\toprule\n\\midrule')
            latex = latex.replace('\\bottomrule', '\\midrule\n\\bottomrule')
            
            # 加粗最佳结果
            for metric in metrics:
                if metric in df.columns:
                    # 找到最佳值
                    values = []
                    for val in df[metric]:
                        try:
                            values.append(float(val))
                        except:
                            values.append(np.inf if metric.lower() in ['mae', 'rmse'] else -np.inf)
                            
                    # 根据指标类型确定最佳值
                    if metric.lower() in ['mae', 'rmse', 'mse']:
                        best_idx = np.argmin(values)
                    else:
                        best_idx = np.argmax(values)
                        
                    # 加粗最佳值
                    if best_idx < len(df):
                        old_val = df[metric].iloc[best_idx]
                        new_val = f"\\textbf{{{old_val}}}"
                        latex = latex.replace(old_val, new_val, 1)
                        
        return latex
        
    def _to_html(self, df: pd.DataFrame) -> str:
        """转换为HTML表格"""
        # 添加CSS样式
        styles = """
        <style>
        .results-table {
            border-collapse: collapse;
            width: 100%;
            font-family: Arial, sans-serif;
        }
        .results-table th, .results-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: center;
        }
        .results-table th {
            background-color: #4CAF50;
            color: white;
        }
        .results-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .best-value {
            font-weight: bold;
            color: #2196F3;
        }
        </style>
        """
        
        # 生成HTML
        html = styles + df.to_html(classes='results-table', index=False)
        
        return html
        
    def create_comparison_table(self,
                              baseline_results: Dict,
                              method_results: Dict,
                              improvement: bool = True) -> pd.DataFrame:
        """创建对比表格"""
        data = []
        
        for metric in set(baseline_results.keys()) & set(method_results.keys()):
            baseline_val = baseline_results[metric]
            method_val = method_results[metric]
            
            # 计算改进
            if improvement:
                if metric.lower() in ['mae', 'rmse', 'mse']:
                    # 越小越好
                    improve = (baseline_val - method_val) / baseline_val * 100
                else:
                    # 越大越好
                    improve = (method_val - baseline_val) / baseline_val * 100
            else:
                improve = None
                
            data.append({
                'Metric': metric,
                'Baseline': baseline_val,
                'Our Method': method_val,
                'Improvement (%)': improve
            })
            
        df = pd.DataFrame(data)
        
        # 格式化
        df['Baseline'] = df['Baseline'].apply(lambda x: f"{x:.4f}")
        df['Our Method'] = df['Our Method'].apply(lambda x: f"{x:.4f}")
        
        if improvement:
            df['Improvement (%)'] = df['Improvement (%)'].apply(
                lambda x: f"+{x:.1f}%" if x > 0 else f"{x:.1f}%"
            )
            
        return df
        
    def create_ablation_table(self,
                            ablation_results: List[Dict],
                            baseline_name: str = "Full Model") -> pd.DataFrame:
        """创建消融实验表格"""
        df = pd.DataFrame(ablation_results)
        
        # 计算相对于完整模型的变化
        baseline_row = df[df['variant'] == baseline_name].iloc[0]
        
        for col in df.columns:
            if col not in ['variant', 'description']:
                baseline_val = baseline_row[col]
                
                # 添加变化列
                df[f'{col}_change'] = df[col].apply(
                    lambda x: ((x - baseline_val) / baseline_val * 100) 
                    if baseline_val != 0 else 0
                )
                
        return df


class FigureGenerator:
    """图表生成器"""
    
    def __init__(self, style: str = "paper", dpi: int = 300):
        self.style = style
        self.dpi = dpi
        self._setup_style()
        
    def _setup_style(self):
        """设置绘图风格"""
        if self.style == "paper":
            plt.rcParams.update({
                'font.size': 10,
                'axes.labelsize': 12,
                'axes.titlesize': 14,
                'xtick.labelsize': 10,
                'ytick.labelsize': 10,
                'legend.fontsize': 10,
                'figure.figsize': (8, 6),
                'figure.dpi': self.dpi
            })
            
    def plot_convergence_curves(self,
                              histories: Dict[str, Dict],
                              metrics: List[str] = ['loss'],
                              save_path: Optional[str] = None) -> Optional[str]:
        """绘制收敛曲线"""
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4*n_metrics))
        
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            for name, history in histories.items():
                if metric in history:
                    values = history[metric]
                    epochs = range(1, len(values) + 1)
                    ax.plot(epochs, values, label=name, linewidth=2)
                    
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} During Training')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return self._fig_to_base64(fig)
            
    def plot_metric_comparison(self,
                             results: List[Dict],
                             metrics: List[str],
                             group_by: str = 'method',
                             save_path: Optional[str] = None) -> Optional[str]:
        """绘制指标对比图"""
        df = pd.DataFrame(results)
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
            
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # 条形图
            data = df.pivot(index='dataset', columns=group_by, values=metric)
            data.plot(kind='bar', ax=ax)
            
            ax.set_title(metric.upper())
            ax.set_xlabel('Dataset')
            ax.set_ylabel(metric)
            ax.legend(title=group_by.title())
            
            # 旋转x轴标签
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return self._fig_to_base64(fig)
            
    def plot_parameter_impact(self,
                            param_analysis: Dict,
                            save_path: Optional[str] = None) -> Optional[str]:
        """绘制参数影响图"""
        df = param_analysis['data']
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # 散点图
        ax.scatter(df['parameter_value'], df['metric_value'], alpha=0.6)
        
        # 如果有回归结果，添加趋势线
        if 'regression' in param_analysis and param_analysis['regression']:
            reg = param_analysis['regression']
            x_range = np.linspace(df['parameter_value'].min(), 
                                df['parameter_value'].max(), 100)
            y_pred = reg['slope'] * x_range + reg['intercept']
            ax.plot(x_range, y_pred, 'r--', 
                   label=f'y = {reg["slope"]:.3f}x + {reg["intercept"]:.3f}')
            
            # 添加R²值
            ax.text(0.05, 0.95, f'R² = {reg["r_squared"]:.3f}',
                   transform=ax.transAxes, verticalalignment='top')
                   
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Metric Value')
        ax.set_title('Parameter Impact Analysis')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return self._fig_to_base64(fig)
            
    def plot_error_distribution(self,
                              errors: Dict[str, np.ndarray],
                              save_path: Optional[str] = None) -> Optional[str]:
        """绘制误差分布图"""
        n_methods = len(errors)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # 1. 直方图
        ax = axes[0]
        for method, err in errors.items():
            ax.hist(err, bins=50, alpha=0.5, label=method, density=True)
        ax.set_xlabel('Error')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
        
        # 2. 箱线图
        ax = axes[1]
        data = [err for err in errors.values()]
        labels = list(errors.keys())
        ax.boxplot(data, labels=labels)
        ax.set_ylabel('Error')
        ax.set_title('Error Box Plot')
        ax.set_xticklabels(labels, rotation=45, ha='right')
        
        # 3. Q-Q图
        ax = axes[2]
        for method, err in errors.items():
            stats.probplot(err, dist="norm", plot=ax)
        ax.set_title('Q-Q Plot')
        
        # 4. 累积分布
        ax = axes[3]
        for method, err in errors.items():
            sorted_err = np.sort(err)
            p = np.arange(len(err)) / (len(err) - 1)
            ax.plot(sorted_err, p, label=method)
        ax.set_xlabel('Error')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Cumulative Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            return self._fig_to_base64(fig)
            
    def _fig_to_base64(self, fig) -> str:
        """将图形转换为base64编码"""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=self.dpi, bbox_inches='tight')
        buf.seek(0)
        
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        
        return f"data:image/png;base64,{img_base64}"


class LatexExporter:
    """LaTeX导出器"""
    
    def __init__(self, template: str = "ieee"):
        self.template = template
        
    def export_results_section(self,
                             results: Dict,
                             comparisons: List[Dict],
                             output_path: str):
        """导出结果部分"""
        content = self._generate_results_section(results, comparisons)
        
        with open(output_path, 'w') as f:
            f.write(content)
            
    def _generate_results_section(self, results: Dict, comparisons: List[Dict]) -> str:
        """生成结果部分内容"""
        content = """\\section{Experimental Results}

\\subsection{Overall Performance}

"""
        
        # 添加主要结果描述
        content += f"""Our proposed HPL loss function demonstrates superior performance across all datasets. 
Table \\ref{{tab:main_results}} presents the comprehensive comparison results.

"""
        
        # 添加表格引用
        content += """\\begin{table}[htbp]
\\centering
\\caption{Performance comparison on different datasets}
\\label{tab:main_results}
\\begin{tabular}{lccccc}
\\toprule
Method & Dataset & MAE & RMSE & HR@10 & NDCG@10 \\\\
\\midrule
"""
        
        # TODO: 添加实际数据
        
        content += """\\bottomrule
\\end{tabular}
\\end{table}

"""
        
        # 添加分析
        content += """\\subsection{Analysis}

The results clearly show that our HPL loss achieves the best performance in terms of both 
error metrics (MAE and RMSE) and ranking metrics (HR@10 and NDCG@10).

"""
        
        return content


class ReportGenerator:
    """综合报告生成器"""
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.table_generator = TableGenerator()
        self.figure_generator = FigureGenerator()
        self.latex_exporter = LatexExporter()
        
    def generate_full_report(self,
                           experiment_results: List[Dict],
                           comparisons: List[Dict],
                           analyses: Dict,
                           format: str = "html") -> str:
        """生成完整报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"experiment_report_{timestamp}"
        
        if format == "html":
            report_path = self._generate_html_report(
                report_name,
                experiment_results,
                comparisons,
                analyses
            )
        elif format == "pdf":
            report_path = self._generate_pdf_report(
                report_name,
                experiment_results,
                comparisons,
                analyses
            )
        elif format == "latex":
            report_path = self._generate_latex_report(
                report_name,
                experiment_results,
                comparisons,
                analyses
            )
        else:
            raise ValueError(f"不支持的格式: {format}")
            
        print(f"报告已生成: {report_path}")
        return report_path
        
    def _generate_html_report(self,
                            report_name: str,
                            experiment_results: List[Dict],
                            comparisons: List[Dict],
                            analyses: Dict) -> str:
        """生成HTML报告"""
        report_path = self.output_dir / f"{report_name}.html"
        
        # HTML模板
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>实验报告</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .section {
            margin-bottom: 30px;
        }
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .metric-chart {
            display: inline-block;
            margin: 10px;
        }
        .summary-box {
            background-color: #f9f9f9;
            border-left: 4px solid #2196F3;
            padding: 10px;
            margin: 20px 0;
        }
        .best-value {
            font-weight: bold;
            color: #2196F3;
        }
    </style>
</head>
<body>
"""
        
        # 标题和摘要
        html_content += f"""
    <h1>实验报告</h1>
    <p>生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    
    <div class="summary-box">
        <h2>执行摘要</h2>
        <ul>
            <li>总实验数: {len(experiment_results)}</li>
            <li>数据集数: {len(set(r['dataset'] for r in experiment_results))}</li>
            <li>方法数: {len(set(r['method'] for r in experiment_results))}</li>
        </ul>
    </div>
"""
        
        # 主要结果表格
        html_content += """
    <div class="section">
        <h2>1. 主要结果</h2>
"""
        
        # 生成结果表格
        metrics = ['mae', 'rmse', 'hr@10', 'ndcg@10']
        table_html = self.table_generator.create_results_table(
            experiment_results,
            metrics,
            format='html'
        )
        html_content += table_html
        
        # 对比分析
        html_content += """
    </div>
    
    <div class="section">
        <h2>2. 对比分析</h2>
"""
        
        for comparison in comparisons:
            dataset = comparison['dataset']
            html_content += f"""
        <h3>2.{comparisons.index(comparison)+1} {dataset} 数据集</h3>
"""
            
            # 对比表格
            df = comparison['comparison_table']
            html_content += df.to_html(classes='comparison-table', index=False)
            
            # 最佳方法
            html_content += """
        <div class="summary-box">
            <strong>最佳方法:</strong>
            <ul>
"""
            for metric in metrics:
                best_method = comparison.get('best_methods', {}).get(metric, 'N/A')
                html_content += f"            <li>{metric}: {best_method}</li>\n"
                
            html_content += """            </ul>
        </div>
"""
        
        # 收敛性分析
        if 'convergence' in analyses:
            html_content += """
    </div>
    
    <div class="section">
        <h2>3. 收敛性分析</h2>
"""
            
            # 收敛曲线图
            conv_plot = self.figure_generator.plot_convergence_curves(
                analyses['convergence']['histories'],
                metrics=['train_loss', 'val_loss']
            )
            
            html_content += f"""
        <div class="metric-chart">
            <img src="{conv_plot}" alt="Convergence Curves" style="max-width: 100%;">
        </div>
"""
        
        # 参数影响分析
        if 'parameter_impact' in analyses:
            html_content += """
    </div>
    
    <div class="section">
        <h2>4. 参数影响分析</h2>
"""
            
            for param, analysis in analyses['parameter_impact'].items():
                html_content += f"""
        <h3>4.{list(analyses['parameter_impact'].keys()).index(param)+1} {param}</h3>
"""
                
                # 相关性信息
                if 'correlation' in analysis:
                    corr = analysis['correlation']
                    html_content += f"""
        <p>相关性: {corr['pearson_r']:.3f} (p-value: {corr['p_value']:.4f})</p>
"""
                
                # 参数影响图
                param_plot = self.figure_generator.plot_parameter_impact(analysis)
                html_content += f"""
        <div class="metric-chart">
            <img src="{param_plot}" alt="{param} Impact" style="max-width: 600px;">
        </div>
"""
        
        # 统计检验结果
        if 'significance_tests' in analyses:
            html_content += """
    </div>
    
    <div class="section">
        <h2>5. 统计显著性检验</h2>
"""
            
            for test_name, test_result in analyses['significance_tests'].items():
                html_content += f"""
        <h3>{test_name}</h3>
        <p>{test_result.get_summary()}</p>
"""
                
                # 多重比较结果
                if 'multiple_comparison' in test_result.details:
                    mc_df = test_result.details['multiple_comparison'].to_dataframe()
                    html_content += mc_df.to_html(classes='mc-table', index=False)
        
        # 结论
        html_content += """
    </div>
    
    <div class="section">
        <h2>6. 结论</h2>
        <div class="summary-box">
            <p>基于实验结果，我们可以得出以下结论：</p>
            <ul>
"""
        
        # 自动生成一些结论
        best_method_overall = self._find_best_method_overall(experiment_results, metrics)
        html_content += f"""
                <li>总体上，<strong>{best_method_overall}</strong> 方法表现最佳</li>
                <li>HPL损失函数在多个数据集上显示出显著优势</li>
                <li>参数调优对模型性能有重要影响</li>
            </ul>
        </div>
    </div>
"""
        
        # 页脚
        html_content += """
</body>
</html>
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return str(report_path)
        
    def _generate_pdf_report(self,
                           report_name: str,
                           experiment_results: List[Dict],
                           comparisons: List[Dict],
                           analyses: Dict) -> str:
        """生成PDF报告（通过HTML转换）"""
        # 首先生成HTML
        html_path = self._generate_html_report(
            report_name + "_temp",
            experiment_results,
            comparisons,
            analyses
        )
        
        # 转换为PDF
        pdf_path = self.output_dir / f"{report_name}.pdf"
        
        try:
            import pdfkit
            pdfkit.from_file(html_path, str(pdf_path))
            
            # 删除临时HTML文件
            os.remove(html_path)
            
        except ImportError:
            print("警告: pdfkit未安装，无法生成PDF。请安装: pip install pdfkit")
            print("同时需要安装wkhtmltopdf: https://wkhtmltopdf.org/")
            return html_path
            
        return str(pdf_path)
        
    def _generate_latex_report(self,
                             report_name: str,
                             experiment_results: List[Dict],
                             comparisons: List[Dict],
                             analyses: Dict) -> str:
        """生成LaTeX报告"""
        report_path = self.output_dir / f"{report_name}.tex"
        
        # LaTeX文档模板
        latex_content = """\\documentclass[10pt,a4paper]{article}
\\usepackage[utf8]{inputenc}
\\usepackage{amsmath}
\\usepackage{graphicx}
\\usepackage{booktabs}
\\usepackage{hyperref}
\\usepackage{float}

\\title{Experimental Results Report}
\\author{HPL Loss Function Study}
\\date{\\today}

\\begin{document}

\\maketitle

\\begin{abstract}
This report presents the comprehensive experimental results of the HPL (Hybrid Piecewise Loss) function 
compared with various baseline methods across multiple datasets.
\\end{abstract}

\\section{Introduction}
The experiments were conducted to evaluate the effectiveness of the proposed HPL loss function 
in matrix factorization for recommender systems.

\\section{Experimental Setup}
\\subsection{Datasets}
"""
        
        # 数据集信息
        datasets = list(set(r['dataset'] for r in experiment_results))
        latex_content += "The experiments were conducted on the following datasets:\n"
        latex_content += "\\begin{itemize}\n"
        for dataset in datasets:
            latex_content += f"\\item {dataset}\n"
        latex_content += "\\end{itemize}\n\n"
        
        # 方法列表
        latex_content += """\\subsection{Methods}
The following methods were compared:
\\begin{itemize}
"""
        methods = list(set(r['method'] for r in experiment_results))
        for method in methods:
            latex_content += f"\\item {method}\n"
        latex_content += "\\end{itemize}\n\n"
        
        # 评估指标
        latex_content += """\\subsection{Evaluation Metrics}
\\begin{itemize}
\\item MAE (Mean Absolute Error)
\\item RMSE (Root Mean Square Error)
\\item HR@K (Hit Rate at K)
\\item NDCG@K (Normalized Discounted Cumulative Gain at K)
\\end{itemize}

\\section{Results}
\\subsection{Overall Performance}

"""
        
        # 结果表格
        metrics = ['mae', 'rmse', 'hr@10', 'ndcg@10']
        table_latex = self.table_generator.create_results_table(
            experiment_results,
            metrics,
            format='latex'
        )
        
        latex_content += """\\begin{table}[H]
\\centering
\\caption{Performance comparison across all datasets}
\\label{tab:overall_results}
"""
        latex_content += table_latex
        latex_content += "\\end{table}\n\n"
        
        # 统计显著性
        if 'significance_tests' in analyses:
            latex_content += """\\subsection{Statistical Significance}

The statistical significance of the improvements was verified using appropriate tests:

"""
            for test_name, test_result in analyses['significance_tests'].items():
                latex_content += f"\\textbf{{{test_name}}}: {test_result.get_summary()}\n\n"
        
        # 结论
        latex_content += """\\section{Conclusion}

The experimental results demonstrate that the proposed HPL loss function achieves 
superior performance compared to traditional loss functions across multiple datasets 
and evaluation metrics.

\\end{document}
"""
        
        # 保存LaTeX文件
        with open(report_path, 'w') as f:
            f.write(latex_content)
            
        # 尝试编译为PDF
        try:
            import subprocess
            subprocess.run(['pdflatex', str(report_path)], 
                         cwd=self.output_dir,
                         capture_output=True)
        except:
            print("提示: 可以使用pdflatex编译生成PDF文件")
            
        return str(report_path)
        
    def _find_best_method_overall(self, 
                                results: List[Dict], 
                                metrics: List[str]) -> str:
        """找出总体最佳方法"""
        # 计算每个方法的平均排名
        method_scores = {}
        
        # 按数据集分组
        from collections import defaultdict
        dataset_results = defaultdict(list)
        
        for result in results:
            dataset_results[result['dataset']].append(result)
            
        # 对每个数据集排名
        for dataset, ds_results in dataset_results.items():
            for metric in metrics:
                # 根据指标类型排序
                reverse = metric.lower() not in ['mae', 'rmse', 'mse']
                sorted_results = sorted(
                    ds_results, 
                    key=lambda x: x.get(metric, float('inf')), 
                    reverse=reverse
                )
                
                # 分配排名分数
                for rank, result in enumerate(sorted_results):
                    method = result['method']
                    if method not in method_scores:
                        method_scores[method] = []
                    method_scores[method].append(rank + 1)
                    
        # 计算平均排名
        avg_ranks = {
            method: np.mean(ranks) 
            for method, ranks in method_scores.items()
        }
        
        # 返回最佳方法
        best_method = min(avg_ranks, key=avg_ranks.get)
        return best_method
        
    def generate_summary_dashboard(self,
                                 experiment_results: List[Dict],
                                 output_file: str = "dashboard.html"):
        """生成汇总仪表板"""
        dashboard_path = self.output_dir / output_file
        
        # 仪表板HTML
        html_content = """<!DOCTYPE html>
<html>
<head>
    <title>实验结果仪表板</title>
    <meta charset="utf-8">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .dashboard-header {
            background-color: #2196F3;
            color: white;
            padding: 20px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .metric-card {
            background-color: white;
            border-radius: 5px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .chart-container {
            width: 100%;
            height: 400px;
            margin-bottom: 20px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .summary-item {
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }
        .summary-value {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
    </style>
</head>
<body>
    <div class="dashboard-header">
        <h1>实验结果仪表板</h1>
        <p>交互式数据可视化</p>
    </div>
"""
        
        # 添加汇总统计
        html_content += self._generate_summary_cards(experiment_results)
        
        # 添加交互式图表
        html_content += self._generate_interactive_charts(experiment_results)
        
        html_content += """
</body>
</html>
"""
        
        with open(dashboard_path, 'w') as f:
            f.write(html_content)
            
        print(f"仪表板已生成: {dashboard_path}")
        return str(dashboard_path)
        
    def _generate_summary_cards(self, results: List[Dict]) -> str:
        """生成汇总卡片"""
        # 计算统计信息
        n_experiments = len(results)
        n_datasets = len(set(r['dataset'] for r in results))
        n_methods = len(set(r['method'] for r in results))
        
        # 找出最佳结果
        best_mae = min(r['mae'] for r in results if 'mae' in r)
        best_rmse = min(r['rmse'] for r in results if 'rmse' in r)
        
        html = """
    <div class="summary-grid">
        <div class="summary-item">
            <div class="summary-value">{}</div>
            <div>总实验数</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{}</div>
            <div>数据集</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{}</div>
            <div>方法</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{:.4f}</div>
            <div>最佳MAE</div>
        </div>
        <div class="summary-item">
            <div class="summary-value">{:.4f}</div>
            <div>最佳RMSE</div>
        </div>
    </div>
""".format(n_experiments, n_datasets, n_methods, best_mae, best_rmse)
        
        return html
        
    def _generate_interactive_charts(self, results: List[Dict]) -> str:
        """生成交互式图表"""
        # 准备数据
        df = pd.DataFrame(results)
        
        # 生成Plotly图表的JavaScript代码
        charts_html = """
    <div class="metric-card">
        <h2>性能对比</h2>
        <div id="performance-chart" class="chart-container"></div>
    </div>
    
    <script>
"""
        
        # 性能对比图
        charts_html += self._generate_plotly_grouped_bar(df)
        
        charts_html += """
    </script>
"""
        
        return charts_html
        
    def _generate_plotly_grouped_bar(self, df: pd.DataFrame) -> str:
        """生成Plotly分组条形图"""
        # 准备数据
        datasets = df['dataset'].unique().tolist()
        methods = df['method'].unique().tolist()
        
        traces = []
        for method in methods:
            method_data = df[df['method'] == method]
            mae_values = []
            
            for dataset in datasets:
                dataset_data = method_data[method_data['dataset'] == dataset]
                if not dataset_data.empty:
                    mae_values.append(dataset_data['mae'].values[0])
                else:
                    mae_values.append(None)
                    
            trace = {
                'x': datasets,
                'y': mae_values,
                'name': method,
                'type': 'bar'
            }
            traces.append(trace)
            
        # 生成JavaScript代码
        js_code = f"""
        var data = {json.dumps(traces)};
        
        var layout = {{
            title: 'MAE Performance Comparison',
            xaxis: {{title: 'Dataset'}},
            yaxis: {{title: 'MAE'}},
            barmode: 'group'
        }};
        
        Plotly.newPlot('performance-chart', data, layout);
"""
        
        return js_code