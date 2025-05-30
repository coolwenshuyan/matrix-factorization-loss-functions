# src/evaluation/utils.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


def create_evaluation_report(results: Dict[str, Any],
                           model_name: str,
                           save_path: Optional[str] = None) -> str:
    """
    创建评估报告
    
    Args:
        results: 评估结果
        model_name: 模型名称
        save_path: 保存路径
        
    Returns:
        报告文本
    """
    report = []
    report.append("=" * 80)
    report.append(f"评估报告 - {model_name}")
    report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")
    
    # 预测准确性指标
    report.append("预测准确性指标:")
    report.append("-" * 40)
    accuracy_metrics = ['mae', 'rmse', 'mse', 'r2']
    for metric in accuracy_metrics:
        if metric in results:
            report.append(f"  {metric.upper()}: {results[metric]:.4f}")
    report.append("")
    
    # 排序质量指标
    report.append("排序质量指标:")
    report.append("-" * 40)
    
    # 按K值分组
    k_metrics = {}
    for metric_name, value in results.items():
        if '@' in metric_name:
            base_name, k = metric_name.split('@')
            k = int(k)
            if k not in k_metrics:
                k_metrics[k] = {}
            k_metrics[k][base_name] = value
    
    for k in sorted(k_metrics.keys()):
        report.append(f"\n  K = {k}:")
        for metric, value in k_metrics[k].items():
            report.append(f"    {metric}: {value:.4f}")
    
    # 其他指标
    other_metrics = {k: v for k, v in results.items() 
                    if k not in accuracy_metrics and '@' not in k}
    if other_metrics:
        report.append("\n其他指标:")
        report.append("-" * 40)
        for metric, value in other_metrics.items():
            report.append(f"  {metric}: {value:.4f}")
    
    report.append("")
    report.append("=" * 80)
    
    report_text = "\n".join(report)
    
    # 保存报告
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"评估报告已保存至: {save_path}")
    
    return report_text


def plot_metrics_comparison(results_dict: Dict[str, Dict[str, float]],
                          metrics: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          figsize: tuple = (12, 8)):
    """
    绘制多个模型的指标对比图
    
    Args:
        results_dict: 结果字典 {model_name: {metric: value}}
        metrics: 要绘制的指标列表
        save_path: 保存路径
        figsize: 图表大小
    """
    # 准备数据
    if metrics is None:
        # 获取所有共同的指标
        all_metrics = set()
        for results in results_dict.values():
            all_metrics.update(results.keys())
        metrics = sorted(list(all_metrics))
    
    # 创建数据框
    data = []
    for model_name, results in results_dict.items():
        for metric in metrics:
            if metric in results:
                data.append({
                    'Model': model_name,
                    'Metric': metric,
                    'Value': results[metric]
                })
    
    df = pd.DataFrame(data)
    
    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # 1. 条形图比较
    ax1 = axes[0]
    pivot_df = df.pivot(index='Metric', columns='Model', values='Value')
    pivot_df.plot(kind='bar', ax=ax1)
    ax1.set_title('模型性能对比 - 条形图')
    ax1.set_xlabel('评估指标')
    ax1.set_ylabel('指标值')
    ax1.legend(title='模型')
    ax1.grid(True, alpha=0.3)
    
    # 旋转x轴标签
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
    
    # 2. 雷达图（如果模型数量合适）
    if len(results_dict) <= 5 and len(metrics) >= 3:
        ax2 = plt.subplot(2, 1, 2, projection='polar')
        
        # 选择部分指标用于雷达图
        radar_metrics = metrics[:8]  # 最多8个指标
        angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False)
        angles = np.concatenate([angles, [angles[0]]])
        
        for model_name, results in results_dict.items():
            values = []
            for metric in radar_metrics:
                if metric in results:
                    # 归一化到0-1范围
                    value = results[metric]
                    all_values = [r.get(metric, 0) for r in results_dict.values()]
                    if max(all_values) > 0:
                        value = value / max(all_values)
                    values.append(value)
                else:
                    values.append(0)
            
            values = values + [values[0]]
            ax2.plot(angles, values, 'o-', linewidth=2, label=model_name)
            ax2.fill(angles, values, alpha=0.15)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(radar_metrics)
        ax2.set_ylim(0, 1)
        ax2.set_title('模型性能对比 - 雷达图')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    else:
        # 如果不适合雷达图，使用热力图
        ax2 = axes[1]
        sns.heatmap(pivot_df.T, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax2)
        ax2.set_title('模型性能对比 - 热力图')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"对比图已保存至: {save_path}")
    
    plt.show()


def save_evaluation_results(results: Dict[str, Any],
                          save_path: str,
                          metadata: Optional[Dict[str, Any]] = None):
    """
    保存评估结果
    
    Args:
        results: 评估结果
        save_path: 保存路径
        metadata: 元数据
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备保存的数据
    save_data = {
        'results': results,
        'timestamp': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # 保存为JSON
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # 同时保存为CSV（如果可能）
    csv_path = save_path.with_suffix('.csv')
    try:
        df = pd.DataFrame([results])
        df.to_csv(csv_path, index=False)
    except:
        pass
    
    logger.info(f"评估结果已保存至: {save_path}")


def load_evaluation_results(load_path: str) -> Dict[str, Any]:
    """
    加载评估结果
    
    Args:
        load_path: 加载路径
        
    Returns:
        评估结果
    """
    load_path = Path(load_path)
    
    if not load_path.exists():
        raise FileNotFoundError(f"文件不存在: {load_path}")
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    return data


def compare_evaluation_results(results_paths: List[str],
                             output_path: Optional[str] = None) -> pd.DataFrame:
    """
    比较多个评估结果
    
    Args:
        results_paths: 结果文件路径列表
        output_path: 输出路径
        
    Returns:
        比较结果DataFrame
    """
    all_results = []
    
    for path in results_paths:
        try:
            data = load_evaluation_results(path)
            results = data['results']
            
            # 添加文件名作为标识
            results['_source'] = Path(path).stem
            all_results.append(results)
        except Exception as e:
            logger.error(f"加载 {path} 失败: {str(e)}")
    
    # 创建比较DataFrame
    df = pd.DataFrame(all_results)
    
    # 重新排列列
    if '_source' in df.columns:
        cols = ['_source'] + [col for col in df.columns if col != '_source']
        df = df[cols]
    
    # 保存比较结果
    if output_path:
        df.to_csv(output_path, index=False)
        logger.info(f"比较结果已保存至: {output_path}")
    
    return df


def create_latex_table(results_dict: Dict[str, Dict[str, float]],
                      metrics: List[str],
                      caption: str = "模型性能对比",
                      label: str = "tab:comparison") -> str:
    """
    创建LaTeX表格
    
    Args:
        results_dict: 结果字典
        metrics: 指标列表
        caption: 表格标题
        label: 表格标签
        
    Returns:
        LaTeX表格代码
    """
    # 开始表格
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    
    # 创建表格头
    num_models = len(results_dict)
    latex.append("\\begin{tabular}{l" + "c" * num_models + "}")
    latex.append("\\toprule")
    
    # 模型名称行
    model_names = list(results_dict.keys())
    header = "Metric & " + " & ".join(model_names) + " \\\\"
    latex.append(header)
    latex.append("\\midrule")
    
    # 数据行
    for metric in metrics:
        row = [metric]
        best_value = None
        best_idx = -1
        
        # 找出最佳值
        values = []
        for i, model in enumerate(model_names):
            if metric in results_dict[model]:
                value = results_dict[model][metric]
                values.append(value)
                
                # 确定是否是最佳值（假设除了MAE、MSE、RMSE外，其他指标越大越好）
                if metric.lower() in ['mae', 'mse', 'rmse']:
                    if best_value is None or value < best_value:
                        best_value = value
                        best_idx = i
                else:
                    if best_value is None or value > best_value:
                        best_value = value
                        best_idx = i
            else:
                values.append(None)
        
        # 格式化值
        for i, value in enumerate(values):
            if value is not None:
                formatted = f"{value:.4f}"
            if i == best_idx:
                formatted = f"\\textbf{{{formatted}}}"
                row.append(formatted)
            else:
               row.append("-")
       
        latex.append(" & ".join(row) + " \\\\")
   
    # 结束表格
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return "\n".join(latex)


def plot_metric_distribution(metric_values: Dict[str, List[float]],
                          metric_name: str,
                          save_path: Optional[str] = None):
   """
   绘制指标分布图
   
   Args:
       metric_values: 指标值字典 {model_name: [values]}
       metric_name: 指标名称
       save_path: 保存路径
   """
   plt.figure(figsize=(10, 6))
   
   # 准备数据
   data = []
   labels = []
   
   for model_name, values in metric_values.items():
       data.append(values)
       labels.append(model_name)
   
   # 创建箱线图
   box_plot = plt.boxplot(data, labels=labels, patch_artist=True)
   
   # 设置颜色
   colors = plt.cm.Set3(np.linspace(0, 1, len(data)))
   for patch, color in zip(box_plot['boxes'], colors):
       patch.set_facecolor(color)
   
   plt.title(f'{metric_name} 分布比较')
   plt.ylabel(metric_name)
   plt.grid(True, alpha=0.3)
   
   # 添加均值标记
   for i, values in enumerate(data):
       mean_val = np.mean(values)
       plt.scatter(i + 1, mean_val, color='red', s=100, zorder=3, marker='D')
   
   plt.legend(['均值'], loc='upper right')
   
   if save_path:
       plt.savefig(save_path, dpi=300, bbox_inches='tight')
   
   plt.show()


def analyze_failure_cases(model, test_data: np.ndarray,
                        threshold: float = 1.0) -> Dict[str, Any]:
   """
   分析预测失败的案例
   
   Args:
       model: 推荐模型
       test_data: 测试数据
       threshold: 错误阈值
       
   Returns:
       失败案例分析
   """
   errors = []
   failure_cases = []
   
   for row in test_data:
       user_id = int(row[0])
       item_id = int(row[1])
       true_rating = float(row[2])
       
       pred_rating = model.predict(user_id, item_id)[0]
       error = abs(pred_rating - true_rating)
       
       errors.append(error)
       
       if error > threshold:
           failure_cases.append({
               'user_id': user_id,
               'item_id': item_id,
               'true_rating': true_rating,
               'pred_rating': pred_rating,
               'error': error
           })
   
   # 分析失败模式
   analysis = {
       'total_cases': len(test_data),
       'failure_cases': len(failure_cases),
       'failure_rate': len(failure_cases) / len(test_data),
       'mean_error': np.mean(errors),
       'max_error': np.max(errors),
       'error_distribution': {
           'q25': np.percentile(errors, 25),
           'q50': np.percentile(errors, 50),
           'q75': np.percentile(errors, 75),
           'q95': np.percentile(errors, 95)
       }
   }
   
   # 分析失败案例的特征
   if failure_cases:
       failure_df = pd.DataFrame(failure_cases)
       
       # 用户维度分析
       user_failures = failure_df.groupby('user_id').size()
       analysis['user_analysis'] = {
           'most_failed_users': user_failures.nlargest(10).to_dict(),
           'users_with_failures': len(user_failures)
       }
       
       # 物品维度分析
       item_failures = failure_df.groupby('item_id').size()
       analysis['item_analysis'] = {
           'most_failed_items': item_failures.nlargest(10).to_dict(),
           'items_with_failures': len(item_failures)
       }
       
       # 评分模式分析
       analysis['rating_analysis'] = {
           'mean_true_rating': failure_df['true_rating'].mean(),
           'mean_pred_rating': failure_df['pred_rating'].mean(),
           'overestimation_rate': (failure_df['pred_rating'] > failure_df['true_rating']).mean()
       }
   
   return analysis


def generate_performance_summary(evaluation_results: Dict[str, Dict[str, float]],
                              baseline_model: Optional[str] = None) -> pd.DataFrame:
   """
   生成性能摘要表
   
   Args:
       evaluation_results: 评估结果
       baseline_model: 基线模型名称
       
   Returns:
       性能摘要DataFrame
   """
   # 创建摘要数据
   summary_data = []
   
   for model_name, results in evaluation_results.items():
       row = {'Model': model_name}
       
       # 添加关键指标
       key_metrics = ['rmse', 'mae', 'hr@10', 'ndcg@10', 'map@10']
       for metric in key_metrics:
           if metric in results:
               row[metric.upper()] = results[metric]
       
       # 如果有基线模型，计算相对改进
       if baseline_model and baseline_model in evaluation_results:
           baseline_results = evaluation_results[baseline_model]
           for metric in key_metrics:
               if metric in results and metric in baseline_results:
                   baseline_value = baseline_results[metric]
                   current_value = results[metric]
                   
                   # 计算相对改进（注意MAE和RMSE是越小越好）
                   if metric in ['mae', 'rmse']:
                       improvement = (baseline_value - current_value) / baseline_value * 100
                   else:
                       improvement = (current_value - baseline_value) / baseline_value * 100
                   
                   row[f'{metric.upper()}_Improvement(%)'] = improvement
       
       summary_data.append(row)
   
   # 创建DataFrame
   df = pd.DataFrame(summary_data)
   
   # 排序（按RMSE或HR@10）
   if 'RMSE' in df.columns:
       df = df.sort_values('RMSE')
   elif 'HR@10' in df.columns:
       df = df.sort_values('HR@10', ascending=False)
   
   return df


class EvaluationVisualizer:
   """评估结果可视化器"""
   
   def __init__(self, style: str = 'seaborn'):
       """
       初始化可视化器
       
       Args:
           style: 绘图风格
       """
       plt.style.use(style)
       self.colors = plt.cm.Set3(np.linspace(0, 1, 10))
   
   def plot_learning_curves(self, training_history: Dict[str, List[float]],
                          save_path: Optional[str] = None):
       """绘制学习曲线"""
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       axes = axes.ravel()
       
       metrics = list(training_history.keys())[:4]
       
       for idx, metric in enumerate(metrics):
           ax = axes[idx]
           values = training_history[metric]
           epochs = range(1, len(values) + 1)
           
           ax.plot(epochs, values, 'b-', linewidth=2)
           ax.set_xlabel('Epoch')
           ax.set_ylabel(metric)
           ax.set_title(f'{metric} vs Epoch')
           ax.grid(True, alpha=0.3)
       
       plt.tight_layout()
       
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
       
       plt.show()
   
   def plot_metric_improvements(self, baseline_results: Dict[str, float],
                              improved_results: Dict[str, float],
                              save_path: Optional[str] = None):
       """绘制指标改进图"""
       # 计算改进
       improvements = {}
       for metric in baseline_results:
           if metric in improved_results:
               baseline = baseline_results[metric]
               improved = improved_results[metric]
               
               if metric.lower() in ['mae', 'rmse', 'mse']:
                   improvement = (baseline - improved) / baseline * 100
               else:
                   improvement = (improved - baseline) / baseline * 100
               
               improvements[metric] = improvement
       
       # 绘图
       plt.figure(figsize=(10, 6))
       
       metrics = list(improvements.keys())
       values = list(improvements.values())
       
       bars = plt.bar(metrics, values, color=['green' if v > 0 else 'red' for v in values])
       
       # 添加数值标签
       for bar, value in zip(bars, values):
           height = bar.get_height()
           plt.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.1f}%',
                   ha='center', va='bottom' if value > 0 else 'top')
       
       plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
       plt.xlabel('Metrics')
       plt.ylabel('Improvement (%)')
       plt.title('Model Performance Improvements')
       plt.xticks(rotation=45)
       plt.grid(True, alpha=0.3, axis='y')
       
       plt.tight_layout()
       
       if save_path:
           plt.savefig(save_path, dpi=300, bbox_inches='tight')
       
       plt.show()