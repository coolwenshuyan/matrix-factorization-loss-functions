"""
结果分析器模块

提供实验结果的深入分析功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .experiment_runner import RunResult


class ResultsAnalyzer:
    """结果分析器"""
    
    def __init__(self, results_dir: str = "./results"):
        self.results_dir = Path(results_dir)
        self.results_cache = {}
        
    def load_result(self, experiment_id: str) -> RunResult:
        """加载实验结果"""
        if experiment_id in self.results_cache:
            return self.results_cache[experiment_id]
            
        # 尝试加载pickle文件
        exp_dir = self.results_dir / experiment_id
        pkl_file = exp_dir / "result.pkl"
        
        if pkl_file.exists():
            import pickle
            with open(pkl_file, 'rb') as f:
                result = pickle.load(f)
                self.results_cache[experiment_id] = result
                return result
                
        # 尝试加载JSON文件
        json_file = exp_dir / "result.json"
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
                # 简化的结果对象（只包含基本信息）
                return data
                
        raise ValueError(f"找不到实验结果: {experiment_id}")
        
    def analyze_single_experiment(self, result: Union[str, RunResult]) -> Dict:
        """分析单个实验"""
        if isinstance(result, str):
            result = self.load_result(result)
            
        analysis = {
            'experiment_id': result.experiment_id if hasattr(result, 'experiment_id') else result.get('experiment_id'),
            'basic_info': self._analyze_basic_info(result),
            'training_analysis': self._analyze_training(result),
            'evaluation_analysis': self._analyze_evaluation(result),
            'convergence_analysis': self._analyze_convergence(result),
            'resource_analysis': self._analyze_resource_usage(result)
        }
        
        return analysis
        
    def _analyze_basic_info(self, result: Union[RunResult, Dict]) -> Dict:
        """分析基本信息"""
        if isinstance(result, dict):
            return {
                'status': result.get('status'),
                'duration': result.get('duration'),
                'config_summary': {
                    'loss_type': result.get('config', {}).get('loss_type'),
                    'dataset': result.get('config', {}).get('dataset'),
                    'epochs': result.get('config', {}).get('epochs')
                }
            }
            
        return {
            'status': result.status.value,
            'duration': result.duration,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'config_summary': {
                'loss_type': result.config.loss_type,
                'dataset': result.config.dataset,
                'latent_factors': result.config.latent_factors,
                'learning_rate': result.config.learning_rate,
                'batch_size': result.config.batch_size,
                'epochs': result.config.epochs
            }
        }
        
    def _analyze_training(self, result: Union[RunResult, Dict]) -> Dict:
        """分析训练过程"""
        if isinstance(result, dict):
            history = result.get('train_history', {})
        else:
            history = result.train_history
            
        if not history:
            return {}
            
        analysis = {
            'total_epochs': len(history.get('train_loss', [])),
            'best_epoch': result.best_epoch if hasattr(result, 'best_epoch') else result.get('best_epoch'),
            'best_val_score': result.best_val_score if hasattr(result, 'best_val_score') else result.get('best_val_score'),
            'final_train_loss': history.get('train_loss', [])[-1] if 'train_loss' in history else None,
            'final_val_loss': history.get('val_loss', [])[-1] if 'val_loss' in history else None
        }
        
        # 计算训练统计
        if 'train_loss' in history:
            train_losses = history['train_loss']
            analysis['train_loss_stats'] = {
                'min': min(train_losses),
                'max': max(train_losses),
                'mean': np.mean(train_losses),
                'std': np.std(train_losses),
                'improvement': train_losses[0] - train_losses[-1] if train_losses else 0
            }
            
        # 早停分析
        if analysis['total_epochs'] < result.config.epochs if hasattr(result, 'config') else result.get('config', {}).get('epochs', 100):
            analysis['early_stopped'] = True
            analysis['epochs_saved'] = result.config.epochs - analysis['total_epochs'] if hasattr(result, 'config') else None
        else:
            analysis['early_stopped'] = False
            
        return analysis
        
    def _analyze_evaluation(self, result: Union[RunResult, Dict]) -> Dict:
        """分析评估结果"""
        if isinstance(result, dict):
            test_results = result.get('test_results', {})
        else:
            test_results = result.test_results
            
        if not test_results:
            return {}
            
        analysis = {
            'metrics': test_results,
            'primary_metric': 'mae',  # 可配置
            'primary_score': test_results.get('mae')
        }
        
        # 分析不同类型的指标
        error_metrics = ['mae', 'rmse', 'mse']
        ranking_metrics = ['hr@5', 'hr@10', 'hr@20', 'ndcg@5', 'ndcg@10', 'ndcg@20']
        
        analysis['error_metrics'] = {k: v for k, v in test_results.items() if k in error_metrics}
        analysis['ranking_metrics'] = {k: v for k, v in test_results.items() if k in ranking_metrics}
        
        return analysis
        
    def _analyze_convergence(self, result: Union[RunResult, Dict]) -> Dict:
        """分析收敛性"""
        if isinstance(result, dict):
            history = result.get('train_history', {})
        else:
            history = result.train_history
            
        if not history or 'train_loss' not in history:
            return {}
            
        train_losses = history['train_loss']
        
        analysis = {
            'converged': False,
            'convergence_epoch': None,
            'convergence_rate': None,
            'final_improvement_rate': None
        }
        
        # 检查收敛
        if len(train_losses) > 10:
            # 计算最后10个epoch的改进率
            recent_losses = train_losses[-10:]
            improvement_rates = []
            for i in range(1, len(recent_losses)):
                rate = abs(recent_losses[i] - recent_losses[i-1]) / recent_losses[i-1]
                improvement_rates.append(rate)
                
            avg_improvement_rate = np.mean(improvement_rates)
            analysis['final_improvement_rate'] = avg_improvement_rate
            
            # 如果平均改进率小于阈值，认为已收敛
            if avg_improvement_rate < 0.001:
                analysis['converged'] = True
                
                # 找到收敛点
                for i in range(len(train_losses) - 10):
                    window = train_losses[i:i+10]
                    window_rates = []
                    for j in range(1, len(window)):
                        rate = abs(window[j] - window[j-1]) / window[j-1]
                        window_rates.append(rate)
                        
                    if np.mean(window_rates) < 0.001:
                        analysis['convergence_epoch'] = i
                        break
                        
        # 计算收敛速度
        if len(train_losses) > 1:
            # 使用指数拟合
            epochs = np.arange(len(train_losses))
            
            # 对数变换后线性拟合
            log_losses = np.log(train_losses)
            slope, intercept = np.polyfit(epochs, log_losses, 1)
            
            analysis['convergence_rate'] = -slope  # 负斜率表示下降速度
            
        return analysis
        
    def _analyze_resource_usage(self, result: Union[RunResult, Dict]) -> Dict:
        """分析资源使用"""
        if isinstance(result, dict):
            resource_usage = result.get('resource_usage', {})
            duration = result.get('duration', 0)
        else:
            resource_usage = result.resource_usage
            duration = result.duration
            
        analysis = {
            'total_time': duration,
            'time_per_epoch': duration / result.train_history.get('total_epochs', 1) if hasattr(result, 'train_history') else None,
            'peak_memory_gb': resource_usage.get('memory_used_gb'),
            'avg_cpu_percent': resource_usage.get('cpu_percent'),
            'gpu_used': bool(resource_usage.get('gpu_usage'))
        }
        
        if resource_usage.get('gpu_usage'):
            gpu_info = resource_usage['gpu_usage'][0] if resource_usage['gpu_usage'] else {}
            analysis['gpu_memory_percent'] = gpu_info.get('memory_percent')
            analysis['gpu_utilization'] = gpu_info.get('load')
            
        return analysis
        
    def compare_experiments(self, 
                          experiment_ids: List[str],
                          metrics: Optional[List[str]] = None) -> pd.DataFrame:
        """比较多个实验"""
        if metrics is None:
            metrics = ['mae', 'rmse', 'hr@10', 'ndcg@10']
            
        data = []
        
        for exp_id in experiment_ids:
            try:
                result = self.load_result(exp_id)
                analysis = self.analyze_single_experiment(result)
                
                row = {
                    'experiment_id': exp_id,
                    'loss_type': analysis['basic_info']['config_summary'].get('loss_type'),
                    'dataset': analysis['basic_info']['config_summary'].get('dataset'),
                    'status': analysis['basic_info']['status'],
                    'duration': analysis['basic_info']['duration'],
                    'epochs': analysis['training_analysis'].get('total_epochs'),
                    'best_val_score': analysis['training_analysis'].get('best_val_score')
                }
                
                # 添加指标
                test_metrics = analysis['evaluation_analysis'].get('metrics', {})
                for metric in metrics:
                    row[metric] = test_metrics.get(metric)
                    
                data.append(row)
                
            except Exception as e:
                print(f"加载实验失败 {exp_id}: {e}")
                
        return pd.DataFrame(data)
        
    def find_best_experiments(self,
                            metric: str = 'mae',
                            minimize: bool = True,
                            filter_criteria: Optional[Dict] = None) -> List[Tuple[str, float]]:
        """找到最佳实验"""
        results = []
        
        # 遍历所有实验目录
        for exp_dir in self.results_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            try:
                result = self.load_result(exp_dir.name)
                
                # 应用过滤条件
                if filter_criteria:
                    config = result.config if hasattr(result, 'config') else result.get('config', {})
                    skip = False
                    
                    for key, value in filter_criteria.items():
                        if hasattr(config, key):
                            if getattr(config, key) != value:
                                skip = True
                                break
                        elif isinstance(config, dict) and config.get(key) != value:
                            skip = True
                            break
                            
                    if skip:
                        continue
                        
                # 获取指标值
                test_results = result.test_results if hasattr(result, 'test_results') else result.get('test_results', {})
                metric_value = test_results.get(metric)
                
                if metric_value is not None:
                    results.append((exp_dir.name, metric_value))
                    
            except Exception as e:
                continue
                
        # 排序
        results.sort(key=lambda x: x[1], reverse=not minimize)
        
        return results


class ResultsAggregator:
    """结果聚合器"""
    
    def __init__(self, analyzer: ResultsAnalyzer):
        self.analyzer = analyzer
        
    def aggregate_by_config(self, 
                          experiment_ids: List[str],
                          group_by: List[str],
                          metrics: List[str]) -> pd.DataFrame:
        """按配置聚合结果"""
        data = []
        
        for exp_id in experiment_ids:
            try:
                result = self.analyzer.load_result(exp_id)
                
                row = {'experiment_id': exp_id}
                
                # 添加分组字段
                config = result.config if hasattr(result, 'config') else result.get('config', {})
                for field in group_by:
                    if hasattr(config, field):
                        row[field] = getattr(config, field)
                    elif isinstance(config, dict):
                        row[field] = config.get(field)
                        
                # 添加指标
                test_results = result.test_results if hasattr(result, 'test_results') else result.get('test_results', {})
                for metric in metrics:
                    row[metric] = test_results.get(metric)
                    
                data.append(row)
                
            except Exception:
                continue
                
        df = pd.DataFrame(data)
        
        # 聚合
        if group_by:
            agg_funcs = {metric: ['mean', 'std', 'min', 'max', 'count'] for metric in metrics}
            grouped = df.groupby(group_by).agg(agg_funcs)
            return grouped
            
        return df
        
    def compute_statistics(self,
                         experiment_ids: List[str],
                         metrics: List[str]) -> Dict:
        """计算统计信息"""
        df = self.analyzer.compare_experiments(experiment_ids, metrics)
        
        statistics = {}
        
        for metric in metrics:
            if metric in df.columns:
                values = df[metric].dropna()
                
                if len(values) > 0:
                    statistics[metric] = {
                        'count': len(values),
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'median': values.median(),
                        'q1': values.quantile(0.25),
                        'q3': values.quantile(0.75)
                    }
                    
                    # 添加置信区间
                    if len(values) > 1:
                        confidence_interval = stats.t.interval(
                            0.95,
                            len(values) - 1,
                            loc=values.mean(),
                            scale=values.sem()
                        )
                        statistics[metric]['ci_lower'] = confidence_interval[0]
                        statistics[metric]['ci_upper'] = confidence_interval[1]
                        
        return statistics


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, analyzer: ResultsAnalyzer):
        self.analyzer = analyzer
        
    def analyze_parameter_impact(self,
                               experiment_ids: List[str],
                               parameter: str,
                               metric: str) -> Dict:
        """分析参数影响"""
        data = []
        
        for exp_id in experiment_ids:
            try:
                result = self.analyzer.load_result(exp_id)
                
                # 获取参数值
                config = result.config if hasattr(result, 'config') else result.get('config', {})
                if hasattr(config, parameter):
                    param_value = getattr(config, parameter)
                elif isinstance(config, dict):
                    param_value = config.get(parameter)
                else:
                    continue
                    
                # 获取指标值
                test_results = result.test_results if hasattr(result, 'test_results') else result.get('test_results', {})
                metric_value = test_results.get(metric)
                
                if metric_value is not None:
                    data.append({
                        'parameter_value': param_value,
                        'metric_value': metric_value
                    })
                    
            except Exception:
                continue
                
        if not data:
            return {}
            
        df = pd.DataFrame(data)
        
        # 分析相关性
        analysis = {
            'data': df,
            'correlation': None,
            'regression': None,
            'anova': None
        }
        
        # 数值参数：计算相关性
        if pd.api.types.is_numeric_dtype(df['parameter_value']):
            correlation, p_value = stats.pearsonr(
                df['parameter_value'],
                df['metric_value']
            )
            analysis['correlation'] = {
                'pearson_r': correlation,
                'p_value': p_value
            }
            
            # 线性回归
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df['parameter_value'],
                df['metric_value']
            )
            analysis['regression'] = {
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err
            }
            
        # 分类参数：ANOVA
        else:
            groups = df.groupby('parameter_value')['metric_value'].apply(list)
            
            if len(groups) > 1:
                f_stat, p_value = stats.f_oneway(*groups)
                analysis['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'group_means': df.groupby('parameter_value')['metric_value'].mean().to_dict()
                }
                
        return analysis
        
    def analyze_interactions(self,
                           experiment_ids: List[str],
                           parameters: List[str],
                           metric: str) -> Dict:
        """分析参数交互效应"""
        data = []
        
        for exp_id in experiment_ids:
            try:
                result = self.analyzer.load_result(exp_id)
                
                row = {}
                
                # 获取参数值
                config = result.config if hasattr(result, 'config') else result.get('config', {})
                for param in parameters:
                    if hasattr(config, param):
                        row[param] = getattr(config, param)
                    elif isinstance(config, dict):
                        row[param] = config.get(param)
                        
                # 获取指标值
                test_results = result.test_results if hasattr(result, 'test_results') else result.get('test_results', {})
                row[metric] = test_results.get(metric)
                
                if metric in row and row[metric] is not None:
                    data.append(row)
                    
            except Exception:
                continue
                
        if not data:
            return {}
            
        df = pd.DataFrame(data)
        
        # 创建交互项
        if len(parameters) == 2:
            param1, param2 = parameters
            
            # 检查是否为数值参数
            if (pd.api.types.is_numeric_dtype(df[param1]) and 
                pd.api.types.is_numeric_dtype(df[param2])):
                
                # 标准化
                scaler = StandardScaler()
                X = scaler.fit_transform(df[[param1, param2]])
                
                # 添加交互项
                df['interaction'] = X[:, 0] * X[:, 1]
                
                # 多元回归分析
                from sklearn.linear_model import LinearRegression
                
                X_with_interaction = np.column_stack([X, df['interaction']])
                y = df[metric].values
                
                model = LinearRegression()
                model.fit(X_with_interaction, y)
                
                return {
                    'coefficients': {
                        param1: model.coef_[0],
                        param2: model.coef_[1],
                        'interaction': model.coef_[2]
                    },
                    'r_squared': model.score(X_with_interaction, y),
                    'data': df
                }
                
        return {'data': df}


class ConvergenceAnalyzer:
    """收敛性分析器"""
    
    def __init__(self, analyzer: ResultsAnalyzer):
        self.analyzer = analyzer
        
    def analyze_convergence_patterns(self,
                                   experiment_ids: List[str]) -> Dict:
        """分析收敛模式"""
        patterns = {
            'fast_converging': [],
            'slow_converging': [],
            'oscillating': [],
            'non_converging': []
        }
        
        for exp_id in experiment_ids:
            try:
                result = self.analyzer.load_result(exp_id)
                analysis = self.analyzer.analyze_single_experiment(result)
                
                conv_analysis = analysis.get('convergence_analysis', {})
                
                if not conv_analysis:
                    continue
                    
                # 分类收敛模式
                if conv_analysis.get('converged'):
                    if conv_analysis.get('convergence_epoch', float('inf')) < 20:
                        patterns['fast_converging'].append(exp_id)
                    else:
                        patterns['slow_converging'].append(exp_id)
                else:
                    # 检查震荡
                    history = result.train_history if hasattr(result, 'train_history') else result.get('train_history', {})
                    
                    if 'train_loss' in history:
                        losses = history['train_loss']
                        
                        # 计算损失变化的方差
                        if len(losses) > 10:
                            changes = np.diff(losses)
                            variance = np.var(changes)
                            
                            if variance > 0.01:
                                patterns['oscillating'].append(exp_id)
                            else:
                                patterns['non_converging'].append(exp_id)
                                
            except Exception:
                continue
                
        return patterns
        
    def plot_convergence_curves(self,
                              experiment_ids: List[str],
                              output_path: str,
                              metrics: List[str] = ['train_loss', 'val_loss']):
        """绘制收敛曲线"""
        plt.figure(figsize=(12, 8))
        
        for i, metric in enumerate(metrics):
            plt.subplot(len(metrics), 1, i+1)
            
            for exp_id in experiment_ids:
                try:
                    result = self.analyzer.load_result(exp_id)
                    history = result.train_history if hasattr(result, 'train_history') else result.get('train_history', {})
                    
                    if metric in history:
                        values = history[metric]
                        epochs = range(1, len(values) + 1)
                        
                        # 获取实验标签
                        config = result.config if hasattr(result, 'config') else result.get('config', {})
                        label = f"{config.get('loss_type', 'unknown')}_{exp_id[:8]}"
                        
                        plt.plot(epochs, values, label=label, alpha=0.7)
                        
                except Exception:
                    continue
                    
            plt.xlabel('Epoch')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.title(f'{metric.replace("_", " ").title()} Convergence')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()