"""
基线模型管理模块

管理基线模型的定义、运行和对比
"""

import os
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .config_manager import ExperimentConfig, ConfigManager
from .experiment_runner import ExperimentRunner, RunResult


@dataclass
class BaselineConfig:
    """基线配置"""
    name: str
    loss_type: str
    description: str = ""
    
    # 默认超参数
    default_params: Dict = field(default_factory=dict)
    
    # 参数搜索范围（可选）
    param_ranges: Dict = field(default_factory=dict)
    
    # 数据集特定的最佳参数（可选）
    dataset_best_params: Dict = field(default_factory=dict)
    
    def get_params_for_dataset(self, dataset: str) -> Dict:
        """获取特定数据集的参数"""
        if dataset in self.dataset_best_params:
            params = self.default_params.copy()
            params.update(self.dataset_best_params[dataset])
            return params
        return self.default_params.copy()


@dataclass
class BaselineResult:
    """基线运行结果"""
    baseline_name: str
    dataset: str
    config: ExperimentConfig
    run_result: RunResult
    
    def get_metrics(self) -> Dict:
        """获取评估指标"""
        return self.run_result.test_results
    
    def get_summary(self) -> Dict:
        """获取结果摘要"""
        return {
            'baseline': self.baseline_name,
            'dataset': self.dataset,
            'status': self.run_result.status.value,
            'duration': self.run_result.duration,
            'best_val_score': self.run_result.best_val_score,
            'metrics': self.get_metrics()
        }


@dataclass
class BaselineComparison:
    """基线对比结果"""
    dataset: str
    target_result: BaselineResult
    baseline_results: List[BaselineResult]
    
    def get_comparison_table(self) -> pd.DataFrame:
        """生成对比表格"""
        rows = []
        
        # 添加目标方法
        target_metrics = self.target_result.get_metrics()
        row = {
            'Method': self.target_result.baseline_name,
            'Dataset': self.dataset,
            **target_metrics
        }
        rows.append(row)
        
        # 添加基线方法
        for baseline in self.baseline_results:
            metrics = baseline.get_metrics()
            row = {
                'Method': baseline.baseline_name,
                'Dataset': self.dataset,
                **metrics
            }
            rows.append(row)
            
        df = pd.DataFrame(rows)
        
        # 计算相对改进
        if len(rows) > 1:
            for col in target_metrics.keys():
                if col in df.columns:
                    baseline_values = df[col].iloc[1:]
                    target_value = df[col].iloc[0]
                    
                    # 根据指标类型计算改进（MAE/RMSE越小越好，其他越大越好）
                    if col.lower() in ['mae', 'rmse', 'mse']:
                        improvements = ((baseline_values - target_value) / baseline_values * 100).tolist()
                    else:
                        improvements = ((target_value - baseline_values) / baseline_values * 100).tolist()
                        
                    df[f'{col}_improvement'] = [np.nan] + improvements
                    
        return df
    
    def get_best_baseline(self, metric: str) -> str:
        """获取在指定指标上表现最好的基线"""
        all_results = [self.target_result] + self.baseline_results
        
        # 根据指标类型确定最优方向
        minimize = metric.lower() in ['mae', 'rmse', 'mse']
        
        best_result = None
        best_value = float('inf') if minimize else float('-inf')
        
        for result in all_results:
            value = result.get_metrics().get(metric)
            if value is not None:
                if (minimize and value < best_value) or (not minimize and value > best_value):
                    best_value = value
                    best_result = result
                    
        return best_result.baseline_name if best_result else None


class BaselineManager:
    """基线模型管理器"""
    
    def __init__(self,
                 config_manager: ConfigManager,
                 experiment_runner: ExperimentRunner,
                 baselines_dir: str = "./configs/baselines"):
        self.config_manager = config_manager
        self.experiment_runner = experiment_runner
        self.baselines_dir = Path(baselines_dir)
        self.baselines_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines = {}
        self._init_default_baselines()
        
    def _init_default_baselines(self):
        """初始化默认基线"""
        # L2损失基线
        self.add_baseline(BaselineConfig(
            name="MF-L2",
            loss_type="l2",
            description="Matrix Factorization with L2 loss",
            default_params={
                'latent_factors': 50,
                'learning_rate': 0.001,
                'lambda_reg': 0.01,
                'epochs': 100,
                'batch_size': 256
            },
            param_ranges={
                'latent_factors': [10, 20, 50, 100],
                'learning_rate': [0.0001, 0.001, 0.01],
                'lambda_reg': [0.001, 0.01, 0.1]
            }
        ))
        
        # L1损失基线
        self.add_baseline(BaselineConfig(
            name="MF-L1",
            loss_type="l1",
            description="Matrix Factorization with L1 loss",
            default_params={
                'latent_factors': 50,
                'learning_rate': 0.001,
                'lambda_reg': 0.01,
                'epochs': 100,
                'batch_size': 256
            }
        ))
        
        # Huber损失基线
        self.add_baseline(BaselineConfig(
            name="MF-Huber",
            loss_type="huber",
            description="Matrix Factorization with Huber loss",
            default_params={
                'latent_factors': 50,
                'learning_rate': 0.001,
                'lambda_reg': 0.01,
                'huber_delta': 1.0,
                'epochs': 100,
                'batch_size': 256
            },
            param_ranges={
                'huber_delta': [0.5, 1.0, 1.5, 2.0]
            }
        ))
        
        # Logcosh损失基线
        self.add_baseline(BaselineConfig(
            name="MF-Logcosh",
            loss_type="logcosh",
            description="Matrix Factorization with Logcosh loss",
            default_params={
                'latent_factors': 50,
                'learning_rate': 0.001,
                'lambda_reg': 0.01,
                'epochs': 100,
                'batch_size': 256
            }
        ))
        
    def add_baseline(self, baseline: BaselineConfig):
        """添加基线"""
        self.baselines[baseline.name] = baseline
        
        # 保存基线配置
        baseline_file = self.baselines_dir / f"{baseline.name}.json"
        with open(baseline_file, 'w') as f:
            json.dump({
                'name': baseline.name,
                'loss_type': baseline.loss_type,
                'description': baseline.description,
                'default_params': baseline.default_params,
                'param_ranges': baseline.param_ranges,
                'dataset_best_params': baseline.dataset_best_params
            }, f, indent=2)
            
    def load_baseline(self, name: str) -> BaselineConfig:
        """加载基线"""
        if name in self.baselines:
            return self.baselines[name]
            
        baseline_file = self.baselines_dir / f"{name}.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                
            baseline = BaselineConfig(
                name=data['name'],
                loss_type=data['loss_type'],
                description=data.get('description', ''),
                default_params=data.get('default_params', {}),
                param_ranges=data.get('param_ranges', {}),
                dataset_best_params=data.get('dataset_best_params', {})
            )
            
            self.baselines[name] = baseline
            return baseline
            
        raise ValueError(f"基线不存在: {name}")
        
    def create_baseline_config(self,
                             baseline_name: str,
                             dataset: str,
                             base_config: Optional[ExperimentConfig] = None,
                             **override_params) -> ExperimentConfig:
        """创建基线实验配置"""
        baseline = self.load_baseline(baseline_name)
        
        # 获取基线参数
        params = baseline.get_params_for_dataset(dataset)
        params.update(override_params)
        
        # 创建配置
        config_name = f"{baseline_name}_{dataset}"
        
        if base_config:
            config = self.config_manager.create_config(
                name=config_name,
                base_config=base_config,
                loss_type=baseline.loss_type,
                dataset=dataset,
                **params
            )
        else:
            config = self.config_manager.create_config(
                name=config_name,
                loss_type=baseline.loss_type,
                dataset=dataset,
                **params
            )
            
        return config
    
    def run_baseline(self,
                    baseline_name: str,
                    dataset: str,
                    **override_params) -> BaselineResult:
        """运行单个基线"""
        # 创建配置
        config = self.create_baseline_config(
            baseline_name,
            dataset,
            **override_params
        )
        
        # 运行实验
        print(f"\n运行基线: {baseline_name} on {dataset}")
        run_result = self.experiment_runner.run_experiment(config)
        
        # 创建基线结果
        result = BaselineResult(
            baseline_name=baseline_name,
            dataset=dataset,
            config=config,
            run_result=run_result
        )
        
        return result
    
    def run_all_baselines(self,
                         dataset: str,
                         baseline_names: Optional[List[str]] = None,
                         **common_params) -> List[BaselineResult]:
        """运行所有基线"""
        if baseline_names is None:
            baseline_names = list(self.baselines.keys())
            
        results = []
        
        for i, baseline_name in enumerate(baseline_names):
            print(f"\n{'='*60}")
            print(f"运行基线 {i+1}/{len(baseline_names)}: {baseline_name}")
            print(f"{'='*60}")
            
            try:
                result = self.run_baseline(
                    baseline_name,
                    dataset,
                    **common_params
                )
                results.append(result)
                
            except Exception as e:
                print(f"基线运行失败: {baseline_name}")
                print(f"错误: {str(e)}")
                
        return results
    
    def compare_with_baselines(self,
                             target_result: BaselineResult,
                             baseline_results: List[BaselineResult]) -> BaselineComparison:
        """与基线进行对比"""
        # 确保所有结果使用相同的数据集
        dataset = target_result.dataset
        baseline_results = [r for r in baseline_results if r.dataset == dataset]
        
        comparison = BaselineComparison(
            dataset=dataset,
            target_result=target_result,
            baseline_results=baseline_results
        )
        
        return comparison
    
    def generate_comparison_report(self,
                                 comparisons: List[BaselineComparison],
                                 output_dir: str) -> str:
        """生成对比报告"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        report_file = output_path / "baseline_comparison_report.md"
        
        with open(report_file, 'w') as f:
            f.write("# 基线对比报告\n\n")
            
            # 汇总表格
            f.write("## 总体结果\n\n")
            
            all_dfs = []
            for comp in comparisons:
                df = comp.get_comparison_table()
                all_dfs.append(df)
                
            if all_dfs:
                combined_df = pd.concat(all_dfs, ignore_index=True)
                
                # 按数据集和方法分组计算平均值
                summary_df = combined_df.groupby(['Dataset', 'Method']).mean()
                
                f.write(summary_df.to_markdown())
                f.write("\n\n")
                
            # 详细结果
            f.write("## 详细结果\n\n")
            
            for comp in comparisons:
                f.write(f"### {comp.dataset}\n\n")
                
                df = comp.get_comparison_table()
                f.write(df.to_markdown())
                f.write("\n\n")
                
                # 最佳方法
                f.write("**最佳方法:**\n")
                for metric in ['mae', 'rmse', 'hr@10', 'ndcg@10']:
                    best = comp.get_best_baseline(metric)
                    if best:
                        f.write(f"- {metric}: {best}\n")
                        
                f.write("\n")
                
        # 生成可视化
        self._generate_comparison_plots(comparisons, output_path)
        
        print(f"对比报告已生成: {report_file}")
        return str(report_file)
    
    def _generate_comparison_plots(self,
                                 comparisons: List[BaselineComparison],
                                 output_dir: Path):
        """生成对比图表"""
        # 收集数据
        data = []
        for comp in comparisons:
            all_results = [comp.target_result] + comp.baseline_results
            for result in all_results:
                metrics = result.get_metrics()
                for metric, value in metrics.items():
                    data.append({
                        'Dataset': comp.dataset,
                        'Method': result.baseline_name,
                        'Metric': metric,
                        'Value': value
                    })
                    
        df = pd.DataFrame(data)
        
        # 为每个指标生成图表
        metrics = df['Metric'].unique()
        
        for metric in metrics:
            plt.figure(figsize=(10, 6))
            
            metric_df = df[df['Metric'] == metric]
            
            # 条形图
            ax = sns.barplot(data=metric_df, x='Dataset', y='Value', hue='Method')
            
            plt.title(f'{metric} Comparison Across Datasets')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            plot_file = output_dir / f'comparison_{metric.replace("@", "_")}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
        # 生成热力图
        plt.figure(figsize=(12, 8))
        
        # 创建数据透视表
        for i, metric in enumerate(['mae', 'rmse']):
            if metric in metrics:
                plt.subplot(2, 2, i+1)
                
                pivot_df = df[df['Metric'] == metric].pivot(
                    index='Method',
                    columns='Dataset',
                    values='Value'
                )
                
                sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='RdYlGn_r')
                plt.title(f'{metric.upper()} Heatmap')
                
        plt.tight_layout()
        heatmap_file = output_dir / 'comparison_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
    def update_baseline_best_params(self,
                                   baseline_name: str,
                                   dataset: str,
                                   best_params: Dict):
        """更新基线的最佳参数"""
        baseline = self.load_baseline(baseline_name)
        baseline.dataset_best_params[dataset] = best_params
        
        # 保存更新
        self.add_baseline(baseline)
        
    def export_baseline_configs(self, output_dir: str):
        """导出所有基线配置"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, baseline in self.baselines.items():
            config_file = output_path / f"{name}_config.yaml"
            
            # 为每个数据集生成配置示例
            example_config = self.create_baseline_config(
                name,
                dataset="ml-100k"  # 示例数据集
            )
            
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(example_config.to_dict(), f, default_flow_style=False)