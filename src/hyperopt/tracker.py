"""
实验追踪器模块

记录和管理超参数优化实验
"""

import os
import json
import time
import pickle
import sqlite3
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import hashlib


class ExperimentTracker:
    """实验追踪器"""
    
    def __init__(self, 
                 experiment_name: str = None,
                 base_dir: str = "./experiments",
                 backend: str = "file"):
        """
        Args:
            experiment_name: 实验名称
            base_dir: 基础目录
            backend: 存储后端 ('file', 'sqlite', 'memory')
        """
        self.experiment_name = experiment_name or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.base_dir = base_dir
        self.backend = backend
        
        # 创建实验目录
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化后端
        if backend == "sqlite":
            self._init_sqlite()
        elif backend == "memory":
            self.memory_store = {
                'trials': [],
                'metadata': {},
                'checkpoints': {}
            }
            
        # 实验元数据
        self.metadata = {
            'experiment_name': self.experiment_name,
            'start_time': time.time(),
            'backend': backend,
            'status': 'running'
        }
        
    def _init_sqlite(self):
        """初始化SQLite数据库"""
        self.db_path = os.path.join(self.experiment_dir, "experiment.db")
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        
        # 创建表
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS trials (
                trial_id INTEGER PRIMARY KEY,
                config TEXT NOT NULL,
                score REAL,
                metrics TEXT,
                start_time REAL,
                end_time REAL,
                duration REAL,
                status TEXT,
                error TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                iteration INTEGER,
                data BLOB,
                timestamp REAL
            )
        """)
        
        self.conn.commit()
        
    def log_trial(self, trial: 'Trial'):
        """记录试验"""
        if self.backend == "file":
            self._log_trial_file(trial)
        elif self.backend == "sqlite":
            self._log_trial_sqlite(trial)
        elif self.backend == "memory":
            self._log_trial_memory(trial)
            
    def _log_trial_file(self, trial: 'Trial'):
        """记录到文件"""
        # 保存为JSON
        trial_path = os.path.join(self.experiment_dir, f"trial_{trial.trial_id}.json")
        
        trial_data = {
            'trial_id': trial.trial_id,
            'config': trial.config,
            'score': trial.score,
            'metrics': trial.metrics,
            'start_time': trial.start_time,
            'end_time': trial.end_time,
            'duration': trial.duration,
            'status': trial.status,
            'error': trial.error
        }
        
        with open(trial_path, 'w') as f:
            json.dump(trial_data, f, indent=2)
            
        # 追加到汇总文件
        summary_path = os.path.join(self.experiment_dir, "trials_summary.jsonl")
        with open(summary_path, 'a') as f:
            f.write(json.dumps(trial_data) + '\n')
            
    def _log_trial_sqlite(self, trial: 'Trial'):
        """记录到SQLite"""
        self.conn.execute("""
            INSERT INTO trials (
                trial_id, config, score, metrics, 
                start_time, end_time, duration, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trial.trial_id,
            json.dumps(trial.config),
            trial.score,
            json.dumps(trial.metrics) if trial.metrics else None,
            trial.start_time,
            trial.end_time,
            trial.duration,
            trial.status,
            trial.error
        ))
        self.conn.commit()
        
    def _log_trial_memory(self, trial: 'Trial'):
        """记录到内存"""
        trial_data = {
            'trial_id': trial.trial_id,
            'config': trial.config,
            'score': trial.score,
            'metrics': trial.metrics,
            'start_time': trial.start_time,
            'end_time': trial.end_time,
            'duration': trial.duration,
            'status': trial.status,
            'error': trial.error
        }
        self.memory_store['trials'].append(trial_data)
        
    def save_checkpoint(self, optimizer: 'HyperOptimizer', 
                       checkpoint_id: Optional[str] = None):
        """保存检查点"""
        if checkpoint_id is None:
            checkpoint_id = f"checkpoint_{optimizer.iteration}"
            
        checkpoint_data = {
            'iteration': optimizer.iteration,
            'n_trials_completed': optimizer.n_trials_completed,
            'best_trial': optimizer.best_trial,
            'sampler_state': getattr(optimizer.sampler, '__dict__', {}),
            'timestamp': time.time()
        }
        
        if self.backend == "file":
            checkpoint_path = os.path.join(
                self.experiment_dir, 
                f"{checkpoint_id}.pkl"
            )
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(checkpoint_data, f)
                
        elif self.backend == "sqlite":
            self.conn.execute("""
                INSERT OR REPLACE INTO checkpoints 
                (checkpoint_id, iteration, data, timestamp)
                VALUES (?, ?, ?, ?)
            """, (
                checkpoint_id,
                checkpoint_data['iteration'],
                pickle.dumps(checkpoint_data),
                checkpoint_data['timestamp']
            ))
            self.conn.commit()
            
        elif self.backend == "memory":
            self.memory_store['checkpoints'][checkpoint_id] = checkpoint_data
            
    def load_checkpoint(self, checkpoint_id: str) -> Dict:
        """加载检查点"""
        if self.backend == "file":
            checkpoint_path = os.path.join(
                self.experiment_dir,
                f"{checkpoint_id}.pkl"
            )
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
                
        elif self.backend == "sqlite":
            cursor = self.conn.execute("""
                SELECT data FROM checkpoints WHERE checkpoint_id = ?
            """, (checkpoint_id,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
            else:
                raise ValueError(f"检查点不存在: {checkpoint_id}")
                
        elif self.backend == "memory":
            if checkpoint_id in self.memory_store['checkpoints']:
                return self.memory_store['checkpoints'][checkpoint_id]
            else:
                raise ValueError(f"检查点不存在: {checkpoint_id}")
                
    def get_trials(self, status: Optional[str] = None) -> List[Dict]:
        """获取试验记录"""
        if self.backend == "file":
            trials = []
            summary_path = os.path.join(self.experiment_dir, "trials_summary.jsonl")
            
            if os.path.exists(summary_path):
                with open(summary_path, 'r') as f:
                    for line in f:
                        trial = json.loads(line.strip())
                        if status is None or trial['status'] == status:
                            trials.append(trial)
                            
            return trials
            
        elif self.backend == "sqlite":
            if status:
                cursor = self.conn.execute(
                    "SELECT * FROM trials WHERE status = ?", (status,)
                )
            else:
                cursor = self.conn.execute("SELECT * FROM trials")
                
            trials = []
            for row in cursor:
                trial = {
                    'trial_id': row[0],
                    'config': json.loads(row[1]),
                    'score': row[2],
                    'metrics': json.loads(row[3]) if row[3] else None,
                    'start_time': row[4],
                    'end_time': row[5],
                    'duration': row[6],
                    'status': row[7],
                    'error': row[8]
                }
                trials.append(trial)
                
            return trials
            
        elif self.backend == "memory":
            if status:
                return [t for t in self.memory_store['trials'] 
                       if t['status'] == status]
            else:
                return self.memory_store['trials'].copy()
                
    def get_best_trial(self, maximize: bool = True) -> Optional[Dict]:
        """获取最佳试验"""
        trials = self.get_trials(status='completed')
        
        if not trials:
            return None
            
        # 过滤出有分数的试验
        scored_trials = [t for t in trials if t['score'] is not None]
        
        if not scored_trials:
            return None
            
        if maximize:
            return max(scored_trials, key=lambda t: t['score'])
        else:
            return min(scored_trials, key=lambda t: t['score'])
            
    def save_metadata(self, key: str, value: Any):
        """保存元数据"""
        if self.backend == "file":
            metadata_path = os.path.join(self.experiment_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
                
            metadata[key] = value
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        elif self.backend == "sqlite":
            self.conn.execute("""
                INSERT OR REPLACE INTO metadata (key, value)
                VALUES (?, ?)
            """, (key, json.dumps(value)))
            self.conn.commit()
            
        elif self.backend == "memory":
            self.memory_store['metadata'][key] = value
            
    def load_metadata(self, key: str) -> Any:
        """加载元数据"""
        if self.backend == "file":
            metadata_path = os.path.join(self.experiment_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get(key)
                    
        elif self.backend == "sqlite":
            cursor = self.conn.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
                
        elif self.backend == "memory":
            return self.memory_store['metadata'].get(key)
            
        return None
        
    def save_final_results(self, optimizer: 'HyperOptimizer'):
        """保存最终结果"""
        results = optimizer.get_results()
        
        # 保存结果摘要
        summary_path = os.path.join(self.experiment_dir, "final_results.json")
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # 保存为CSV便于分析
        if results['all_trials']:
            df = pd.DataFrame(results['all_trials'])
            csv_path = os.path.join(self.experiment_dir, "trials.csv")
            df.to_csv(csv_path, index=False)
            
        # 更新元数据
        self.metadata['end_time'] = time.time()
        self.metadata['status'] = 'completed'
        self.metadata['best_score'] = results.get('best_score')
        self.metadata['n_trials'] = results.get('n_trials')
        
        self.save_metadata('experiment_metadata', self.metadata)
        
    def create_report(self) -> Dict:
        """创建实验报告"""
        trials = self.get_trials()
        
        if not trials:
            return {'error': '没有试验记录'}
            
        completed_trials = [t for t in trials if t['status'] == 'completed']
        failed_trials = [t for t in trials if t['status'] == 'failed']
        
        # 基础统计
        report = {
            'experiment_name': self.experiment_name,
            'total_trials': len(trials),
            'completed_trials': len(completed_trials),
            'failed_trials': len(failed_trials),
            'success_rate': len(completed_trials) / len(trials) if trials else 0
        }
        
        if completed_trials:
            scores = [t['score'] for t in completed_trials if t['score'] is not None]
            
            if scores:
                report.update({
                    'best_score': max(scores),
                    'worst_score': min(scores),
                    'mean_score': sum(scores) / len(scores),
                    'std_score': pd.Series(scores).std()
                })
                
            # 时间统计
            durations = [t['duration'] for t in completed_trials if t['duration']]
            if durations:
                report.update({
                    'mean_duration': sum(durations) / len(durations),
                    'total_duration': sum(durations)
                })
                
        # 参数分析
        if completed_trials:
            # 提取所有参数
            all_params = set()
            for trial in completed_trials:
                all_params.update(trial['config'].keys())
                
            # 参数与分数的相关性（简化版）
            param_impacts = {}
            for param in all_params:
                values = []
                scores = []
                
                for trial in completed_trials:
                    if param in trial['config'] and trial['score'] is not None:
                        values.append(trial['config'][param])
                        scores.append(trial['score'])
                        
                if len(set(values)) > 1:  # 参数有变化
                    # 简单的方差分析
                    if isinstance(values[0], (int, float)):
                        # 数值参数：计算相关系数
                        correlation = pd.Series(values).corr(pd.Series(scores))
                        param_impacts[param] = {
                            'type': 'numeric',
                            'correlation': correlation,
                            'min_value': min(values),
                            'max_value': max(values)
                        }
                    else:
                        # 分类参数：计算每个值的平均分数
                        value_scores = {}
                        for v, s in zip(values, scores):
                            if v not in value_scores:
                                value_scores[v] = []
                            value_scores[v].append(s)
                            
                        avg_scores = {v: sum(s)/len(s) for v, s in value_scores.items()}
                        param_impacts[param] = {
                            'type': 'categorical',
                            'value_scores': avg_scores,
                            'best_value': max(avg_scores, key=avg_scores.get)
                        }
                        
            report['parameter_impacts'] = param_impacts
            
        return report
        
    def export_to_dataframe(self) -> pd.DataFrame:
        """导出为DataFrame"""
        trials = self.get_trials()
        
        if not trials:
            return pd.DataFrame()
            
        # 展开配置参数
        rows = []
        for trial in trials:
            row = {
                'trial_id': trial['trial_id'],
                'score': trial['score'],
                'status': trial['status'],
                'duration': trial['duration']
            }
            
            # 添加配置参数
            if trial['config']:
                for k, v in trial['config'].items():
                    row[f'param_{k}'] = v
                    
            # 添加度量指标
            if trial.get('metrics'):
                for k, v in trial['metrics'].items():
                    row[f'metric_{k}'] = v
                    
            rows.append(row)
            
        return pd.DataFrame(rows)
        
    def plot_optimization_history(self, save_path: Optional[str] = None):
        """绘制优化历史"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告：matplotlib未安装，无法绘图")
            return
            
        trials = self.get_trials(status='completed')
        
        if not trials:
            print("没有完成的试验")
            return
            
        # 提取数据
        trial_ids = []
        scores = []
        best_scores = []
        
        current_best = None
        for trial in sorted(trials, key=lambda t: t['trial_id']):
            if trial['score'] is not None:
                trial_ids.append(trial['trial_id'])
                scores.append(trial['score'])
                
                if current_best is None or trial['score'] > current_best:
                    current_best = trial['score']
                best_scores.append(current_best)
                
        # 绘图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 分数历史
        ax1.scatter(trial_ids, scores, alpha=0.6, label='Trial scores')
        ax1.plot(trial_ids, best_scores, 'r-', linewidth=2, label='Best score')
        ax1.set_xlabel('Trial ID')
        ax1.set_ylabel('Score')
        ax1.set_title('Optimization History')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 分数分布
        ax2.hist(scores, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(max(scores), color='red', linestyle='--', 
                   label=f'Best: {max(scores):.4f}')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Count')
        ax2.set_title('Score Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.experiment_dir, 'optimization_history.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
        
    def plot_parameter_importance(self, save_path: Optional[str] = None):
        """绘制参数重要性"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("警告：matplotlib未安装，无法绘图")
            return
            
        report = self.create_report()
        
        if 'parameter_impacts' not in report:
            print("没有参数影响分析数据")
            return
            
        # 提取数值参数的相关性
        param_names = []
        correlations = []
        
        for param, impact in report['parameter_impacts'].items():
            if impact['type'] == 'numeric' and not pd.isna(impact['correlation']):
                param_names.append(param)
                correlations.append(abs(impact['correlation']))
                
        if not param_names:
            print("没有数值参数的相关性数据")
            return
            
        # 排序
        sorted_idx = sorted(range(len(correlations)), 
                          key=lambda i: correlations[i], 
                          reverse=True)
        param_names = [param_names[i] for i in sorted_idx]
        correlations = [correlations[i] for i in sorted_idx]
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        y_pos = range(len(param_names))
        ax.barh(y_pos, correlations, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(param_names)
        ax.set_xlabel('Absolute Correlation with Score')
        ax.set_title('Parameter Importance (Numeric Parameters)')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 添加数值标签
        for i, v in enumerate(correlations):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            save_path = os.path.join(self.experiment_dir, 'parameter_importance.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.close()
        
    def compare_experiments(self, other_experiments: List[str]) -> pd.DataFrame:
        """比较多个实验"""
        all_experiments = [self.experiment_name] + other_experiments
        comparison_data = []
        
        for exp_name in all_experiments:
            # 加载其他实验
            if exp_name == self.experiment_name:
                tracker = self
            else:
                tracker = ExperimentTracker(
                    experiment_name=exp_name,
                    base_dir=self.base_dir,
                    backend=self.backend
                )
                
            # 获取统计信息
            report = tracker.create_report()
            
            comparison_data.append({
                'experiment': exp_name,
                'total_trials': report.get('total_trials', 0),
                'success_rate': report.get('success_rate', 0),
                'best_score': report.get('best_score', None),
                'mean_score': report.get('mean_score', None),
                'std_score': report.get('std_score', None),
                'total_duration': report.get('total_duration', None)
            })
            
        return pd.DataFrame(comparison_data)
        
    def cleanup(self, keep_best_n: int = 10):
        """清理实验数据，只保留最好的N个试验"""
        trials = self.get_trials(status='completed')
        
        if len(trials) <= keep_best_n:
            return
            
        # 按分数排序
        sorted_trials = sorted(
            trials, 
            key=lambda t: t['score'] if t['score'] is not None else float('-inf'),
            reverse=True
        )
        
        # 保留最好的N个
        keep_trial_ids = {t['trial_id'] for t in sorted_trials[:keep_best_n]}
        
        if self.backend == "file":
            # 删除不需要的文件
            for filename in os.listdir(self.experiment_dir):
                if filename.startswith('trial_'):
                    trial_id = int(filename.split('_')[1].split('.')[0])
                    if trial_id not in keep_trial_ids:
                        os.remove(os.path.join(self.experiment_dir, filename))
                        
        elif self.backend == "sqlite":
            # 删除数据库记录
            trial_ids_str = ','.join(str(tid) for tid in keep_trial_ids)
            self.conn.execute(
                f"DELETE FROM trials WHERE trial_id NOT IN ({trial_ids_str})"
            )
            self.conn.commit()
            
        elif self.backend == "memory":
            # 过滤内存中的数据
            self.memory_store['trials'] = [
                t for t in self.memory_store['trials'] 
                if t['trial_id'] in keep_trial_ids
            ]
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.backend == "sqlite":
            self.conn.close()