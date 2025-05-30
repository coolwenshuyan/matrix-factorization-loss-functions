"""
实验运行器模块

管理实验的执行、监控和结果收集
"""

import os
import time
import json
import pickle
import traceback
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import psutil
import GPUtil

from .config_manager import ExperimentConfig
from .reproducibility import ReproducibilityManager


class ExperimentStatus(Enum):
    """实验状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


@dataclass
class RunResult:
    """运行结果"""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus
    start_time: float
    end_time: Optional[float] = None
    
    # 训练结果
    train_history: Dict = field(default_factory=dict)
    best_epoch: Optional[int] = None
    best_val_score: Optional[float] = None
    
    # 评估结果
    test_results: Dict = field(default_factory=dict)
    predictions: Optional[Any] = None
    
    # 系统信息
    system_info: Dict = field(default_factory=dict)
    resource_usage: Dict = field(default_factory=dict)
    
    # 错误信息
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None
    
    # 文件路径
    checkpoint_path: Optional[str] = None
    log_path: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'experiment_id': self.experiment_id,
            'config': self.config.to_dict(),
            'status': self.status.value,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'train_history': self.train_history,
            'best_epoch': self.best_epoch,
            'best_val_score': self.best_val_score,
            'test_results': self.test_results,
            'system_info': self.system_info,
            'resource_usage': self.resource_usage,
            'error_message': self.error_message,
            'checkpoint_path': self.checkpoint_path,
            'log_path': self.log_path
        }


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self,
                 train_fn: Callable,
                 eval_fn: Callable,
                 output_dir: str = "./results",
                 use_gpu: bool = True,
                 gpu_memory_fraction: float = 0.9):
        """
        Args:
            train_fn: 训练函数
            eval_fn: 评估函数
            output_dir: 输出目录
            use_gpu: 是否使用GPU
            gpu_memory_fraction: GPU内存使用比例
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_gpu = use_gpu
        self.gpu_memory_fraction = gpu_memory_fraction
        
        self.reproducibility_manager = ReproducibilityManager()
        self.running_experiments = {}
        
    def run_experiment(self, 
                      config: ExperimentConfig,
                      resume_from: Optional[str] = None) -> RunResult:
        """运行单个实验"""
        # 生成实验ID
        experiment_id = f"{config.name}_{config.get_hash()}_{int(time.time())}"
        
        # 创建实验目录
        exp_dir = self.output_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)
        
        # 初始化结果
        result = RunResult(
            experiment_id=experiment_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            start_time=time.time()
        )
        
        # 记录系统信息
        result.system_info = self._get_system_info()
        
        # 设置随机种子
        self.reproducibility_manager.set_all_seeds(config.seed)
        
        try:
            # 保存配置
            config_path = exp_dir / "config.yaml"
            with open(config_path, 'w') as f:
                import yaml
                yaml.dump(config.to_dict(), f)
            
            # 设置日志
            log_path = exp_dir / "experiment.log"
            result.log_path = str(log_path)
            
            # 运行训练
            print(f"开始训练实验: {experiment_id}")
            train_result = self._run_training(config, exp_dir, resume_from)
            
            result.train_history = train_result['history']
            result.best_epoch = train_result['best_epoch']
            result.best_val_score = train_result['best_val_score']
            result.checkpoint_path = train_result['checkpoint_path']
            
            # 运行评估
            print(f"开始评估实验: {experiment_id}")
            eval_result = self._run_evaluation(
                config, 
                result.checkpoint_path,
                exp_dir
            )
            
            result.test_results = eval_result['metrics']
            result.predictions = eval_result.get('predictions')
            
            # 记录资源使用
            result.resource_usage = self._get_resource_usage()
            
            # 标记完成
            result.status = ExperimentStatus.COMPLETED
            result.end_time = time.time()
            
            print(f"实验完成: {experiment_id}")
            print(f"耗时: {result.duration:.2f}秒")
            print(f"测试结果: {result.test_results}")
            
        except KeyboardInterrupt:
            print(f"\n实验被中断: {experiment_id}")
            result.status = ExperimentStatus.CANCELLED
            result.end_time = time.time()
            
        except Exception as e:
            print(f"\n实验失败: {experiment_id}")
            print(f"错误: {str(e)}")
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            result.error_traceback = traceback.format_exc()
            result.end_time = time.time()
            
        finally:
            # 保存结果
            self._save_result(result, exp_dir)
            
        return result
    
    def _run_training(self, 
                     config: ExperimentConfig,
                     exp_dir: Path,
                     resume_from: Optional[str]) -> Dict:
        """运行训练"""
        # 准备训练参数
        train_args = {
            'config': config,
            'output_dir': str(exp_dir),
            'resume_from': resume_from,
            'device': self._get_device(config),
            'use_tensorboard': config.tensorboard,
            'log_interval': config.log_interval,
            'checkpoint_interval': config.checkpoint_interval
        }
        
        # 调用训练函数
        result = self.train_fn(**train_args)
        
        return result
    
    def _run_evaluation(self,
                       config: ExperimentConfig,
                       checkpoint_path: str,
                       exp_dir: Path) -> Dict:
        """运行评估"""
        # 准备评估参数
        eval_args = {
            'config': config,
            'checkpoint_path': checkpoint_path,
            'output_dir': str(exp_dir),
            'device': self._get_device(config),
            'metrics': config.eval_metrics,
            'top_k': config.top_k,
            'save_predictions': True
        }
        
        # 调用评估函数
        result = self.eval_fn(**eval_args)
        
        return result
    
    def _get_device(self, config: ExperimentConfig) -> str:
        """获取设备"""
        if config.device == "cuda" and self.use_gpu:
            import torch
            if torch.cuda.is_available():
                # 选择空闲的GPU
                gpus = GPUtil.getGPUs()
                if gpus:
                    # 按内存使用率排序
                    gpus.sort(key=lambda x: x.memoryUsed)
                    return f"cuda:{gpus[0].id}"
            return "cpu"
        return config.device
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        import platform
        
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'cpu_count': mp.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3)
        }
        
        # GPU信息
        if self.use_gpu:
            gpus = GPUtil.getGPUs()
            if gpus:
                info['gpu_count'] = len(gpus)
                info['gpu_info'] = []
                for gpu in gpus:
                    info['gpu_info'].append({
                        'name': gpu.name,
                        'memory_total': gpu.memoryTotal,
                        'memory_used': gpu.memoryUsed,
                        'memory_free': gpu.memoryFree,
                        'temperature': gpu.temperature
                    })
                    
        return info
    
    def _get_resource_usage(self) -> Dict:
        """获取资源使用情况"""
        usage = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_used_gb': psutil.virtual_memory().used / (1024**3)
        }
        
        # GPU使用情况
        if self.use_gpu:
            gpus = GPUtil.getGPUs()
            if gpus:
                usage['gpu_usage'] = []
                for gpu in gpus:
                    usage['gpu_usage'].append({
                        'id': gpu.id,
                        'load': gpu.load * 100,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100
                    })
                    
        return usage
    
    def _save_result(self, result: RunResult, exp_dir: Path):
        """保存实验结果"""
        # 保存为JSON
        result_dict = result.to_dict()
        json_path = exp_dir / "result.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
            
        # 保存为pickle（包含完整对象）
        pkl_path = exp_dir / "result.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(result, f)
            
        # 保存简要摘要
        summary_path = exp_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"实验ID: {result.experiment_id}\n")
            f.write(f"状态: {result.status.value}\n")
            f.write(f"耗时: {result.duration:.2f}秒\n")
            f.write(f"最佳验证分数: {result.best_val_score}\n")
            f.write(f"测试结果:\n")
            for metric, value in result.test_results.items():
                f.write(f"  {metric}: {value}\n")


class BatchRunner:
    """批量实验运行器"""
    
    def __init__(self,
                 runner: ExperimentRunner,
                 max_parallel: int = 1,
                 use_gpu_queue: bool = True):
        """
        Args:
            runner: 实验运行器
            max_parallel: 最大并行数
            use_gpu_queue: 是否使用GPU队列
        """
        self.runner = runner
        self.max_parallel = max_parallel
        self.use_gpu_queue = use_gpu_queue
        
        self.experiment_queue = []
        self.running_experiments = {}
        self.completed_experiments = []
        self.failed_experiments = []
        
    def add_experiment(self, 
                      config: ExperimentConfig,
                      priority: int = 0,
                      dependencies: Optional[List[str]] = None):
        """添加实验到队列"""
        experiment = {
            'config': config,
            'priority': priority,
            'dependencies': dependencies or [],
            'status': ExperimentStatus.PENDING,
            'id': f"{config.name}_{config.get_hash()}"
        }
        
        self.experiment_queue.append(experiment)
        
        # 按优先级排序
        self.experiment_queue.sort(key=lambda x: x['priority'], reverse=True)
        
    def add_experiments_from_configs(self, configs: List[ExperimentConfig]):
        """批量添加实验"""
        for config in configs:
            self.add_experiment(config)
            
    def run_all(self, 
                continue_on_failure: bool = True,
                save_interval: int = 10) -> List[RunResult]:
        """运行所有实验"""
        results = []
        
        if self.max_parallel == 1:
            # 串行执行
            results = self._run_serial(continue_on_failure)
        else:
            # 并行执行
            results = self._run_parallel(continue_on_failure, save_interval)
            
        # 生成批量运行报告
        self._generate_batch_report(results)
        
        return results
    
    def _run_serial(self, continue_on_failure: bool) -> List[RunResult]:
        """串行执行实验"""
        results = []
        
        while self.experiment_queue:
            experiment = self._get_next_experiment()
            if not experiment:
                break
                
            print(f"\n{'='*60}")
            print(f"运行实验 {len(results)+1}/{len(self.experiment_queue)+len(results)+1}")
            print(f"{'='*60}\n")
            
            # 运行实验
            result = self.runner.run_experiment(experiment['config'])
            results.append(result)
            
            # 更新状态
            if result.status == ExperimentStatus.COMPLETED:
                self.completed_experiments.append(experiment)
            else:
                self.failed_experiments.append(experiment)
                if not continue_on_failure:
                    print("实验失败，停止批量运行")
                    break
                    
        return results
    
    def _run_parallel(self, 
                     continue_on_failure: bool,
                     save_interval: int) -> List[RunResult]:
        """并行执行实验"""
        results = []
        
        with ProcessPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}
            
            while self.experiment_queue or futures:
                # 提交新实验
                while len(futures) < self.max_parallel and self.experiment_queue:
                    experiment = self._get_next_experiment()
                    if experiment:
                        future = executor.submit(
                            self.runner.run_experiment,
                            experiment['config']
                        )
                        futures[future] = experiment
                        self.running_experiments[experiment['id']] = experiment
                        
                # 等待完成
                if futures:
                    done_futures = []
                    for future in futures:
                        if future.done():
                            done_futures.append(future)
                            
                    # 处理完成的实验
                    for future in done_futures:
                        experiment = futures.pop(future)
                        self.running_experiments.pop(experiment['id'])
                        
                        try:
                            result = future.result()
                            results.append(result)
                            
                            if result.status == ExperimentStatus.COMPLETED:
                                self.completed_experiments.append(experiment)
                            else:
                                self.failed_experiments.append(experiment)
                                
                        except Exception as e:
                            print(f"实验执行异常: {experiment['id']}")
                            print(f"错误: {str(e)}")
                            self.failed_experiments.append(experiment)
                            
                            if not continue_on_failure:
                                # 取消所有运行中的实验
                                executor.shutdown(wait=False)
                                return results
                                
                # 定期保存进度
                if len(results) % save_interval == 0 and results:
                    self._save_progress(results)
                    
                time.sleep(1)  # 避免CPU占用过高
                
        return results
    
    def _get_next_experiment(self) -> Optional[Dict]:
        """获取下一个可运行的实验"""
        for i, experiment in enumerate(self.experiment_queue):
            # 检查依赖
            if self._check_dependencies(experiment):
                return self.experiment_queue.pop(i)
                
        return None
    
    def _check_dependencies(self, experiment: Dict) -> bool:
        """检查实验依赖是否满足"""
        for dep_id in experiment['dependencies']:
            # 检查依赖实验是否完成
            completed_ids = [e['id'] for e in self.completed_experiments]
            if dep_id not in completed_ids:
                return False
                
        return True
    
    def _save_progress(self, results: List[RunResult]):
        """保存运行进度"""
        progress_file = self.runner.output_dir / "batch_progress.json"
        
        progress = {
            'total': len(self.experiment_queue) + len(results),
            'completed': len(self.completed_experiments),
            'failed': len(self.failed_experiments),
            'running': len(self.running_experiments),
            'pending': len(self.experiment_queue),
            'results_summary': []
        }
        
        for result in results:
            progress['results_summary'].append({
                'experiment_id': result.experiment_id,
                'status': result.status.value,
                'duration': result.duration,
                'test_results': result.test_results
            })
            
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
            
    def _generate_batch_report(self, results: List[RunResult]):
        """生成批量运行报告"""
        report_file = self.runner.output_dir / "batch_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("批量实验运行报告\n")
            f.write("="*60 + "\n\n")
            
            # 总体统计
            f.write(f"总实验数: {len(results)}\n")
            f.write(f"成功: {len(self.completed_experiments)}\n")
            f.write(f"失败: {len(self.failed_experiments)}\n")
            
            total_time = sum(r.duration for r in results if r.duration)
            f.write(f"总耗时: {total_time:.2f}秒\n\n")
            
            # 成功实验摘要
            if self.completed_experiments:
                f.write("成功实验:\n")
                f.write("-"*60 + "\n")
                
                for exp in self.completed_experiments:
                    result = next(r for r in results 
                                if r.config.name == exp['config'].name)
                    
                    f.write(f"\n实验: {result.experiment_id}\n")
                    f.write(f"配置: {result.config.name}\n")
                    f.write(f"耗时: {result.duration:.2f}秒\n")
                    f.write(f"测试结果:\n")
                    
                    for metric, value in result.test_results.items():
                        f.write(f"  {metric}: {value}\n")
                        
            # 失败实验摘要
            if self.failed_experiments:
                f.write("\n失败实验:\n")
                f.write("-"*60 + "\n")
                
                for exp in self.failed_experiments:
                    result = next(r for r in results 
                                if r.config.name == exp['config'].name)
                    
                    f.write(f"\n实验: {result.experiment_id}\n")
                    f.write(f"错误: {result.error_message}\n")
                    
        print(f"\n批量运行报告已保存到: {report_file}")
        
    def get_status(self) -> Dict:
        """获取批量运行状态"""
        return {
            'total': len(self.experiment_queue) + len(self.completed_experiments) + 
                    len(self.failed_experiments) + len(self.running_experiments),
            'pending': len(self.experiment_queue),
            'running': len(self.running_experiments),
            'completed': len(self.completed_experiments),
            'failed': len(self.failed_experiments),
            'queue': [e['config'].name for e in self.experiment_queue],
            'running_experiments': list(self.running_experiments.keys())
        }