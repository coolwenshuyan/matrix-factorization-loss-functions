"""
优化器主类

管理整个超参数优化流程
"""

import time
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from .space import ParameterSpace
from .sampler import Sampler, RandomSampler
from .constraints import ConstraintManager
from .parallel import ParallelExecutor
from .tracker import ExperimentTracker


class OptimizationStatus(Enum):
    """优化状态"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Trial:
    """单次试验"""
    trial_id: int
    config: Dict
    score: Optional[float] = None
    metrics: Optional[Dict] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    status: str = "pending"
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class HyperOptimizer:
    """超参数优化器"""
    
    def __init__(self,
                 objective_fn: Callable[[Dict], float],
                 space: ParameterSpace,
                 sampler: Optional[Sampler] = None,
                 constraints: Optional[ConstraintManager] = None,
                 parallel_executor: Optional[ParallelExecutor] = None,
                 tracker: Optional[ExperimentTracker] = None,
                 maximize: bool = True,
                 seed: Optional[int] = None):
        """
        Args:
            objective_fn: 目标函数，输入配置，返回分数
            space: 参数空间
            sampler: 采样器
            constraints: 约束管理器
            parallel_executor: 并行执行器
            tracker: 实验追踪器
            maximize: 是否最大化目标
            seed: 随机种子
        """
        self.objective_fn = objective_fn
        self.space = space
        self.sampler = sampler or RandomSampler(space, seed)
        self.constraints = constraints or ConstraintManager()
        self.parallel_executor = parallel_executor
        self.tracker = tracker or ExperimentTracker()
        self.maximize = maximize
        self.seed = seed
        
        # 优化状态
        self.status = OptimizationStatus.INITIALIZED
        self.trials: List[Trial] = []
        self.best_trial: Optional[Trial] = None
        self.n_trials_completed = 0
        self.start_time = None
        self.iteration = 0
        
        # 停止条件
        self.max_trials = None
        self.max_time = None
        self.target_score = None
        self.no_improvement_rounds = None
        self._no_improvement_count = 0
        
    def optimize(self,
                n_trials: Optional[int] = None,
                max_time: Optional[float] = None,
                target_score: Optional[float] = None,
                no_improvement_rounds: Optional[int] = None,
                batch_size: int = 1,
                n_initial_points: Optional[int] = None):
        """
        执行优化
        
        Args:
            n_trials: 最大试验次数
            max_time: 最大运行时间（秒）
            target_score: 目标分数（达到即停止）
            no_improvement_rounds: 无改进轮数（达到即停止）
            batch_size: 批量大小（并行评估）
            n_initial_points: 初始随机点数
        """
        # 设置停止条件
        self.max_trials = n_trials
        self.max_time = max_time
        self.target_score = target_score
        self.no_improvement_rounds = no_improvement_rounds
        
        # 开始优化
        self.status = OptimizationStatus.RUNNING
        self.start_time = time.time()
        
        # 初始随机探索
        if n_initial_points and n_initial_points > 0:
            self._run_initial_points(n_initial_points)
            
        # 主优化循环
        try:
            while not self._should_stop():
                self._run_iteration(batch_size)
                
        except KeyboardInterrupt:
            print("\n优化被用户中断")
            self.status = OptimizationStatus.PAUSED
            
        except Exception as e:
            print(f"\n优化过程出错: {e}")
            self.status = OptimizationStatus.FAILED
            raise
            
        else:
            self.status = OptimizationStatus.COMPLETED
            
        # 保存最终结果
        self.tracker.save_final_results(self)
        
        return self.best_trial
        
    def _run_initial_points(self, n_points: int):
        """运行初始随机点"""
        print(f"运行 {n_points} 个初始随机点...")
        
        # 使用随机采样器
        random_sampler = RandomSampler(self.space, self.seed)
        
        for i in range(0, n_points, self.parallel_executor.n_workers if self.parallel_executor else 1):
            batch_size = min(
                self.parallel_executor.n_workers if self.parallel_executor else 1,
                n_points - i
            )
            
            # 采样配置
            configs = random_sampler.sample(batch_size)
            
            # 应用约束
            valid_configs = []
            for config in configs:
                fixed_config = self.constraints.fix_config(config)
                if fixed_config is not None:
                    valid_configs.append(fixed_config)
                    
            # 评估
            if valid_configs:
                self._evaluate_configs(valid_configs)
                
    def _run_iteration(self, batch_size: int):
        """运行一次迭代"""
        self.iteration += 1
        
        # 采样新配置
        configs = []
        attempts = 0
        max_attempts = batch_size * 10
        
        while len(configs) < batch_size and attempts < max_attempts:
            # 采样
            sampled = self.sampler.sample(1)[0]
            
            # 检查和修正约束
            fixed_config = self.constraints.fix_config(sampled)
            if fixed_config is not None:
                configs.append(fixed_config)
                
            attempts += 1
            
        if not configs:
            print("警告：无法生成满足约束的配置")
            return
            
        # 评估配置
        trials = self._evaluate_configs(configs)
        
        # 更新采样器
        if hasattr(self.sampler, 'update'):
            scores = [t.score for t in trials if t.score is not None]
            valid_configs = [t.config for t in trials if t.score is not None]
            if valid_configs:
                self.sampler.update(valid_configs, scores)
                
        # 检查是否有改进
        self._check_improvement()
        
        # 打印进度
        if self.iteration % 10 == 0:
            self._print_progress()
            
    def _evaluate_configs(self, configs: List[Dict]) -> List[Trial]:
        """评估配置"""
        trials = []
        
        for i, config in enumerate(configs):
            trial = Trial(
                trial_id=len(self.trials) + i,
                config=config,
                start_time=time.time()
            )
            trials.append(trial)
            
        # 并行或串行评估
        if self.parallel_executor and len(configs) > 1:
            # 并行评估
            results = self.parallel_executor.map(self.objective_fn, configs)
            
            for trial, result in zip(trials, results):
                if isinstance(result, Exception):
                    trial.status = "failed"
                    trial.error = str(result)
                else:
                    trial.score = result
                    trial.status = "completed"
                trial.end_time = time.time()
                
        else:
            # 串行评估
            for trial in trials:
                try:
                    trial.score = self.objective_fn(trial.config)
                    trial.status = "completed"
                except Exception as e:
                    trial.status = "failed"
                    trial.error = str(e)
                trial.end_time = time.time()
                
        # 记录试验
        for trial in trials:
            self.trials.append(trial)
            self.tracker.log_trial(trial)
            
            if trial.status == "completed":
                self.n_trials_completed += 1
                self._update_best_trial(trial)
                
        return trials
        
    def _update_best_trial(self, trial: Trial):
        """更新最佳试验"""
        if trial.score is None:
            return
            
        if self.best_trial is None:
            self.best_trial = trial
            self._no_improvement_count = 0
            
        else:
            is_better = (trial.score > self.best_trial.score) if self.maximize \
                       else (trial.score < self.best_trial.score)
                       
            if is_better:
                self.best_trial = trial
                self._no_improvement_count = 0
                print(f"\n发现更好的配置！分数: {trial.score:.6f}")
                
    def _check_improvement(self):
        """检查是否有改进"""
        if self.best_trial is None:
            return
            
        # 检查最近的试验
        recent_trials = [t for t in self.trials[-10:] if t.score is not None]
        if not recent_trials:
            return
            
        best_recent = max(recent_trials, key=lambda t: t.score) if self.maximize \
                     else min(recent_trials, key=lambda t: t.score)
                     
        if best_recent.score == self.best_trial.score:
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
            
    def _should_stop(self) -> bool:
        """检查是否应该停止"""
        # 达到最大试验数
        if self.max_trials and self.n_trials_completed >= self.max_trials:
            print(f"\n达到最大试验数 {self.max_trials}")
            return True
            
        # 超过最大时间
        if self.max_time and (time.time() - self.start_time) > self.max_time:
            print(f"\n达到最大运行时间 {self.max_time}秒")
            return True
            
        # 达到目标分数
        if self.target_score and self.best_trial:
            if self.maximize and self.best_trial.score >= self.target_score:
                print(f"\n达到目标分数 {self.target_score}")
                return True
            elif not self.maximize and self.best_trial.score <= self.target_score:
                print(f"\n达到目标分数 {self.target_score}")
                return True
                
        # 无改进轮数
        if self.no_improvement_rounds and \
           self._no_improvement_count >= self.no_improvement_rounds:
            print(f"\n连续 {self.no_improvement_rounds} 轮无改进")
            return True
            
        return False
        
    def _print_progress(self):
        """打印进度信息"""
        elapsed = time.time() - self.start_time
        
        print(f"\n--- 迭代 {self.iteration} ---")
        print(f"已完成试验: {self.n_trials_completed}")
        print(f"运行时间: {elapsed:.1f}秒")
        
        if self.best_trial:
            print(f"当前最佳分数: {self.best_trial.score:.6f}")
            print(f"最佳配置: {self.best_trial.config}")
            
        # 约束统计
        stats = self.constraints.get_statistics()
        if stats['n_constraints'] > 0:
            print(f"约束拒绝率: {stats['rejection_rate']:.2%}")
            
    def resume(self, additional_trials: Optional[int] = None):
        """恢复优化"""
        if self.status != OptimizationStatus.PAUSED:
            raise ValueError("只能恢复暂停的优化")
            
        print("恢复优化...")
        
        if additional_trials:
            if self.max_trials:
                self.max_trials += additional_trials
            else:
                self.max_trials = self.n_trials_completed + additional_trials
                
        self.optimize()
        
    def get_results(self) -> Dict:
        """获取优化结果"""
        if not self.trials:
            return {}
            
        completed_trials = [t for t in self.trials if t.status == "completed"]
        
        return {
            'best_config': self.best_trial.config if self.best_trial else None,
            'best_score': self.best_trial.score if self.best_trial else None,
            'n_trials': len(self.trials),
            'n_completed': len(completed_trials),
            'n_failed': len([t for t in self.trials if t.status == "failed"]),
            'total_time': time.time() - self.start_time if self.start_time else 0,
            'status': self.status.value,
            'all_trials': [
                {
                    'trial_id': t.trial_id,
                    'config': t.config,
                    'score': t.score,
                    'duration': t.duration
                }
                for t in completed_trials
            ]
        }
        
    def get_history(self) -> List[Tuple[Dict, float]]:
        """获取历史记录（配置，分数）"""
        return [
            (t.config, t.score)
            for t in self.trials
            if t.status == "completed" and t.score is not None
        ]