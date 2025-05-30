"""
工作流管理模块

定义和执行实验工作流
"""

import time
import json
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import threading
import queue
from datetime import datetime
import traceback

from .config_manager import ConfigManager, ExperimentConfig
from .experiment_runner import ExperimentRunner, BatchRunner
from .baseline_manager import BaselineManager
from .results_analyzer import ResultsAnalyzer
from .significance_test import SignificanceTest
from .report_generator import ReportGenerator


class StepStatus(Enum):
    """步骤状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """工作流步骤"""
    name: str
    function: Callable
    inputs: Dict[str, Any] = field(default_factory=dict)
    outputs: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # 运行时信息
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
        
    def can_run(self, completed_steps: List[str]) -> bool:
        """检查是否可以运行"""
        return all(dep in completed_steps for dep in self.dependencies)


class ExperimentWorkflow:
    """实验工作流"""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.steps: Dict[str, WorkflowStep] = {}
        self.context: Dict[str, Any] = {}
        
    def add_step(self, step: WorkflowStep):
        """添加步骤"""
        if step.name in self.steps:
            raise ValueError(f"步骤已存在: {step.name}")
            
        self.steps[step.name] = step
        
    def add_steps(self, steps: List[WorkflowStep]):
        """批量添加步骤"""
        for step in steps:
            self.add_step(step)
            
    def validate(self) -> List[str]:
        """验证工作流"""
        errors = []
        
        # 检查依赖关系
        all_steps = set(self.steps.keys())
        
        for step_name, step in self.steps.items():
            for dep in step.dependencies:
                if dep not in all_steps:
                    errors.append(f"步骤 {step_name} 依赖不存在的步骤: {dep}")
                    
        # 检查循环依赖
        if self._has_circular_dependency():
            errors.append("存在循环依赖")
            
        return errors
        
    def _has_circular_dependency(self) -> bool:
        """检查循环依赖"""
        visited = set()
        rec_stack = set()
        
        def _visit(step_name: str) -> bool:
            visited.add(step_name)
            rec_stack.add(step_name)
            
            step = self.steps.get(step_name)
            if step:
                for dep in step.dependencies:
                    if dep not in visited:
                        if _visit(dep):
                            return True
                    elif dep in rec_stack:
                        return True
                        
            rec_stack.remove(step_name)
            return False
            
        for step_name in self.steps:
            if step_name not in visited:
                if _visit(step_name):
                    return True
                    
        return False
        
    def get_execution_order(self) -> List[str]:
        """获取执行顺序（拓扑排序）"""
        # 计算入度
        in_degree = {name: 0 for name in self.steps}
        
        for step in self.steps.values():
            for dep in step.dependencies:
                if dep in in_degree:
                    in_degree[dep] += 1
                    
        # 拓扑排序
        queue = [name for name, degree in in_degree.items() if degree == 0]
        order = []
        
        while queue:
            current = queue.pop(0)
            order.append(current)
            
            # 更新依赖此步骤的其他步骤
            for name, step in self.steps.items():
                if current in step.dependencies:
                    in_degree[name] -= 1
                    if in_degree[name] == 0:
                        queue.append(name)
                        
        return order
        
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'description': self.description,
            'steps': {
                name: {
                    'function': step.function.__name__,
                    'inputs': step.inputs,
                    'outputs': step.outputs,
                    'dependencies': step.dependencies,
                    'status': step.status.value
                }
                for name, step in self.steps.items()
            }
        }


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, 
                 max_parallel: int = 1,
                 checkpoint_dir: Optional[str] = None):
        self.max_parallel = max_parallel
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
    def execute(self, 
                workflow: ExperimentWorkflow,
                resume_from: Optional[str] = None) -> Dict[str, Any]:
        """执行工作流"""
        # 验证工作流
        errors = workflow.validate()
        if errors:
            raise ValueError(f"工作流验证失败:\n" + "\n".join(errors))
            
        # 恢复检查点
        if resume_from and self.checkpoint_dir:
            self._load_checkpoint(workflow, resume_from)
            
        # 执行
        if self.max_parallel > 1:
            return self._execute_parallel(workflow)
        else:
            return self._execute_sequential(workflow)
            
    def _execute_sequential(self, workflow: ExperimentWorkflow) -> Dict[str, Any]:
        """顺序执行"""
        execution_order = workflow.get_execution_order()
        completed_steps = []
        
        print(f"开始执行工作流: {workflow.name}")
        print(f"步骤数: {len(execution_order)}")
        
        for step_name in execution_order:
            step = workflow.steps[step_name]
            
            # 检查是否已完成
            if step.status == StepStatus.COMPLETED:
                completed_steps.append(step_name)
                continue
                
            # 检查依赖
            if not step.can_run(completed_steps):
                print(f"跳过步骤 {step_name}: 依赖未满足")
                step.status = StepStatus.SKIPPED
                continue
                
            # 执行步骤
            print(f"\n执行步骤: {step_name}")
            step.status = StepStatus.RUNNING
            step.start_time = time.time()
            
            try:
                # 准备输入
                inputs = self._prepare_inputs(step, workflow.context)
                
                # 执行
                result = step.function(**inputs)
                
                # 保存结果
                step.result = result
                step.status = StepStatus.COMPLETED
                
                # 更新上下文
                if step.outputs:
                    for output_name in step.outputs:
                        workflow.context[output_name] = result
                        
                completed_steps.append(step_name)
                
            except Exception as e:
                step.status = StepStatus.FAILED
                step.error = str(e)
                print(f"步骤失败: {step_name}")
                print(f"错误: {e}")
                traceback.print_exc()
                
                # 保存检查点
                if self.checkpoint_dir:
                    self._save_checkpoint(workflow, f"failed_{step_name}")
                    
                raise
                
            finally:
                step.end_time = time.time()
                
            print(f"步骤完成: {step_name} (耗时: {step.duration:.2f}秒)")
            
            # 定期保存检查点
            if self.checkpoint_dir and len(completed_steps) % 5 == 0:
                self._save_checkpoint(workflow, f"step_{len(completed_steps)}")
                
        print(f"\n工作流执行完成: {workflow.name}")
        
        return workflow.context
        
    def _execute_parallel(self, workflow: ExperimentWorkflow) -> Dict[str, Any]:
        """并行执行"""
        import concurrent.futures
        
        completed_steps = []
        running_steps = set()
        step_futures = {}
        
        print(f"开始并行执行工作流: {workflow.name}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            while True:
                # 找出可以执行的步骤
                ready_steps = []
                for name, step in workflow.steps.items():
                    if (name not in completed_steps and 
                        name not in running_steps and
                        step.status != StepStatus.COMPLETED and
                        step.can_run(completed_steps)):
                        ready_steps.append(name)
                        
                # 提交新任务
                for step_name in ready_steps:
                    step = workflow.steps[step_name]
                    print(f"提交步骤: {step_name}")
                    
                    future = executor.submit(self._execute_step, step, workflow.context)
                    step_futures[future] = step_name
                    running_steps.add(step_name)
                    
                # 等待任务完成
                if step_futures:
                    done, pending = concurrent.futures.wait(
                        step_futures.keys(),
                        return_when=concurrent.futures.FIRST_COMPLETED
                    )
                    
                    for future in done:
                        step_name = step_futures.pop(future)
                        step = workflow.steps[step_name]
                        
                        try:
                            result = future.result()
                            step.result = result
                            step.status = StepStatus.COMPLETED
                            
                            # 更新上下文
                            if step.outputs:
                                for output_name in step.outputs:
                                    workflow.context[output_name] = result
                                    
                            completed_steps.append(step_name)
                            print(f"步骤完成: {step_name}")
                            
                        except Exception as e:
                            step.status = StepStatus.FAILED
                            step.error = str(e)
                            print(f"步骤失败: {step_name}")
                            print(f"错误: {e}")
                            
                        finally:
                            running_steps.remove(step_name)
                            
                # 检查是否完成
                if len(completed_steps) == len(workflow.steps):
                    break
                    
                # 检查是否有步骤无法执行
                if not ready_steps and not running_steps:
                    print("警告: 存在无法执行的步骤")
                    break
                    
        print(f"\n工作流执行完成: {workflow.name}")
        
        return workflow.context
        
    def _execute_step(self, step: WorkflowStep, context: Dict) -> Any:
        """执行单个步骤"""
        step.status = StepStatus.RUNNING
        step.start_time = time.time()
        
        try:
            # 准备输入
            inputs = self._prepare_inputs(step, context)
            
            # 执行
            result = step.function(**inputs)
            
            return result
            
        finally:
            step.end_time = time.time()
            
    def _prepare_inputs(self, step: WorkflowStep, context: Dict) -> Dict:
        """准备步骤输入"""
        inputs = {}
        
        for key, value in step.inputs.items():
            if isinstance(value, str) and value.startswith("$"):
                # 从上下文获取
                context_key = value[1:]
                if context_key in context:
                    inputs[key] = context[context_key]
                else:
                    raise ValueError(f"上下文中找不到: {context_key}")
            else:
                inputs[key] = value
                
        return inputs
        
    def _save_checkpoint(self, workflow: ExperimentWorkflow, name: str):
        """保存检查点"""
        checkpoint_file = self.checkpoint_dir / f"{workflow.name}_{name}.json"
        
        checkpoint_data = {
            'workflow': workflow.to_dict(),
            'context': workflow.context,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
    def _load_checkpoint(self, workflow: ExperimentWorkflow, checkpoint_name: str):
        """加载检查点"""
        checkpoint_file = self.checkpoint_dir / f"{workflow.name}_{checkpoint_name}.json"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点不存在: {checkpoint_name}")
            
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            
        # 恢复上下文
        workflow.context = checkpoint_data['context']
        
        # 恢复步骤状态
        for step_name, step_data in checkpoint_data['workflow']['steps'].items():
            if step_name in workflow.steps:
                workflow.steps[step_name].status = StepStatus(step_data['status'])


class WorkflowMonitor:
    """工作流监控器"""
    
    def __init__(self):
        self.active_workflows: Dict[str, Dict] = {}
        self.event_queue = queue.Queue()
        self.monitor_thread = None
        self.running = False
        
    def start(self):
        """启动监控"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def register_workflow(self, workflow_id: str, workflow: ExperimentWorkflow):
        """注册工作流"""
        self.active_workflows[workflow_id] = {
            'workflow': workflow,
            'start_time': time.time(),
            'status': 'running',
            'progress': 0.0
        }
        
    def update_progress(self, workflow_id: str, progress: float):
        """更新进度"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]['progress'] = progress
            
    def complete_workflow(self, workflow_id: str):
        """完成工作流"""
        if workflow_id in self.active_workflows:
            self.active_workflows[workflow_id]['status'] = 'completed'
            self.active_workflows[workflow_id]['end_time'] = time.time()
            
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            # 更新状态
            for workflow_id, info in self.active_workflows.items():
                workflow = info['workflow']
                
                # 计算进度
                completed = sum(1 for s in workflow.steps.values() 
                              if s.status == StepStatus.COMPLETED)
                total = len(workflow.steps)
                progress = completed / total if total > 0 else 0
                
                info['progress'] = progress
                
                # 发送事件
                event = {
                    'type': 'progress_update',
                    'workflow_id': workflow_id,
                    'progress': progress,
                    'timestamp': time.time()
                }
                self.event_queue.put(event)
                
            time.sleep(1)  # 每秒更新一次
            
    def get_status(self) -> Dict:
        """获取所有工作流状态"""
        status = {}
        
        for workflow_id, info in self.active_workflows.items():
            workflow = info['workflow']
            
            # 统计步骤状态
            step_stats = {
                'total': len(workflow.steps),
                'completed': 0,
                'running': 0,
                'failed': 0,
                'pending': 0
            }
            
            for step in workflow.steps.values():
                if step.status == StepStatus.COMPLETED:
                    step_stats['completed'] += 1
                elif step.status == StepStatus.RUNNING:
                    step_stats['running'] += 1
                elif step.status == StepStatus.FAILED:
                    step_stats['failed'] += 1
                else:
                    step_stats['pending'] += 1
                    
            status[workflow_id] = {
                'name': workflow.name,
                'status': info['status'],
                'progress': info['progress'],
                'step_stats': step_stats,
                'duration': time.time() - info['start_time']
            }
            
        return status


# 预定义工作流模板
class WorkflowTemplates:
    """工作流模板"""
    
    @staticmethod
    def create_full_experiment_workflow(
        config_manager: ConfigManager,
        experiment_runner: ExperimentRunner,
        baseline_manager: BaselineManager,
        analyzer: ResultsAnalyzer,
        report_generator: ReportGenerator
    ) -> ExperimentWorkflow:
        """创建完整实验工作流"""
        workflow = ExperimentWorkflow(
            name="full_experiment",
            description="完整的实验评估流程"
        )
        
        # 1. 准备实验配置
        def prepare_configs(dataset: str, n_trials: int = 5):
            configs = []
            
            # HPL配置
            for i in range(n_trials):
                config = config_manager.create_config(
                    name=f"hpl_{dataset}_trial{i}",
                    dataset=dataset,
                    loss_type="hpl",
                    seed=42 + i
                )
                configs.append(config)
                
            return configs
            
        # 2. 运行基线实验
        def run_baselines(dataset: str):
            baseline_names = ["MF-L2", "MF-L1", "MF-Huber", "MF-Logcosh"]
            results = baseline_manager.run_all_baselines(dataset, baseline_names)
            return results
            
        # 3. 运行主实验
        def run_main_experiments(configs: List[ExperimentConfig]):
            batch_runner = BatchRunner(experiment_runner)
            batch_runner.add_experiments_from_configs(configs)
            results = batch_runner.run_all()
            return results
            
        # 4. 分析结果
        def analyze_results(main_results, baseline_results):
            # 比较分析
            all_results = main_results + baseline_results
            comparison = analyzer.compare_experiments(
                [r.experiment_id for r in all_results]
            )
            
            # 统计检验
            sig_test = SignificanceTest()
            test_results = {}
            
            # HPL vs 各基线
            hpl_scores = [r.test_results['mae'] for r in main_results]
            
            for baseline in baseline_results:
                baseline_scores = [baseline.run_result.test_results['mae']]
                test_result = sig_test.t_test(
                    hpl_scores,
                    baseline_scores * len(hpl_scores),  # 扩展到相同长度
                    paired=False
                )
                test_results[baseline.baseline_name] = test_result
                
            return {
                'comparison': comparison,
                'significance_tests': test_results
            }
            
        # 5. 生成报告
        def generate_report(analysis_results):
            report_path = report_generator.generate_full_report(
                experiment_results=analysis_results['comparison'],
                comparisons=[],
                analyses=analysis_results,
                format='html'
            )
            return report_path
            
        # 添加步骤
        workflow.add_steps([
            WorkflowStep(
                name="prepare_configs",
                function=prepare_configs,
                inputs={"dataset": "ml-100k", "n_trials": 5},
                outputs=["configs"]
            ),
            WorkflowStep(
                name="run_baselines",
                function=run_baselines,
                inputs={"dataset": "ml-100k"},
                outputs=["baseline_results"]
            ),
            WorkflowStep(
                name="run_main_experiments",
                function=run_main_experiments,
                inputs={"configs": "$configs"},
                outputs=["main_results"],
                dependencies=["prepare_configs"]
            ),
            WorkflowStep(
                name="analyze_results",
                function=analyze_results,
                inputs={
                    "main_results": "$main_results",
                    "baseline_results": "$baseline_results"
                },
                outputs=["analysis"],
                dependencies=["run_main_experiments", "run_baselines"]
            ),
            WorkflowStep(
                name="generate_report",
                function=generate_report,
                inputs={"analysis_results": "$analysis"},
                outputs=["report_path"],
                dependencies=["analyze_results"]
            )
        ])
        
        return workflow
        
    @staticmethod
    def create_hyperparameter_search_workflow(
        config_manager: ConfigManager,
        experiment_runner: ExperimentRunner,
        hyperopt_module
    ) -> ExperimentWorkflow:
        """创建超参数搜索工作流"""
        workflow = ExperimentWorkflow(
            name="hyperparameter_search",
            description="超参数优化工作流"
        )
        
        # 定义步骤函数
        def define_search_space():
            from hyperopt import ParameterSpace
            
            space = ParameterSpace()
            space.add_continuous('learning_rate', 0.0001, 0.1, scale='log')
            space.add_discrete('latent_factors', 10, 100, step=10)
            space.add_continuous('lambda_reg', 0.001, 1.0)
            space.add_continuous('delta1', 0.1, 2.0)
            space.add_continuous('delta2', 0.5, 5.0)
            
            return space
            
        def run_optimization(space, n_trials: int = 100):
            from hyperopt import HyperOptimizer, RandomSampler, ConstraintManager
            
            # 定义目标函数
            def objective(config_dict):
                config = config_manager.create_config(
                    name="hpo_trial",
                    **config_dict
                )
                result = experiment_runner.run_experiment(config)
                return result.test_results['mae']
                
            # 添加约束
            constraints = ConstraintManager()
            constraints.add_relation('delta1', 'delta2', '<')
            
            # 创建优化器
            optimizer = HyperOptimizer(
                objective_fn=objective,
                space=space,
                sampler=RandomSampler(space),
                constraints=constraints,
                maximize=False
            )
            
            # 运行优化
            best_trial = optimizer.optimize(n_trials=n_trials)
            
            return {
                'best_config': best_trial.config,
                'best_score': best_trial.score,
                'all_trials': optimizer.get_history()
            }
            
        # 添加步骤
        workflow.add_steps([
            WorkflowStep(
                name="define_space",
                function=define_search_space,
                outputs=["search_space"]
            ),
            WorkflowStep(
                name="optimize",
                function=run_optimization,
                inputs={"space": "$search_space", "n_trials": 100},
                outputs=["optimization_result"],
                dependencies=["define_space"]
            )
        ])
        
        return workflow