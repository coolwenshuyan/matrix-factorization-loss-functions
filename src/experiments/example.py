# 创建完整的实验工作流
from experiments import *

# 初始化各个组件
config_manager = ConfigManager()
experiment_runner = ExperimentRunner(train_fn, eval_fn)
baseline_manager = BaselineManager(config_manager, experiment_runner)
analyzer = ResultsAnalyzer()
report_generator = ReportGenerator()

# 创建工作流
workflow = WorkflowTemplates.create_full_experiment_workflow(
    config_manager,
    experiment_runner,
    baseline_manager,
    analyzer,
    report_generator
)

# 执行工作流
executor = WorkflowExecutor(max_parallel=4)
results = executor.execute(workflow)

# 生成报告
report_path = results['report_path']
print(f"实验报告: {report_path}")