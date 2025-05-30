import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hyperopt import (
    ParameterSpace, 
    HyperOptimizer,
    RandomSampler,
    ConstraintManager,
    ExperimentTracker
)

# 1. 定义参数空间
space = ParameterSpace()
space.add_continuous('learning_rate', 0.0001, 0.1, scale='log')
space.add_discrete('latent_factors', 10, 100, step=10)
space.add_continuous('lambda_reg', 0.001, 1.0)
space.add_continuous('delta1', 0.1, 2.0)
space.add_continuous('delta2', 0.5, 5.0)
space.add_categorical('loss_type', ['hpl', 'l2', 'l1', 'huber'])

# 2. 添加约束
constraints = ConstraintManager()
constraints.add_relation('delta1', 'delta2', '<')  # δ1 < δ2

# 3. 定义目标函数
def objective(config):
    # 训练模型并返回验证集分数
    model = train_model(**config)
    score = evaluate_model(model)
    return score

# 4. 创建优化器
optimizer = HyperOptimizer(
    objective_fn=objective,
    space=space,
    sampler=RandomSampler(space),
    constraints=constraints,
    tracker=ExperimentTracker('hpl_optimization'),
    maximize=False  # 最小化损失
)

# 5. 运行优化
best_trial = optimizer.optimize(
    n_trials=100,
    batch_size=4,  # 并行评估4个配置
    no_improvement_rounds=20
)

print(f"最佳配置: {best_trial.config}")
print(f"最佳分数: {best_trial.score}")
