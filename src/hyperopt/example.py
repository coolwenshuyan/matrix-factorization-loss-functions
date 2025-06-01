import sys
import os
import numpy as np
import time

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.hyperopt import (
    ParameterSpace,
    HyperOptimizer,
    RandomSampler,
    LatinHypercubeSampler,
    ConstraintManager,
    ExperimentTracker
)

# 模拟数据和模型类
class MockModel:
    """模拟的矩阵分解模型"""

    def __init__(self, **config):
        self.config = config
        self.learning_rate = config.get('learning_rate', 0.01)
        self.latent_factors = config.get('latent_factors', 50)
        self.lambda_reg = config.get('lambda_reg', 0.01)
        self.delta1 = config.get('delta1', 1.0)
        self.delta2 = config.get('delta2', 2.0)
        self.loss_type = config.get('loss_type', 'l2')

    def fit(self, train_data, val_data):
        """模拟训练过程"""
        # 模拟训练时间
        time.sleep(0.1)

    def predict(self, data):
        """模拟预测"""
        return np.random.random(len(data))

def train_model(**config):
    """训练模型的实现"""
    model = MockModel(**config)

    # 模拟训练数据
    train_data = np.random.random((1000, 3))
    val_data = np.random.random((200, 3))

    # 训练模型
    model.fit(train_data, val_data)

    return model

def evaluate_model(model):
    """评估模型的实现"""
    # 模拟评估过程，基于配置计算RMSE
    lr = model.config.get('learning_rate', 0.01)
    factors = model.config.get('latent_factors', 50)
    reg = model.config.get('lambda_reg', 0.01)
    delta1 = model.config.get('delta1', 1.0)
    delta2 = model.config.get('delta2', 2.0)
    loss_type = model.config.get('loss_type', 'l2')

    # 模拟RMSE计算 - 基于参数的合理性
    base_rmse = 0.85

    # 学习率影响
    if lr < 0.001 or lr > 0.05:
        base_rmse += 0.1

    # 潜在因子数影响
    if factors < 20 or factors > 80:
        base_rmse += 0.05

    # 正则化影响
    if reg < 0.005 or reg > 0.5:
        base_rmse += 0.03

    # HPL损失函数的delta参数影响
    if loss_type == 'hpl':
        if delta1 >= delta2:  # 违反约束
            base_rmse += 0.2
        elif delta1 < 0.5 or delta2 > 4.0:
            base_rmse += 0.1

    # 添加随机噪声
    rmse = base_rmse + np.random.normal(0, 0.02)

    return max(0.5, rmse)  # 确保RMSE不会太小

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
    """目标函数：训练模型并返回验证集分数"""
    try:
        model = train_model(**config)
        score = evaluate_model(model)
        return score
    except Exception as e:
        print(f"配置 {config} 评估失败: {e}")
        return 10.0  # 返回一个较大的值表示失败

def run_optimization_example():
    """运行超参数优化示例"""
    print("="*60)
    print("Hyperopt 超参数优化示例")
    print("="*60)

    # 4. 创建优化器
    optimizer = HyperOptimizer(
        objective_fn=objective,
        space=space,
        sampler=RandomSampler(space, seed=42),
        constraints=constraints,
        tracker=ExperimentTracker('hpl_optimization'),
        maximize=False  # 最小化损失
    )

    print("开始优化...")
    start_time = time.time()

    # 5. 运行优化
    best_trial = optimizer.optimize(
        n_trials=20,  # 减少试验次数以加快示例运行
        batch_size=1,  # 串行执行以简化示例
        no_improvement_rounds=10
    )

    end_time = time.time()

    # 6. 显示结果
    print("\n" + "="*60)
    print("优化结果")
    print("="*60)

    if best_trial:
        print(f"最佳配置: {best_trial.config}")
        print(f"最佳分数: {best_trial.score:.4f}")
        print(f"优化耗时: {end_time - start_time:.2f}秒")

        # 显示参数分析
        print("\n参数分析:")
        config = best_trial.config
        print(f"  学习率: {config['learning_rate']:.6f}")
        print(f"  潜在因子数: {config['latent_factors']}")
        print(f"  正则化系数: {config['lambda_reg']:.4f}")
        print(f"  损失函数类型: {config['loss_type']}")
        if config['loss_type'] == 'hpl':
            print(f"  Delta1: {config['delta1']:.2f}")
            print(f"  Delta2: {config['delta2']:.2f}")
            print(f"  约束满足: {'是' if config['delta1'] < config['delta2'] else '否'}")
    else:
        print("优化失败，未找到有效配置")

    return optimizer, best_trial

def demonstrate_samplers():
    """演示不同采样器的效果"""
    print("\n" + "="*60)
    print("采样器对比演示")
    print("="*60)

    # 创建不同的采样器
    samplers = {
        'Random': RandomSampler(space, seed=42),
        'LatinHypercube': LatinHypercubeSampler(space, seed=42),
    }

    # 尝试导入高级采样器
    try:
        from src.hyperopt.samplers import SobolSampler, AdaptiveSampler
        samplers['Sobol'] = SobolSampler(space, seed=42)
        samplers['Adaptive'] = AdaptiveSampler(space, seed=42)
    except ImportError as e:
        print(f"部分采样器不可用: {e}")

    results = {}

    for name, sampler in samplers.items():
        print(f"\n测试 {name} 采样器...")

        # 创建优化器
        optimizer = HyperOptimizer(
            objective_fn=objective,
            space=space,
            sampler=sampler,
            constraints=constraints,
            tracker=ExperimentTracker(f'{name}_optimization'),
            maximize=False,
            seed=42
        )

        # 运行优化
        start_time = time.time()
        best_trial = optimizer.optimize(n_trials=10, batch_size=1)
        end_time = time.time()

        # 保存结果
        results[name] = {
            'best_score': best_trial.score if best_trial else float('inf'),
            'best_config': best_trial.config if best_trial else None,
            'time': end_time - start_time
        }

        print(f"  最佳分数: {results[name]['best_score']:.4f}")
        print(f"  耗时: {results[name]['time']:.2f}秒")

    # 显示对比结果
    print("\n" + "="*40)
    print("采样器对比结果")
    print("="*40)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['best_score'])
    for i, (name, result) in enumerate(sorted_results, 1):
        print(f"{i}. {name}: {result['best_score']:.4f} (耗时: {result['time']:.2f}s)")

if __name__ == "__main__":
    try:
        # 运行主要示例
        optimizer, best_trial = run_optimization_example()

        # 演示采样器对比
        demonstrate_samplers()

        print("\n" + "="*60)
        print("示例运行完成！")
        print("="*60)

    except Exception as e:
        print(f"示例运行出错: {e}")
        import traceback
        traceback.print_exc()
