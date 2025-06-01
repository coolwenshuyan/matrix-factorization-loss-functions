#!/usr/bin/env python3
"""
损失函数模块使用示例
演示如何使用自定义的损失函数模块进行机器学习任务
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple


import sys
import os
# 获取项目根目录（当前文件所在目录的上一级）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print(project_root)
sys.path.append(project_root)


from src.losses import (
    L1Loss, L2Loss, HuberLoss, LogcoshLoss, 
    HybridPiecewiseLoss, HPLVariants, SigmoidLikeLoss,
    check_gradient, plot_loss_comparison, analyze_loss_properties
)


def generate_sample_data(n_samples: int = 100, noise_level: float = 0.2, 
                        outlier_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成带有噪声和异常值的回归数据
    
    Args:
        n_samples: 样本数量
        noise_level: 噪声水平
        outlier_ratio: 异常值比例
        
    Returns:
        (X, y): 特征和目标值
    """
    np.random.seed(42)
    
    # 生成基础数据
    X = np.linspace(0, 10, n_samples)
    y_true = 2 * X + 1 + np.sin(X)  # 真实函数
    
    # 添加正常噪声
    noise = np.random.normal(0, noise_level, n_samples)
    y = y_true + noise
    
    # 添加异常值
    n_outliers = int(n_samples * outlier_ratio)
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    outlier_values = np.random.normal(0, 5, n_outliers)  # 大噪声
    y[outlier_indices] += outlier_values
    
    return X.reshape(-1, 1), y


class SimpleLossBasedRegressor:
    """
    基于自定义损失函数的简单回归器
    使用梯度下降优化
    """
    
    def __init__(self, loss_function, learning_rate: float = 0.01, 
                 max_iterations: int = 1000, tolerance: float = 1e-6):
        """
        初始化回归器
        
        Args:
            loss_function: 损失函数实例
            learning_rate: 学习率
            max_iterations: 最大迭代次数
            tolerance: 收敛容忍度
        """
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            # 前向传播
            predictions = X @ self.weights + self.bias
            
            # 计算损失和梯度
            loss = self.loss_function.forward(predictions, y)
            grad = self.loss_function.gradient(predictions, y)
            
            # 反向传播 - 计算参数梯度
            dw = X.T @ grad / n_samples
            db = np.mean(grad)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 记录损失
            self.loss_history.append(loss)
            
            # 检查收敛
            if abs(prev_loss - loss) < self.tolerance:
                print(f"Converged at iteration {iteration}")
                break
            
            prev_loss = loss
            
            if iteration % 100 == 0:
                print(f"Iteration {iteration}: Loss = {loss:.6f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算R²分数"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


def example_1_basic_usage():
    """Example 1: Basic usage of loss functions"""
    print("=" * 60)
    print("Example 1: Basic Loss Function Usage")
    print("=" * 60)
    
    # 创建不同的损失函数实例
    loss_functions = {
        'L2 (MSE)': L2Loss(),
        'L1 (MAE)': L1Loss(),
        'Huber': HuberLoss(delta=1.0),
        'Logcosh': LogcoshLoss(),
        'HPL': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
        'Sigmoid-like': SigmoidLikeLoss(alpha=1.0, l_max=3.0)
    }
    
    # 测试数据
    predictions = np.array([1.0, 2.5, -1.5, 0.0, 3.0])
    targets = np.array([1.2, 2.0, -1.0, 0.5, 2.8])
    
    print("Test Data:")
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    print(f"Errors: {predictions - targets}")
    print()
    
    # 计算不同损失函数的值和梯度
    print("Loss Function Results:")
    print(f"{'Loss Function':<15} {'Loss Value':<10} {'Gradient(first 3)':<20}")
    print("-" * 50)
    
    for name, loss_fn in loss_functions.items():
        loss_value = loss_fn.forward(predictions, targets)
        gradients = loss_fn.gradient(predictions, targets)
        
        print(f"{name:<15} {loss_value:<10.4f} {str(gradients[:3]):<20}")
    
    print()
    return loss_functions


def example_2_gradient_checking():
    """Example 2: Gradient checking"""
    print("=" * 60)
    print("Example 2: Gradient Checking")
    print("=" * 60)
    
    # 创建损失函数
    hpl = HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0)
    
    # 检查梯度
    result = check_gradient(
    hpl,
    test_points=np.array([-3, -1.5, -0.8, -0.2, 0.2, 0.8, 1.5, 3]),  # 避开边界点
    h=1e-4,        # 较大步长
    rtol=1e-3,     # 宽松相对容忍度  
    atol=1e-4      # 宽松绝对容忍度
)
    
    print(f"Gradient check result: {'Passed' if result['passed'] else 'Failed'}")
    print(f"Maximum absolute error: {result['max_abs_error']:.2e}")
    print(f"Maximum relative error: {result['max_rel_error']:.2e}")
    
    # 详细结果
    print("\n详细对比:")
    print(f"{'测试点':<8} {'解析梯度':<12} {'数值梯度':<12} {'绝对误差':<12}")
    print("-" * 48)
    for i in range(len(result['test_points'])):
        print(f"{result['test_points'][i]:<8.2f} "
              f"{result['analytical_gradient'][i]:<12.6f} "
              f"{result['numerical_gradient'][i]:<12.6f} "
              f"{result['absolute_error'][i]:<12.2e}")
    
    print()


def example_3_visualization():
    """Example 3: Loss function visualization"""
    print("=" * 60)
    print("Example 3: Loss Function Visualization")
    print("=" * 60)
    
    # 创建要对比的损失函数
    loss_functions = {
        'L2': L2Loss(),
        'L1': L1Loss(),
        'Huber (δ=1.0)': HuberLoss(delta=1.0),
        'HPL': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
        'Sigmoid-like': SigmoidLikeLoss(alpha=0.5, l_max=3.0)
    }
    
    # 绘制对比图
    plot_data = plot_loss_comparison(
        loss_functions, 
        error_range=(-4, 4),
        show_gradient=True,
        show_hessian=True,
        mark_special_points=True
    )
    
    print("Loss function comparison chart generated")
    print()


def example_4_regression_comparison():
    """Example 4: Comparison of loss functions in regression tasks"""
    print("=" * 60)
    print("Example 4: Loss Function Comparison in Regression")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_sample_data(n_samples=100, noise_level=0.3, outlier_ratio=0.15)
    
    # 创建不同损失函数的模型
    models = {
        'L2 (MSE)': SimpleLossBasedRegressor(L2Loss(), learning_rate=0.001),
        'L1 (MAE)': SimpleLossBasedRegressor(L1Loss(), learning_rate=0.001),
        'Huber': SimpleLossBasedRegressor(HuberLoss(delta=1.0), learning_rate=0.001),
        'HPL': SimpleLossBasedRegressor(
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=5.0), 
            learning_rate=0.001
        )
    }
    
    # 训练所有模型
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name} model:")
        model.fit(X, y)
        
        # 评估
        predictions = model.predict(X)
        r2_score = model.score(X, y)
        
        results[name] = {
            'model': model,
            'predictions': predictions,
            'r2_score': r2_score,
            'final_loss': model.loss_history[-1] if model.loss_history else None
        }
        
        print(f"R² score: {r2_score:.4f}")
        print(f"Final loss: {results[name]['final_loss']:.6f}")
    
    # 可视化结果
    plt.figure(figsize=(15, 10))
    
    # 子图1: 数据和拟合结果
    plt.subplot(2, 2, 1)
    plt.scatter(X.flatten(), y, alpha=0.6, label='Training Data')
    
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    for name, result in results.items():
        y_plot = result['model'].predict(X_plot)
        plt.plot(X_plot.flatten(), y_plot, label=f"{name} (R²={result['r2_score']:.3f})")
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Regression Results Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 损失历史
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if result['model'].loss_history:
            plt.plot(result['model'].loss_history, label=name)
    
    plt.xlabel('Iterations')
    plt.ylabel('Loss Value')
    plt.title('Training Loss History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # 子图3: 残差分析
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        residuals = y - result['predictions']
        plt.scatter(result['predictions'], residuals, alpha=0.6, label=name, s=20)
    
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 性能对比柱状图
    plt.subplot(2, 2, 4)
    names = list(results.keys())
    r2_scores = [results[name]['r2_score'] for name in names]
    
    bars = plt.bar(range(len(names)), r2_scores)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison')
    plt.xticks(range(len(names)), names, rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标签
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    print("\nPerformance Summary:")
    for name, result in results.items():
        print(f"{name}: R² = {result['r2_score']:.4f}, "
              f"Final Loss = {result['final_loss']:.6f}")
    
    print()


def example_5_hpl_variants():
    """Example 5: HPL variants ablation study"""
    print("=" * 60)
    print("Example 5: HPL Variants Ablation Study")
    print("=" * 60)
    
    # 创建HPL及其变体
    loss_functions = {
        'HPL (Complete)': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
        'HPL (No Saturation)': HPLVariants.no_saturation(delta1=0.5),
        'HPL (No Linear)': HPLVariants.no_linear(delta1=0.5, l_max=3.0),
        'L2 (Baseline)': L2Loss(),
        'Huber (Baseline)': HuberLoss(delta=0.5)
    }
    
    # 分析各变体的性质
    print("Variant Property Analysis:")
    print(f"{'Variant Name':<15} {'Loss at Zero':<10} {'Gradient at Zero':<10} {'Symmetry':<8} {'Max Gradient':<10}")
    print("-" * 65)
    
    for name, loss_fn in loss_functions.items():
        props = analyze_loss_properties(loss_fn, error_range=(-3, 3))
        print(f"{name:<15} {props['loss_at_zero']:<10.4f} "
              f"{props['gradient_at_zero']:<10.4f} "
              f"{'Yes' if props['is_symmetric'] else 'No':<8} "
              f"{props['lipschitz_constant']:<10.2f}")
    
    # 可视化对比
    plot_loss_comparison(loss_functions, error_range=(-3, 3), show_gradient=True)
    
    print()


def example_6_parameter_sensitivity():
    """Example 6: Parameter sensitivity analysis"""
    print("=" * 60)
    print("Example 6: Parameter Sensitivity Analysis")
    print("=" * 60)
    
    # 创建不同参数设置的HPL变体
    hpl_variants = {
        'HPL (δ₁=0.3)': HybridPiecewiseLoss(delta1=0.3, delta2=2.0, l_max=3.0),
        'HPL (δ₁=0.8)': HybridPiecewiseLoss(delta1=0.8, delta2=2.0, l_max=3.0),
        'HPL (δ₂=1.5)': HybridPiecewiseLoss(delta1=0.5, delta2=1.5, l_max=3.0),
        'HPL (δ₂=3.0)': HybridPiecewiseLoss(delta1=0.5, delta2=3.0, l_max=4.0),
        'HPL (L_max=2.0)': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=2.0),
        'HPL (L_max=5.0)': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=5.0),
    }
    
    # 生成测试数据
    X, y = generate_sample_data(n_samples=80, noise_level=0.2, outlier_ratio=0.2)
    
    # 比较不同参数设置的性能
    print("Parameter Sensitivity Test:")
    print(f"{'Parameter Set':<12} {'δ₁':<6} {'δ₂':<6} {'L_max':<7} {'R² Score':<8} {'Final Loss':<10}")
    print("-" * 55)
    
    for name, loss_fn in hpl_variants.items():
        model = SimpleLossBasedRegressor(loss_fn, learning_rate=0.001, max_iterations=500)
        model.fit(X, y)
        
        r2_score = model.score(X, y)
        final_loss = model.loss_history[-1] if model.loss_history else 0
        
        print(f"{name:<12} {loss_fn.delta1:<6.1f} {loss_fn.delta2:<6.1f} "
              f"{loss_fn.l_max:<7.1f} {r2_score:<8.4f} {final_loss:<10.6f}")
    
    # 可视化不同参数设置
    plot_loss_comparison(hpl_variants, error_range=(-4, 4))
    
    print()


def example_7_advanced_features():
    """示例7: 高级功能演示"""
    print("=" * 60)
    print("示例7: 高级功能演示")
    print("=" * 60)
    
    # 创建HPL损失函数
    hpl = HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0)
    
    # 1. 连续性验证
    print("1. HPL连续性验证:")
    continuity_result = hpl.verify_continuity()
    for check, result in continuity_result.items():
        print(f"   {check}: {'通过' if result else '失败'}")
    
    # 2. 配置管理
    print("\n2. 配置管理:")
    config = hpl.get_config()
    print(f"   当前配置: {config}")
    
    # 保存和加载配置
    hpl.save_config('hpl_config.json')
    print("   配置已保存到 hpl_config.json")
    
    # 3. 可导性检查
    print("\n3. 可导性检查:")
    test_points = [-2, -0.5, 0, 0.5, 2, 10]
    for point in test_points:
        is_diff = hpl.is_differentiable_at(point)
        print(f"   在点 {point:4.1f} 处可导: {'是' if is_diff else '否'}")
    
    # 4. 使用 __call__ 方法
    print("\n4. 使用可调用接口:")
    test_pred = np.array([1.0, 2.0])
    test_target = np.array([0.8, 1.5])
    
    # 只计算损失
    loss_only = hpl(test_pred, test_target)
    print(f"   损失值: {loss_only:.4f}")
    
    # 同时计算损失和梯度
    loss_and_grad = hpl(test_pred, test_target, return_gradient=True)
    print(f"   损失值: {loss_and_grad[0]:.4f}")
    print(f"   梯度: {loss_and_grad[1]}")
    
    # 5. 字符串表示
    print(f"\n5. 字符串表示: {repr(hpl)}")
    
    print()


def main():
    """主函数 - 运行所有示例"""
    print("自定义损失函数模块使用示例")
    print("=" * 60)
    
    try:
        # 运行所有示例
        loss_functions = example_1_basic_usage()
        example_2_gradient_checking()
        example_3_visualization()
        example_4_regression_comparison()
        example_5_hpl_variants()
        example_6_parameter_sensitivity()
        example_7_advanced_features()
        
        print("所有示例运行完成！")
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保损失函数模块在Python路径中")
        print("你可能需要调整import语句中的路径")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
