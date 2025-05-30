# src/losses/utils.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Callable, Tuple
from .base import BaseLoss


def check_gradient(loss_fn: BaseLoss, 
                  test_points: Optional[np.ndarray] = None,
                  h: float = 1e-5,
                  rtol: float = 1e-5,
                  atol: float = 1e-8) -> Dict[str, any]:
    """
    使用数值方法检查解析梯度的正确性
    
    Args:
        loss_fn: 损失函数实例
        test_points: 测试点，如果为None则使用默认测试点
        h: 有限差分步长
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
        
    Returns:
        检查结果字典
    """
    if test_points is None:
        # 默认测试点：包括0、正负值、边界点等
        test_points = np.array([-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5])
    
    targets = np.zeros_like(test_points)
    
    # 解析梯度
    analytical_grad = loss_fn.gradient(test_points, targets)
    
    # 数值梯度（中心差分）
    numerical_grad = np.zeros_like(test_points)
    
    for i in range(len(test_points)):
        # 前向点
        test_plus = test_points.copy()
        test_plus[i] += h
        loss_plus = loss_fn.forward(test_plus, targets)
        
        # 后向点
        test_minus = test_points.copy()
        test_minus[i] -= h
        loss_minus = loss_fn.forward(test_minus, targets)
        
        # 中心差分
        numerical_grad[i] = (loss_plus - loss_minus) / (2 * h)
    
    # 计算误差
    abs_error = np.abs(analytical_grad - numerical_grad)
    rel_error = abs_error / (np.abs(numerical_grad) + 1e-10)
    
    # 检查是否通过
    passed = np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)
    
    results = {
        'passed': passed,
        'test_points': test_points,
        'analytical_gradient': analytical_grad,
        'numerical_gradient': numerical_grad,
        'absolute_error': abs_error,
        'relative_error': rel_error,
        'max_abs_error': np.max(abs_error),
        'max_rel_error': np.max(rel_error)
    }
    
    return results


def plot_loss_comparison(loss_functions: Dict[str, BaseLoss],
                        error_range: Tuple[float, float] = (-5, 5),
                        num_points: int = 1000,
                        save_path: Optional[str] = None,
                        show_gradient: bool = True,
                        show_hessian: bool = False,
                        mark_special_points: bool = True):
    """
    绘制多个损失函数的对比图
    
    Args:
        loss_functions: 损失函数字典 {名称: 损失函数实例}
        error_range: 误差范围
        num_points: 采样点数
        save_path: 保存路径
        show_gradient: 是否显示梯度
        show_hessian: 是否显示二阶导数
        mark_special_points: 是否标记特殊点（如HPL的阈值）
    """
    # 创建误差点
    errors = np.linspace(error_range[0], error_range[1], num_points)
    targets = np.zeros_like(errors)
    
    # 计算每个损失函数的值
    loss_values = {}
    gradient_values = {}
    hessian_values = {}
    
    for name, loss_fn in loss_functions.items():
        # 批量计算损失值
        losses = []
        for e in errors:
            loss = loss_fn.forward(np.array([e]), np.array([0]))
            losses.append(loss)
        loss_values[name] = np.array(losses)
        
        # 计算梯度
        if show_gradient:
            gradient_values[name] = loss_fn.gradient(errors, targets)
        
        # 计算二阶导数
        if show_hessian and hasattr(loss_fn, 'hessian'):
            hess = loss_fn.hessian(errors, targets)
            if hess is not None:
                hessian_values[name] = hess
    
    # 创建图表
    num_plots = 1 + (1 if show_gradient else 0) + (1 if show_hessian else 0)
    fig, axes = plt.subplots(num_plots, 1, figsize=(12, 4 * num_plots))
    
    if num_plots == 1:
        axes = [axes]
    
    # 绘制损失函数
    ax_idx = 0
    ax = axes[ax_idx]
    
    for name, values in loss_values.items():
        ax.plot(errors, values, label=name, linewidth=2)
    
    # 标记HPL的特殊点
    if mark_special_points:
        for name, loss_fn in loss_functions.items():
            if hasattr(loss_fn, 'delta1') and hasattr(loss_fn, 'delta2'):
                # HPL损失函数
                ax.axvline(x=loss_fn.delta1, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=-loss_fn.delta1, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=loss_fn.delta2, color='gray', linestyle='--', alpha=0.5)
                ax.axvline(x=-loss_fn.delta2, color='gray', linestyle='--', alpha=0.5)
                
                # 添加标签
                ax.text(loss_fn.delta1, ax.get_ylim()[1] * 0.9, f'δ₁={loss_fn.delta1}',
                       ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                ax.text(loss_fn.delta2, ax.get_ylim()[1] * 0.9, f'δ₂={loss_fn.delta2}',
                       ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Prediction Error')
    ax.set_ylabel('Loss Value')
    ax.set_title('Loss Functions Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 绘制梯度
    if show_gradient:
        ax_idx += 1
        ax = axes[ax_idx]
        
        for name, values in gradient_values.items():
            ax.plot(errors, values, label=name, linewidth=2)
        
        if mark_special_points:
            for name, loss_fn in loss_functions.items():
                if hasattr(loss_fn, 'delta1') and hasattr(loss_fn, 'delta2'):
                    ax.axvline(x=loss_fn.delta1, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=-loss_fn.delta1, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=loss_fn.delta2, color='gray', linestyle='--', alpha=0.5)
                    ax.axvline(x=-loss_fn.delta2, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Gradient Value')
        ax.set_title('Gradient Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 绘制二阶导数
    if show_hessian and hessian_values:
        ax_idx += 1
        ax = axes[ax_idx]
        
        for name, values in hessian_values.items():
            ax.plot(errors, values, label=name, linewidth=2)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Hessian Value')
        ax.set_title('Hessian (Second Derivative) Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 创建放大视图
    _plot_zoomed_views(errors, loss_values, gradient_values, save_path)
    
    return {
        'errors': errors,
        'loss_values': loss_values,
        'gradient_values': gradient_values,
        'hessian_values': hessian_values
    }


def _plot_zoomed_views(errors: np.ndarray, 
                      loss_values: Dict[str, np.ndarray],
                      gradient_values: Dict[str, np.ndarray],
                      save_path: Optional[str] = None):
    """绘制局部放大视图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 定义放大区域
    regions = [
        ('Small Errors', -1, 1),
        ('Medium Errors', 0.5, 2.5),
        ('Large Errors', 2, 5)
    ]
    
    # 绘制损失函数放大图
    for i, (title, x_min, x_max) in enumerate(regions):
        ax = axes[0, i]
        mask = (errors >= x_min) & (errors <= x_max)
        
        for name, values in loss_values.items():
            ax.plot(errors[mask], values[mask], label=name, linewidth=2)
        
        ax.set_xlabel('Prediction Error')
        ax.set_ylabel('Loss Value')
        ax.set_title(f'Loss Functions - {title}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 绘制梯度放大图
    if gradient_values:
        for i, (title, x_min, x_max) in enumerate(regions):
            ax = axes[1, i]
            mask = (errors >= x_min) & (errors <= x_max)
            
            for name, values in gradient_values.items():
                ax.plot(errors[mask], values[mask], label=name, linewidth=2)
            
            ax.set_xlabel('Prediction Error')
            ax.set_ylabel('Gradient Value')
            ax.set_title(f'Gradients - {title}')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        zoom_path = save_path.replace('.png', '_zoomed.png')
        plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def analyze_loss_properties(loss_fn: BaseLoss, 
                           error_range: Tuple[float, float] = (-5, 5),
                           num_points: int = 1000) -> Dict[str, any]:
    """
    分析损失函数的性质
    
    Args:
        loss_fn: 损失函数实例
        error_range: 分析的误差范围
        num_points: 采样点数
        
    Returns:
        分析结果字典
    """
    errors = np.linspace(error_range[0], error_range[1], num_points)
    targets = np.zeros_like(errors)
    
    # 计算损失值和梯度
    losses = []
    for e in errors:
        loss = loss_fn.forward(np.array([e]), np.array([0]))
        losses.append(loss)
    losses = np.array(losses)
    
    gradients = loss_fn.gradient(errors, targets)
    
    # 分析性质
    properties = {
        'name': loss_fn.name,
        'config': loss_fn.get_config(),
        
        # 损失值范围
        'loss_range': (np.min(losses), np.max(losses)),
        'loss_at_zero': loss_fn.forward(np.array([0]), np.array([0])),
        
        # 梯度性质
        'gradient_at_zero': loss_fn.gradient(np.array([0]), np.array([0]))[0],
        'gradient_range': (np.min(gradients), np.max(gradients)),
        
        # 对称性
        'is_symmetric': np.allclose(losses[:num_points//2], losses[num_points//2:][::-1]),
        
        # 凸性（通过二阶导数近似检查）
        'is_convex': None,
        
        # 利普希茨常数估计
        'lipschitz_constant': np.max(np.abs(gradients))
    }
    
    # 检查凸性
    if hasattr(loss_fn, 'hessian'):
        hessian = loss_fn.hessian(errors, targets)
        if hessian is not None:
            properties['is_convex'] = np.all(hessian >= -1e-10)
    
    # 特殊点检测
    if hasattr(loss_fn, 'delta1'):
        properties['delta1'] = loss_fn.delta1
    if hasattr(loss_fn, 'delta2'):
        properties['delta2'] = loss_fn.delta2
    if hasattr(loss_fn, 'l_max'):
        properties['l_max'] = loss_fn.l_max
    
    # 连续性验证（针对HPL）
    if hasattr(loss_fn, 'verify_continuity'):
        properties['continuity'] = loss_fn.verify_continuity()
    
    return properties


# 测试示例
if __name__ == "__main__":
    from standard import L1Loss, L2Loss
    from robust import HuberLoss, LogcoshLoss
    from hpl import HybridPiecewiseLoss, HPLVariants
    from sigmoid import SigmoidLikeLoss
    
    # 创建损失函数实例
    loss_functions = {
        'L2': L2Loss(),
        'L1': L1Loss(),
        'Huber': HuberLoss(delta=1.0),
        'Logcosh': LogcoshLoss(),
        'HPL': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
        'Sigmoid-like': SigmoidLikeLoss(alpha=1.0, l_max=3.0)
    }
    
    # 测试梯度检查
    print("梯度检查结果：")
    for name, loss_fn in loss_functions.items():
        result = check_gradient(loss_fn)
        print(f"{name}: {'通过' if result['passed'] else '失败'}")
        if not result['passed']:
            print(f"  最大绝对误差: {result['max_abs_error']:.2e}")
            print(f"  最大相对误差: {result['max_rel_error']:.2e}")
    
    # 绘制对比图
    plot_loss_comparison(loss_functions, save_path='loss_comparison.png')
    
    # 分析性质
    print("\n损失函数性质分析：")
    for name, loss_fn in loss_functions.items():
        props = analyze_loss_properties(loss_fn)
        print(f"\n{name}:")
        print(f"  损失范围: [{props['loss_range'][0]:.3f}, {props['loss_range'][1]:.3f}]")
        print(f"  零点损失: {props['loss_at_zero']:.3f}")
        print(f"  零点梯度: {props['gradient_at_zero']:.3f}")
        print(f"  对称性: {'是' if props['is_symmetric'] else '否'}")
        if props['is_convex'] is not None:
            print(f"  凸性: {'是' if props['is_convex'] else '否'}")
        print(f"  Lipschitz常数: {props['lipschitz_constant']:.3f}")
        
        # HPL特殊信息
        if 'continuity' in props:
            print(f"  连续性验证: {props['continuity']}")