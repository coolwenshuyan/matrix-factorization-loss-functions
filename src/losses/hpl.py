# src/losses/hpl.py
import numpy as np
from .base import BaseLoss
from typing import Dict, Optional


class HybridPiecewiseLoss(BaseLoss):
    """
    混合分段损失函数（Hybrid Piecewise Loss）
    
    三段式设计：
    1. 小误差（|e| < δ₁）：L(e) = 0.5 * e²
    2. 中等误差（δ₁ ≤ |e| < δ₂）：L(e) = δ₁ * |e| - 0.5 * δ₁²
    3. 大误差（|e| ≥ δ₂）：L(e) = L_max - (L_max - L_lin(δ₂)) * exp(-B'(|e| - δ₂))
    
    其中：
    - L_lin(δ₂) = δ₁ * δ₂ - 0.5 * δ₁²
    - B' = C_sigmoid * δ₁ / (L_max - L_lin(δ₂))
    """
    
    def __init__(self, delta1: float = 0.5, delta2: float = 2.0, 
                 c_sigmoid: float = 1.0, l_max: float = 3.0, 
                 epsilon: float = 1e-8):
        """
        初始化HPL损失函数
        
        Args:
            delta1: 第一个阈值（二次段到线性段）
            delta2: 第二个阈值（线性段到饱和段）
            c_sigmoid: 饱和速率控制参数
            l_max: 最大损失值
            epsilon: 数值稳定性常数
        """
        super().__init__("HybridPiecewise")
        
        # 参数验证
        if delta1 <= 0:
            raise ValueError("delta1必须大于0")
        if delta2 <= delta1:
            raise ValueError("delta2必须大于delta1")
        if l_max <= 0:
            raise ValueError("l_max必须大于0")
        if c_sigmoid <= 0:
            raise ValueError("c_sigmoid必须大于0")
        
        self.delta1 = delta1
        self.delta2 = delta2
        self.c_sigmoid = c_sigmoid
        self.l_max = l_max
        self.epsilon = epsilon
        
        # 预计算常量
        self.l_lin_delta2 = delta1 * delta2 - 0.5 * delta1 ** 2
        
        # 验证l_max约束
        if l_max <= self.l_lin_delta2:
            raise ValueError(f"l_max ({l_max}) 必须大于 L_lin(δ₂) ({self.l_lin_delta2})")
        
        # 计算B'以确保C¹连续性
        self.b_prime = c_sigmoid * delta1 / (l_max - self.l_lin_delta2 + epsilon)
        
        # 保存配置
        self._config = {
            'delta1': delta1,
            'delta2': delta2,
            'c_sigmoid': c_sigmoid,
            'l_max': l_max,
            'epsilon': epsilon
        }
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """计算HPL损失"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        
        # 初始化损失数组
        loss = np.zeros_like(errors, dtype=np.float64)
        
        # 第一段：二次损失（小误差）
        mask1 = abs_errors < self.delta1
        if np.any(mask1):
            loss[mask1] = 0.5 * errors[mask1] ** 2
        
        # 第二段：线性损失（中等误差）
        mask2 = (abs_errors >= self.delta1) & (abs_errors < self.delta2)
        if np.any(mask2):
            loss[mask2] = self.delta1 * abs_errors[mask2] - 0.5 * self.delta1 ** 2
        
        # 第三段：饱和损失（大误差）
        mask3 = abs_errors >= self.delta2
        if np.any(mask3):
            # 防止指数溢出
            exp_arg = -self.b_prime * (abs_errors[mask3] - self.delta2)
            exp_arg = np.clip(exp_arg, -50, 50)  # 限制指数参数范围
            
            exp_term = np.exp(exp_arg)
            loss[mask3] = self.l_max - (self.l_max - self.l_lin_delta2) * exp_term
        
        return float(np.mean(loss))
    
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """计算HPL损失的梯度"""
        self.validate_inputs(predictions, targets)
        
        errors = predictions - targets
        abs_errors = np.abs(errors)
        sign_errors = np.sign(errors)
        
        # 初始化梯度数组
        grad = np.zeros_like(errors, dtype=np.float64)
        
        # 第一段梯度：e
        mask1 = abs_errors < self.delta1
        if np.any(mask1):
            grad[mask1] = errors[mask1]
        
        # 第二段梯度：δ₁ * sign(e)
        mask2 = (abs_errors >= self.delta1) & (abs_errors < self.delta2)
        if np.any(mask2):
            grad[mask2] = self.delta1 * sign_errors[mask2]
        
        # 第三段梯度：C_sigmoid * δ₁ * exp(-B'(|e| - δ₂)) * sign(e)
        mask3 = abs_errors >= self.delta2
        if np.any(mask3):
            # 防止指数溢出
            exp_arg = -self.b_prime * (abs_errors[mask3] - self.delta2)
            exp_arg = np.clip(exp_arg, -50, 50)
            
            exp_term = np.exp(exp_arg)
            grad[mask3] = self.c_sigmoid * self.delta1 * exp_term * sign_errors[mask3]
        
        return grad
    
    def verify_continuity(self, num_points: int = 1000) -> Dict[str, bool]:
        """
        验证函数的连续性（C⁰和C¹）
        
        Args:
            num_points: 测试点数量
            
        Returns:
            连续性验证结果
        """
        results = {}
        
        # 在阈值附近创建测试点
        eps = 1e-6
        
        # 测试δ₁处的连续性
        test_points_1 = np.array([self.delta1 - eps, self.delta1, self.delta1 + eps])
        targets = np.zeros_like(test_points_1)
        
        losses_1 = [self.forward(np.array([p]), np.array([0])) for p in test_points_1]
        grads_1 = [self.gradient(np.array([p]), np.array([0]))[0] for p in test_points_1]
        
        # C⁰连续性（函数值）
        c0_at_delta1 = abs(losses_1[0] - losses_1[2]) < 1e-4
        results['C0_continuity_at_delta1'] = c0_at_delta1
        
        # C¹连续性（导数）
        c1_at_delta1 = abs(grads_1[0] - grads_1[2]) < 1e-4
        results['C1_continuity_at_delta1'] = c1_at_delta1
        
        # 测试δ₂处的连续性
        test_points_2 = np.array([self.delta2 - eps, self.delta2, self.delta2 + eps])
        
        losses_2 = [self.forward(np.array([p]), np.array([0])) for p in test_points_2]
        grads_2 = [self.gradient(np.array([p]), np.array([0]))[0] for p in test_points_2]
        
        # C⁰连续性（函数值）
        c0_at_delta2 = abs(losses_2[0] - losses_2[2]) < 1e-4
        results['C0_continuity_at_delta2'] = c0_at_delta2
        
        # C¹连续性（导数）
        c1_at_delta2 = abs(grads_2[0] - grads_2[2]) < 1e-4
        results['C1_continuity_at_delta2'] = c1_at_delta2
        
        return results


class HPLVariants:
    """HPL的变体实现，用于消融研究"""
    
    @staticmethod
    def no_saturation(delta1: float = 0.5) -> BaseLoss:
        """
        无饱和段的HPL变体
        只有二次段和线性段
        """
        class NoSaturationLoss(BaseLoss):
            def __init__(self):
                super().__init__("HPL-NoSaturation")
                self.delta1 = delta1
                self._config = {'delta1': delta1}
            
            def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
                errors = predictions - targets
                abs_errors = np.abs(errors)
                
                loss = np.zeros_like(errors)
                
                # 二次段
                mask1 = abs_errors < self.delta1
                loss[mask1] = 0.5 * errors[mask1] ** 2
                
                # 线性段（延伸到无穷）
                mask2 = abs_errors >= self.delta1
                loss[mask2] = self.delta1 * abs_errors[mask2] - 0.5 * self.delta1 ** 2
                
                return float(np.mean(loss))
            
            def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
                errors = predictions - targets
                abs_errors = np.abs(errors)
                
                grad = np.zeros_like(errors)
                
                # 二次段梯度
                mask1 = abs_errors < self.delta1
                grad[mask1] = errors[mask1]
                
                # 线性段梯度
                mask2 = abs_errors >= self.delta1
                grad[mask2] = self.delta1 * np.sign(errors[mask2])
                
                return grad
        
        return NoSaturationLoss()
    
    @staticmethod
    def no_linear(delta1: float = 0.5, l_max: float = 3.0, c_sigmoid: float = 1.0) -> BaseLoss:
        """
        无线性段的HPL变体
        只有二次段和饱和段
        """
        class NoLinearLoss(BaseLoss):
            def __init__(self):
                super().__init__("HPL-NoLinear")
                self.delta1 = delta1
                self.l_max = l_max
                self.c_sigmoid = c_sigmoid
                self.l_quad_delta1 = 0.5 * delta1 ** 2
                self.b_prime = c_sigmoid * delta1 / (l_max - self.l_quad_delta1)
                self._config = {
                    'delta1': delta1,
                    'l_max': l_max,
                    'c_sigmoid': c_sigmoid
                }
            
            def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
                errors = predictions - targets
                abs_errors = np.abs(errors)
                
                loss = np.zeros_like(errors)
                
                # 二次段
                mask1 = abs_errors < self.delta1
                loss[mask1] = 0.5 * errors[mask1] ** 2
                
                # 饱和段（直接从二次段过渡）
                mask2 = abs_errors >= self.delta1
                if np.any(mask2):
                    exp_arg = -self.b_prime * (abs_errors[mask2] - self.delta1)
                    exp_arg = np.clip(exp_arg, -50, 50)
                    exp_term = np.exp(exp_arg)
                    loss[mask2] = self.l_max - (self.l_max - self.l_quad_delta1) * exp_term
                
                return float(np.mean(loss))
            
            def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
                errors = predictions - targets
                abs_errors = np.abs(errors)
                
                grad = np.zeros_like(errors)
                
                # 二次段梯度
                mask1 = abs_errors < self.delta1
                grad[mask1] = errors[mask1]
                
                # 饱和段梯度
                mask2 = abs_errors >= self.delta1
                if np.any(mask2):
                    exp_arg = -self.b_prime * (abs_errors[mask2] - self.delta1)
                    exp_arg = np.clip(exp_arg, -50, 50)
                    exp_term = np.exp(exp_arg)
                    grad[mask2] = self.c_sigmoid * self.delta1 * exp_term * np.sign(errors[mask2])
                
                return grad
        
        return NoLinearLoss()
