#!/usr/bin/env python3
"""
损失函数完整单元测试模块
涵盖基类功能、标准损失函数、鲁棒损失函数、HPL损失函数及其变体、Sigmoid-like损失函数的全面测试
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import json
from unittest.mock import patch

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.losses import (
    BaseLoss, L1Loss, L2Loss, HuberLoss, LogcoshLoss,
    HybridPiecewiseLoss, HPLVariants, SigmoidLikeLoss,
    check_gradient
)


class TestBaseLoss(unittest.TestCase):
    """1.1 基类功能测试"""
    
    def setUp(self):
        """创建一个简单的测试损失函数"""
        class TestLoss(BaseLoss):
            def __init__(self):
                super().__init__("TestLoss")
                self._config = {'param1': 1.0, 'param2': 'test'}
            
            def forward(self, predictions, targets):
                return np.mean((predictions - targets) ** 2)
            
            def gradient(self, predictions, targets):
                return 2 * (predictions - targets) / len(predictions)
        
        self.test_loss = TestLoss()
        self.valid_pred = np.array([1.0, 2.0, 3.0])
        self.valid_target = np.array([1.5, 1.5, 2.5])
    
    def test_input_validation_shape_mismatch(self):
        """测试形状不匹配的输入验证"""
        pred_wrong_shape = np.array([1.0, 2.0])
        target_wrong_shape = np.array([1.5, 1.5, 2.5])
        
        with self.assertRaises(ValueError) as context:
            self.test_loss.validate_inputs(pred_wrong_shape, target_wrong_shape)
        
        self.assertIn("形状不匹配", str(context.exception))
    
    def test_input_validation_nan_values(self):
        """测试NaN值的输入验证"""
        pred_with_nan = np.array([1.0, np.nan, 3.0])
        target_with_nan = np.array([1.5, 1.5, np.nan])
        
        with self.assertRaises(ValueError) as context:
            self.test_loss.validate_inputs(pred_with_nan, self.valid_target)
        self.assertIn("NaN值", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.test_loss.validate_inputs(self.valid_pred, target_with_nan)
        self.assertIn("NaN值", str(context.exception))
    
    def test_input_validation_inf_values(self):
        """测试无穷大值的输入验证"""
        pred_with_inf = np.array([1.0, np.inf, 3.0])
        target_with_inf = np.array([1.5, 1.5, -np.inf])
        
        with self.assertRaises(ValueError) as context:
            self.test_loss.validate_inputs(pred_with_inf, self.valid_target)
        self.assertIn("无穷大值", str(context.exception))
        
        with self.assertRaises(ValueError) as context:
            self.test_loss.validate_inputs(self.valid_pred, target_with_inf)
        self.assertIn("无穷大值", str(context.exception))
    
    def test_config_management(self):
        """测试配置管理功能"""
        # 测试get_config
        config = self.test_loss.get_config()
        expected_config = {
            'name': 'TestLoss',
            'class': 'TestLoss',
            'param1': 1.0,
            'param2': 'test'
        }
        self.assertEqual(config, expected_config)
        
        # 测试set_config
        new_config = {'param3': 'new_value'}
        self.test_loss.set_config(new_config)
        updated_config = self.test_loss.get_config()
        self.assertEqual(updated_config['param3'], 'new_value')
        
        # 测试save_config
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.test_loss.save_config(temp_path)
            
            # 验证保存的文件
            with open(temp_path, 'r') as f:
                saved_config = json.load(f)
            
            self.assertEqual(saved_config['name'], 'TestLoss')
            self.assertEqual(saved_config['param1'], 1.0)
        finally:
            os.unlink(temp_path)
    
    def test_differentiability_check(self):
        """测试可导性检查功能"""
        # 测试可导点
        self.assertTrue(self.test_loss.is_differentiable_at(1.0))
        self.assertTrue(self.test_loss.is_differentiable_at(0.0))
        self.assertTrue(self.test_loss.is_differentiable_at(-1.0))
    
    def test_string_representation(self):
        """测试字符串表示"""
        repr_str = repr(self.test_loss)
        self.assertIn("TestLoss", repr_str)
        self.assertIn("param1=1.0", repr_str)
        self.assertIn("param2=test", repr_str)
    
    def test_call_interface(self):
        """测试调用接口"""
        # 测试只返回损失值
        loss = self.test_loss(self.valid_pred, self.valid_target)
        expected_loss = self.test_loss.forward(self.valid_pred, self.valid_target)
        self.assertEqual(loss, expected_loss)
        
        # 测试同时返回损失值和梯度
        loss, grad = self.test_loss(self.valid_pred, self.valid_target, return_gradient=True)
        expected_grad = self.test_loss.gradient(self.valid_pred, self.valid_target)
        self.assertEqual(loss, expected_loss)
        np.testing.assert_allclose(grad, expected_grad)


class TestStandardLosses(unittest.TestCase):
    """1.2 标准损失函数测试"""
    
    def setUp(self):
        self.pred = np.array([1.0, 2.0, 3.0, 4.0])
        self.target = np.array([1.5, 1.5, 2.5, 3.5])
        self.errors = self.pred - self.target  # [-0.5, 0.5, 0.5, 0.5]
        
    def test_l1_loss_computation(self):
        """测试L1损失计算的正确性"""
        loss_fn = L1Loss()
        
        # 测试损失值计算
        loss = loss_fn.forward(self.pred, self.target)
        expected_loss = np.mean(np.abs(self.errors))  # (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        self.assertAlmostEqual(loss, expected_loss, places=7)
        
        # 测试零误差情况
        zero_loss = loss_fn.forward(self.pred, self.pred)
        self.assertEqual(zero_loss, 0.0)
    
    def test_l1_loss_subgradient_at_zero(self):
        """验证L1损失在零点处的次梯度行为"""
        loss_fn = L1Loss()
        
        # 测试零误差点的梯度
        zero_pred = np.array([1.0, 2.0, 3.0])
        zero_target = np.array([1.0, 2.0, 3.0])
        grad_zero = loss_fn.gradient(zero_pred, zero_target)
        np.testing.assert_allclose(grad_zero, np.zeros(3), atol=1e-10)
        
        # 测试非零误差点的梯度
        grad = loss_fn.gradient(self.pred, self.target)
        expected_grad = np.sign(self.errors) / len(self.errors)
        np.testing.assert_allclose(grad, expected_grad)
        
        # 测试可导性检查
        self.assertFalse(loss_fn.is_differentiable_at(0.0))  # 在0处不可导
        self.assertTrue(loss_fn.is_differentiable_at(1.0))   # 在非零处可导
    
    def test_l2_loss_computation(self):
        """测试L2损失计算的正确性"""
        loss_fn = L2Loss()
        
        # 测试损失值计算
        loss = loss_fn.forward(self.pred, self.target)
        expected_loss = np.mean(self.errors ** 2)  # (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
        self.assertAlmostEqual(loss, expected_loss, places=7)
    
    def test_l2_loss_gradient_and_hessian(self):
        """验证L2损失的梯度和Hessian矩阵"""
        loss_fn = L2Loss()
        
        # 测试梯度
        grad = loss_fn.gradient(self.pred, self.target)
        expected_grad = 2 * self.errors / len(self.errors)
        np.testing.assert_allclose(grad, expected_grad)
        
        # 测试Hessian矩阵
        hess = loss_fn.hessian(self.pred, self.target)
        expected_hess = np.ones_like(self.pred)
        np.testing.assert_allclose(hess, expected_hess)


class TestRobustLosses(unittest.TestCase):
    """1.3 鲁棒损失函数测试"""
    
    def test_huber_loss_continuity_at_threshold(self):
        """测试Huber损失在阈值δ处的连续性"""
        delta = 1.0
        loss_fn = HuberLoss(delta=delta)
        
        # 在阈值附近测试连续性
        eps = 1e-6
        test_points = np.array([delta - eps, delta, delta + eps])
        targets = np.zeros_like(test_points)
        
        losses = []
        grads = []
        for point in test_points:
            loss = loss_fn.forward(np.array([point]), np.array([0]))
            grad = loss_fn.gradient(np.array([point]), np.array([0]))[0]
            losses.append(loss)
            grads.append(grad)
        
        # 检查函数值连续性（C⁰）
        self.assertAlmostEqual(losses[0], losses[2], places=4, 
                              msg="Huber损失在δ处不连续")
        
        # 检查梯度连续性（C¹）
        self.assertAlmostEqual(grads[0], grads[2], places=4,
                              msg="Huber损失梯度在δ处不连续")
    
    def test_huber_loss_segment_switching(self):
        """验证Huber损失二次段和线性段的正确切换"""
        delta = 1.0
        loss_fn = HuberLoss(delta=delta)
        
        # 测试小误差（二次段）
        small_errors = np.array([0.5, -0.5])
        targets = np.zeros_like(small_errors)
        
        loss_small = loss_fn.forward(small_errors, targets)
        grad_small = loss_fn.gradient(small_errors, targets)
        
        # 二次段：L(e) = 0.5 * e²
        expected_loss_small = np.mean(0.5 * small_errors ** 2)
        expected_grad_small = small_errors / len(small_errors)
        
        self.assertAlmostEqual(loss_small, expected_loss_small, places=7)
        np.testing.assert_allclose(grad_small, expected_grad_small)
        
        # 测试大误差（线性段）
        large_errors = np.array([2.0, -2.0])
        
        loss_large = loss_fn.forward(large_errors, targets[:2])
        grad_large = loss_fn.gradient(large_errors, targets[:2])
        
        # 线性段：L(e) = δ * |e| - 0.5 * δ²
        expected_loss_large = np.mean(delta * np.abs(large_errors) - 0.5 * delta ** 2)
        expected_grad_large = delta * np.sign(large_errors) / len(large_errors)
        
        self.assertAlmostEqual(loss_large, expected_loss_large, places=7)
        np.testing.assert_allclose(grad_large, expected_grad_large)
    
    def test_logcosh_numerical_stability(self):
        """测试Logcosh损失的数值稳定性"""
        loss_fn = LogcoshLoss()
        
        # 测试大误差值的处理
        large_errors = np.array([50.0, -50.0, 100.0, -100.0])
        targets = np.zeros_like(large_errors)
        
        # 应该不会出现溢出或NaN
        loss = loss_fn.forward(large_errors, targets)
        grad = loss_fn.gradient(large_errors, targets)
        
        self.assertTrue(np.isfinite(loss), "大误差时Logcosh损失出现非有限值")
        self.assertTrue(np.all(np.isfinite(grad)), "大误差时Logcosh梯度出现非有限值")
        
        # 梯度应该接近±1（tanh的饱和值）
        np.testing.assert_allclose(np.abs(grad), 1.0 / len(grad), atol=0.1)


class TestHPLLoss(unittest.TestCase):
    """1.4 HPL损失函数测试"""
    
    def setUp(self):
        self.delta1 = 0.5
        self.delta2 = 2.0
        self.l_max = 3.0
        self.c_sigmoid = 1.0
        self.loss_fn = HybridPiecewiseLoss(
            delta1=self.delta1, 
            delta2=self.delta2, 
            l_max=self.l_max,
            c_sigmoid=self.c_sigmoid
        )
    
    def test_parameter_constraints(self):
        """测试参数约束检查"""
        # δ₁ > 0
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=0, delta2=2.0, l_max=3.0)
        
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=-0.5, delta2=2.0, l_max=3.0)
        
        # δ₂ > δ₁
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=2.0, delta2=1.0, l_max=3.0)
        
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=1.0, delta2=1.0, l_max=3.0)
        
        # l_max > 0
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=0)
        
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=-1.0)
        
        # c_sigmoid > 0
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0, c_sigmoid=0)
        
        # l_max > L_lin(δ₂)
        l_lin_delta2 = 0.5 * 2.0 - 0.5 * 0.5 ** 2  # 0.875
        with self.assertRaises(ValueError):
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=0.8)  # 小于L_lin(δ₂)
    
    def test_three_segment_implementation(self):
        """测试三段函数的正确实现"""
        # 测试第一段（二次段）：|e| < δ₁
        small_errors = np.array([0.3, -0.3])
        targets = np.zeros_like(small_errors)
        
        loss_small = self.loss_fn.forward(small_errors, targets)
        expected_small = np.mean(0.5 * small_errors ** 2)
        self.assertAlmostEqual(loss_small, expected_small, places=7)
        
        # 测试第二段（线性段）：δ₁ ≤ |e| < δ₂
        medium_errors = np.array([1.0, -1.0])
        
        loss_medium = self.loss_fn.forward(medium_errors, targets[:2])
        expected_medium = np.mean(self.delta1 * np.abs(medium_errors) - 0.5 * self.delta1 ** 2)
        self.assertAlmostEqual(loss_medium, expected_medium, places=7)
        
        # 测试第三段（饱和段）：|e| ≥ δ₂
        large_errors = np.array([3.0, -3.0])
        
        loss_large = self.loss_fn.forward(large_errors, targets[:2])
        
        # 手动计算饱和段损失
        l_lin_delta2 = self.delta1 * self.delta2 - 0.5 * self.delta1 ** 2
        b_prime = self.c_sigmoid * self.delta1 / (self.l_max - l_lin_delta2)
        
        expected_losses = []
        for err in large_errors:
            abs_err = abs(err)
            exp_arg = -b_prime * (abs_err - self.delta2)
            exp_term = np.exp(exp_arg)
            expected_loss = self.l_max - (self.l_max - l_lin_delta2) * exp_term
            expected_losses.append(expected_loss)
        
        expected_large = np.mean(expected_losses)
        self.assertAlmostEqual(loss_large, expected_large, places=6)
    
    def test_continuity_at_thresholds(self):
        """验证在δ₁和δ₂处的C⁰和C¹连续性"""
        continuity_result = self.loss_fn.verify_continuity()
        
        self.assertTrue(continuity_result['C0_continuity_at_delta1'], 
                       "HPL在δ₁处不满足C⁰连续性")
        self.assertTrue(continuity_result['C1_continuity_at_delta1'], 
                       "HPL在δ₁处不满足C¹连续性")
        self.assertTrue(continuity_result['C0_continuity_at_delta2'], 
                       "HPL在δ₂处不满足C⁰连续性")
        self.assertTrue(continuity_result['C1_continuity_at_delta2'], 
                       "HPL在δ₂处不满足C¹连续性")
    
    def test_b_prime_calculation(self):
        """测试B'参数的自动计算"""
        l_lin_delta2 = self.delta1 * self.delta2 - 0.5 * self.delta1 ** 2
        expected_b_prime = self.c_sigmoid * self.delta1 / (self.l_max - l_lin_delta2)
        
        self.assertAlmostEqual(self.loss_fn.b_prime, expected_b_prime, places=7)
    
    def test_numerical_stability(self):
        """验证数值稳定性（指数溢出防护）"""
        # 测试极大误差值
        extreme_errors = np.array([1000.0, -1000.0])
        targets = np.zeros_like(extreme_errors)
        
        # 应该不会出现溢出
        loss = self.loss_fn.forward(extreme_errors, targets)
        grad = self.loss_fn.gradient(extreme_errors, targets)
        
        self.assertTrue(np.isfinite(loss), "极大误差时HPL损失出现非有限值")
        self.assertTrue(np.all(np.isfinite(grad)), "极大误差时HPL梯度出现非有限值")
        
        # 损失应该接近l_max
        self.assertLess(loss, self.l_max + 0.1, "HPL损失超过了l_max上界")


class TestHPLVariants(unittest.TestCase):
    """1.5 HPL变体测试"""
    
    def test_no_saturation_variant(self):
        """测试无饱和段变体"""
        delta1 = 0.5
        no_sat_loss = HPLVariants.no_saturation(delta1=delta1)
        
        # 测试小误差（二次段）
        small_errors = np.array([0.3, -0.3])
        targets = np.zeros_like(small_errors)
        
        loss_small = no_sat_loss.forward(small_errors, targets)
        expected_small = np.mean(0.5 * small_errors ** 2)
        self.assertAlmostEqual(loss_small, expected_small, places=7)
        
        # 测试大误差（应该是线性段，无饱和）
        large_errors = np.array([5.0, -5.0])
        
        loss_large = no_sat_loss.forward(large_errors, targets[:2])
        expected_large = np.mean(delta1 * np.abs(large_errors) - 0.5 * delta1 ** 2)
        self.assertAlmostEqual(loss_large, expected_large, places=7)
        
        # 验证梯度
        grad_large = no_sat_loss.gradient(large_errors, targets[:2])
        expected_grad = delta1 * np.sign(large_errors) / len(large_errors)
        np.testing.assert_allclose(grad_large, expected_grad)
    
    def test_no_linear_variant(self):
        """测试无线性段变体"""
        delta1 = 0.5
        l_max = 3.0
        c_sigmoid = 1.0
        no_lin_loss = HPLVariants.no_linear(
            delta1=delta1, l_max=l_max, c_sigmoid=c_sigmoid
        )
        
        # 测试小误差（二次段）
        small_errors = np.array([0.3, -0.3])
        targets = np.zeros_like(small_errors)
        
        loss_small = no_lin_loss.forward(small_errors, targets)
        expected_small = np.mean(0.5 * small_errors ** 2)
        self.assertAlmostEqual(loss_small, expected_small, places=7)
        
        # 测试大误差（应该直接从二次段跳到饱和段）
        large_errors = np.array([2.0, -2.0])
        
        loss_large = no_lin_loss.forward(large_errors, targets[:2])
        
        # 手动计算：从二次段直接到饱和段
        l_quad_delta1 = 0.5 * delta1 ** 2
        b_prime = c_sigmoid * delta1 / (l_max - l_quad_delta1)
        
        expected_losses = []
        for err in large_errors:
            abs_err = abs(err)
            exp_arg = -b_prime * (abs_err - delta1)
            exp_term = np.exp(exp_arg)
            expected_loss = l_max - (l_max - l_quad_delta1) * exp_term
            expected_losses.append(expected_loss)
        
        expected_large = np.mean(expected_losses)
        self.assertAlmostEqual(loss_large, expected_large, places=6)
    
    def test_variant_consistency(self):
        """验证变体与原版HPL的一致性"""
        # 创建标准HPL
        delta1 = 0.5
        delta2 = 2.0
        l_max = 3.0
        hpl_standard = HybridPiecewiseLoss(delta1=delta1, delta2=delta2, l_max=l_max)
        
        # 创建变体
        hpl_no_sat = HPLVariants.no_saturation(delta1=delta1)
        hpl_no_lin = HPLVariants.no_linear(delta1=delta1, l_max=l_max)
        
        # 在小误差区域，所有版本应该一致（都是二次段）
        small_errors = np.array([0.3, -0.3])
        targets = np.zeros_like(small_errors)
        
        loss_std = hpl_standard.forward(small_errors, targets)
        loss_no_sat = hpl_no_sat.forward(small_errors, targets)
        loss_no_lin = hpl_no_lin.forward(small_errors, targets)
        
        self.assertAlmostEqual(loss_std, loss_no_sat, places=7)
        self.assertAlmostEqual(loss_std, loss_no_lin, places=7)


class TestSigmoidLikeLoss(unittest.TestCase):
    """1.6 Sigmoid-like损失函数测试"""
    
    def setUp(self):
        self.alpha = 1.0
        self.l_max = 3.0
        self.loss_fn = SigmoidLikeLoss(alpha=self.alpha, l_max=self.l_max)
    
    def test_loss_upper_bound(self):
        """测试损失上界约束"""
        # 测试各种误差值，损失不应超过l_max
        test_errors = np.array([0, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0])
        targets = np.zeros_like(test_errors)
        
        for err in test_errors:
            loss = self.loss_fn.forward(np.array([err]), np.array([0]))
            self.assertLessEqual(loss, self.l_max + 1e-10, 
                               f"损失值 {loss} 超过上界 {self.l_max}，误差为 {err}")
    
    def test_smoothness_and_differentiability(self):
        """验证平滑性和可导性"""
        # Sigmoid-like函数应该在所有点都可导
        test_points = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
        
        for point in test_points:
            self.assertTrue(self.loss_fn.is_differentiable_at(point),
                          f"Sigmoid-like损失在点 {point} 处不可导")
    
    def test_numerical_stability(self):
        """测试数值稳定性"""
        # 测试极大误差值
        extreme_errors = np.array([100.0, -100.0])
        targets = np.zeros_like(extreme_errors)
        
        # 应该不会出现溢出
        loss = self.loss_fn.forward(extreme_errors, targets)
        grad = self.loss_fn.gradient(extreme_errors, targets)
        
        self.assertTrue(np.isfinite(loss), "极大误差时Sigmoid-like损失出现非有限值")
        self.assertTrue(np.all(np.isfinite(grad)), "极大误差时Sigmoid-like梯度出现非有限值")
        
        # 损失应该接近l_max
        self.assertLess(loss, self.l_max + 0.1, "Sigmoid-like损失超过了l_max上界")
    
    def test_parameter_validation(self):
        """测试参数验证"""
        # alpha > 0
        with self.assertRaises(ValueError):
            SigmoidLikeLoss(alpha=0, l_max=3.0)
        
        with self.assertRaises(ValueError):
            SigmoidLikeLoss(alpha=-1.0, l_max=3.0)
        
        # l_max > 0
        with self.assertRaises(ValueError):
            SigmoidLikeLoss(alpha=1.0, l_max=0)
        
        with self.assertRaises(ValueError):
            SigmoidLikeLoss(alpha=1.0, l_max=-1.0)


class TestGradientValidation(unittest.TestCase):
    """梯度验证测试（所有损失函数的数值梯度检查）"""
    
    def setUp(self):
        self.loss_functions = {
            'L1': L1Loss(),
            'L2': L2Loss(),
            'Huber': HuberLoss(delta=1.0),
            'Logcosh': LogcoshLoss(),
            'HPL': HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
            'HPL_NoSat': HPLVariants.no_saturation(delta1=0.5),
            'HPL_NoLin': HPLVariants.no_linear(delta1=0.5, l_max=3.0),
            'SigmoidLike': SigmoidLikeLoss(alpha=1.0, l_max=3.0)
        }
    
    def test_gradient_accuracy(self):
        """使用数值微分验证所有损失函数的解析梯度"""
        for name, loss_fn in self.loss_functions.items():
            with self.subTest(loss_function=name):
                # 使用多个测试点，包括边界点
                test_points = np.array([-5, -2, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2, 5])
                
                # 对于HPL，添加阈值附近的测试点
                if hasattr(loss_fn, 'delta1'):
                    delta1 = loss_fn.delta1
                    additional_points = np.array([
                        delta1 - 0.01, delta1, delta1 + 0.01
                    ])
                    test_points = np.concatenate([test_points, additional_points])
                    
                    if hasattr(loss_fn, 'delta2'):
                        delta2 = loss_fn.delta2
                        additional_points = np.array([
                            delta2 - 0.01, delta2, delta2 + 0.01
                        ])
                        test_points = np.concatenate([test_points, additional_points])
                
                result = check_gradient(loss_fn, test_points=test_points)
                
                self.assertTrue(
                    result['passed'],
                    f"梯度检查失败: {name}\n"
                    f"最大绝对误差: {result['max_abs_error']:.2e}\n"
                    f"最大相对误差: {result['max_rel_error']:.2e}\n"
                    f"测试点: {test_points}\n"
                    f"解析梯度: {result['analytical_gradient']}\n"
                    f"数值梯度: {result['numerical_gradient']}"
                )


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def setUp(self):
        self.loss_functions = [
            L1Loss(),
            L2Loss(),
            HuberLoss(delta=1.0),
            LogcoshLoss(),
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
            SigmoidLikeLoss(alpha=1.0, l_max=3.0)
        ]
    
    def test_zero_error_handling(self):
        """测试零误差处理"""
        zero_pred = np.array([1.0, 2.0, 3.0])
        zero_target = np.array([1.0, 2.0, 3.0])
        
        for loss_fn in self.loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                loss = loss_fn.forward(zero_pred, zero_target)
                
                # L2损失在零误差时应该为0，其他损失函数也应该为0或接近0
                if isinstance(loss_fn, L2Loss):
                    self.assertEqual(loss, 0.0)
                else:
                    self.assertAlmostEqual(loss, 0.0, places=10)
    
    def test_large_error_stability(self):
        """测试大误差值的稳定性"""
        large_pred = np.array([1000.0, -1000.0])
        large_target = np.array([0.0, 0.0])
        
        for loss_fn in self.loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                try:
                    loss = loss_fn.forward(large_pred, large_target)
                    grad = loss_fn.gradient(large_pred, large_target)
                    
                    # 检查是否有非有限值
                    self.assertTrue(np.isfinite(loss), 
                                  f"{loss_fn.__class__.__name__} 在大误差时产生非有限损失值")
                    self.assertTrue(np.all(np.isfinite(grad)), 
                                  f"{loss_fn.__class__.__name__} 在大误差时产生非有限梯度")
                    
                    # 对于有上界的损失函数，检查上界约束
                    if hasattr(loss_fn, 'l_max'):
                        self.assertLessEqual(loss, loss_fn.l_max + 1e-6,
                                           f"{loss_fn.__class__.__name__} 损失值超过上界")
                
                except Exception as e:
                    self.fail(f"{loss_fn.__class__.__name__} 在大误差时抛出异常: {e}")
    
    def test_single_sample_handling(self):
        """测试单样本处理"""
        single_pred = np.array([2.0])
        single_target = np.array([1.0])
        
        for loss_fn in self.loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                loss = loss_fn.forward(single_pred, single_target)
                grad = loss_fn.gradient(single_pred, single_target)
                
                self.assertIsInstance(loss, float)
                self.assertEqual(grad.shape, single_pred.shape)
    
    def test_empty_array_handling(self):
        """测试空数组处理"""
        empty_pred = np.array([])
        empty_target = np.array([])
        
        for loss_fn in self.loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                if len(empty_pred) == 0:
                    # 空数组应该导致适当的错误或返回0
                    try:
                        loss = loss_fn.forward(empty_pred, empty_target)
                        # 如果没有抛出异常，损失应该是0或NaN
                        self.assertTrue(loss == 0.0 or np.isnan(loss))
                    except (ValueError, ZeroDivisionError):
                        # 这些异常是可以接受的
                        pass


class TestSpecialFunctionProperties(unittest.TestCase):
    """特殊函数性质测试"""
    
    def test_symmetry(self):
        """测试函数对称性：L(e) = L(-e)"""
        test_errors = np.array([0.5, 1.0, 2.0, 5.0])
        
        loss_functions = [
            L1Loss(),
            L2Loss(),
            HuberLoss(delta=1.0),
            LogcoshLoss(),
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
            SigmoidLikeLoss(alpha=1.0, l_max=3.0)
        ]
        
        for loss_fn in loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                for err in test_errors:
                    pos_loss = loss_fn.forward(np.array([err]), np.array([0]))
                    neg_loss = loss_fn.forward(np.array([-err]), np.array([0]))
                    
                    self.assertAlmostEqual(pos_loss, neg_loss, places=10,
                                         msg=f"{loss_fn.__class__.__name__} 不满足对称性，"
                                             f"误差 {err}: L({err})={pos_loss}, L(-{err})={neg_loss}")
    
    def test_monotonicity(self):
        """测试单调性：|e₁| < |e₂| 时损失值的关系"""
        errors = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
        
        loss_functions = [
            L1Loss(),
            L2Loss(),
            HuberLoss(delta=1.0),
            LogcoshLoss(),
            SigmoidLikeLoss(alpha=1.0, l_max=3.0)  # HPL可能不严格单调（有饱和段）
        ]
        
        for loss_fn in loss_functions:
            with self.subTest(loss_function=loss_fn.__class__.__name__):
                losses = []
                for err in errors:
                    loss = loss_fn.forward(np.array([err]), np.array([0]))
                    losses.append(loss)
                
                # 检查单调性（损失值应该随误差增大而增大或保持不变）
                for i in range(len(losses) - 1):
                    if not isinstance(loss_fn, SigmoidLikeLoss):  # Sigmoid-like可能饱和
                        self.assertLessEqual(losses[i], losses[i + 1] + 1e-10,
                                           f"{loss_fn.__class__.__name__} 不满足单调性，"
                                           f"L({errors[i]})={losses[i]} > L({errors[i+1]})={losses[i+1]}")


if __name__ == '__main__':
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestBaseLoss,
        TestStandardLosses,
        TestRobustLosses,
        TestHPLLoss,
        TestHPLVariants,
        TestSigmoidLikeLoss,
        TestGradientValidation,
        TestEdgeCases,
        TestSpecialFunctionProperties
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"测试总结:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError: ')[-1].split(chr(10))[0]}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2]}")
    
    print(f"{'='*60}")
    
    # 退出代码
    exit_code = 0 if result.wasSuccessful() else 1
    print(f"测试{'通过' if exit_code == 0 else '失败'}")
    
    import sys
    sys.exit(exit_code)