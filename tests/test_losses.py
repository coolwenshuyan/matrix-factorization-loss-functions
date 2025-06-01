#!/usr/bin/env python3
"""
损失函数测试模块
用于验证各种损失函数的计算是否正确
"""

import unittest
import numpy as np
import sys
import os

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.losses import (
    L1Loss, L2Loss, HuberLoss, LogcoshLoss,
    HybridPiecewiseLoss, SigmoidLikeLoss,
    check_gradient
)


class TestLossFunctions(unittest.TestCase):
    """测试各种损失函数的计算是否正确"""

    def setUp(self):
        """设置测试数据"""
        # 简单测试数据
        self.pred_simple = np.array([1.0, 2.0, 3.0])
        self.target_simple = np.array([1.5, 1.5, 2.5])
        
        # 包含正负误差的测试数据
        self.pred_mixed = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.target_mixed = np.array([1.5, 1.5, 3.0, 3.5, 6.0])
        
        # 零误差测试数据
        self.pred_zero = np.array([1.0, 2.0, 3.0])
        self.target_zero = np.array([1.0, 2.0, 3.0])
        
        # 大误差测试数据
        self.pred_large = np.array([10.0, -5.0, 8.0])
        self.target_large = np.array([0.0, 5.0, 0.0])

    def test_l1_loss(self):
        """测试L1损失函数"""
        loss_fn = L1Loss()
        
        # 测试简单数据
        loss = loss_fn.forward(self.pred_simple, self.target_simple)
        expected_loss = (0.5 + 0.5 + 0.5) / 3  # |1.0-1.5| + |2.0-1.5| + |3.0-2.5| / 3
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # 测试梯度
        grad = loss_fn.gradient(self.pred_simple, self.target_simple)
        expected_grad = np.array([-1.0, 1.0, 1.0]) / 3  # sign(pred - target) / n
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)
        
        # 测试零误差
        loss_zero = loss_fn.forward(self.pred_zero, self.target_zero)
        self.assertEqual(loss_zero, 0.0)

    def test_l2_loss(self):
        """测试L2损失函数"""
        loss_fn = L2Loss()
        
        # 测试简单数据
        loss = loss_fn.forward(self.pred_simple, self.target_simple)
        expected_loss = (0.5**2 + 0.5**2 + 0.5**2) / 3  # ((1.0-1.5)^2 + (2.0-1.5)^2 + (3.0-2.5)^2) / 3
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # 测试梯度
        grad = loss_fn.gradient(self.pred_simple, self.target_simple)
        expected_grad = 2 * np.array([-0.5, 0.5, 0.5]) / 3  # 2 * (pred - target) / n
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_huber_loss(self):
        """测试Huber损失函数"""
        delta = 1.0
        loss_fn = HuberLoss(delta=delta)
        
        # 测试小误差（二次区域）
        small_pred = np.array([1.0, 2.0])
        small_target = np.array([1.4, 1.7])
        loss = loss_fn.forward(small_pred, small_target)
        
        # 手动计算小误差损失: 0.5 * (error)^2 for |error| <= delta
        expected_loss = (0.5 * 0.4**2 + 0.5 * 0.3**2) / 2
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # 测试大误差（线性区域）
        large_pred = np.array([3.0, 4.0])
        large_target = np.array([5.0, 1.0])
        loss = loss_fn.forward(large_pred, large_target)
        
        # 手动计算大误差损失: delta * (|error| - 0.5*delta) for |error| > delta
        err1 = abs(3.0 - 5.0)  # 2.0
        err2 = abs(4.0 - 1.0)  # 3.0
        expected_loss = (delta * (err1 - 0.5*delta) + delta * (err2 - 0.5*delta)) / 2
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # 测试梯度
        grad = loss_fn.gradient(large_pred, large_target)
        expected_grad = np.array([-1.0, 1.0]) * delta / 2  # sign(error) * delta / n for |error| > delta
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_logcosh_loss(self):
        """测试Logcosh损失函数"""
        loss_fn = LogcoshLoss()
        
        # 测试简单数据
        loss = loss_fn.forward(self.pred_simple, self.target_simple)
        
        # 手动计算: log(cosh(pred - target))
        errors = self.pred_simple - self.target_simple
        expected_loss = np.mean(np.log(np.cosh(errors)))
        self.assertAlmostEqual(loss, expected_loss, places=6)
        
        # 测试梯度
        grad = loss_fn.gradient(self.pred_simple, self.target_simple)
        expected_grad = np.tanh(errors) / len(errors)
        np.testing.assert_allclose(grad, expected_grad, rtol=1e-5)

    def test_hpl_loss(self):
        """测试HPL损失函数"""
        delta1 = 0.5
        delta2 = 2.0
        l_max = 3.0
        loss_fn = HybridPiecewiseLoss(delta1=delta1, delta2=delta2, l_max=l_max)
        
        # 测试不同区域的误差
        test_errors = np.array([-3.0, -1.5, -0.3, 0.0, 0.3, 1.5, 3.0])
        test_preds = np.zeros_like(test_errors)
        test_targets = -test_errors  # 使得 error = pred - target = test_errors
        
        loss = loss_fn.forward(test_preds, test_targets)
        
        # 手动计算每个区域的损失
        expected_losses = []
        for err in test_errors:
            abs_err = abs(err)
            if abs_err < delta1:
                # 二次区域
                expected_losses.append(0.5 * err**2)
            elif abs_err < delta2:
                # 线性区域
                expected_losses.append(delta1 * (abs_err - 0.5 * delta1))
            else:
                # 饱和区域：使用实际的HPL公式
                l_lin_delta2 = delta1 * delta2 - 0.5 * delta1**2
                b_prime = loss_fn.c_sigmoid * delta1 / (l_max - l_lin_delta2)
                exp_arg = -b_prime * (abs_err - delta2)
                exp_term = np.exp(exp_arg)
                expected_losses.append(l_max - (l_max - l_lin_delta2) * exp_term)

        expected_loss = np.mean(expected_losses)
        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_sigmoid_like_loss(self):
        """测试Sigmoid-like损失函数"""
        alpha = 1.0
        l_max = 3.0
        loss_fn = SigmoidLikeLoss(alpha=alpha, l_max=l_max)
        
        # 测试简单数据
        loss = loss_fn.forward(self.pred_simple, self.target_simple)
        
        # 手动计算: l_max * (2 / (1 + exp(-alpha * error^2)) - 1)
        errors = self.pred_simple - self.target_simple
        expected_losses = l_max * (2 / (1 + np.exp(-alpha * errors**2)) - 1)
        expected_loss = np.mean(expected_losses)
        self.assertAlmostEqual(loss, expected_loss, places=6)

    def test_gradient_check(self):
        """使用梯度检查工具验证所有损失函数的梯度计算"""
        loss_functions = [
            L1Loss(),
            L2Loss(),
            HuberLoss(delta=1.0),
            LogcoshLoss(),
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0),
            SigmoidLikeLoss(alpha=1.0, l_max=3.0)
        ]
        
        for loss_fn in loss_functions:
            result = check_gradient(loss_fn)
            self.assertTrue(
                result['passed'],
                f"梯度检查失败: {loss_fn.__class__.__name__}, "
                f"最大绝对误差: {result['max_abs_error']:.2e}, "
                f"最大相对误差: {result['max_rel_error']:.2e}"
            )


if __name__ == '__main__':
    unittest.main()