#!/usr/bin/env python3
"""
models模块单元测试
测试矩阵分解模型的各个组件功能
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import json
from unittest.mock import patch, MagicMock

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from src.models.base_mf import BaseMatrixFactorization
    from src.models.mf_sgd import MatrixFactorizationSGD
    from src.models.initializers import (
        NormalInitializer, UniformInitializer, 
        XavierInitializer, TruncatedNormalInitializer
    )
    from src.models.regularizers import (
        L2Regularizer, L1Regularizer, ElasticNetRegularizer
    )
    from src.losses.standard import L1Loss, L2Loss
    from src.losses.hpl import HybridPiecewiseLoss
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目路径正确且所有依赖模块存在")


class TestInitializers(unittest.TestCase):
    """测试参数初始化器"""
    
    def setUp(self):
        self.shape = (100, 10)
        self.random_seed = 42
    
    def test_normal_initializer(self):
        """测试正态分布初始化器"""
        initializer = NormalInitializer(mean=0.0, std=0.1, random_seed=self.random_seed)
        
        # 测试初始化
        params = initializer.initialize(self.shape)
        
        # 验证形状
        self.assertEqual(params.shape, self.shape)
        
        # 验证数据类型
        self.assertEqual(params.dtype, np.float32)
        
        # 验证统计特性（近似）
        self.assertAlmostEqual(np.mean(params), 0.0, places=2)
        self.assertAlmostEqual(np.std(params), 0.1, places=1)
        
        # 测试可调用性
        params2 = initializer(self.shape)
        self.assertEqual(params2.shape, self.shape)
    
    def test_uniform_initializer(self):
        """测试均匀分布初始化器"""
        low, high = -0.1, 0.1
        initializer = UniformInitializer(low=low, high=high, random_seed=self.random_seed)
        
        params = initializer.initialize(self.shape)
        
        # 验证范围
        self.assertTrue(np.all(params >= low))
        self.assertTrue(np.all(params <= high))
        
        # 验证分布（近似均匀）
        self.assertAlmostEqual(np.mean(params), 0.0, places=2)
    
    def test_xavier_initializer(self):
        """测试Xavier初始化器"""
        initializer = XavierInitializer(mode='fan_avg', random_seed=self.random_seed)
        
        params = initializer.initialize(self.shape)
        
        # 验证形状
        self.assertEqual(params.shape, self.shape)
        
        # 验证Xavier初始化的方差
        fan_in, fan_out = self.shape
        expected_std = np.sqrt(2.0 / (fan_in + fan_out))
        actual_std = np.std(params)
        self.assertAlmostEqual(actual_std, expected_std, places=1)
        
        # 测试不同模式
        for mode in ['fan_in', 'fan_out']:
            init = XavierInitializer(mode=mode, random_seed=self.random_seed)
            params = init.initialize(self.shape)
            self.assertEqual(params.shape, self.shape)
        
        # 测试错误输入
        with self.assertRaises(ValueError):
            initializer.initialize((10,))  # 不是2D
    
    def test_truncated_normal_initializer(self):
        """测试截断正态分布初始化器"""
        initializer = TruncatedNormalInitializer(
            mean=0.0, std=0.1, num_std=2.0, random_seed=self.random_seed
        )
        
        params = initializer.initialize(self.shape)
        
        # 验证截断范围
        lower = 0.0 - 2.0 * 0.1
        upper = 0.0 + 2.0 * 0.1
        self.assertTrue(np.all(params >= lower))
        self.assertTrue(np.all(params <= upper))
    
    def test_random_seed_consistency(self):
        """测试随机种子的一致性"""
        seed = 123
        
        init1 = NormalInitializer(random_seed=seed)
        init2 = NormalInitializer(random_seed=seed)
        
        params1 = init1.initialize(self.shape)
        params2 = init2.initialize(self.shape)
        
        np.testing.assert_array_equal(params1, params2)


class TestRegularizers(unittest.TestCase):
    """测试正则化器"""
    
    def setUp(self):
        self.parameters = {
            'user_factors': np.random.randn(100, 10).astype(np.float32),
            'item_factors': np.random.randn(50, 10).astype(np.float32),
            'user_bias': np.random.randn(100).astype(np.float32),
            'item_bias': np.random.randn(50).astype(np.float32)
        }
    
    def test_l2_regularizer(self):
        """测试L2正则化器"""
        lambda_reg = 0.01
        regularizer = L2Regularizer(lambda_reg=lambda_reg)
        
        # 测试惩罚计算
        penalty = regularizer.compute_penalty(self.parameters)
        
        # 手动计算L2惩罚
        expected_penalty = 0
        for name, param in self.parameters.items():
            lambda_val = regularizer._get_lambda(name)
            expected_penalty += lambda_val * np.sum(param ** 2)
        
        self.assertAlmostEqual(penalty, expected_penalty, places=5)
        
        # 测试梯度计算
        for name, param in self.parameters.items():
            gradient = regularizer.compute_gradient(name, param)
            expected_gradient = 2 * regularizer._get_lambda(name) * param
            np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-5)
    
    def test_l2_regularizer_different_lambdas(self):
        """测试L2正则化器的不同lambda值"""
        regularizer = L2Regularizer(
            lambda_reg=0.01,
            lambda_user=0.02,
            lambda_item=0.03,
            lambda_bias=0.005
        )
        
        # 验证不同参数使用不同的lambda
        self.assertEqual(regularizer._get_lambda('user_factors'), 0.02)
        self.assertEqual(regularizer._get_lambda('item_factors'), 0.03)
        self.assertEqual(regularizer._get_lambda('user_bias'), 0.005)
        self.assertEqual(regularizer._get_lambda('item_bias'), 0.005)
    
    def test_l1_regularizer(self):
        """测试L1正则化器"""
        lambda_reg = 0.01
        regularizer = L1Regularizer(lambda_reg=lambda_reg)
        
        # 测试惩罚计算
        penalty = regularizer.compute_penalty(self.parameters)
        
        # 手动计算L1惩罚
        expected_penalty = 0
        for param in self.parameters.values():
            expected_penalty += lambda_reg * np.sum(np.abs(param))
        
        self.assertAlmostEqual(penalty, expected_penalty, places=5)
        
        # 测试梯度计算（平滑版本）
        for name, param in self.parameters.items():
            gradient = regularizer.compute_gradient(name, param)
            
            # 验证梯度形状
            self.assertEqual(gradient.shape, param.shape)
            
            # 验证梯度符号
            expected_sign = param / (np.abs(param) + regularizer.epsilon)
            np.testing.assert_allclose(
                np.sign(gradient), 
                np.sign(expected_sign * lambda_reg), 
                rtol=1e-3
            )
    
    def test_elastic_net_regularizer(self):
        """测试弹性网络正则化器"""
        lambda_reg = 0.01
        l1_ratio = 0.3
        regularizer = ElasticNetRegularizer(lambda_reg=lambda_reg, l1_ratio=l1_ratio)
        
        # 测试惩罚计算
        penalty = regularizer.compute_penalty(self.parameters)
        
        # 手动计算弹性网络惩罚
        l1_penalty = sum(np.sum(np.abs(param)) for param in self.parameters.values())
        l2_penalty = sum(np.sum(param ** 2) for param in self.parameters.values())
        expected_penalty = lambda_reg * (l1_ratio * l1_penalty + (1 - l1_ratio) * l2_penalty)
        
        self.assertAlmostEqual(penalty, expected_penalty, places=5)
        
        # 测试梯度计算
        for name, param in self.parameters.items():
            gradient = regularizer.compute_gradient(name, param)
            self.assertEqual(gradient.shape, param.shape)


class ConcreteMatrixFactorization(BaseMatrixFactorization):
    """BaseMatrixFactorization的具体实现用于测试"""
    
    def initialize_parameters(self, initializer=None):
        if initializer is None:
            initializer = NormalInitializer(std=0.1)
        
        self.user_factors = initializer.initialize((self.n_users, self.n_factors))
        self.item_factors = initializer.initialize((self.n_items, self.n_factors))
        
        if self.use_bias:
            self.user_bias = np.zeros(self.n_users, dtype=self.dtype)
            self.item_bias = np.zeros(self.n_items, dtype=self.dtype)
    
    def fit(self, train_data, val_data=None, **kwargs):
        # 简单的模拟训练
        self.train_history['loss'] = [1.0, 0.8, 0.6, 0.4]
        self.train_history['val_loss'] = [1.2, 1.0, 0.8, 0.6]


class TestBaseMatrixFactorization(unittest.TestCase):
    """测试矩阵分解基类"""
    
    def setUp(self):
        self.n_users = 100
        self.n_items = 50
        self.n_factors = 10
        
        self.model = ConcreteMatrixFactorization(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            use_bias=True,
            global_mean=3.5
        )
        self.model.initialize_parameters()
    
    def test_initialization(self):
        """测试模型初始化"""
        self.assertEqual(self.model.n_users, self.n_users)
        self.assertEqual(self.model.n_items, self.n_items)
        self.assertEqual(self.model.n_factors, self.n_factors)
        self.assertTrue(self.model.use_bias)
        self.assertEqual(self.model.global_mean, 3.5)
        
        # 验证参数形状
        self.assertEqual(self.model.user_factors.shape, (self.n_users, self.n_factors))
        self.assertEqual(self.model.item_factors.shape, (self.n_items, self.n_factors))
        self.assertEqual(self.model.user_bias.shape, (self.n_users,))
        self.assertEqual(self.model.item_bias.shape, (self.n_items,))
    
    def test_predict_single(self):
        """测试单样本预测"""
        user_id = 0
        item_id = 0
        
        prediction = self.model.predict(user_id, item_id)
        
        # 验证预测形状
        self.assertEqual(prediction.shape, (1,))
        
        # 手动计算预测值
        expected = (
            np.dot(self.model.user_factors[user_id], self.model.item_factors[item_id]) +
            self.model.user_bias[user_id] +
            self.model.item_bias[item_id] +
            self.model.global_mean
        )
        
        self.assertAlmostEqual(prediction[0], expected, places=5)
    
    def test_predict_batch(self):
        """测试批量预测"""
        user_ids = np.array([0, 1, 2])
        item_ids = np.array([0, 1, 2])
        
        predictions = self.model.predict(user_ids, item_ids)
        
        # 验证形状
        self.assertEqual(predictions.shape, (3,))
        
        # 验证每个预测值
        for i in range(3):
            expected = (
                np.dot(self.model.user_factors[user_ids[i]], 
                      self.model.item_factors[item_ids[i]]) +
                self.model.user_bias[user_ids[i]] +
                self.model.item_bias[item_ids[i]] +
                self.model.global_mean
            )
            self.assertAlmostEqual(predictions[i], expected, places=5)
    
    def test_predict_all(self):
        """测试全量预测"""
        all_predictions = self.model.predict_all()
        
        # 验证形状
        self.assertEqual(all_predictions.shape, (self.n_users, self.n_items))
        
        # 验证部分预测值
        for i in range(min(5, self.n_users)):
            for j in range(min(5, self.n_items)):
                expected = (
                    np.dot(self.model.user_factors[i], self.model.item_factors[j]) +
                    self.model.user_bias[i] +
                    self.model.item_bias[j] +
                    self.model.global_mean
                )
                self.assertAlmostEqual(all_predictions[i, j], expected, places=5)
    
    def test_predict_input_validation(self):
        """测试预测输入验证"""
        # 测试形状不匹配
        with self.assertRaises(ValueError):
            self.model.predict([0, 1], [0])  # 不同长度
    
    def test_parameter_management(self):
        """测试参数管理"""
        # 测试获取参数
        params = self.model.get_parameters()
        
        self.assertIn('user_factors', params)
        self.assertIn('item_factors', params)
        self.assertIn('user_bias', params)
        self.assertIn('item_bias', params)
        
        # 测试设置参数
        new_params = {
            'user_factors': np.random.randn(self.n_users, self.n_factors).astype(np.float32),
            'item_factors': np.random.randn(self.n_items, self.n_factors).astype(np.float32),
            'user_bias': np.random.randn(self.n_users).astype(np.float32),
            'item_bias': np.random.randn(self.n_items).astype(np.float32)
        }
        
        self.model.set_parameters(new_params)
        
        np.testing.assert_array_equal(self.model.user_factors, new_params['user_factors'])
        np.testing.assert_array_equal(self.model.item_factors, new_params['item_factors'])
    
    def test_similarity_functions(self):
        """测试相似度计算"""
        # 测试物品相似度
        item_id = 0
        n_similar = 5
        
        similar_items, scores = self.model.get_similar_items(item_id, n_similar)
        
        self.assertEqual(len(similar_items), n_similar)
        self.assertEqual(len(scores), n_similar)
        self.assertTrue(all(item_id != sim_id for sim_id in similar_items))
        
        # 测试用户相似度
        user_id = 0
        similar_users, scores = self.model.get_similar_users(user_id, n_similar)
        
        self.assertEqual(len(similar_users), n_similar)
        self.assertEqual(len(scores), n_similar)
        self.assertTrue(all(user_id != sim_id for sim_id in similar_users))
    
    def test_model_persistence(self):
        """测试模型保存和加载"""
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            # 保存模型
            self.model.save_model(save_path)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(save_path))
            
            # 创建新模型并加载
            new_model = ConcreteMatrixFactorization(
                n_users=1, n_items=1, n_factors=1  # 临时值
            )
            new_model.load_model(save_path)
            
            # 验证配置
            self.assertEqual(new_model.n_users, self.n_users)
            self.assertEqual(new_model.n_items, self.n_items)
            self.assertEqual(new_model.n_factors, self.n_factors)
            
            # 验证参数
            np.testing.assert_array_equal(new_model.user_factors, self.model.user_factors)
            np.testing.assert_array_equal(new_model.item_factors, self.model.item_factors)
            
        finally:
            # 清理临时文件
            if os.path.exists(save_path):
                os.unlink(save_path)
            config_path = save_path.replace('.npz', '_config.json')
            if os.path.exists(config_path):
                os.unlink(config_path)


class TestMatrixFactorizationSGD(unittest.TestCase):
    """测试SGD矩阵分解模型"""
    
    def setUp(self):
        self.n_users = 100
        self.n_items = 50
        self.n_factors = 10
        
        self.model = MatrixFactorizationSGD(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            learning_rate=0.01,
            regularizer=L2Regularizer(lambda_reg=0.01),
            loss_function=L2Loss(),
            use_bias=True,
            clip_gradient=5.0,
            momentum=0.9
        )
        
        # 创建测试数据
        self.train_data = self._create_test_data(1000)
        self.val_data = self._create_test_data(200)
    
    def _create_test_data(self, n_samples):
        """创建测试数据"""
        user_ids = np.random.randint(0, self.n_users, n_samples)
        item_ids = np.random.randint(0, self.n_items, n_samples)
        ratings = np.random.normal(3.5, 1.0, n_samples)
        
        return np.column_stack([user_ids, item_ids, ratings])
    
    def test_initialization(self):
        """测试SGD模型初始化"""
        self.assertEqual(self.model.learning_rate, 0.01)
        self.assertEqual(self.model.momentum, 0.9)
        self.assertEqual(self.model.clip_gradient, 5.0)
        self.assertIsInstance(self.model.regularizer, L2Regularizer)
        self.assertIsInstance(self.model.loss_function, L2Loss)
    
    def test_parameter_initialization(self):
        """测试参数初始化"""
        self.model.initialize_parameters()
        
        # 验证参数存在
        self.assertIsNotNone(self.model.user_factors)
        self.assertIsNotNone(self.model.item_factors)
        self.assertIsNotNone(self.model.user_bias)
        self.assertIsNotNone(self.model.item_bias)
        
        # 验证形状
        self.assertEqual(self.model.user_factors.shape, (self.n_users, self.n_factors))
        self.assertEqual(self.model.item_factors.shape, (self.n_items, self.n_factors))
        
        # 验证动量项（如果使用）
        if self.model.momentum > 0:
            self.assertIn('user_factors', self.model.velocity)
            self.assertIn('item_factors', self.model.velocity)
    
    def test_loss_computation(self):
        """测试损失计算"""
        self.model.initialize_parameters()
        
        predictions = np.array([3.0, 4.0, 2.5])
        targets = np.array([3.5, 3.8, 2.2])
        
        loss = self.model.compute_loss(predictions, targets)
        
        # 验证损失是正数
        self.assertGreater(loss, 0)
        self.assertTrue(np.isfinite(loss))
    
    def test_sgd_update(self):
        """测试SGD更新"""
        self.model.initialize_parameters()
        
        # 保存更新前的参数
        old_user_factor = self.model.user_factors[0].copy()
        old_item_factor = self.model.item_factors[0].copy()
        
        # 执行SGD更新
        sample_loss = self.model.sgd_update(0, 0, 3.5, 0)
        
        # 验证参数已更新
        self.assertFalse(np.array_equal(old_user_factor, self.model.user_factors[0]))
        self.assertFalse(np.array_equal(old_item_factor, self.model.item_factors[0]))
        
        # 验证损失值
        self.assertTrue(np.isfinite(sample_loss))
    
    def test_gradient_clipping(self):
        """测试梯度裁剪"""
        # 创建大梯度
        large_gradient = np.array([10.0, 15.0, 20.0])
        clipped = self.model._clip_gradient(large_gradient)
        
        # 验证梯度被裁剪
        self.assertLessEqual(np.linalg.norm(clipped), self.model.clip_gradient)
    
    def test_learning_rate_schedule(self):
        """测试学习率调度"""
        # 测试指数衰减
        self.model.lr_schedule = 'exponential'
        lr_0 = self.model._get_learning_rate(0)
        lr_10 = self.model._get_learning_rate(10)
        
        self.assertLessEqual(lr_10, lr_0)
        
        # 测试反比例衰减
        self.model.lr_schedule = 'inverse'
        lr_0 = self.model._get_learning_rate(0)
        lr_10 = self.model._get_learning_rate(10)
        
        self.assertLessEqual(lr_10, lr_0)
    
    def test_training(self):
        """测试训练过程"""
        # 短时间训练
        self.model.fit(
            self.train_data,
            val_data=self.val_data,
            n_epochs=3,
            verbose=0
        )
        
        # 验证训练历史
        self.assertGreater(len(self.model.train_history['loss']), 0)
        self.assertGreater(len(self.model.train_history['val_loss']), 0)
        
        # 验证损失趋势（应该下降或保持稳定）
        losses = self.model.train_history['loss']
        if len(losses) > 1:
            # 允许一定的波动
            self.assertLessEqual(losses[-1], losses[0] * 1.1)
    
    def test_prediction_with_different_loss_functions(self):
        """测试不同损失函数的预测"""
        loss_functions = [
            L1Loss(),
            L2Loss(),
            HybridPiecewiseLoss(delta1=0.5, delta2=2.0, l_max=3.0)
        ]
        
        for loss_fn in loss_functions:
            model = MatrixFactorizationSGD(
                n_users=50, n_items=30, n_factors=5,
                loss_function=loss_fn
            )
            model.initialize_parameters()
            
            # 测试预测
            prediction = model.predict([0], [0])
            self.assertEqual(len(prediction), 1)
            self.assertTrue(np.isfinite(prediction[0]))


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整的训练预测流程"""
        # 创建模型
        model = MatrixFactorizationSGD(
            n_users=50,
            n_items=30,
            n_factors=5,
            learning_rate=0.1,
            regularizer=L2Regularizer(lambda_reg=0.01),
            loss_function=L2Loss(),
            use_bias=True
        )
        
        # 创建训练数据
        n_samples = 500
        train_data = np.column_stack([
            np.random.randint(0, 50, n_samples),
            np.random.randint(0, 30, n_samples),
            np.random.normal(3.5, 1.0, n_samples)
        ])
        
        # 训练模型
        model.fit(train_data, n_epochs=5, verbose=0)
        
        # 预测
        test_users = np.array([0, 1, 2])
        test_items = np.array([0, 1, 2])
        predictions = model.predict(test_users, test_items)
        
        # 验证预测结果
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(np.isfinite(p) for p in predictions))
        
        # 测试相似度计算
        similar_items, scores = model.get_similar_items(0, n_similar=5)
        self.assertEqual(len(similar_items), 5)
        self.assertEqual(len(scores), 5)
        
        # 测试模型保存和加载
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
            save_path = tmp.name
        
        try:
            model.save_model(save_path)
            
            new_model = MatrixFactorizationSGD(1, 1, 1)  # 临时参数
            new_model.load_model(save_path)
            
            # 验证加载后的预测一致性
            new_predictions = new_model.predict(test_users, test_items)
            np.testing.assert_allclose(predictions, new_predictions, rtol=1e-5)
            
        finally:
            if os.path.exists(save_path):
                os.unlink(save_path)
            config_path = save_path.replace('.npz', '_config.json')
            if os.path.exists(config_path):
                os.unlink(config_path)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestInitializers,
        TestRegularizers, 
        TestBaseMatrixFactorization,
        TestMatrixFactorizationSGD,
        TestIntegration
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
    
    if result.failures:
        print(f"\n失败的测试:")
        for test, traceback in result.failures:
            print(f"- {test}")
    
    if result.errors:
        print(f"\n错误的测试:")
        for test, traceback in result.errors:
            print(f"- {test}")
    
    print(f"{'='*60}")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)