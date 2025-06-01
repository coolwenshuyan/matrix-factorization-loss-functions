#!/usr/bin/env python3
"""
快速模型功能测试
用于验证models模块的基本功能是否正常
"""

import sys
import os
import numpy as np
import traceback

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def test_imports():
    """测试模块导入"""
    print("1. 测试模块导入...")
    try:
        from src.models.base_mf import BaseMatrixFactorization
        from src.models.mf_sgd import MatrixFactorizationSGD
        from src.models.initializers import NormalInitializer, XavierInitializer
        from src.models.regularizers import L2Regularizer, L1Regularizer
        from src.losses.standard import L2Loss, L1Loss
        from src.losses.hpl import HybridPiecewiseLoss
        
        print("   ✓ 所有模块导入成功")
        return True, {
            'BaseMatrixFactorization': BaseMatrixFactorization,
            'MatrixFactorizationSGD': MatrixFactorizationSGD,
            'NormalInitializer': NormalInitializer,
            'XavierInitializer': XavierInitializer,
            'L2Regularizer': L2Regularizer,
            'L1Regularizer': L1Regularizer,
            'L2Loss': L2Loss,
            'L1Loss': L1Loss,
            'HybridPiecewiseLoss': HybridPiecewiseLoss
        }
    except Exception as e:
        print(f"   ✗ 导入失败: {e}")
        return False, {}

def test_initializers(modules):
    """测试初始化器"""
    print("\n2. 测试参数初始化器...")
    
    try:
        # 测试正态分布初始化器
        normal_init = modules['NormalInitializer'](mean=0.0, std=0.1, random_seed=42)
        params = normal_init.initialize((100, 10))
        
        assert params.shape == (100, 10), "正态初始化器形状错误"
        assert abs(np.mean(params)) < 0.05, "正态初始化器均值错误"
        assert abs(np.std(params) - 0.1) < 0.02, "正态初始化器标准差错误"
        print("   ✓ 正态分布初始化器测试通过")
        
        # 测试Xavier初始化器
        xavier_init = modules['XavierInitializer'](random_seed=42)
        params = xavier_init.initialize((50, 20))
        
        assert params.shape == (50, 20), "Xavier初始化器形状错误"
        expected_std = np.sqrt(2.0 / (50 + 20))
        assert abs(np.std(params) - expected_std) < 0.02, "Xavier初始化器方差错误"
        print("   ✓ Xavier初始化器测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 初始化器测试失败: {e}")
        return False

def test_regularizers(modules):
    """测试正则化器"""
    print("\n3. 测试正则化器...")
    
    try:
        # 创建测试参数
        test_params = {
            'user_factors': np.random.randn(10, 5).astype(np.float32),
            'item_factors': np.random.randn(8, 5).astype(np.float32)
        }
        
        # 测试L2正则化器
        l2_reg = modules['L2Regularizer'](lambda_reg=0.01)
        l2_penalty = l2_reg.compute_penalty(test_params)
        
        assert l2_penalty > 0, "L2正则化惩罚应该大于0"
        print("   ✓ L2正则化器测试通过")
        
        # 测试梯度计算
        for name, param in test_params.items():
            gradient = l2_reg.compute_gradient(name, param)
            assert gradient.shape == param.shape, "梯度形状错误"
            expected_grad = 2 * l2_reg.lambda_reg * param
            assert np.allclose(gradient, expected_grad), "L2梯度计算错误"
        
        print("   ✓ L2正则化器梯度测试通过")
        
        # 测试L1正则化器
        l1_reg = modules['L1Regularizer'](lambda_reg=0.01)
        l1_penalty = l1_reg.compute_penalty(test_params)
        
        assert l1_penalty > 0, "L1正则化惩罚应该大于0"
        print("   ✓ L1正则化器测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 正则化器测试失败: {e}")
        return False

def test_matrix_factorization_sgd(modules):
    """测试SGD矩阵分解模型"""
    print("\n4. 测试SGD矩阵分解模型...")
    
    try:
        # 创建模型
        model = modules['MatrixFactorizationSGD'](
            n_users=50,
            n_items=30,
            n_factors=10,
            learning_rate=0.01,
            regularizer=modules['L2Regularizer'](lambda_reg=0.01),
            loss_function=modules['L2Loss'](),
            use_bias=True
        )
        
        print("   ✓ 模型创建成功")
        
        # 初始化参数
        model.initialize_parameters()
        
        # 验证参数形状
        assert model.user_factors.shape == (50, 10), "用户因子形状错误"
        assert model.item_factors.shape == (30, 10), "物品因子形状错误"
        assert model.user_bias.shape == (50,), "用户偏差形状错误"
        assert model.item_bias.shape == (30,), "物品偏差形状错误"
        
        print("   ✓ 参数初始化成功")
        
        # 测试预测
        user_ids = np.array([0, 1, 2])
        item_ids = np.array([0, 1, 2])
        predictions = model.predict(user_ids, item_ids)
        
        assert predictions.shape == (3,), "预测形状错误"
        assert all(np.isfinite(p) for p in predictions), "预测值包含非有限值"
        
        print("   ✓ 预测功能测试通过")
        
        # 创建训练数据
        n_samples = 200
        train_data = np.column_stack([
            np.random.randint(0, 50, n_samples),
            np.random.randint(0, 30, n_samples),
            np.random.normal(3.5, 1.0, n_samples)
        ])
        
        # 测试SGD更新
        initial_loss = model.sgd_update(0, 0, 3.5, 0)
        assert np.isfinite(initial_loss), "SGD更新损失值异常"
        
        print("   ✓ SGD更新测试通过")
        
        # 测试短期训练
        model.fit(train_data, n_epochs=3, verbose=0)
        
        # 验证训练历史
        assert len(model.train_history['loss']) > 0, "训练历史记录为空"
        
        print("   ✓ 训练功能测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ SGD矩阵分解模型测试失败: {e}")
        traceback.print_exc()
        return False

def test_loss_function_integration(modules):
    """测试损失函数集成"""
    print("\n5. 测试损失函数集成...")
    
    try:
        loss_functions = [
            ('L2Loss', modules['L2Loss']()),
            ('L1Loss', modules['L1Loss']()),
            ('HPL', modules['HybridPiecewiseLoss'](delta1=0.5, delta2=2.0, l_max=3.0))
        ]
        
        for loss_name, loss_fn in loss_functions:
            # 创建使用特定损失函数的模型
            model = modules['MatrixFactorizationSGD'](
                n_users=20,
                n_items=15,
                n_factors=5,
                learning_rate=0.05,
                loss_function=loss_fn
            )
            
            model.initialize_parameters()
            
            # 测试损失计算
            predictions = np.array([3.0, 4.0, 2.5])
            targets = np.array([3.5, 3.8, 2.2])
            
            loss = model.compute_loss(predictions, targets)
            assert np.isfinite(loss) and loss >= 0, f"{loss_name}损失计算异常"
            
            # 测试SGD更新
            sample_loss = model.sgd_update(0, 0, 3.5, 0)
            assert np.isfinite(sample_loss), f"{loss_name}的SGD更新异常"
            
            print(f"   ✓ {loss_name}集成测试通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 损失函数集成测试失败: {e}")
        return False

def test_model_persistence(modules):
    """测试模型持久化"""
    print("\n6. 测试模型保存和加载...")
    
    try:
        # 创建并训练模型
        model = modules['MatrixFactorizationSGD'](
            n_users=20,
            n_items=15,
            n_factors=5,
            learning_rate=0.1
        )
        
        model.initialize_parameters()
        
        # 创建少量训练数据
        train_data = np.column_stack([
            np.random.randint(0, 20, 50),
            np.random.randint(0, 15, 50),
            np.random.normal(3.5, 1.0, 50)
        ])
        
        model.fit(train_data, n_epochs=2, verbose=0)
        
        # 保存模型
        save_path = 'test_model.npz'
        model.save_model(save_path)
        
        assert os.path.exists(save_path), "模型文件未保存"
        print("   ✓ 模型保存成功")
        
        # 创建新模型并加载
        new_model = modules['MatrixFactorizationSGD'](1, 1, 1)  # 临时参数
        new_model.load_model(save_path)
        
        # 验证配置一致性
        assert new_model.n_users == model.n_users, "用户数量不一致"
        assert new_model.n_items == model.n_items, "物品数量不一致"
        assert new_model.n_factors == model.n_factors, "因子数量不一致"
        
        # 验证参数一致性
        np.testing.assert_allclose(new_model.user_factors, model.user_factors, 
                                 rtol=1e-5, err_msg="用户因子不一致")
        np.testing.assert_allclose(new_model.item_factors, model.item_factors, 
                                 rtol=1e-5, err_msg="物品因子不一致")
        
        print("   ✓ 模型加载成功")
        
        # 验证预测一致性
        test_users = np.array([0, 1])
        test_items = np.array([0, 1])
        
        pred1 = model.predict(test_users, test_items)
        pred2 = new_model.predict(test_users, test_items)
        
        np.testing.assert_allclose(pred1, pred2, rtol=1e-5, 
                                 err_msg="加载后预测不一致")
        
        print("   ✓ 预测一致性验证通过")
        
        # 清理文件
        os.unlink(save_path)
        config_path = save_path.replace('.npz', '_config.json')
        if os.path.exists(config_path):
            os.unlink(config_path)
        
        return True
        
    except Exception as e:
        print(f"   ✗ 模型持久化测试失败: {e}")
        # 清理可能的文件
        try:
            if os.path.exists('test_model.npz'):
                os.unlink('test_model.npz')
            if os.path.exists('test_model_config.json'):
                os.unlink('test_model_config.json')
        except:
            pass
        return False

def test_similarity_functions(modules):
    """测试相似度计算功能"""
    print("\n7. 测试相似度计算...")
    
    try:
        # 创建小模型用于测试
        model = modules['MatrixFactorizationSGD'](
            n_users=10,
            n_items=8,
            n_factors=3
        )
        
        model.initialize_parameters()
        
        # 测试物品相似度
        similar_items, scores = model.get_similar_items(0, n_similar=3)
        
        assert len(similar_items) == 3, "相似物品数量错误"
        assert len(scores) == 3, "相似度分数数量错误"
        assert 0 not in similar_items, "相似物品包含查询物品本身"
        assert all(0 <= item < 8 for item in similar_items), "相似物品ID超出范围"
        
        print("   ✓ 物品相似度计算通过")
        
        # 测试用户相似度
        similar_users, scores = model.get_similar_users(0, n_similar=3)
        
        assert len(similar_users) == 3, "相似用户数量错误"
        assert len(scores) == 3, "相似度分数数量错误"
        assert 0 not in similar_users, "相似用户包含查询用户本身"
        assert all(0 <= user < 10 for user in similar_users), "相似用户ID超出范围"
        
        print("   ✓ 用户相似度计算通过")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 相似度计算测试失败: {e}")
        return False

def test_edge_cases(modules):
    """测试边界情况"""
    print("\n8. 测试边界情况...")
    
    try:
        model = modules['MatrixFactorizationSGD'](
            n_users=5,
            n_items=5,
            n_factors=2
        )
        
        model.initialize_parameters()
        
        # 测试单样本预测
        single_pred = model.predict(0, 0)
        assert len(single_pred) == 1, "单样本预测长度错误"
        
        # 测试空数组输入（应该报错）
        try:
            model.predict(np.array([]), np.array([]))
            print("   ⚠ 空数组预测应该报错但没有")
        except:
            print("   ✓ 空数组输入正确处理")
        
        # 测试形状不匹配输入
        try:
            model.predict([0, 1], [0])  # 不同长度
            print("   ⚠ 形状不匹配应该报错但没有")
        except ValueError:
            print("   ✓ 形状不匹配正确处理")
        
        # 测试超出范围的索引
        try:
            model.predict([10], [0])  # 用户ID超出范围
            print("   ⚠ 索引超出范围可能导致问题")
        except IndexError:
            print("   ✓ 索引范围检查正确")
        except:
            print("   ✓ 超出范围索引被处理")
        
        return True
        
    except Exception as e:
        print(f"   ✗ 边界情况测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 60)
    print("Models模块快速功能测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        test_imports,
        test_initializers,
        test_regularizers,
        test_matrix_factorization_sgd,
        test_loss_function_integration,
        test_model_persistence,
        test_similarity_functions,
        test_edge_cases
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    modules = {}
    
    for i, test_func in enumerate(tests):
        try:
            if i == 0:  # 第一个测试返回模块字典
                success, modules = test_func()
            else:
                success = test_func(modules)
            
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"\n测试异常: {e}")
            traceback.print_exc()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过! Models模块功能正常")
        return True
    else:
        print("❌ 有测试失败，请检查相关功能")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)