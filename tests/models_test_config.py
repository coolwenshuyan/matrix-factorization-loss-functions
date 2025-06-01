#!/usr/bin/env python3
"""
Models模块测试配置和运行脚本
提供不同级别的测试和配置选项
"""

import sys
import os
import argparse
import time

# 获取项目根目录
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class TestConfiguration:
    """测试配置类"""
    
    @staticmethod
    def get_minimal_config():
        """最小测试配置 - 快速验证核心功能"""
        return {
            'test_groups': ['imports', 'basic_functionality'],
            'model_params': {
                'n_users': 10,
                'n_items': 8,
                'n_factors': 3,
                'n_epochs': 2,
                'n_samples': 50
            },
            'timeout': 30,
            'description': '最小测试集 - 验证基本导入和核心功能'
        }
    
    @staticmethod
    def get_standard_config():
        """标准测试配置 - 完整功能测试"""
        return {
            'test_groups': ['imports', 'initializers', 'regularizers', 
                           'sgd_model', 'loss_integration', 'persistence'],
            'model_params': {
                'n_users': 50,
                'n_items': 30,
                'n_factors': 10,
                'n_epochs': 5,
                'n_samples': 200
            },
            'timeout': 120,
            'description': '标准测试集 - 完整功能验证'
        }
    
    @staticmethod
    def get_comprehensive_config():
        """全面测试配置 - 包括边界情况和性能测试"""
        return {
            'test_groups': ['imports', 'initializers', 'regularizers', 
                           'sgd_model', 'loss_integration', 'persistence',
                           'similarity', 'edge_cases', 'performance'],
            'model_params': {
                'n_users': 100,
                'n_items': 80,
                'n_factors': 20,
                'n_epochs': 10,
                'n_samples': 1000
            },
            'timeout': 300,
            'description': '全面测试集 - 包括性能和边界情况'
        }
    
    @staticmethod
    def get_loss_function_config():
        """损失函数专项测试配置"""
        return {
            'test_groups': ['imports', 'loss_integration', 'hpl_specific'],
            'model_params': {
                'n_users': 30,
                'n_items': 20,
                'n_factors': 8,
                'n_epochs': 8,
                'n_samples': 300
            },
            'loss_functions': [
                ('L2Loss', {}),
                ('L1Loss', {}),
                ('HybridPiecewiseLoss', {'delta1': 0.5, 'delta2': 2.0, 'l_max': 3.0}),
                ('HuberLoss', {'delta': 1.0}),
                ('LogcoshLoss', {})
            ],
            'timeout': 180,
            'description': '损失函数专项测试 - 重点验证各种损失函数的集成'
        }


class TestRunner:
    """测试运行器"""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
        self.start_time = None
        
    def run_tests(self, verbose=True):
        """运行测试"""
        if verbose:
            print(f"运行测试配置: {self.config['description']}")
            print(f"测试组: {', '.join(self.config['test_groups'])}")
            print("-" * 60)
        
        self.start_time = time.time()
        
        try:
            # 导入测试模块
            if 'imports' in self.config['test_groups']:
                self._test_imports(verbose)
            
            # 其他测试组
            if 'initializers' in self.config['test_groups']:
                self._test_initializers(verbose)
            
            if 'regularizers' in self.config['test_groups']:
                self._test_regularizers(verbose)
            
            if 'basic_functionality' in self.config['test_groups']:
                self._test_basic_functionality(verbose)
            
            if 'sgd_model' in self.config['test_groups']:
                self._test_sgd_model(verbose)
            
            if 'loss_integration' in self.config['test_groups']:
                self._test_loss_integration(verbose)
            
            if 'persistence' in self.config['test_groups']:
                self._test_persistence(verbose)
            
            if 'similarity' in self.config['test_groups']:
                self._test_similarity(verbose)
            
            if 'edge_cases' in self.config['test_groups']:
                self._test_edge_cases(verbose)
            
            if 'performance' in self.config['test_groups']:
                self._test_performance(verbose)
            
            if 'hpl_specific' in self.config['test_groups']:
                self._test_hpl_specific(verbose)
                
        except Exception as e:
            print(f"测试运行异常: {e}")
            return False
        
        return self._summarize_results(verbose)
    
    def _test_imports(self, verbose):
        """测试导入"""
        if verbose:
            print("测试模块导入...")
        
        try:
            from src.models import (
                BaseMatrixFactorization, MatrixFactorizationSGD,
                L2Regularizer, L1Regularizer, 
                NormalInitializer, XavierInitializer
            )
            from src.losses import L2Loss, L1Loss, HybridPiecewiseLoss
            
            self.results['imports'] = True
            if verbose:
                print("  ✓ 模块导入成功")
            
            # 保存模块引用
            self.modules = {
                'MatrixFactorizationSGD': MatrixFactorizationSGD,
                'L2Regularizer': L2Regularizer,
                'L1Regularizer': L1Regularizer,
                'NormalInitializer': NormalInitializer,
                'XavierInitializer': XavierInitializer,
                'L2Loss': L2Loss,
                'L1Loss': L1Loss,
                'HybridPiecewiseLoss': HybridPiecewiseLoss
            }
            
        except Exception as e:
            self.results['imports'] = False
            if verbose:
                print(f"  ✗ 导入失败: {e}")
    
    def _test_initializers(self, verbose):
        """测试初始化器"""
        if verbose:
            print("测试参数初始化器...")
        
        try:
            # 测试正态分布初始化
            normal_init = self.modules['NormalInitializer'](std=0.1, random_seed=42)
            params = normal_init.initialize((20, 5))
            assert params.shape == (20, 5)
            assert abs(np.std(params) - 0.1) < 0.02
            
            # 测试Xavier初始化
            xavier_init = self.modules['XavierInitializer'](random_seed=42)
            params = xavier_init.initialize((10, 10))
            assert params.shape == (10, 10)
            
            self.results['initializers'] = True
            if verbose:
                print("  ✓ 初始化器测试通过")
                
        except Exception as e:
            self.results['initializers'] = False
            if verbose:
                print(f"  ✗ 初始化器测试失败: {e}")
    
    def _test_regularizers(self, verbose):
        """测试正则化器"""
        if verbose:
            print("测试正则化器...")
        
        try:
            import numpy as np
            
            test_params = {
                'user_factors': np.random.randn(10, 5).astype(np.float32),
                'item_factors': np.random.randn(8, 5).astype(np.float32)
            }
            
            # 测试L2正则化
            l2_reg = self.modules['L2Regularizer'](lambda_reg=0.01)
            penalty = l2_reg.compute_penalty(test_params)
            assert penalty > 0
            
            # 测试梯度
            grad = l2_reg.compute_gradient('user_factors', test_params['user_factors'])
            assert grad.shape == test_params['user_factors'].shape
            
            self.results['regularizers'] = True
            if verbose:
                print("  ✓ 正则化器测试通过")
                
        except Exception as e:
            self.results['regularizers'] = False
            if verbose:
                print(f"  ✗ 正则化器测试失败: {e}")
    
    def _test_basic_functionality(self, verbose):
        """测试基本功能"""
        if verbose:
            print("测试基本功能...")
        
        try:
            import numpy as np
            
            params = self.config['model_params']
            
            # 创建模型
            model = self.modules['MatrixFactorizationSGD'](
                n_users=params['n_users'],
                n_items=params['n_items'],
                n_factors=params['n_factors']
            )
            
            # 初始化参数
            model.initialize_parameters()
            
            # 测试预测
            predictions = model.predict([0, 1], [0, 1])
            assert len(predictions) == 2
            assert all(np.isfinite(p) for p in predictions)
            
            self.results['basic_functionality'] = True
            if verbose:
                print("  ✓ 基本功能测试通过")
                
        except Exception as e:
            self.results['basic_functionality'] = False
            if verbose:
                print(f"  ✗ 基本功能测试失败: {e}")
    
    def _test_sgd_model(self, verbose):
        """测试SGD模型"""
        if verbose:
            print("测试SGD模型...")
        
        try:
            import numpy as np
            
            params = self.config['model_params']
            
            model = self.modules['MatrixFactorizationSGD'](
                n_users=params['n_users'],
                n_items=params['n_items'],
                n_factors=params['n_factors'],
                learning_rate=0.05,
                regularizer=self.modules['L2Regularizer'](lambda_reg=0.01),
                loss_function=self.modules['L2Loss']()
            )
            
            # 创建训练数据
            train_data = np.column_stack([
                np.random.randint(0, params['n_users'], params['n_samples']),
                np.random.randint(0, params['n_items'], params['n_samples']),
                np.random.normal(3.5, 1.0, params['n_samples'])
            ])
            
            # 训练模型
            model.fit(train_data, n_epochs=params['n_epochs'], verbose=0)
            
            # 验证训练结果
            assert len(model.train_history['loss']) > 0
            
            self.results['sgd_model'] = True
            if verbose:
                print("  ✓ SGD模型测试通过")
                
        except Exception as e:
            self.results['sgd_model'] = False
            if verbose:
                print(f"  ✗ SGD模型测试失败: {e}")
    
    def _test_loss_integration(self, verbose):
        """测试损失函数集成"""
        if verbose:
            print("测试损失函数集成...")
        
        try:
            import numpy as np
            
            params = self.config['model_params']
            loss_functions = [
                self.modules['L2Loss'](),
                self.modules['L1Loss'](),
                self.modules['HybridPiecewiseLoss'](delta1=0.5, delta2=2.0, l_max=3.0)
            ]
            
            for loss_fn in loss_functions:
                model = self.modules['MatrixFactorizationSGD'](
                    n_users=params['n_users'],
                    n_items=params['n_items'], 
                    n_factors=params['n_factors'],
                    loss_function=loss_fn
                )
                
                model.initialize_parameters()
                
                # 测试损失计算
                predictions = np.array([3.0, 4.0])
                targets = np.array([3.5, 3.8])
                loss = model.compute_loss(predictions, targets)
                
                assert np.isfinite(loss) and loss >= 0
            
            self.results['loss_integration'] = True
            if verbose:
                print("  ✓ 损失函数集成测试通过")
                
        except Exception as e:
            self.results['loss_integration'] = False
            if verbose:
                print(f"  ✗ 损失函数集成测试失败: {e}")
    
    def _test_persistence(self, verbose):
        """测试模型持久化"""
        if verbose:
            print("测试模型持久化...")
        
        try:
            import numpy as np
            
            params = self.config['model_params']
            
            # 创建和训练模型
            model = self.modules['MatrixFactorizationSGD'](
                n_users=params['n_users'],
                n_items=params['n_items'],
                n_factors=params['n_factors']
            )
            
            model.initialize_parameters()
            
            # 保存模型
            save_path = 'test_model_temp.npz'
            model.save_model(save_path)
            
            # 加载模型
            new_model = self.modules['MatrixFactorizationSGD'](1, 1, 1)
            new_model.load_model(save_path)
            
            # 验证一致性
            assert new_model.n_users == model.n_users
            assert new_model.n_items == model.n_items
            
            # 清理文件
            os.unlink(save_path)
            config_path = save_path.replace('.npz', '_config.json')
            if os.path.exists(config_path):
                os.unlink(config_path)
            
            self.results['persistence'] = True
            if verbose:
                print("  ✓ 模型持久化测试通过")
                
        except Exception as e:
            self.results['persistence'] = False
            if verbose:
                print(f"  ✗ 模型持久化测试失败: {e}")
    
    def _test_similarity(self, verbose):
        """测试相似度计算"""
        if verbose:
            print("测试相似度计算...")
        
        try:
            params = self.config['model_params']
            
            model = self.modules['MatrixFactorizationSGD'](
                n_users=params['n_users'],
                n_items=params['n_items'],
                n_factors=params['n_factors']
            )
            
            model.initialize_parameters()
            
            # 测试物品相似度
            similar_items, scores = model.get_similar_items(0, n_similar=3)
            assert len(similar_items) == 3
            assert 0 not in similar_items
            
            # 测试用户相似度
            similar_users, scores = model.get_similar_users(0, n_similar=3) 
            assert len(similar_users) == 3
            assert 0 not in similar_users
            
            self.results['similarity'] = True
            if verbose:
                print("  ✓ 相似度计算测试通过")
                
        except Exception as e:
            self.results['similarity'] = False
            if verbose:
                print(f"  ✗ 相似度计算测试失败: {e}")
    
    def _test_edge_cases(self, verbose):
        """测试边界情况"""
        if verbose:
            print("测试边界情况...")
        
        try:
            import numpy as np
            
            model = self.modules['MatrixFactorizationSGD'](
                n_users=5, n_items=5, n_factors=2
            )
            model.initialize_parameters()
            
            # 测试单样本预测
            pred = model.predict(0, 0)
            assert len(pred) == 1
            
            # 测试形状不匹配
            try:
                model.predict([0, 1], [0])
                edge_case_passed = False
            except ValueError:
                edge_case_passed = True
            
            assert edge_case_passed, "应该捕获形状不匹配错误"
            
            self.results['edge_cases'] = True
            if verbose:
                print("  ✓ 边界情况测试通过")
                
        except Exception as e:
            self.results['edge_cases'] = False
            if verbose:
                print(f"  ✗ 边界情况测试失败: {e}")
    
    def _test_performance(self, verbose):
        """测试性能"""
        if verbose:
            print("测试性能...")
        
        try:
            import numpy as np
            import time
            
            params = self.config['model_params']
            
            # 创建较大的模型进行性能测试
            large_model = self.modules['MatrixFactorizationSGD'](
                n_users=params['n_users'] * 2,
                n_items=params['n_items'] * 2,
                n_factors=params['n_factors']
            )
            
            large_model.initialize_parameters()
            
            # 测试预测性能
            start_time = time.time()
            
            test_users = np.random.randint(0, params['n_users'] * 2, 100)
            test_items = np.random.randint(0, params['n_items'] * 2, 100)
            predictions = large_model.predict(test_users, test_items)
            
            prediction_time = time.time() - start_time
            
            # 验证预测时间合理（应该很快）
            assert prediction_time < 1.0, f"预测时间过长: {prediction_time:.3f}秒"
            assert len(predictions) == 100
            
            self.results['performance'] = True
            if verbose:
                print(f"  ✓ 性能测试通过 (预测时间: {prediction_time:.3f}秒)")
                
        except Exception as e:
            self.results['performance'] = False
            if verbose:
                print(f"  ✗ 性能测试失败: {e}")
    
    def _test_hpl_specific(self, verbose):
        """测试HPL特定功能"""
        if verbose:
            print("测试HPL特定功能...")
        
        try:
            import numpy as np
            
            params = self.config['model_params']
            
            # 测试不同HPL参数配置
            hpl_configs = [
                {'delta1': 0.5, 'delta2': 2.0, 'l_max': 3.0},
                {'delta1': 1.0, 'delta2': 3.0, 'l_max': 5.0},
                {'delta1': 0.3, 'delta2': 1.5, 'l_max': 2.0}
            ]
            
            for config in hpl_configs:
                hpl_loss = self.modules['HybridPiecewiseLoss'](**config)
                
                model = self.modules['MatrixFactorizationSGD'](
                    n_users=params['n_users'],
                    n_items=params['n_items'],
                    n_factors=params['n_factors'],
                    loss_function=hpl_loss
                )
                
                model.initialize_parameters()
                
                # 测试HPL损失计算
                predictions = np.array([1.0, 3.0, 5.0])
                targets = np.array([1.5, 2.5, 4.0])
                loss = model.compute_loss(predictions, targets)
                
                assert np.isfinite(loss) and loss >= 0
                assert loss <= config['l_max'] * 1.1  # 允许一定误差
                
                # 测试HPL的连续性验证
                continuity = hpl_loss.verify_continuity()
                assert all(continuity.values()), f"HPL连续性验证失败: {continuity}"
            
            self.results['hpl_specific'] = True
            if verbose:
                print("  ✓ HPL特定功能测试通过")
                
        except Exception as e:
            self.results['hpl_specific'] = False
            if verbose:
                print(f"  ✗ HPL特定功能测试失败: {e}")
    
    def _summarize_results(self, verbose):
        """总结测试结果"""
        end_time = time.time()
        total_time = end_time - self.start_time
        
        passed_count = sum(1 for result in self.results.values() if result)
        total_count = len(self.results)
        
        if verbose:
            print("\n" + "=" * 60)
            print("测试结果总结")
            print("=" * 60)
            print(f"总耗时: {total_time:.2f}秒")
            print(f"通过测试: {passed_count}/{total_count}")
            
            for test_name, passed in self.results.items():
                status = "✓" if passed else "✗"
                print(f"  {status} {test_name}")
            
            if passed_count == total_count:
                print("\n🎉 所有测试通过!")
            else:
                print(f"\n❌ {total_count - passed_count} 个测试失败")
            
            print("=" * 60)
        
        return passed_count == total_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Models模块测试运行器')
    
    parser.add_argument(
        '--config', 
        choices=['minimal', 'standard', 'comprehensive', 'loss_function'],
        default='standard',
        help='选择测试配置'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='详细输出'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='静默模式'
    )
    
    args = parser.parse_args()
    
    # 获取配置
    config_map = {
        'minimal': TestConfiguration.get_minimal_config(),
        'standard': TestConfiguration.get_standard_config(),
        'comprehensive': TestConfiguration.get_comprehensive_config(),
        'loss_function': TestConfiguration.get_loss_function_config()
    }
    
    config = config_map[args.config]
    verbose = args.verbose and not args.quiet
    
    if verbose:
        print("Models模块测试运行器")
        print("=" * 60)
        print(f"选择的配置: {args.config}")
        print(f"描述: {config['description']}")
        print(f"预计耗时: 最多 {config['timeout']} 秒")
        print()
    
    # 运行测试
    runner = TestRunner(config)
    success = runner.run_tests(verbose=verbose)
    
    if not verbose:
        # 简化输出模式
        passed = sum(1 for r in runner.results.values() if r)
        total = len(runner.results)
        print(f"测试结果: {passed}/{total} {'通过' if success else '失败'}")
    
    return 0 if success else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)