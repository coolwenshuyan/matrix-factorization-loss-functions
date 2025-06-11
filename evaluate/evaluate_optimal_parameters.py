#!/usr/bin/env python3
# evaluate\evaluate_optimal_parameters.py
"""
使用最优参数进行模型评估并保存结果 - 支持批量损失函数评估
"""
import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import pandas as pd
from collections import defaultdict

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # 上一级目录才是项目根目录
sys.path.append(project_root)

print(f"项目根目录: {project_root}")
print(f"当前目录: {current_dir}")

# 导入模块 - 必须在添加项目根目录到sys.path之后
from utils.sms_notification import send_sms_notification
from data.data_manager import DataManager
from src.models.mf_sgd import MatrixFactorizationSGD
from src.models.initializers import NormalInitializer
from src.models.regularizers import L2Regularizer
# 导入所有损失函数
from src.losses.standard import L1Loss, L2Loss
from src.losses.hpl import HybridPiecewiseLoss
from src.losses.robust import HuberLoss, LogcoshLoss
from src.losses.sigmoid import SigmoidLikeLoss

# 导入标准evaluation模块
# from src.evaluation import (
#     MAE, RMSE, MSE, R2Score,
#     HitRate, Precision, Recall, MAP, NDCG, MRR,
#     CatalogCoverage, UserCoverage, Diversity,
#     Novelty, Serendipity, MetricFactory
# )
from src.evaluation.ranking import TopKGenerator
from src.evaluation.evaluator import ModelEvaluator

print("成功导入所有模块")


class OptimalParameterEvaluator:
    """使用最优参数评估模型性能"""

    def __init__(self, dataset_name, dataset_file):
        """初始化评估器"""
        self.dataset_name = dataset_name
        self.dataset_file = dataset_file
        self.data_manager = self._setup_data_manager()

        # 获取数据集显示名称
        self.dataset_display_name = self._get_dataset_display_name()

        # 固定保存到当前目录的evaluation_results中，以数据集名字命名
        self.results_dir = Path(current_dir) / 'evaluation_results' / self.dataset_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"结果将保存到: {self.results_dir}")

    def _get_dataset_display_name(self):
        """获取数据集显示名称"""
        if hasattr(self.data_manager.dataset, 'name'):
            return self.data_manager.dataset.name
        else:
            # 生成显示名称的映射
            display_names = {
                'ml100k': 'MovieLens 100K',
                'movielens100k': 'MovieLens 100K',
                'netflix': 'Netflix',
                'filmtrust': 'FilmTrust',
                'ciaodvd': 'CiaoDVD',
                'epinions': 'Epinions',
                'amazon': 'Amazon',
                'movielens1m': 'MovieLens 1M',
                'movietweetings': 'MovieTweetings'
            }
            return display_names.get(self.dataset_name.lower(), self.dataset_name.title())

    def _setup_data_manager(self):
        """设置数据管理器"""
        print(f"加载数据集: {self.dataset_name} ({self.dataset_file})")

        # 创建配置
        config = {
            'random_seed': 42,
            'train_ratio': 0.8,
            'val_ratio': 0.1,
            'test_ratio': 0.1,
            'center_data': True,
            'ensure_user_in_train': True
        }

        # 创建数据管理器
        data_manager = DataManager(config)

        # 加载和预处理数据（链式调用）
        data_manager.load_dataset_from_path(self.dataset_file, self.dataset_name).preprocess()

        # 打印数据摘要
        data_manager.print_summary()

        # 获取数据集划分
        train_data, val_data, test_data = data_manager.get_splits()

        print(f"数据加载成功:")
        print(f"  训练集: {len(train_data)} 条记录")
        print(f"  验证集: {len(val_data)} 条记录")
        print(f"  测试集: {len(test_data)} 条记录")

        return data_manager

    def load_config_from_file(self, config_file):
        """从配置文件加载完整配置信息"""
        print(f"加载配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取数据集信息
        dataset_info = data.get('dataset_info', {})
        dataset_name = dataset_info.get('dataset_name', '')
        dataset_display_name = dataset_info.get('dataset_display_name', '')

        # 提取最优配置
        if 'results' in data and 'best_config' in data['results']:
            best_config = data['results']['best_config']
            best_score = data['results'].get('best_score', 0)

            # 查找最佳试验的trial_id
            trial_id = 'N/A'
            if 'optimizer_results' in data['results'] and 'all_trials' in data['results']['optimizer_results']:
                all_trials = data['results']['optimizer_results']['all_trials']
                for trial in all_trials:
                    if abs(trial.get('score', 0) - best_score) < 1e-10:  # 浮点数比较
                        trial_id = trial.get('trial_id', 'N/A')
                        break
        elif 'best_config' in data:
            best_config = data['best_config']
            trial_id = data.get('best_trial_id', 'N/A')
        else:
            raise ValueError("无法从文件中提取最优配置")

        return {
            'dataset_name': dataset_name,
            'dataset_display_name': dataset_display_name,
            'best_config': best_config,
            'trial_id': trial_id
        }

    @staticmethod
    def get_available_loss_functions():
        """获取可用的损失函数列表"""
        return {
            'l1': {'class': L1Loss, 'name': 'L1损失 (Mean Absolute Error)', 'params': []},
            'l2': {'class': L2Loss, 'name': 'L2损失 (Mean Squared Error)', 'params': []},
            'hpl': {'class': HybridPiecewiseLoss, 'name': 'HPL混合分段损失',
                   'params': ['delta1', 'delta2', 'l_max', 'c_sigmoid']},
            'huber': {'class': HuberLoss, 'name': 'Huber损失', 'params': ['delta']},
            'logcosh': {'class': LogcoshLoss, 'name': 'LogCosh损失', 'params': []},
            'sigmoid': {'class': SigmoidLikeLoss, 'name': 'Sigmoid-like损失',
                       'params': ['alpha', 'l_max']}
        }

    @staticmethod
    def display_loss_functions():
        """显示可用的损失函数"""
        loss_functions = OptimalParameterEvaluator.get_available_loss_functions()
        print("\n" + "="*60)
        print("📊 可用损失函数列表")
        print("="*60)
        for i, (key, info) in enumerate(loss_functions.items(), 1):
            print(f"[{i}] {key.upper()}: {info['name']}")
        print(f"[0] 运行所有损失函数")
        print("="*60)

    def create_model(self, config, loss_type='l2'):
        """创建模型 - 添加错误处理"""
        try:
            # 获取损失函数信息
            loss_functions = self.get_available_loss_functions()

            if loss_type.lower() not in loss_functions:
                raise ValueError(f"不支持的损失函数类型: {loss_type}")

            loss_info = loss_functions[loss_type.lower()]
            loss_class = loss_info['class']

            # 验证配置参数
            required_params = loss_info.get('params', [])
            missing_params = []

            # 根据损失函数类型创建实例，添加参数验证
            if loss_type.lower() == 'hpl':
                # 验证HPL所需参数
                hpl_params = ['delta1', 'delta2', 'l_max', 'c_sigmoid']
                for param in hpl_params:
                    if param not in config:
                        missing_params.append(param)

                if missing_params:
                    print(f"警告: HPL损失函数缺少参数 {missing_params}，使用默认值")

                loss_function = loss_class(
                    delta1=config.get('delta1', 0.5),
                    delta2=config.get('delta2', 1.5),
                    l_max=config.get('l_max', 4.0),
                    c_sigmoid=config.get('c_sigmoid', 1.0)
                )
            elif loss_type.lower() == 'huber':
                if 'delta' not in config:
                    print("警告: Huber损失函数缺少delta参数，使用默认值1.0")
                loss_function = loss_class(delta=config.get('delta', 1.0))
            elif loss_type.lower() == 'sigmoid':
                sigmoid_params = ['alpha', 'l_max']
                for param in sigmoid_params:
                    if param not in config:
                        missing_params.append(param)

                if missing_params:
                    print(f"警告: Sigmoid损失函数缺少参数 {missing_params}，使用默认值")

                loss_function = loss_class(
                    alpha=config.get('alpha', 1.0),
                    l_max=config.get('l_max', 3.0)
                )
            elif loss_type.lower() == 'l1':
                loss_function = loss_class(epsilon=config.get('epsilon', 1e-8))
            else:  # l2, logcosh等无参数损失函数
                loss_function = loss_class()

            # 获取用户和物品数量
            stats = self.data_manager.get_statistics()
            n_users = stats['n_users']
            n_items = stats['n_items']

            # 验证基本配置参数
            if 'latent_factors' not in config:
                print("警告: 缺少latent_factors参数，使用默认值20")
            if 'learning_rate' not in config:
                print("警告: 缺少learning_rate参数，使用默认值0.01")
            if 'lambda_reg' not in config:
                print("警告: 缺少lambda_reg参数，使用默认值0.01")

            print(f"\n关键统计信息:")
            print(f"用户数: {stats['n_users']}")
            print(f"物品数: {stats['n_items']}")
            print(f"稀疏度: {stats['sparsity']:.4f}")
            print(f"平均评分: {stats['rating_mean']:.2f}")
            print(f"使用损失函数: {loss_info['name']}")

            # 创建模型
            model = MatrixFactorizationSGD(
                n_users=n_users,
                n_items=n_items,
                n_factors=config.get('latent_factors', 20),
                learning_rate=config.get('learning_rate', 0.01),
                loss_function=loss_function,
                regularizer=L2Regularizer(lambda_reg=config.get('lambda_reg', 0.01)),
                use_bias=True,
                global_mean=self.data_manager.global_mean or 0.0
            )

            # 初始化模型参数
            initializer = NormalInitializer(mean=0, std=0.1)
            model.initialize_parameters(initializer)

            return model

        except Exception as e:
            print(f"创建模型时出错: {e}")
            raise

    def train_and_evaluate(self, model, n_epochs=50):
        """训练和评估模型 - 使用标准evaluation模块"""
        try:
            print("="*60)
            print("开始训练和评估")
            #  添加调试步骤
            print("\n  预训练调试...")
            self.debug_recommendation_generation(model)

            if not hasattr(self.data_manager, 'train_data') or self.data_manager.train_data is None:
                raise ValueError("训练数据未加载")

            if not hasattr(self.data_manager, 'val_data') or self.data_manager.val_data is None:
                raise ValueError("验证数据未加载")

            # 打印模型配置参数
            print("\n" + "="*60)
            print("模型配置参数:")
            print("="*60)
            print(f"用户数量: {model.n_users}")
            print(f"物品数量: {model.n_items}")
            print(f"潜在因子数: {model.n_factors}")
            print(f"学习率: {model.learning_rate}")
            print(f"使用偏差项: {model.use_bias}")
            print(f"全局均值: {model.global_mean}")
            print(f"数据类型: {model.dtype}")

            # 损失函数信息
            if hasattr(model, 'loss_function') and model.loss_function is not None:
                print(f"损失函数: {model.loss_function.__class__.__name__}")
                if hasattr(model.loss_function, 'get_config'):
                    loss_config = model.loss_function.get_config()
                    for key, value in loss_config.items():
                        if key not in ['name', 'class']:
                            print(f"  {key}: {value}")
            else:
                print("损失函数: 默认MSE")

            # 正则化器信息
            if hasattr(model, 'regularizer') and model.regularizer is not None:
                print(f"正则化器: {model.regularizer.__class__.__name__}")
                if hasattr(model.regularizer, 'lambda_reg'):
                    print(f"  正则化系数: {model.regularizer.lambda_reg}")
            else:
                print("正则化器: 无")

            print(f"训练轮数: {n_epochs}")
            print("="*60)

            start_time = time.time()

            # 训练模型
            model.fit(
                train_data=self.data_manager.train_data,
                val_data=self.data_manager.val_data,
                n_epochs=n_epochs,
                verbose=1,
                early_stopping_patience=None  # 禁用早停
            )

            train_time = time.time() - start_time
            print(f"训练完成，耗时: {train_time:.2f}秒")

            # 训练后再次调试
            print("\n 训练后调试...")
            self.debug_recommendation_generation(model)
            # 评估模型
            print("评估模型...")
            evaluation_results = self.evaluate_model(model)
            evaluation_results['training_time'] = train_time

            return evaluation_results

        except Exception as e:
            print(f"训练和评估过程中出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回基本结果，避免程序崩溃
            return {
                'training_time': 0,
                'mae': float('inf'),
                'rmse': float('inf'),
                'error': str(e)
            }

    def evaluate_model(self, model):
        """全面评估模型性能 - 使用标准evaluation模块"""
        print("使用标准evaluation模块评估模型...")

        try:
            # 创建ModelEvaluator实例
            model_evaluator = ModelEvaluator(
                model=model,
                test_data=self.data_manager.test_data,
                k_values=[5, 10, 20],
                metrics=['mae', 'rmse', 'mse', 'r2', 'hr', 'precision', 'recall', 'map', 'ndcg', 'mrr'],
                n_users_sample=100  # 🔧 先限制为100个用户进行测试
            )

            # 执行完整评估
            results = model_evaluator.evaluate_all()

            # 计算额外的多样性指标
            additional_results = self._calculate_diversity_metrics(model)
            results.update(additional_results)

            return results

        except Exception as e:
            print(f"评估模型时出错: {e}")
            import traceback
            traceback.print_exc()

            # 返回基本结果
            return {
                'mae': float('inf'),
                'rmse': float('inf'),
                'error': str(e)
            }

    def _calculate_diversity_metrics(self, model):
        """计算多样性相关指标 - 修复版本"""
        results = {}

        try:
            # 🔧 创建TopKGenerator实例
            topk_gen = TopKGenerator(model)

            # 获取测试集用户（限制数量以提高效率）
            unique_users = np.unique(self.data_manager.test_data[:, 0].astype(int))

            if len(unique_users) > 200:  # 限制用户数量
                np.random.seed(42)
                unique_users = np.random.choice(unique_users, 200, replace=False)

            print(f"评估 {len(unique_users)} 个用户的多样性指标...")

            # 获取训练集用户物品
            train_user_items = self._get_train_user_items()

            # 生成推荐列表
            all_recommendations = []
            successful_users = 0
            total_items = model.n_items

            for user_id in unique_users:
                try:
                    # 获取该用户的已知物品
                    known_items = set(train_user_items.get(user_id, []))

                    # 🔧 关键修复：检查候选物品数量
                    candidate_items = total_items - len(known_items)

                    if candidate_items < 20:  # 如果候选物品太少
                        # 不排除已知物品，或只排除部分
                        print(f"用户{user_id}候选物品不足({candidate_items})，降低排除策略")
                        exclude_items = None  # 不排除任何物品
                    else:
                        exclude_items = known_items

                    # 生成推荐
                    user_recs = topk_gen.generate_top_k_for_user(
                        user_id=user_id,
                        k=10,
                        exclude_items=exclude_items
                    )

                    if len(user_recs) >= 5:  # 确保至少有5个推荐
                        all_recommendations.extend(user_recs)
                        successful_users += 1
                    else:
                        print(f"用户{user_id}推荐数量不足: {len(user_recs)}")
                except Exception as e:
                    print(f"为用户{user_id}生成推荐失败: {e}")
                    continue

            print(f"成功为 {successful_users}/{len(unique_users)} 个用户生成推荐")

            # 🔧 关键检查：如果成功用户太少，返回0
            if successful_users < len(unique_users) * 0.1:  # 少于10%的用户成功
                print("⚠️ 推荐生成成功率过低，多样性指标设为0")
                return {metric: 0.0 for metric in ['catalog_coverage', 'user_coverage', 'diversity', 'novelty', 'serendipity']}

            # 准备评估数据
            stats = self.data_manager.get_statistics()
            additional_data = {
                'recommendations': {user_id: all_recommendations[i*10:(i+1)*10]
                                  for i, user_id in enumerate(unique_users[:successful_users])},
                'n_items': stats['n_items'],
                'n_users': successful_users,
                'train_data': self.data_manager.train_data
            }

            if not all_recommendations:
                print("⚠️ 没有生成任何推荐，多样性指标设为0")
                return {metric: 0.0 for metric in ['catalog_coverage', 'user_coverage', 'diversity', 'novelty', 'serendipity']}


            # 计算多样性指标
            diversity_metrics = ['catalog_coverage', 'user_coverage', 'diversity', 'novelty', 'serendipity']

            for metric_name in diversity_metrics:
                try:
                    # 🔧 修复：根据指标类型提供合适的数据
                    if metric_name == 'catalog_coverage':
                        # 计算目录覆盖率
                        unique_items = set(all_recommendations)
                        results[metric_name] = len(unique_items) / stats['n_items']

                    elif metric_name == 'user_coverage':
                        # 计算用户覆盖率
                        results[metric_name] = successful_users / len(unique_users)

                    elif metric_name == 'diversity':
                        # 计算多样性（熵）
                        from collections import Counter
                        if all_recommendations:
                            item_counts = Counter(all_recommendations)
                            total_recs = len(all_recommendations)
                            diversity = 0.0
                            for count in item_counts.values():
                                p = count / total_recs
                                if p > 0:
                                    diversity -= p * np.log2(p)
                            max_diversity = np.log2(len(set(all_recommendations))) if len(set(all_recommendations)) > 1 else 1
                            results[metric_name] = diversity / max_diversity if max_diversity > 0 else 0.0
                        else:
                            results[metric_name] = 0.0

                    elif metric_name == 'novelty':
                        # 计算新颖性
                        if all_recommendations:
                            from collections import Counter
                            item_popularity = Counter()
                            for row in self.data_manager.train_data:
                                item_id = int(row[1])
                                item_popularity[item_id] += 1

                            total_interactions = len(self.data_manager.train_data)
                            novelty_scores = []

                            for item in all_recommendations:
                                popularity = item_popularity.get(item, 0)
                                novelty = -np.log2((popularity + 1) / (total_interactions + len(item_popularity)))
                                novelty_scores.append(novelty)

                            if novelty_scores:
                                max_novelty = -np.log2(1 / (total_interactions + len(item_popularity)))
                                results[metric_name] = np.mean(novelty_scores) / max_novelty if max_novelty > 0 else 0
                            else:
                                results[metric_name] = 0.0
                        else:
                            results[metric_name] = 0.0

                    elif metric_name == 'serendipity':
                        # 意外性（简化为新颖性和多样性的乘积）
                        results[metric_name] = results.get('novelty', 0) * results.get('diversity', 0)

                    print(f"✅ {metric_name}: {results[metric_name]:.4f}")

                except Exception as e:
                    print(f"❌ 计算{metric_name}指标时出错: {e}")
                    results[metric_name] = 0.0

        except Exception as e:
            print(f"计算多样性指标时出错: {e}")
            # 设置默认值
            for metric_name in ['catalog_coverage', 'user_coverage', 'diversity', 'novelty', 'serendipity']:
                results[metric_name] = 0.0

        return results

    def _get_train_user_items(self):
        """获取训练集中用户的物品"""
        user_items = {}
        for row in self.data_manager.train_data:
            user_id = int(row[0])
            item_id = int(row[1])
            if user_id not in user_items:
                user_items[user_id] = []
            user_items[user_id].append(item_id)
        return user_items

    def save_results(self, results, config, loss_type='l2', trial_id='N/A'):
        """保存评估结果"""
        # 创建结果字典
        save_data = {
            'dataset_info': {
                'dataset_name': self.dataset_name,
                'dataset_display_name': self.dataset_display_name,
                'dataset_file': self.dataset_file,
            },
            'model_config': config,
            'loss_type': loss_type,
            'best_trial_id': trial_id,
            'evaluation_results': results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # 保存为JSON文件
        filename = f"{self.dataset_name}_{loss_type}_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        save_path = self.results_dir / filename

        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"结果已保存至: {save_path}")

        # 同时保存为CSV格式，方便后续分析
        results_df = pd.DataFrame([results])
        results_df['loss_type'] = loss_type
        results_df['trial_id'] = trial_id

        # 添加配置参数到DataFrame
        for param, value in config.items():
            results_df[f'config_{param}'] = value

        csv_path = self.results_dir / f"{self.dataset_name}_{loss_type}_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(csv_path, index=False)

        print(f"CSV结果已保存至: {csv_path}")

        return save_path

    def save_combined_results(self, all_results):
        """保存所有损失函数的合并结果"""
        # 创建综合结果DataFrame
        combined_df = pd.DataFrame()

        for loss_type, results in all_results.items():
            results_copy = results.copy()
            results_copy['loss_type'] = loss_type
            combined_df = pd.concat([combined_df, pd.DataFrame([results_copy])], ignore_index=True)

        # 保存综合CSV文件
        combined_csv_path = self.results_dir / f"{self.dataset_name}_all_losses_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"\n综合结果CSV已保存至: {combined_csv_path}")

        # 保存综合JSON文件
        combined_json = {
            'dataset_info': {
                'dataset_name': self.dataset_name,
                'dataset_display_name': self.dataset_display_name,
                'dataset_file': self.dataset_file,
            },
            'results_by_loss': all_results,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

        combined_json_path = self.results_dir / f"{self.dataset_name}_all_losses_evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(combined_json_path, 'w', encoding='utf-8') as f:
            json.dump(combined_json, f, indent=2, ensure_ascii=False)
        print(f"综合结果JSON已保存至: {combined_json_path}")


    def debug_recommendation_generation(self, model):
        """调试推荐生成过程"""
        print("\n 调试推荐生成...")

        # 随机选择5个用户测试
        test_users = np.random.choice(
            np.unique(self.data_manager.test_data[:, 0].astype(int)),
            min(5, len(np.unique(self.data_manager.test_data[:, 0]))),
            replace=False
        )

        train_user_items = self._get_train_user_items()

        for user_id in test_users:
            print(f"\n用户 {user_id}:")

            # 检查训练集交互
            known_items = train_user_items.get(user_id, [])
            print(f"  训练集已知物品数: {len(known_items)}")
            print(f"  候选物品数: {model.n_items - len(known_items)}")

            # 尝试生成推荐
            try:
                topk_gen = TopKGenerator(model)

                # 先尝试排除已知物品
                recs_excluded = topk_gen.generate_top_k_for_user(
                    user_id=user_id,
                    k=10,
                    exclude_items=set(known_items)
                )
                print(f"  排除已知物品后推荐数: {len(recs_excluded)}")

                # 再尝试不排除
                recs_all = topk_gen.generate_top_k_for_user(
                    user_id=user_id,
                    k=10,
                    exclude_items=None
                )
                print(f"  不排除任何物品推荐数: {len(recs_all)}")

                if len(recs_excluded) < 5:
                    print(f"  ⚠️ 该用户推荐生成有问题！")

            except Exception as e:
                print(f"  ❌ 推荐生成失败: {e}")

def main():
    """主函数"""
    # 获取用户输入
    print("="*60)
    print("最优参数评估工具 - 支持批量损失函数评估")
    print("="*60)

    # 记录实验开始时间
    experiment_start_time = time.time()

    # 1. 输入配置文件路径
    config_file = input("请输入配置文件的完整路径: ").strip()
    if not os.path.exists(config_file):
        print(f"错误: 文件不存在 {config_file}")
        return

    # 2. 选择损失函数
    OptimalParameterEvaluator.display_loss_functions()
    loss_functions = OptimalParameterEvaluator.get_available_loss_functions()
    loss_keys = list(loss_functions.keys())

    while True:
        try:
            choice = input(f"请选择损失函数 (0-{len(loss_keys)}, 默认0运行所有): ").strip()
            if not choice:  # 用户直接按回车，默认运行所有
                choice = "0"

            choice_idx = int(choice)
            if choice_idx == 0:
                selected_losses = loss_keys  # 运行所有损失函数
                break
            elif 1 <= choice_idx <= len(loss_keys):
                selected_losses = [loss_keys[choice_idx - 1]]  # 运行单个损失函数
                break
            else:
                print(f"请输入0到{len(loss_keys)}之间的数字")
        except ValueError:
            print("请输入有效的数字")

    # 3. 输入数据集文件的完整路径
    dataset_file = input("请输入数据集文件的完整路径: ").strip()
    if not os.path.exists(dataset_file):
        print(f"错误: 文件不存在 {dataset_file}")
        return

    try:
        # 先加载配置文件获取基本信息（不创建数据管理器）
        print(f"加载配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 提取数据集信息
        dataset_info = data.get('dataset_info', {})
        dataset_name = dataset_info.get('dataset_name', '')
        dataset_display_name = dataset_info.get('dataset_display_name', '')

        # 提取最优配置和trial_id
        if 'results' in data and 'best_config' in data['results']:
            best_config = data['results']['best_config']
            best_score = data['results'].get('best_score', 0)

            # 查找最佳试验的trial_id
            trial_id = 'N/A'
            if 'optimizer_results' in data['results'] and 'all_trials' in data['results']['optimizer_results']:
                all_trials = data['results']['optimizer_results']['all_trials']
                for trial in all_trials:
                    if abs(trial.get('score', 0) - best_score) < 1e-10:  # 浮点数比较
                        trial_id = trial.get('trial_id', 'N/A')
                        break
        elif 'best_config' in data:
            best_config = data['best_config']
            trial_id = data.get('best_trial_id', 'N/A')
        else:
            raise ValueError("无法从文件中提取最优配置")

        # 4. 设置训练轮数（默认为trial_id）
        default_epochs = trial_id if isinstance(trial_id, int) else 50
        n_epochs = int(input(f"请输入训练轮数 (默认: {default_epochs}): ").strip() or str(default_epochs))

        config_info = {
            'dataset_name': dataset_name,
            'dataset_display_name': dataset_display_name,
            'best_config': best_config,
            'trial_id': trial_id
        }

        print("\n从配置文件中读取的信息:")
        print(f"数据集名称: {config_info['dataset_name']}")
        print(f"数据集显示名称: {config_info['dataset_display_name']}")
        print(f"最优迭代次数 (trial_id): {config_info['trial_id']}")
        print(f"选择的损失函数: {', '.join(selected_losses) if len(selected_losses) > 1 else loss_functions[selected_losses[0]]['name']}")
        print(f"最优参数配置:")
        for key, value in config_info['best_config'].items():
            print(f"  {key}: {value}")

        # 创建正式的评估器
        evaluator = OptimalParameterEvaluator(
            config_info['dataset_name'],
            dataset_file
        )

        # 存储所有损失函数的结果
        all_results = {}

        # 对每个选择的损失函数进行评估
        for loss_type in selected_losses:
            print(f"\n{'='*80}")
            print(f"开始评估损失函数: {loss_functions[loss_type]['name']}")
            print(f"{'='*80}")

            # 创建模型
            model = evaluator.create_model(
                config_info['best_config'],
                loss_type
            )

            # 训练和评估模型
            results = evaluator.train_and_evaluate(model, n_epochs)

            # 打印评估结果
            print("\n评估结果:")
            print("-" * 60)

            # 评分预测指标
            print("评分预测指标:")
            for metric in ['mae', 'rmse', 'mse', 'r2']:
                if metric in results:
                    print(f"  {metric.upper()}: {results[metric]:.4f}")

            # 排序指标
            print("\n排序指标:")
            for k in [5, 10, 20]:
                print(f"  @{k}:")
                for metric in ['hr', 'precision', 'recall', 'map', 'ndcg']:
                    key = f'{metric}@{k}'
                    if key in results:
                        print(f"    {metric.upper()}: {results[key]:.4f}")

            # MRR指标
            if 'mrr' in results:
                print(f"  MRR: {results['mrr']:.4f}")

            # 多样性和覆盖度指标
            print("\n多样性和覆盖度指标:")
            for metric in ['catalog_coverage', 'user_coverage', 'diversity', 'novelty', 'serendipity']:
                if metric in results:
                    print(f"  {metric.replace('_', ' ').title()}: {results[metric]:.4f}")

            print(f"\n训练时间: {results.get('training_time', 0):.2f} 秒")

            # 保存单个损失函数的结果
            evaluator.save_results(
                results,
                config_info['best_config'],
                loss_type,
                config_info['trial_id']
            )

            # 存储结果用于后续合并
            all_results[loss_type] = results

            # 如果是多个损失函数评估，发送单个损失函数完成通知
            if len(selected_losses) > 1:
                single_loss_message = f"✅ 损失函数 {loss_functions[loss_type]['name']} 评估完成\n" \
                                     f"数据集: {config_info['dataset_display_name']}\n" \
                                     f"RMSE: {results.get('rmse', 0):.4f}\n" \
                                     f"MAE: {results.get('mae', 0):.4f}\n" \
                                     f"HR@10: {results.get('hr@10', 0):.4f}\n" \
                                     f"训练时间: {results.get('training_time', 0):.2f}秒"

                try:
                    send_sms_notification(single_loss_message)
                    print(f"✅ {loss_type.upper()} 完成通知已发送")
                except Exception as e:
                    print(f"⚠️ 发送 {loss_type.upper()} 完成通知失败: {e}")

        # 如果评估了多个损失函数，保存合并结果
        if len(selected_losses) > 1:
            print(f"\n{'='*80}")
            print("保存所有损失函数的综合结果...")
            evaluator.save_combined_results(all_results)

            # 打印所有损失函数的对比摘要
            print(f"\n{'='*80}")
            print("损失函数性能对比摘要:")
            print(f"{'='*80}")

            # 创建对比表格
            print(f"{'损失函数':<15} {'RMSE':<10} {'MAE':<10} {'HR@10':<12} {'NDCG@10':<10} {'训练时间(秒)':<12}")
            print("-" * 80)

            for loss_type in selected_losses:
                if loss_type in all_results:
                    r = all_results[loss_type]
                    print(f"{loss_type.upper():<15} "
                          f"{r.get('rmse', 0):<10.4f} "
                          f"{r.get('mae', 0):<10.4f} "
                          f"{r.get('hr@10', 0):<12.4f} "
                          f"{r.get('ndcg@10', 0):<10.4f} "
                          f"{r.get('training_time', 0):<12.2f}")

        print(f"\n{'='*80}")
        print("所有评估完成!")
        print(f"结果保存在: {evaluator.results_dir}")

        # 计算总实验时间
        experiment_end_time = time.time()
        total_experiment_time = experiment_end_time - experiment_start_time

        # 发送实验完成总结通知
        if len(selected_losses) == 1:
            # 单个损失函数的完成通知
            loss_type = selected_losses[0]
            results = all_results[loss_type]
            completion_message = f"🎉 最优参数评估实验完成!\n" \
                               f"数据集: {config_info['dataset_display_name']}\n" \
                               f"损失函数: {loss_functions[loss_type]['name']}\n" \
                               f"RMSE: {results.get('rmse', 0):.4f}\n" \
                               f"MAE: {results.get('mae', 0):.4f}\n" \
                               f"HR@10: {results.get('hr@10', 0):.4f}\n" \
                               f"NDCG@10: {results.get('ndcg@10', 0):.4f}\n" \
                               f"总耗时: {total_experiment_time/60:.1f}分钟"
        else:
            # 多个损失函数的完成通知 - 找到最佳表现
            best_loss_type = ""
            best_rmse = float('inf')
            for loss_type in selected_losses:
                if loss_type in all_results:
                    rmse = all_results[loss_type].get('rmse', float('inf'))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_loss_type = loss_type

            completion_message = f"🎉 批量评估实验完成!\n" \
                               f"数据集: {config_info['dataset_display_name']}\n" \
                               f"评估了 {len(selected_losses)} 个损失函数\n" \
                               f"最佳损失函数: {loss_functions[best_loss_type]['name']}\n" \
                               f"最佳RMSE: {best_rmse:.4f}\n" \
                               f"总耗时: {total_experiment_time/60:.1f}分钟\n" \
                               f"结果已保存到: {evaluator.results_dir.name}"

        try:
            send_sms_notification(completion_message)
            print("✅ 实验完成通知已发送")
        except Exception as e:
            print(f"⚠️ 发送完成通知失败: {e}")

    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

        # 发送错误通知
        error_message = f"❌ 最优参数评估实验出错!\n" \
                       f"数据集: {config_info.get('dataset_display_name', '未知') if 'config_info' in locals() else '未知'}\n" \
                       f"错误信息: {str(e)[:100]}...\n" \
                       f"发生时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"

        try:
            send_sms_notification(error_message)
            print("✅ 错误通知已发送")
        except Exception as sms_e:
            print(f"⚠️ 发送错误通知失败: {sms_e}")


if __name__ == "__main__":
    main()
