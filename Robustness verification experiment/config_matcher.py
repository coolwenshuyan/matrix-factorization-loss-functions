#!/usr/bin/env python3
"""
配置匹配器模块
负责从优化结果中匹配和加载最优配置
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class OptimalConfig:
    """最优配置类"""

    def __init__(self, dataset_name: str, config_data: Dict[str, Any],
                 source_file: str = None, confidence: float = 1.0):
        self.dataset_name = dataset_name
        self.config_data = config_data
        self.source_file = source_file
        self.confidence = confidence  # 配置匹配的置信度
        self.timestamp = self._extract_timestamp()

    def _extract_timestamp(self) -> Optional[datetime]:
        """从文件名中提取时间戳"""
        if not self.source_file:
            return None

        try:
            # 尝试从文件名中提取时间戳
            timestamp_pattern = r'(\d{8}_\d{6})'
            match = re.search(timestamp_pattern, self.source_file)
            if match:
                timestamp_str = match.group(1)
                return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        except:
            pass

        return None

    def get_best_config(self) -> Dict[str, Any]:
        """获取最佳配置参数"""
        if 'results' in self.config_data and 'best_config' in self.config_data['results']:
            return self.config_data['results']['best_config']
        elif 'best_config' in self.config_data:
            return self.config_data['best_config']
        else:
            return {}

    def get_best_score(self) -> Optional[float]:
        """获取最佳得分"""
        if 'results' in self.config_data and 'best_score' in self.config_data['results']:
            return self.config_data['results']['best_score']
        elif 'best_score' in self.config_data:
            return self.config_data['best_score']
        else:
            return None


class ConfigMatcher:
    """配置匹配器"""

    def __init__(self, results_dir: str = "Optimal parameter experiment/results"):
        self.results_dir = Path(results_dir)
        self.configs: Dict[str, List[OptimalConfig]] = {}
        self.dataset_aliases = {
            'ml100k': ['movielens100k', 'm100k', '100k'],
            'movielens100k': ['ml100k', 'm100k', '100k'],
            'netflix': ['nf'],
            'filmtrust': ['flimtrust', 'ft'],
            'flimtrust': ['filmtrust', 'ft'],
            'ciaodvd': ['ciao'],
            'movielens1m': ['ml1m', 'm1m', '1m'],
            'amazon': ['amz']
        }
        self._scan_config_files()

    def _scan_config_files(self):
        """扫描配置文件"""
        if not self.results_dir.exists():
            logger.warning(f"结果目录不存在: {self.results_dir}")
            return

        logger.info(f"扫描配置目录: {self.results_dir}")

        # 递归搜索JSON文件
        json_files = list(self.results_dir.rglob("*.json"))

        for json_file in json_files:
            try:
                config = self._parse_config_file(json_file)
                if config:
                    dataset_name = config.dataset_name
                    if dataset_name not in self.configs:
                        self.configs[dataset_name] = []
                    self.configs[dataset_name].append(config)
            except Exception as e:
                logger.warning(f"解析配置文件失败 {json_file}: {e}")

        # 按时间戳排序（最新的在前）
        for dataset_configs in self.configs.values():
            dataset_configs.sort(key=lambda x: x.timestamp or datetime.min, reverse=True)

        logger.info(f"加载了 {sum(len(configs) for configs in self.configs.values())} 个配置文件")

    def _parse_config_file(self, json_file: Path) -> Optional[OptimalConfig]:
        """解析配置文件"""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 识别数据集名称
            dataset_name = self._identify_dataset_from_file(json_file, data)
            if not dataset_name:
                return None

            # 验证配置完整性
            if not self._validate_config_data(data):
                logger.warning(f"配置文件格式不完整: {json_file}")
                return None

            return OptimalConfig(
                dataset_name=dataset_name,
                config_data=data,
                source_file=str(json_file),
                confidence=1.0
            )

        except json.JSONDecodeError as e:
            logger.error(f"JSON格式错误 {json_file}: {e}")
            return None
        except Exception as e:
            logger.error(f"读取配置文件失败 {json_file}: {e}")
            return None

    def _identify_dataset_from_file(self, json_file: Path, data: Dict) -> Optional[str]:
        """从文件路径和内容识别数据集名称"""
        # 1. 从文件路径识别
        path_parts = json_file.parts
        for part in path_parts:
            part_lower = part.lower()
            for dataset_name, aliases in self.dataset_aliases.items():
                if (dataset_name in part_lower or
                    any(alias in part_lower for alias in aliases)):
                    return dataset_name

        # 2. 从文件名识别
        filename = json_file.stem.lower()
        for dataset_name, aliases in self.dataset_aliases.items():
            if (dataset_name in filename or
                any(alias in filename for alias in aliases)):
                return dataset_name

        # 3. 从配置内容识别
        if 'dataset_info' in data:
            dataset_info = data['dataset_info']
            if 'dataset_name' in dataset_info:
                return dataset_info['dataset_name'].lower()
            if 'dataset_file' in dataset_info:
                file_path = dataset_info['dataset_file'].lower()
                for dataset_name, aliases in self.dataset_aliases.items():
                    if (dataset_name in file_path or
                        any(alias in file_path for alias in aliases)):
                        return dataset_name

        # 4. 从实验信息识别
        if 'experiment_info' in data:
            exp_info = data['experiment_info']
            if 'dataset' in exp_info:
                return exp_info['dataset'].lower()

        return None

    def _validate_config_data(self, data: Dict) -> bool:
        """验证配置数据完整性"""
        # 检查是否有最佳配置
        has_best_config = (
            ('results' in data and 'best_config' in data['results']) or
            'best_config' in data
        )

        if not has_best_config:
            return False

        # 检查最佳配置的关键参数
        best_config = data.get('results', {}).get('best_config') or data.get('best_config', {})
        required_params = ['learning_rate', 'latent_factors', 'lambda_reg']

        return all(param in best_config for param in required_params)

    def find_config(self, dataset_name: str) -> Optional[OptimalConfig]:
        """查找数据集的最优配置"""
        dataset_name_lower = dataset_name.lower()

        # 1. 精确匹配
        if dataset_name_lower in self.configs:
            return self.configs[dataset_name_lower][0]  # 返回最新的配置

        # 2. 别名匹配
        for stored_name, aliases in self.dataset_aliases.items():
            if dataset_name_lower in aliases and stored_name in self.configs:
                return self.configs[stored_name][0]

        # 3. 模糊匹配
        for stored_name in self.configs.keys():
            if (dataset_name_lower in stored_name or
                stored_name in dataset_name_lower):
                logger.info(f"通过模糊匹配找到配置: {dataset_name} -> {stored_name}")
                config = self.configs[stored_name][0]
                config.confidence = 0.8  # 降低置信度
                return config

        return None

    def get_similar_config(self, dataset_name: str) -> Optional[OptimalConfig]:
        """获取相似数据集的配置"""
        # 定义数据集相似性
        similarity_groups = [
            ['ml100k', 'movielens100k', 'movielens1m'],  # MovieLens系列
            ['filmtrust', 'flimtrust', 'ciaodvd'],       # 电影评分系列
            ['netflix', 'amazon', 'epinions']            # 大型评分系列
        ]

        dataset_name_lower = dataset_name.lower()

        # 查找相似组
        for group in similarity_groups:
            if any(name in dataset_name_lower for name in group):
                # 在同组中查找可用配置
                for similar_name in group:
                    if similar_name in self.configs:
                        logger.info(f"使用相似数据集配置: {dataset_name} -> {similar_name}")
                        config = self.configs[similar_name][0]
                        config.confidence = 0.6  # 进一步降低置信度
                        return config

        # 如果没有找到相似的，返回任意一个配置
        if self.configs:
            logger.info(f"使用默认配置: {dataset_name}")
            first_config = next(iter(self.configs.values()))[0]
            first_config.confidence = 0.3  # 最低置信度
            return first_config

        return None

    def build_model_configs(self, optimal_config: OptimalConfig) -> List[Dict[str, Any]]:
        """基于最优配置构建不同损失函数的模型配置"""
        base_config = optimal_config.get_best_config()

        # 提取基础参数
        base_params = {
            'n_factors': base_config.get('latent_factors', 50),
            'learning_rate': base_config.get('learning_rate', 0.01),
            'n_epochs': 100,  # 可以从配置中获取或使用默认值
            'use_bias': True,
            'regularizer': {
                'type': 'L2',
                'lambda_reg': base_config.get('lambda_reg', 0.01)
            },
            'initializer': {'type': 'Normal', 'mean': 0.0, 'std': 0.01}
        }

        # 构建不同损失函数的配置
        model_configs = []

        # 1. HPL优化模型
        hpl_config = base_params.copy()
        hpl_config.update({
            'name': 'HPL_optimized',
            'description': '优化后的HPL损失函数模型',
            'loss_function': {
                'type': 'HPL',
                'delta1': base_config.get('delta1', 0.5),
                'delta2': base_config.get('delta2', 2.0),
                'l_max': base_config.get('l_max', 3.0)
            }
        })
        model_configs.append(hpl_config)

        # 2. L2基线模型
        l2_config = base_params.copy()
        l2_config.update({
            'name': 'L2_baseline',
            'description': 'L2损失函数基线模型',
            'loss_function': {'type': 'L2'}
        })
        model_configs.append(l2_config)

        # 3. L1基线模型
        l1_config = base_params.copy()
        l1_config.update({
            'name': 'L1_baseline',
            'description': 'L1损失函数基线模型',
            'loss_function': {'type': 'L1'}
        })
        model_configs.append(l1_config)

        # 4. Huber损失模型
        huber_config = base_params.copy()
        huber_config.update({
            'name': 'Huber_robust',
            'description': 'Huber损失函数鲁棒模型',
            'loss_function': {'type': 'Huber', 'delta': 1.0}
        })
        model_configs.append(huber_config)

        # 5. Logcosh损失模型
        logcosh_config = base_params.copy()
        logcosh_config.update({
            'name': 'Logcosh_robust',
            'description': 'Logcosh损失函数鲁棒模型',
            'loss_function': {'type': 'Logcosh'}
        })
        model_configs.append(logcosh_config)

        # 6. SigmoidLike损失模型
        sigmoid_config = base_params.copy()
        sigmoid_config.update({
            'name': 'Sigmoid_robust',
            'description': 'Sigmoid类损失函数模型',
            'loss_function': {
                'type': 'SigmoidLike',
                'alpha': 1.0,
                'l_max': base_config.get('l_max', 3.0)
            }
        })
        model_configs.append(sigmoid_config)

        return model_configs

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'learning_rate': 0.01,
            'latent_factors': 50,
            'lambda_reg': 0.01,
            'delta1': 0.5,
            'delta2': 2.0,
            'l_max': 3.0
        }

    def list_available_configs(self) -> Dict[str, List[str]]:
        """列出可用的配置"""
        result = {}
        for dataset_name, configs in self.configs.items():
            result[dataset_name] = [
                f"{config.source_file} (confidence: {config.confidence:.2f})"
                for config in configs
            ]
        return result

    def print_config_summary(self):
        """打印配置摘要"""
        print("\n" + "="*80)
        print("📋 可用配置摘要")
        print("="*80)

        if not self.configs:
            print("❌ 未发现任何配置文件")
            return

        for dataset_name, configs in self.configs.items():
            print(f"\n📊 {dataset_name.upper()}:")
            for i, config in enumerate(configs, 1):
                best_score = config.get_best_score()
                timestamp = config.timestamp.strftime('%Y-%m-%d %H:%M') if config.timestamp else 'Unknown'
                # 修复格式化问题
                score_str = f"{best_score:.4f}" if best_score is not None else 'N/A'
                print(f"  [{i}] 得分: {score_str} | "
                      f"时间: {timestamp} | 置信度: {config.confidence:.2f}")
                print(f"      文件: {Path(config.source_file).name if config.source_file else 'N/A'}")

        print("="*80)

    def print_model_configs(self, model_configs: List[Dict[str, Any]]):
        """打印模型配置信息"""
        print("\n" + "="*80)
        print("🔧 生成的模型配置详情")
        print("="*80)

        for i, config in enumerate(model_configs, 1):
            print(f"\n【模型 {i}】{config['name']}")
            print(f"描述: {config['description']}")
            print("-" * 60)

            # 基础参数
            print("📊 基础参数:")
            print(f"  • 潜在因子数量: {config['n_factors']}")
            print(f"  • 学习率: {config['learning_rate']}")
            print(f"  • 训练轮数: {config['n_epochs']}")
            print(f"  • 使用偏置: {config['use_bias']}")

            # 损失函数参数
            print("🎯 损失函数:")
            loss_func = config['loss_function']
            print(f"  • 类型: {loss_func['type']}")

            if loss_func['type'] == 'HPL':
                print(f"  • Delta1: {loss_func['delta1']}")
                print(f"  • Delta2: {loss_func['delta2']}")
                print(f"  • L_max: {loss_func['l_max']}")
            elif loss_func['type'] == 'Huber':
                print(f"  • Delta: {loss_func['delta']}")
            elif loss_func['type'] == 'SigmoidLike':
                print(f"  • Alpha: {loss_func['alpha']}")
                print(f"  • L_max: {loss_func['l_max']}")

            # 正则化参数
            print("🛡️ 正则化:")
            reg = config['regularizer']
            print(f"  • 类型: {reg['type']}")
            print(f"  • Lambda: {reg['lambda_reg']}")

            # 初始化参数
            print("🎲 初始化:")
            init = config['initializer']
            print(f"  • 类型: {init['type']}")
            print(f"  • 均值: {init['mean']}")
            print(f"  • 标准差: {init['std']}")

        print("="*80)

def main():
    """测试函数"""
    matcher = ConfigMatcher()
    matcher.print_config_summary()

    # 测试配置查找
    test_datasets = ['ml100k', 'netflix', 'filmtrust']
    for dataset in test_datasets:
        print(f"\n🔍 查找 {dataset} 的配置:")
        config = matcher.find_config(dataset)
        if config:
            print(f"  ✅ 找到配置 (置信度: {config.confidence:.2f})")

            # 生成模型配置
            model_configs = matcher.build_model_configs(config)
            print(f"  📋 生成了 {len(model_configs)} 个模型配置")

            # 打印详细的模型配置信息
            matcher.print_model_configs(model_configs)

            # 打印最优配置的原始参数
            print(f"\n📈 {dataset} 的最优参数:")
            best_config = config.get_best_config()
            for key, value in best_config.items():
                print(f"  • {key}: {value}")

            best_score = config.get_best_score()
            if best_score:
                print(f"  • 最佳得分: {best_score:.4f}")

        else:
            print(f"  ❌ 未找到配置，尝试相似配置...")
            similar_config = matcher.get_similar_config(dataset)
            if similar_config:
                print(f"  🔄 使用相似配置 (置信度: {similar_config.confidence:.2f})")
                model_configs = matcher.build_model_configs(similar_config)
                matcher.print_model_configs(model_configs)
            else:
                print(f"  ❌ 无可用配置")


if __name__ == "__main__":
    main()


