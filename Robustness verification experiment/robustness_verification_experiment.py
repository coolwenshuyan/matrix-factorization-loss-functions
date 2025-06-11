#!/usr/bin/env python3
"""
鲁棒性验证实验主程序
Robustness Verification Experiment Main Program

该模块整合数据集选择、配置匹配、噪声注入和实验执行功能，
提供完整的矩阵分解模型鲁棒性验证实验流程。

主要功能:
1. 自动化数据集选择和验证
2. 智能配置匹配和优化
3. 多种噪声类型的鲁棒性测试
4. 实验结果的统计分析和可视化
5. 完整的实验报告生成

作者: 鲁棒性验证实验系统
版本: 1.0
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# 添加项目根目录到系统路径
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# 导入必要的模块
from dataset_selector import DatasetSelector
from config_matcher import ConfigMatcher
# from experiment_executor import ExperimentExecutor  # 将在函数内部导入
from noise_injection_system import NoiseConfig, ExperimentRunner
from result_manager import ResultManager  # 导入ResultManager
from utils.sms_notification import send_sms_notification

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_noise_config(config_path):
    """加载噪声配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"成功加载噪声配置: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载噪声配置失败: {e}")
        return None


def main():
    """主函数"""
    print("\n" + "="*80)
    print("🧪 矩阵分解模型鲁棒性验证实验系统")
    print("="*80)

    try:
        # 步骤1: 数据集选择
        print("\n📊 步骤1: 数据集选择")

        # 设置正确的数据集路径（使用项目根目录）
        dataset_path = project_root / "dataset"
        print(f"  🔍 当前目录: {current_dir.absolute()}")
        print(f"  🔍 项目根目录: {project_root.absolute()}")
        print(f"  🔍 数据集路径: {dataset_path.absolute()}")
        print(f"  🔍 数据集路径存在: {dataset_path.exists()}")
        dataset_selector = DatasetSelector(str(dataset_path))
        selected_datasets = dataset_selector.get_user_selection()

        if not selected_datasets:
            print("❌ 未选择任何数据集，实验终止")
            return

        print(f"✅ 已选择 {len(selected_datasets)} 个数据集:")
        for dataset in selected_datasets:
            print(f"  • {dataset.display_name} ({dataset.name})")

        # 步骤2: 配置匹配
        print("\n⚙️ 步骤2: 配置匹配")
        config_matcher = ConfigMatcher()
        config_matcher.print_config_summary()

        # 为每个数据集查找最优配置
        matched_configs = []
        dataset_model_configs = {}  # 存储每个数据集的模型配置

        for dataset in selected_datasets:
            print(f"\n🔍 查找 {dataset.display_name} 的配置:")
            config = config_matcher.find_config(dataset.name)

            if config:
                print(f"  ✅ 找到配置 (置信度: {config.confidence:.2f})")
                matched_configs.append(config)
            else:
                print(f"  ❌ 未找到配置，尝试查找相似配置...")
                similar_config = config_matcher.get_similar_config(dataset.name)
                if similar_config:
                    print(f"  🔄 使用相似配置 (置信度: {similar_config.confidence:.2f})")
                    config = similar_config
                    matched_configs.append(config)
                else:
                    print(f"  ⚠️ 无可用配置，使用默认配置")
                    # 创建默认配置
                    from config_matcher import OptimalConfig
                    config = OptimalConfig(
                        dataset_name=dataset.name,
                        config_data={'results': {'best_config': config_matcher.get_default_config()}}
                    )
                    matched_configs.append(config)

            # 生成不同损失函数的模型配置
            print(f"\n📋 为 {dataset.display_name} 生成模型配置:")
            model_configs = config_matcher.build_model_configs(config)
            dataset_model_configs[dataset.name] = model_configs
            print(f"  ✅ 生成了 {len(model_configs)} 个模型配置")

            # 打印模型配置详情
            config_matcher.print_model_configs(model_configs)

        # 步骤3: 加载噪声配置
        print("\n🔊 步骤3: 加载噪声配置")
        noise_config_path = Path(current_dir) / "configs" / "noise_config.json"

        if not noise_config_path.exists():
            print(f"  ⚠️ 噪声配置文件不存在: {noise_config_path}")
            print("  ⚠️ 将使用默认噪声配置")
            noise_config = None
        else:
            noise_config = load_noise_config(noise_config_path)
            if noise_config:
                print(f"  ✅ 成功加载噪声配置")
                print(f"  📊 实验名称: {noise_config.get('experiment_name', '未命名实验')}")
                print(f"  📊 噪声实验数量: {len(noise_config.get('noise_experiments', []))}")

                # 打印噪声实验概要
                for i, exp in enumerate(noise_config.get('noise_experiments', []), 1):
                    print(f"    {i}. {exp.get('name', f'实验{i}')} - {exp.get('description', '无描述')}")
                    print(f"       噪声类型: {', '.join(exp.get('noise_types', []))}")
                    print(f"       噪声强度: {exp.get('noise_strengths', [])}")
                    print(f"       影响比例: {exp.get('noise_ratios', [])}")
            else:
                print("  ⚠️ 加载噪声配置失败，将使用默认配置")

        # 步骤4: 实验执行
        print("\n🔬 步骤4: 实验执行")

        # 清理缓存文件以确保使用最新代码
        import shutil
        cache_dir = Path(current_dir) / "__pycache__"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            print("  🧹 清理缓存文件")

        # 强制重新导入模块
        import importlib
        if 'experiment_executor' in sys.modules:
            importlib.reload(sys.modules['experiment_executor'])
            print("  🔄 重新加载实验执行器模块")

        # 创建实验执行器
        from experiment_executor import ExperimentExecutor
        executor = ExperimentExecutor()

        # 如果有噪声配置，设置噪声实验配置
        if noise_config and 'noise_experiments' in noise_config:
            executor.noise_experiment_configs = noise_config['noise_experiments']
            print(f"  ✅ 使用加载的噪声配置: {len(executor.noise_experiment_configs)} 个实验")
        else:
            print(f"  ℹ️ 使用默认噪声配置: {len(executor.noise_experiment_configs)} 个实验")

        # 为每个数据集执行实验
        dataset_results = {}
        for dataset_idx, dataset in enumerate(selected_datasets):
            dataset_config = matched_configs[dataset_idx]

            # 执行单个数据集的实验
            print(f"\n🔬 执行 {dataset.display_name} 的实验:")

            # 准备数据
            print(f"  📊 准备 {dataset.display_name} 数据...")

            # 对每个模型配置执行实验
            model_configs = dataset_model_configs[dataset.name]
            for model_idx, model_config in enumerate(model_configs):
                print(f"  🧪 执行模型 {model_idx+1}/{len(model_configs)}: {model_config['name']}")

                # 对每个噪声实验配置执行实验
                noise_experiments = noise_config.get('noise_experiments', executor.noise_experiment_configs) if noise_config else executor.noise_experiment_configs
                for exp_idx, exp_config in enumerate(noise_experiments):
                    print(f"    🔊 噪声实验 {exp_idx+1}/{len(noise_experiments)}: {exp_config.get('name', f'实验{exp_idx+1}')}")

                    # 创建噪声配置列表
                    noise_configs = []
                    for noise_type in exp_config.get('noise_types', []):
                        for strength in exp_config.get('noise_strengths', []):
                            for ratio in exp_config.get('noise_ratios', []):
                                noise_configs.append(NoiseConfig(
                                    noise_type=noise_type,
                                    noise_strength=strength,
                                    noise_ratio=ratio
                                ))

                    print(f"      📊 生成了 {len(noise_configs)} 个噪声配置")

                    # 导入单个实验执行模块
                    from execute_single_experiment import execute_single_experiment

                    # 实际执行实验
                    try:
                        # 创建实验名称
                        experiment_name = f"{dataset.name}_{model_config['name']}_{exp_config.get('name', 'exp')}"

                        # 执行单个实验
                        result = execute_single_experiment(
                            dataset=dataset,
                            model_config=model_config,
                            noise_configs=noise_configs,
                            experiment_name=experiment_name
                        )

                        # 保存结果
                        if result and result['status'] == 'success':
                            if dataset.name not in dataset_results:
                                dataset_results[dataset.name] = {}
                            if model_config['name'] not in dataset_results[dataset.name]:
                                dataset_results[dataset.name][model_config['name']] = {}

                            dataset_results[dataset.name][model_config['name']][exp_config.get('name', f'实验{exp_idx+1}')] = result
                            print(f"      ✅ 噪声实验完成并保存结果: {result['result_path']}")
                        else:
                            print(f"      ⚠️ 噪声实验失败: {result.get('error', '未知错误')}")
                    except Exception as exp_error:
                        print(f"      ❌ 噪声实验执行失败: {exp_error}")
                        logger.error(f"噪声实验执行失败: {exp_error}")

                print(f"  ✅ 模型 {model_config['name']} 实验完成")

            print(f"✅ 数据集 {dataset.display_name} 实验完成")

        print("\n🎉 所有实验执行完毕!")

        # 实验完成后发送短信通知
        experiment_summary = f"鲁棒性验证实验已完成，数据集: {', '.join(selected_datasets)}"
        send_sms_notification(experiment_summary)

        print("\n✅ 实验完成，已发送短信通知!")

    except KeyboardInterrupt:
        print("\n\n👋 用户中断实验")
    except Exception as e:
        logger.exception(f"实验执行过程中发生错误: {e}")
        print(f"\n❌ 实验失败: {e}")
        send_sms_notification(f"鲁棒性验证实验失败: {str(e)[:50]}...")


if __name__ == "__main__":
    main()







