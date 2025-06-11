#!/usr/bin/env python3
"""
结果管理器模块
负责实验结果的存储、组织和管理
"""

import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
import hashlib
import numpy as np

logger = logging.getLogger(__name__)


class DateTimeEncoder(json.JSONEncoder):
    """自定义JSON编码器，用于处理datetime对象和数值类型"""
    
    def default(self, obj):
        # 处理datetime对象
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # 处理NumPy数值类型
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        
        # 处理复数类型
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        
        # 处理set类型
        elif isinstance(obj, set):
            return list(obj)
        
        # 处理bytes类型
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
            
        return super().default(obj)


class ResultManager:
    """结果管理器"""

    def __init__(self, base_dir: str):
        """
        初始化结果管理器

        Args:
            base_dir: 基础目录
        """
        # 确保base_dir是Path对象
        self.base_dir = Path(base_dir)
        self.structure = {
            'experiments': 'experiments',      # 具体实验结果
            'summaries': 'summaries',         # 实验摘要
            'metadata': 'metadata',           # 元数据
            'reports': 'reports',             # 报告文件
            'visualizations': 'visualizations', # 可视化文件
            'backups': 'backups'              # 备份文件
        }
        self._create_directory_structure()

    def _create_directory_structure(self):
        """创建目录结构"""
        for dir_name in self.structure.values():
            (self.base_dir / dir_name).mkdir(exist_ok=True)

    def save_experiment_metadata(self, metadata: Dict[str, Any], dataset_name: str = None) -> str:
        """
        保存实验元数据

        Args:
            metadata: 元数据字典
            dataset_name: 数据集名称（可选）

        Returns:
            保存的文件路径
        """
        # 生成时间戳文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果提供了数据集名称，将其添加到文件名中
        if dataset_name:
            filename = f"experiment_metadata_{dataset_name}_{timestamp}.json"
        else:
            filename = f"experiment_metadata_{timestamp}.json"

        # 确保使用Path对象进行路径操作
        file_path = self.base_dir / self.structure['metadata'] / filename

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 添加系统信息
        enhanced_metadata = metadata.copy()
        enhanced_metadata.update({
            'save_time': datetime.now().isoformat(),
            'file_path': str(file_path),
            'version': '1.0',
            'dataset_name': dataset_name
        })

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理datetime对象
                json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

            logger.info(f"元数据已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
            raise

    def save_experiment_results(self,
                               results: Dict[str, Any],
                               experiment_name: str,
                               dataset_name: str = None,
                               experiment_type: str = "robustness") -> str:
        """
        保存实验结果

        Args:
            results: 实验结果
            experiment_name: 实验名称
            dataset_name: 数据集名称（可选）
            experiment_type: 实验类型

        Returns:
            保存的文件路径
        """
        # 创建实验目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果提供了数据集名称，将其添加到目录名中
        if dataset_name:
            exp_dir_name = f"{experiment_type}_{dataset_name}_{experiment_name}_{timestamp}"
        else:
            exp_dir_name = f"{experiment_type}_{experiment_name}_{timestamp}"

        # 确保使用Path对象进行路径操作
        exp_dir = self.base_dir / self.structure['experiments'] / exp_dir_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # 保存主要结果
        main_result_file = exp_dir / "results.json"
        enhanced_results = results.copy()
        enhanced_results.update({
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'dataset_name': dataset_name,
            'save_time': datetime.now().isoformat(),
            'result_hash': self._calculate_result_hash(results)
        })

        try:
            with open(main_result_file, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理datetime对象
                json.dump(enhanced_results, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

            # 保存简化版本（只包含关键信息）
            summary_file = exp_dir / "summary.json"
            summary = self._extract_result_summary(enhanced_results)
            with open(summary_file, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理datetime对象
                json.dump(summary, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

            logger.info(f"实验结果已保存: {exp_dir}")
            return str(exp_dir)

        except Exception as e:
            logger.error(f"保存实验结果失败: {e}")
            raise

    def save_experiment_summary(self, summary: Dict[str, Any], dataset_name: str = None) -> str:
        """
        保存实验摘要

        Args:
            summary: 实验摘要
            dataset_name: 数据集名称（可选）

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果提供了数据集名称，将其添加到文件名中
        if dataset_name:
            filename = f"experiment_summary_{dataset_name}_{timestamp}.json"
        else:
            filename = f"experiment_summary_{timestamp}.json"

        # 确保使用Path对象进行路径操作
        file_path = self.base_dir / self.structure['summaries'] / filename

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 增强摘要信息
        enhanced_summary = summary.copy()
        enhanced_summary.update({
            'summary_generated_time': datetime.now().isoformat(),
            'summary_version': '1.0',
            'dataset_name': dataset_name
        })

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # 使用自定义编码器处理datetime对象
                json.dump(enhanced_summary, f, indent=2, ensure_ascii=False, cls=DateTimeEncoder)

            logger.info(f"实验摘要已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"保存实验摘要失败: {e}")
            raise

    def save_robustness_report(self,
                              report_content: str,
                              experiment_name: str,
                              report_type: str = "text",
                              dataset_name: str = None) -> str:
        """
        保存鲁棒性报告

        Args:
            report_content: 报告内容
            experiment_name: 实验名称
            report_type: 报告类型 (text, html, markdown)
            dataset_name: 数据集名称（可选）

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果提供了数据集名称，将其添加到文件名中
        if dataset_name:
            filename = f"{dataset_name}_{experiment_name}_report_{timestamp}.{report_type}"
        else:
            filename = f"{experiment_name}_report_{timestamp}.{report_type}"

        # 确保使用Path对象进行路径操作
        file_path = self.base_dir / self.structure['reports'] / filename

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            logger.info(f"报告已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"保存报告失败: {e}")
            raise

    def save_visualization(self,
                          figure_object: Any,
                          experiment_name: str,
                          plot_type: str,
                          format: str = "png",
                          dataset_name: str = None) -> str:
        """
        保存可视化图表

        Args:
            figure_object: matplotlib figure对象
            experiment_name: 实验名称
            plot_type: 图表类型
            format: 保存格式 (png, pdf, svg)
            dataset_name: 数据集名称（可选）

        Returns:
            保存的文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 如果提供了数据集名称，将其添加到文件名中
        if dataset_name:
            filename = f"{dataset_name}_{experiment_name}_{plot_type}_{timestamp}.{format}"
        else:
            filename = f"{experiment_name}_{plot_type}_{timestamp}.{format}"

        # 确保使用Path对象进行路径操作
        file_path = self.base_dir / self.structure['visualizations'] / filename

        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            figure_object.savefig(file_path, dpi=300, bbox_inches='tight')
            logger.info(f"可视化图表已保存: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"保存可视化图表失败: {e}")
            raise

    def load_experiment_results(self, experiment_path: str) -> Dict[str, Any]:
        """
        加载实验结果

        Args:
            experiment_path: 实验路径

        Returns:
            实验结果字典
        """
        exp_path = Path(experiment_path)

        if not exp_path.exists():
            raise FileNotFoundError(f"实验路径不存在: {experiment_path}")

        # 尝试加载主要结果文件
        result_file = exp_path / "results.json"
        if result_file.exists():
            try:
                with open(result_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.info(f"加载实验结果: {result_file}")
                return results
            except Exception as e:
                logger.error(f"加载实验结果失败: {e}")
                raise

        # 如果主要结果文件不存在，尝试加载摘要
        summary_file = exp_path / "summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                logger.warning(f"只找到摘要文件: {summary_file}")
                return results
            except Exception as e:
                logger.error(f"加载摘要文件失败: {e}")
                raise

        raise FileNotFoundError(f"在 {experiment_path} 中未找到有效的结果文件")

    def list_experiments(self, experiment_type: str = None) -> List[Dict[str, Any]]:
        """
        列出所有实验

        Args:
            experiment_type: 实验类型过滤（可选）

        Returns:
            实验信息列表
        """
        experiments = []
        exp_dir = self.base_dir / self.structure['experiments']

        if not exp_dir.exists():
            return experiments

        for exp_folder in exp_dir.iterdir():
            if exp_folder.is_dir():
                try:
                    # 尝试加载实验信息
                    summary_file = exp_folder / "summary.json"
                    result_file = exp_folder / "results.json"

                    exp_info = {
                        'name': exp_folder.name,
                        'path': str(exp_folder),
                        'created_time': datetime.fromtimestamp(exp_folder.stat().st_ctime).isoformat()
                    }

                    # 尝试从摘要文件获取信息
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                        exp_info.update({
                            'experiment_type': summary.get('experiment_type', 'unknown'),
                            'experiment_name': summary.get('experiment_name', 'unknown'),
                            'status': summary.get('status', 'unknown')
                        })
                    elif result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            results = json.load(f)
                        exp_info.update({
                            'experiment_type': results.get('experiment_type', 'unknown'),
                            'experiment_name': results.get('experiment_name', 'unknown'),
                            'status': 'completed'
                        })

                    # 应用类型过滤
                    if experiment_type is None or exp_info.get('experiment_type') == experiment_type:
                        experiments.append(exp_info)

                except Exception as e:
                    logger.warning(f"读取实验信息失败 {exp_folder}: {e}")
                    continue

        # 按创建时间排序（最新的在前）
        experiments.sort(key=lambda x: x['created_time'], reverse=True)
        return experiments

    def backup_results(self, backup_name: str = None) -> str:
        """
        备份实验结果

        Args:
            backup_name: 备份名称（可选）

        Returns:
            备份文件路径
        """
        if backup_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"results_backup_{timestamp}"

        backup_dir = self.base_dir / self.structure['backups']
        backup_dir.mkdir(exist_ok=True)

        backup_path = backup_dir / f"{backup_name}.zip"

        try:
            # 创建zip备份
            import zipfile
            with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.base_dir):
                    # 跳过备份目录本身
                    if 'backups' in root:
                        continue

                    for file in files:
                        file_path = os.path.join(root, file)
                        arc_name = os.path.relpath(file_path, self.base_dir)
                        zipf.write(file_path, arc_name)

            logger.info(f"结果已备份到: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"备份失败: {e}")
            raise

    def cleanup_old_results(self, days_old: int = 30, keep_count: int = 10):
        """
        清理旧的实验结果

        Args:
            days_old: 保留天数
            keep_count: 最少保留的实验数量
        """
        from datetime import timedelta

        cutoff_date = datetime.now() - timedelta(days=days_old)
        experiments = self.list_experiments()

        # 按时间排序，保留最新的keep_count个
        experiments.sort(key=lambda x: x['created_time'], reverse=True)

        to_delete = []
        for i, exp in enumerate(experiments):
            exp_time = datetime.fromisoformat(exp['created_time'])
            if i >= keep_count and exp_time < cutoff_date:
                to_delete.append(exp)

        # 执行删除
        deleted_count = 0
        for exp in to_delete:
            try:
                exp_path = Path(exp['path'])
                if exp_path.exists():
                    shutil.rmtree(exp_path)
                    deleted_count += 1
                    logger.info(f"删除旧实验: {exp['name']}")
            except Exception as e:
                logger.warning(f"删除实验失败 {exp['name']}: {e}")

        logger.info(f"清理完成，删除了 {deleted_count} 个旧实验")

    def get_experiment_statistics(self) -> Dict[str, Any]:
        """
        获取实验统计信息

        Returns:
            统计信息字典
        """
        experiments = self.list_experiments()

        stats = {
            'total_experiments': len(experiments),
            'experiment_types': {},
            'storage_info': {},
            'recent_activity': []
        }

        # 统计实验类型
        for exp in experiments:
            exp_type = exp.get('experiment_type', 'unknown')
            if exp_type not in stats['experiment_types']:
                stats['experiment_types'][exp_type] = 0
            stats['experiment_types'][exp_type] += 1

        # 计算存储信息
        total_size = 0
        for root, dirs, files in os.walk(self.base_dir):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                except:
                    pass

        stats['storage_info'] = {
            'total_size_mb': round(total_size / (1024 * 1024), 2),
            'total_size_gb': round(total_size / (1024 * 1024 * 1024), 3)
        }

        # 最近活动（最新5个实验）
        stats['recent_activity'] = experiments[:5]

        return stats

    def _calculate_result_hash(self, results: Dict[str, Any]) -> str:
        """计算结果哈希值"""
        try:
            result_str = json.dumps(results, sort_keys=True, default=str)
            return hashlib.md5(result_str.encode()).hexdigest()
        except:
            return "hash_calculation_failed"

    def _extract_result_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """提取结果摘要"""
        summary = {
            'experiment_name': results.get('experiment_name'),
            'experiment_type': results.get('experiment_type'),
            'save_time': results.get('save_time'),
            'result_hash': results.get('result_hash')
        }

        # 提取关键统计信息
        if 'experiment_info' in results:
            exp_info = results['experiment_info']
            summary['experiment_stats'] = {
                'total_datasets': exp_info.get('total_datasets'),
                'total_tasks': exp_info.get('total_tasks'),
                'successful_tasks': exp_info.get('successful_tasks'),
                'failed_tasks': exp_info.get('failed_tasks'),
                'status': exp_info.get('status')
            }

        # 提取数据集摘要
        if 'dataset_summaries' in results:
            summary['datasets'] = list(results['dataset_summaries'].keys())

        return summary

    def export_results(self,
                      experiment_names: List[str],
                      export_format: str = "json",
                      output_file: str = None) -> str:
        """
        导出实验结果

        Args:
            experiment_names: 要导出的实验名称列表
            export_format: 导出格式 (json, csv, excel)
            output_file: 输出文件路径（可选）

        Returns:
            导出文件路径
        """
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"exported_results_{timestamp}.{export_format}"

        output_path = self.base_dir / output_file

        # 收集实验数据
        export_data = {}
        for exp_name in experiment_names:
            # 查找匹配的实验
            experiments = self.list_experiments()
            for exp in experiments:
                if exp_name in exp['name'] or exp_name == exp['experiment_name']:
                    try:
                        results = self.load_experiment_results(exp['path'])
                        export_data[exp['name']] = results
                        break
                    except Exception as e:
                        logger.warning(f"加载实验数据失败 {exp_name}: {e}")

        # 根据格式导出
        try:
            if export_format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)

            elif export_format.lower() == 'csv':
                import pandas as pd
                # 将嵌套数据展平为CSV格式
                flattened_data = self._flatten_for_csv(export_data)
                df = pd.DataFrame(flattened_data)
                df.to_csv(output_path, index=False)

            elif export_format.lower() == 'excel':
                import pandas as pd
                # 创建多个工作表的Excel文件
                with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                    for exp_name, exp_data in export_data.items():
                        flattened_data = self._flatten_for_csv({exp_name: exp_data})
                        df = pd.DataFrame(flattened_data)
                        sheet_name = exp_name[:31]  # Excel工作表名称限制
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

            else:
                raise ValueError(f"不支持的导出格式: {export_format}")

            logger.info(f"结果已导出到: {output_path}")
            return str(output_path)

        except Exception as e:
            logger.error(f"导出失败: {e}")
            raise

    def _flatten_for_csv(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """将嵌套数据展平为CSV格式"""
        flattened = []

        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(flatten_dict(item, f"{new_key}{sep}{i}", sep=sep).items())
                        else:
                            items.append((f"{new_key}{sep}{i}", item))
                else:
                    items.append((new_key, v))
            return dict(items)

        for exp_name, exp_data in data.items():
            flat_data = flatten_dict(exp_data)
            flat_data['experiment_name'] = exp_name
            flattened.append(flat_data)

        return flattened

    def print_storage_summary(self):
        """打印存储摘要"""
        stats = self.get_experiment_statistics()

        print("\n" + "="*80)
        print("📁 实验结果存储摘要")
        print("="*80)
        print(f"存储位置: {self.base_dir}")
        print(f"总实验数: {stats['total_experiments']}")
        print(f"存储大小: {stats['storage_info']['total_size_mb']} MB "
              f"({stats['storage_info']['total_size_gb']} GB)")

        print(f"\n实验类型分布:")
        for exp_type, count in stats['experiment_types'].items():
            print(f"  - {exp_type}: {count} 个实验")

        if stats['recent_activity']:
            print(f"\n最近实验:")
            for exp in stats['recent_activity']:
                print(f"  - {exp['name']} ({exp['created_time'][:10]})")

        print("="*80)


def main():
    """测试函数"""
    # 创建结果管理器
    manager = ResultManager("test_results")

    # 测试保存元数据
    metadata = {
        'experiment_name': 'test_robustness',
        'datasets': ['ml100k', 'netflix'],
        'start_time': datetime.now().isoformat()
    }
    manager.save_experiment_metadata(metadata)

    # 测试保存实验结果
    results = {
        'dataset_results': {
            'ml100k': {'mae': 0.85, 'rmse': 1.02},
            'netflix': {'mae': 0.92, 'rmse': 1.15}
        },
        'summary': 'Test experiment completed successfully'
    }
    manager.save_experiment_results(results, 'test_experiment', 'robustness')

    # 打印存储摘要
    manager.print_storage_summary()

    # 列出所有实验
    experiments = manager.list_experiments()
    print(f"\n发现 {len(experiments)} 个实验:")
    for exp in experiments:
        print(f"  - {exp['name']}")


if __name__ == "__main__":
    main()















