#!/usr/bin/env python3
"""
数据集选择器模块
负责扫描、展示和选择可用的数据集
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetInfo:
    """数据集信息类"""

    def __init__(self, name: str, file_path: str, display_name: str = None):
        self.name = name
        self.file_path = file_path
        self.display_name = display_name or name
        self.file_size = self._get_file_size()
        self.estimated_records = self._estimate_records()

    def _get_file_size(self) -> str:
        """获取文件大小"""
        try:
            size_bytes = os.path.getsize(self.file_path)
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            elif size_bytes < 1024 * 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
        except:
            return "Unknown"

    def _estimate_records(self) -> str:
        """估算记录数量"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                # 读取前几行估算
                lines = 0
                for _ in range(1000):
                    if f.readline():
                        lines += 1
                    else:
                        break

                if lines < 1000:
                    return f"~{lines}"
                else:
                    # 估算总行数
                    f.seek(0, 2)  # 移到文件末尾
                    file_size = f.tell()
                    f.seek(0)
                    avg_line_size = f.tell() / lines if lines > 0 else 100
                    estimated_lines = int(file_size / avg_line_size) if avg_line_size > 0 else 0

                    if estimated_lines < 1000:
                        return f"~{estimated_lines}"
                    elif estimated_lines < 1000000:
                        return f"~{estimated_lines/1000:.1f}K"
                    else:
                        return f"~{estimated_lines/1000000:.1f}M"
        except:
            return "Unknown"


class DatasetSelector:
    """数据集选择器"""

    def __init__(self, dataset_dir: str = "dataset"):
        self.dataset_dir = Path(dataset_dir)
        self.datasets: List[DatasetInfo] = []
        self.dataset_patterns = {
            'ml100k': r'.*100k.*',
            'movielens100k': r'.*M100K.*',
            'netflix': r'.*netflix.*',
            'filmtrust': r'.*filmtrust.*',
            'flimtrust': r'.*flimtrust.*',
            'ciaodvd': r'.*ciaodvd.*',
            'epinions': r'.*epinions.*',
            'amazon': r'.*amazon.*',
            'movielens1m': r'.*1m.*|.*movie.*1m.*',
            'tweetings': r'.*tweetings.*'
        }
        self._scan_datasets()

    def _scan_datasets(self):
        """扫描数据集目录"""
        if not self.dataset_dir.exists():
            logger.warning(f"数据集目录不存在: {self.dataset_dir}")
            return

        logger.info(f"扫描数据集目录: {self.dataset_dir}")

        # 获取所有数据文件
        data_files = []
        for ext in ['*.txt', '*.csv', '*.dat']:
            data_files.extend(self.dataset_dir.glob(ext))

        # 解析数据集信息
        for file_path in data_files:
            dataset_info = self._parse_dataset_file(file_path)
            if dataset_info:
                self.datasets.append(dataset_info)

        # 按名称排序
        self.datasets.sort(key=lambda x: x.name)
        logger.info(f"发现 {len(self.datasets)} 个数据集")

    def _parse_dataset_file(self, file_path: Path) -> Optional[DatasetInfo]:
        """解析数据集文件信息"""
        filename = file_path.name.lower()

        # 跳过明显的非数据集文件
        skip_patterns = ['test', 'sample', 'small', 'demo', 'temp']
        if any(pattern in filename for pattern in skip_patterns):
            # 除非是小数据集用于测试
            if not any(size in filename for size in ['1percent', 'small_']):
                return None

        # 识别数据集类型
        dataset_name = self._identify_dataset_type(filename)
        display_name = self._generate_display_name(filename, dataset_name)

        return DatasetInfo(
            name=dataset_name,
            file_path=str(file_path),
            display_name=display_name
        )

    def _identify_dataset_type(self, filename: str) -> str:
        """识别数据集类型"""
        filename_lower = filename.lower()

        for dataset_type, pattern in self.dataset_patterns.items():
            if re.search(pattern, filename_lower, re.IGNORECASE):
                return dataset_type

        # 如果没有匹配，使用文件名
        base_name = Path(filename).stem
        return re.sub(r'[0-9]+|random|data|all|_', '', base_name).strip('_')

    def _generate_display_name(self, filename: str, dataset_type: str) -> str:
        """生成显示名称"""
        display_names = {
            'ml100k': 'MovieLens 100K',
            'movielens100k': 'MovieLens 100K',
            'netflix': 'Netflix',
            'filmtrust': 'FilmTrust',
            'flimtrust': 'FilmTrust',
            'ciaodvd': 'CiaoDVD',
            'epinions': 'Epinions',
            'amazon': 'Amazon',
            'movielens1m': 'MovieLens 1M',
            'tweetings': 'MovieTweetings'
        }

        base_display = display_names.get(dataset_type, dataset_type.title())

        # 添加特殊标识
        if 'small' in filename.lower() or '1percent' in filename.lower():
            base_display += ' (Small)'

        return base_display

    def display_datasets(self) -> None:
        """显示可用数据集列表"""
        if not self.datasets:
            print("❌ 未发现可用的数据集文件")
            return

        print("\n" + "="*80)
        print("📊 可用数据集列表")
        print("="*80)

        for i, dataset in enumerate(self.datasets, 1):
            print(f"[{i:2d}] {dataset.display_name}")
            print(f"     文件: {Path(dataset.file_path).name}")
            print(f"     大小: {dataset.file_size}")
            print()

        print(f"[{len(self.datasets)+1:2d}] 全部选择")
        print(f"[{len(self.datasets)+2:2d}] 退出")
        print("="*80)

    def get_user_selection(self) -> List[DatasetInfo]:
        """获取用户选择的数据集"""
        while True:
            self.display_datasets()

            try:
                choice = input("\n请选择数据集 (输入序号，多个用逗号分隔): ").strip()

                if not choice:
                    continue

                # 解析用户输入
                selected_indices = []
                for item in choice.split(','):
                    item = item.strip()
                    if item.isdigit():
                        selected_indices.append(int(item))

                if not selected_indices:
                    print("❌ 请输入有效的序号")
                    continue

                # 处理特殊选择
                if len(self.datasets) + 2 in selected_indices:  # 退出
                    return []

                if len(self.datasets) + 1 in selected_indices:  # 全部选择
                    print(f"✅ 已选择全部 {len(self.datasets)} 个数据集")
                    return self.datasets.copy()

                # 验证选择
                selected_datasets = []
                for idx in selected_indices:
                    if 1 <= idx <= len(self.datasets):
                        selected_datasets.append(self.datasets[idx-1])
                    else:
                        print(f"❌ 序号 {idx} 超出范围")
                        break
                else:
                    # 确认选择
                    print(f"\n✅ 已选择 {len(selected_datasets)} 个数据集:")
                    for dataset in selected_datasets:
                        print(f"   - {dataset.display_name}")

                    confirm = input("\n确认选择? (y/n): ").strip().lower()
                    if confirm in ['y', 'yes', '是']:
                        return selected_datasets

            except KeyboardInterrupt:
                print("\n\n👋 用户取消选择")
                return []
            except Exception as e:
                print(f"❌ 输入错误: {e}")
                continue

    def get_dataset_by_name(self, name: str) -> Optional[DatasetInfo]:
        """根据名称获取数据集"""
        name_lower = name.lower()
        for dataset in self.datasets:
            if (dataset.name.lower() == name_lower or
                dataset.display_name.lower() == name_lower or
                name_lower in dataset.file_path.lower()):
                return dataset
        return None

    def get_all_datasets(self) -> List[DatasetInfo]:
        """获取所有数据集"""
        return self.datasets.copy()

    def validate_dataset(self, dataset: DatasetInfo) -> bool:
        """验证数据集文件"""
        try:
            file_path = Path(dataset.file_path)

            # 检查文件是否存在
            if not file_path.exists():
                logger.error(f"数据集文件不存在: {file_path}")
                return False

            # 检查文件是否可读
            if not os.access(file_path, os.R_OK):
                logger.error(f"数据集文件不可读: {file_path}")
                return False

            # 检查文件格式
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if not first_line:
                    logger.error(f"数据集文件为空: {file_path}")
                    return False

                # 简单格式验证（假设是用户ID 物品ID 评分的格式）
                parts = first_line.split()
                if len(parts) < 3:
                    logger.warning(f"数据集格式可能不正确: {file_path}")
                    # 不阻止使用，只是警告

            return True

        except Exception as e:
            logger.error(f"验证数据集时出错 {dataset.file_path}: {e}")
            return False


def main():
    """测试函数"""
    selector = DatasetSelector()
    selected = selector.get_user_selection()

    if selected:
        print(f"\n最终选择了 {len(selected)} 个数据集:")
        for dataset in selected:
            print(f"  - {dataset.display_name}: {dataset.file_path}")
    else:
        print("\n未选择任何数据集")


if __name__ == "__main__":
    main()

