"""
导出管理模块

管理图表和表格的导出
"""

import os
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
import json
from datetime import datetime


class ExportManager:
    """导出管理器"""
    
    def __init__(self, base_dir: str = "./exports"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.figures_dir = self.base_dir / "figures"
        self.tables_dir = self.base_dir / "tables"
        self.data_dir = self.base_dir / "data"
        
        for dir_path in [self.figures_dir, self.tables_dir, self.data_dir]:
            dir_path.mkdir(exist_ok=True)
            
        # 导出记录
        self.export_log = []
        
    def export_figure(self,
                     fig: plt.Figure,
                     name: str,
                     formats: List[str] = ['pdf', 'png'],
                     dpi: int = 300,
                     category: Optional[str] = None) -> Dict[str, str]:
        """导出图形
        
        Args:
            fig: matplotlib图形对象
            name: 文件名（不含扩展名）
            formats: 导出格式列表
            dpi: 分辨率
            category: 分类目录
            
        Returns:
            格式到文件路径的字典
        """
        # 创建分类目录
        if category:
            export_dir = self.figures_dir / category
            export_dir.mkdir(exist_ok=True)
        else:
            export_dir = self.figures_dir
            
        exported_files = {}
        
        for fmt in formats:
            filename = f"{name}.{fmt}"
            filepath = export_dir / filename
            
            # 导出设置
            save_kwargs = {
                'dpi': dpi,
                'bbox_inches': 'tight',
                'pad_inches': 0.1,
                'facecolor': 'white',
                'edgecolor': 'none'
            }
            
            # 格式特定设置
            if fmt == 'pdf':
                save_kwargs['backend'] = 'pdf'
            elif fmt == 'svg':
                save_kwargs['format'] = 'svg'
            elif fmt == 'eps':
                save_kwargs['format'] = 'eps'
                
            # 保存图形
            fig.savefig(filepath, **save_kwargs)
            exported_files[fmt] = str(filepath)
            
            # 记录导出
            self._log_export('figure', name, filepath, {
                'format': fmt,
                'dpi': dpi,
                'category': category
            })
            
        return exported_files
        
    def export_table(self,
                    table_str: str,
                    name: str,
                    format_type: str = 'latex',
                    category: Optional[str] = None) -> str:
        """导出表格
        
        Args:
            table_str: 表格字符串
            name: 文件名（不含扩展名）
            format_type: 格式类型
            category: 分类目录
            
        Returns:
            文件路径
        """
        # 创建分类目录
        if category:
            export_dir = self.tables_dir / category
            export_dir.mkdir(exist_ok=True)
        else:
            export_dir = self.tables_dir
            
        # 确定扩展名
        ext_map = {
            'latex': '.tex',
            'markdown': '.md',
            'html': '.html',
            'csv': '.csv'
        }
        ext = ext_map.get(format_type, '.txt')
        
        filename = f"{name}{ext}"
        filepath = export_dir / filename
        
        # 保存表格
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(table_str)
            
        # 记录导出
        self._log_export('table', name, filepath, {
            'format': format_type,
            'category': category
        })
        
        return str(filepath)
        
    def export_data(self,
                   data: Union[Dict, List],
                   name: str,
                   format_type: str = 'json') -> str:
        """导出数据
        
        Args:
            data: 数据对象
            name: 文件名（不含扩展名）
            format_type: 格式类型 ('json', 'pickle')
            
        Returns:
            文件路径
        """
        if format_type == 'json':
            filename = f"{name}.json"
            filepath = self.data_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
                
        elif format_type == 'pickle':
            import pickle
            filename = f"{name}.pkl"
            filepath = self.data_dir / filename
            
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
                
        else:
            raise ValueError(f"不支持的格式: {format_type}")
            
        # 记录导出
        self._log_export('data', name, filepath, {
            'format': format_type
        })
        
        return str(filepath)
        
    def create_figure_package(self,
                            figure_list: List[Tuple[str, plt.Figure]],
                            package_name: str = "figures",
                            formats: List[str] = ['pdf', 'png'],
                            include_source: bool = True) -> str:
        """创建图形包
        
        Args:
            figure_list: (名称, 图形)元组列表
            package_name: 包名称
            formats: 导出格式
            include_source: 是否包含源代码
            
        Returns:
            包文件路径
        """
        # 创建临时目录
        temp_dir = self.base_dir / f"temp_{package_name}"
        temp_dir.mkdir(exist_ok=True)
        
        # 导出所有图形
        for name, fig in figure_list:
            for fmt in formats:
                fmt_dir = temp_dir / fmt
                fmt_dir.mkdir(exist_ok=True)
                
                filepath = fmt_dir / f"{name}.{fmt}"
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                
        # 创建README
        readme_content = f"""# Figure Package: {package_name}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

This package contains {len(figure_list)} figures in the following formats:
"""
        
        for fmt in formats:
            readme_content += f"- {fmt.upper()}: /{fmt}/\n"
            
        readme_content += "\n## Figure List\n\n"
        
        for i, (name, _) in enumerate(figure_list, 1):
            readme_content += f"{i}. {name}\n"
            
        readme_path = temp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
            
        # 创建ZIP包
        zip_path = self.base_dir / f"{package_name}.zip"
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(temp_dir)
                    zipf.write(file_path, arcname)
                    
        # 清理临时目录
        shutil.rmtree(temp_dir)
        
        print(f"图形包已创建: {zip_path}")
        return str(zip_path)
        
    def export_all_figures(self,
                         figures: Dict[str, plt.Figure],
                         formats: List[str] = ['pdf', 'png'],
                         create_package: bool = True) -> Dict[str, str]:
        """批量导出所有图形
        
        Args:
            figures: 名称到图形的字典
            formats: 导出格式
            create_package: 是否创建打包文件
            
        Returns:
            导出文件路径字典
        """
        exported_files = {}
        
        # 导出每个图形
        for name, fig in figures.items():
            files = self.export_figure(fig, name, formats)
            exported_files[name] = files
            
        # 创建打包文件
        if create_package:
            figure_list = list(figures.items())
            package_path = self.create_figure_package(
                figure_list, 
                "all_figures",
                formats
            )
            exported_files['_package'] = package_path
            
        return exported_files
        
    def generate_latex_figures_file(self,
                                  figure_names: List[str],
                                  output_file: str = "figures.tex") -> str:
        """生成LaTeX图形包含文件
        
        Args:
            figure_names: 图形名称列表
            output_file: 输出文件名
            
        Returns:
            文件路径
        """
        latex_content = """% Auto-generated figure inclusion file
% Generated on: {}

""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # 添加图形命令定义
        for name in figure_names:
            # 转换为合法的LaTeX命令名
            cmd_name = name.replace('_', '').replace('-', '')
            
            latex_content += f"""
% Figure: {name}
\\newcommand{{\\fig{cmd_name}}}[1][]{{%
    \\begin{{figure}}[#1]
        \\centering
        \\includegraphics[width=\\linewidth]{{figures/{name}}}
        \\caption{{\\label{{fig:{name}}}}}
    \\end{{figure}}
}}
"""
        
        # 保存文件
        filepath = self.base_dir / output_file
        with open(filepath, 'w') as f:
            f.write(latex_content)
            
        print(f"LaTeX图形文件已生成: {filepath}")
        return str(filepath)
        
    def _log_export(self,
                   export_type: str,
                   name: str,
                   filepath: Path,
                   metadata: Dict):
        """记录导出信息"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'type': export_type,
            'name': name,
            'filepath': str(filepath),
            'metadata': metadata
        }
        
        self.export_log.append(log_entry)
        
        # 保存日志
        log_file = self.base_dir / "export_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.export_log, f, indent=2)
            
    def get_export_summary(self) -> Dict:
        """获取导出摘要"""
        summary = {
            'total_exports': len(self.export_log),
            'figures': 0,
            'tables': 0,
            'data': 0,
            'by_format': {},
            'by_category': {}
        }
        
        for entry in self.export_log:
            # 按类型统计
            if entry['type'] == 'figure':
                summary['figures'] += 1
            elif entry['type'] == 'table':
                summary['tables'] += 1
            elif entry['type'] == 'data':
                summary['data'] += 1
                
            # 按格式统计
            fmt = entry['metadata'].get('format', 'unknown')
            summary['by_format'][fmt] = summary['by_format'].get(fmt, 0) + 1
            
            # 按分类统计
            category = entry['metadata'].get('category', 'uncategorized')
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            
        return summary
        
    def clean_exports(self, keep_latest: int = 5):
        """清理旧的导出文件
        
        Args:
            keep_latest: 保留最新的N个导出
        """
        # 按时间戳排序
        sorted_log = sorted(self.export_log, 
                          key=lambda x: x['timestamp'], 
                          reverse=True)
        
        # 保留最新的条目
        self.export_log = sorted_log[:keep_latest]
        
        # 获取要保留的文件路径
        keep_files = set()
        for entry in self.export_log:
            keep_files.add(entry['filepath'])
            
        # 删除旧文件
        for dir_path in [self.figures_dir, self.tables_dir, self.data_dir]:
            for file_path in dir_path.rglob('*'):
                if file_path.is_file() and str(file_path) not in keep_files:
                    file_path.unlink()
                    
        # 更新日志
        log_file = self.base_dir / "export_log.json"
        with open(log_file, 'w') as f:
            json.dump(self.export_log, f, indent=2)
            
        print(f"清理完成，保留了 {len(self.export_log)} 个最新导出")


# 便捷函数
def export_figure(fig: plt.Figure,
                 name: str,
                 formats: List[str] = ['pdf', 'png'],
                 export_dir: Optional[str] = None) -> Dict[str, str]:
    """快速导出图形"""
    if export_dir:
        manager = ExportManager(export_dir)
    else:
        manager = ExportManager()
        
    return manager.export_figure(fig, name, formats)


def export_table(table_str: str,
                name: str,
                format_type: str = 'latex',
                export_dir: Optional[str] = None) -> str:
    """快速导出表格"""
    if export_dir:
        manager = ExportManager(export_dir)
    else:
        manager = ExportManager()
        
    return manager.export_table(table_str, name, format_type)


def export_all_figures(figures: Dict[str, plt.Figure],
                      formats: List[str] = ['pdf', 'png'],
                      export_dir: Optional[str] = None) -> Dict[str, str]:
    """快速导出所有图形"""
    if export_dir:
        manager = ExportManager(export_dir)
    else:
        manager = ExportManager()
        
    return manager.export_all_figures(figures, formats)


def create_figure_package(figure_list: List[Tuple[str, plt.Figure]],
                         package_name: str = "figures",
                         export_dir: Optional[str] = None) -> str:
    """快速创建图形包"""
    if export_dir:
        manager = ExportManager(export_dir)
    else:
        manager = ExportManager()
        
    return manager.create_figure_package(figure_list, package_name)