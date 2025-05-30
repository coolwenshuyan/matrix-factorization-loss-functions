"""
工具函数模块

提供超参数优化的辅助功能
"""

import numpy as np
import json
import yaml
import pickle
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
import os
from datetime import datetime


def normalize_config(config: Dict, space: 'ParameterSpace') -> np.ndarray:
    """
    将配置字典归一化为向量
    
    Args:
        config: 参数配置
        space: 参数空间
        
    Returns:
        归一化的向量
    """
    vector = []
    
    for name, param in space.parameters.items():
        if name in config:
            normalized_value = param.normalize(config[name])
            vector.append(normalized_value)
        else:
            # 条件参数可能不存在
            vector.append(0.0)
            
    return np.array(vector)


def denormalize_config(vector: np.ndarray, space: 'ParameterSpace') -> Dict:
    """
    将归一化向量还原为配置字典
    
    Args:
        vector: 归一化向量
        space: 参数空间
        
    Returns:
        参数配置
    """
    config = {}
    param_names = list(space.parameters.keys())
    
    for i, (name, param) in enumerate(space.parameters.items()):
        if i < len(vector):
            # 检查条件参数
            if hasattr(param, 'is_active'):
                if param.is_active(config):
                    config[name] = param.denormalize(vector[i])
            else:
                config[name] = param.denormalize(vector[i])
                
    return config


def config_to_vector(config: Dict, param_names: List[str]) -> np.ndarray:
    """
    将配置转换为向量（用于机器学习模型）
    
    Args:
        config: 参数配置
        param_names: 参数名列表
        
    Returns:
        参数向量
    """
    vector = []
    
    for name in param_names:
        if name in config:
            value = config[name]
            
            # 处理不同类型的参数
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(1.0 if value else 0.0)
            elif isinstance(value, str):
                # 字符串参数需要预先编码
                # 这里使用简单的哈希值
                hash_value = int(hashlib.md5(value.encode()).hexdigest()[:8], 16)
                vector.append(float(hash_value % 1000) / 1000.0)
            else:
                # 其他类型默认为0
                vector.append(0.0)
        else:
            vector.append(0.0)
            
    return np.array(vector)


def vector_to_config(vector: np.ndarray, param_specs: Dict) -> Dict:
    """
    将向量转换为配置（需要参数规格信息）
    
    Args:
        vector: 参数向量
        param_specs: 参数规格字典
        
    Returns:
        参数配置
    """
    config = {}
    param_names = list(param_specs.keys())
    
    for i, name in enumerate(param_names):
        if i < len(vector):
            spec = param_specs[name]
            value = vector[i]
            
            # 根据参数类型转换
            if spec['type'] == 'int':
                config[name] = int(round(value))
            elif spec['type'] == 'float':
                config[name] = float(value)
            elif spec['type'] == 'bool':
                config[name] = value > 0.5
            elif spec['type'] == 'choice':
                # 选择最近的选项
                choices = spec['choices']
                idx = int(round(value * (len(choices) - 1)))
                config[name] = choices[idx]
            else:
                config[name] = value
                
    return config


def calculate_config_distance(config1: Dict, config2: Dict, 
                            space: 'ParameterSpace') -> float:
    """
    计算两个配置之间的距离
    
    Args:
        config1: 第一个配置
        config2: 第二个配置
        space: 参数空间
        
    Returns:
        欧氏距离
    """
    vec1 = normalize_config(config1, space)
    vec2 = normalize_config(config2, space)
    
    return np.linalg.norm(vec1 - vec2)


def generate_config_hash(config: Dict) -> str:
    """
    生成配置的哈希值（用于去重）
    
    Args:
        config: 参数配置
        
    Returns:
        哈希字符串
    """
    # 确保键的顺序一致
    sorted_config = dict(sorted(config.items()))
    config_str = json.dumps(sorted_config, sort_keys=True)
    
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_config_file(filepath: str) -> Dict:
    """
    从文件加载配置
    
    Args:
        filepath: 文件路径
        
    Returns:
        配置字典
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    with open(filepath, 'r') as f:
        if ext == '.json':
            return json.load(f)
        elif ext in ['.yaml', '.yml']:
            return yaml.safe_load(f)
        elif ext == '.pkl':
            return pickle.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


def save_config_file(config: Dict, filepath: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        filepath: 文件路径
    """
    ext = os.path.splitext(filepath)[1].lower()
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        if ext == '.json':
            json.dump(config, f, indent=2)
        elif ext in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False)
        elif ext == '.pkl':
            pickle.dump(config, f)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")


def merge_configs(base_config: Dict, update_config: Dict) -> Dict:
    """
    合并两个配置（深度合并）
    
    Args:
        base_config: 基础配置
        update_config: 更新配置
        
    Returns:
        合并后的配置
    """
    result = base_config.copy()
    
    for key, value in update_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # 递归合并字典
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
            
    return result


def validate_config_schema(config: Dict, schema: Dict) -> Tuple[bool, List[str]]:
    """
    验证配置是否符合模式
    
    Args:
        config: 配置字典
        schema: 模式定义
        
    Returns:
        (是否有效, 错误信息列表)
    """
    errors = []
    
    # 检查必需参数
    if 'required' in schema:
        for param in schema['required']:
            if param not in config:
                errors.append(f"缺少必需参数: {param}")
                
    # 检查参数类型和范围
    if 'properties' in schema:
        for param, spec in schema['properties'].items():
            if param in config:
                value = config[param]
                
                # 类型检查
                if 'type' in spec:
                    expected_type = spec['type']
                    if expected_type == 'number':
                        if not isinstance(value, (int, float)):
                            errors.append(f"{param} 应该是数字类型")
                    elif expected_type == 'integer':
                        if not isinstance(value, int):
                            errors.append(f"{param} 应该是整数类型")
                    elif expected_type == 'string':
                        if not isinstance(value, str):
                            errors.append(f"{param} 应该是字符串类型")
                    elif expected_type == 'boolean':
                        if not isinstance(value, bool):
                            errors.append(f"{param} 应该是布尔类型")
                            
                # 范围检查
                if isinstance(value, (int, float)):
                    if 'minimum' in spec and value < spec['minimum']:
                        errors.append(f"{param} 不能小于 {spec['minimum']}")
                    if 'maximum' in spec and value > spec['maximum']:
                        errors.append(f"{param} 不能大于 {spec['maximum']}")
                        
                # 枚举检查
                if 'enum' in spec and value not in spec['enum']:
                    errors.append(f"{param} 必须是以下之一: {spec['enum']}")
                    
    return len(errors) == 0, errors


def interpolate_configs(config1: Dict, config2: Dict, 
                       alpha: float = 0.5,
                       space: Optional['ParameterSpace'] = None) -> Dict:
    """
    在两个配置之间进行插值
    
    Args:
        config1: 第一个配置
        config2: 第二个配置
        alpha: 插值系数 (0到1)
        space: 参数空间（可选）
        
    Returns:
        插值后的配置
    """
    if space:
        # 使用参数空间进行插值
        vec1 = normalize_config(config1, space)
        vec2 = normalize_config(config2, space)
        
        # 线性插值
        vec_interp = vec1 * (1 - alpha) + vec2 * alpha
        
        return denormalize_config(vec_interp, space)
        
    else:
        # 简单插值
        result = {}
        
        # 获取所有参数
        all_params = set(config1.keys()) | set(config2.keys())
        
        for param in all_params:
            if param in config1 and param in config2:
                val1 = config1[param]
                val2 = config2[param]
                
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    # 数值插值
                    result[param] = val1 * (1 - alpha) + val2 * alpha
                    if isinstance(val1, int) and isinstance(val2, int):
                        result[param] = int(round(result[param]))
                else:
                    # 非数值参数选择其中一个
                    result[param] = val1 if alpha < 0.5 else val2
                    
            elif param in config1:
                result[param] = config1[param]
            else:
                result[param] = config2[param]
                
        return result


def get_config_summary(configs: List[Dict]) -> Dict:
    """
    获取配置集合的摘要统计
    
    Args:
        configs: 配置列表
        
    Returns:
        摘要统计信息
    """
    if not configs:
        return {}
        
    summary = {}
    
    # 获取所有参数
    all_params = set()
    for config in configs:
        all_params.update(config.keys())
        
    for param in all_params:
        values = [c.get(param) for c in configs if param in c]
        
        if not values:
            continue
            
        param_info = {
            'count': len(values),
            'missing': len(configs) - len(values)
        }
        
        # 根据类型计算统计信息
        if all(isinstance(v, (int, float)) for v in values):
            # 数值参数
            param_info.update({
                'type': 'numeric',
                'min': min(values),
                'max': max(values),
                'mean': sum(values) / len(values),
                'std': np.std(values) if len(values) > 1 else 0
            })
        else:
            # 分类参数
            unique_values = list(set(values))
            value_counts = {v: values.count(v) for v in unique_values}
            
            param_info.update({
                'type': 'categorical',
                'unique_values': len(unique_values),
                'value_counts': value_counts,
                'most_common': max(value_counts, key=value_counts.get)
            })
            
        summary[param] = param_info
        
    return summary


def format_duration(seconds: float) -> str:
    """
    格式化时间长度
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"


def create_experiment_id() -> str:
    """
    创建唯一的实验ID
    
    Returns:
        实验ID
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_suffix = hashlib.md5(os.urandom(16)).hexdigest()[:6]
    
    return f"{timestamp}_{random_suffix}"