"""
约束条件管理模块

管理超参数搜索中的各种约束条件
"""

import numpy as np
from typing import Dict, List, Callable, Optional, Any, Tuple
from abc import ABC, abstractmethod


class Constraint(ABC):
    """约束基类"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def check(self, config: Dict) -> bool:
        """检查配置是否满足约束"""
        pass
    
    @abstractmethod
    def fix(self, config: Dict) -> Optional[Dict]:
        """尝试修正配置以满足约束"""
        pass
    
    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class RangeConstraint(Constraint):
    """范围约束"""
    
    def __init__(self, name: str, param_name: str, 
                 min_value: Optional[float] = None,
                 max_value: Optional[float] = None):
        super().__init__(name)
        self.param_name = param_name
        self.min_value = min_value
        self.max_value = max_value
        
    def check(self, config: Dict) -> bool:
        """检查参数是否在范围内"""
        if self.param_name not in config:
            return True  # 参数不存在时认为满足约束
            
        value = config[self.param_name]
        
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
            
        return True
        
    def fix(self, config: Dict) -> Optional[Dict]:
        """修正参数到范围内"""
        if self.param_name not in config:
            return config
            
        fixed_config = config.copy()
        value = config[self.param_name]
        
        if self.min_value is not None and value < self.min_value:
            fixed_config[self.param_name] = self.min_value
        elif self.max_value is not None and value > self.max_value:
            fixed_config[self.param_name] = self.max_value
            
        return fixed_config


class RelationalConstraint(Constraint):
    """关系约束（如 a < b）"""
    
    def __init__(self, name: str, param1: str, param2: str, 
                 relation: str = '<'):
        super().__init__(name)
        self.param1 = param1
        self.param2 = param2
        self.relation = relation
        
        # 支持的关系
        self.relations = {
            '<': lambda a, b: a < b,
            '<=': lambda a, b: a <= b,
            '>': lambda a, b: a > b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b
        }
        
        if relation not in self.relations:
            raise ValueError(f"不支持的关系: {relation}")
            
    def check(self, config: Dict) -> bool:
        """检查关系是否满足"""
        if self.param1 not in config or self.param2 not in config:
            return True
            
        val1 = config[self.param1]
        val2 = config[self.param2]
        
        return self.relations[self.relation](val1, val2)
        
    def fix(self, config: Dict) -> Optional[Dict]:
        """尝试修正以满足关系"""
        if self.param1 not in config or self.param2 not in config:
            return config
            
        if self.check(config):
            return config
            
        fixed_config = config.copy()
        val1 = config[self.param1]
        val2 = config[self.param2]
        
        # 简单的修正策略
        if self.relation in ['<', '<=']:
            # 如果需要 val1 < val2，但实际 val1 >= val2
            # 可以减小 val1 或增大 val2
            # 这里选择调整 val1 到 val2 的某个比例
            if self.relation == '<':
                fixed_config[self.param1] = val2 * 0.99
            else:
                fixed_config[self.param1] = val2
                
        elif self.relation in ['>', '>=']:
            # 相反的情况
            if self.relation == '>':
                fixed_config[self.param1] = val2 * 1.01
            else:
                fixed_config[self.param1] = val2
                
        return fixed_config


class ConditionalConstraint(Constraint):
    """条件约束（if A then B）"""
    
    def __init__(self, name: str, 
                 condition: Callable[[Dict], bool],
                 constraint: Constraint):
        super().__init__(name)
        self.condition = condition
        self.constraint = constraint
        
    def check(self, config: Dict) -> bool:
        """只有条件满足时才检查约束"""
        if not self.condition(config):
            return True
        return self.constraint.check(config)
        
    def fix(self, config: Dict) -> Optional[Dict]:
        """只有条件满足时才修正"""
        if not self.condition(config):
            return config
        return self.constraint.fix(config)


class FunctionalConstraint(Constraint):
    """函数约束（自定义验证函数）"""
    
    def __init__(self, name: str, 
                 check_fn: Callable[[Dict], bool],
                 fix_fn: Optional[Callable[[Dict], Dict]] = None):
        super().__init__(name)
        self.check_fn = check_fn
        self.fix_fn = fix_fn
        
    def check(self, config: Dict) -> bool:
        """使用自定义函数检查"""
        return self.check_fn(config)
        
    def fix(self, config: Dict) -> Optional[Dict]:
        """使用自定义函数修正"""
        if self.fix_fn is None:
            return None
        return self.fix_fn(config)


class CompositeConstraint(Constraint):
    """复合约束（AND/OR组合）"""
    
    def __init__(self, name: str, constraints: List[Constraint], 
                 mode: str = 'AND'):
        super().__init__(name)
        self.constraints = constraints
        self.mode = mode.upper()
        
        if self.mode not in ['AND', 'OR']:
            raise ValueError("mode必须是'AND'或'OR'")
            
    def check(self, config: Dict) -> bool:
        """检查复合约束"""
        if self.mode == 'AND':
            return all(c.check(config) for c in self.constraints)
        else:  # OR
            return any(c.check(config) for c in self.constraints)
            
    def fix(self, config: Dict) -> Optional[Dict]:
        """尝试修正以满足复合约束"""
        if self.check(config):
            return config
            
        if self.mode == 'AND':
            # 需要满足所有约束
            fixed_config = config.copy()
            for constraint in self.constraints:
                if not constraint.check(fixed_config):
                    result = constraint.fix(fixed_config)
                    if result is None:
                        return None
                    fixed_config = result
            return fixed_config
            
        else:  # OR
            # 只需要满足一个约束
            for constraint in self.constraints:
                result = constraint.fix(config)
                if result is not None and constraint.check(result):
                    return result
            return None


class ConstraintManager:
    """约束管理器"""
    
    def __init__(self):
        self.constraints: List[Constraint] = []
        self.rejection_count = 0
        self.fix_count = 0
        
    def add_constraint(self, constraint: Constraint):
        """添加约束"""
        self.constraints.append(constraint)
        
    def add_range(self, param_name: str, 
                  min_value: Optional[float] = None,
                  max_value: Optional[float] = None):
        """添加范围约束"""
        name = f"range_{param_name}"
        self.add_constraint(
            RangeConstraint(name, param_name, min_value, max_value)
        )
        
    def add_relation(self, param1: str, param2: str, relation: str = '<'):
        """添加关系约束"""
        name = f"{param1}_{relation}_{param2}"
        self.add_constraint(
            RelationalConstraint(name, param1, param2, relation)
        )
        
    def add_conditional(self, condition: Callable[[Dict], bool], 
                       constraint: Constraint):
        """添加条件约束"""
        name = f"conditional_{constraint.name}"
        self.add_constraint(
            ConditionalConstraint(name, condition, constraint)
        )
        
    def add_functional(self, name: str, 
                      check_fn: Callable[[Dict], bool],
                      fix_fn: Optional[Callable[[Dict], Dict]] = None):
        """添加函数约束"""
        self.add_constraint(
            FunctionalConstraint(name, check_fn, fix_fn)
        )
        
    def check_all(self, config: Dict) -> Tuple[bool, List[str]]:
        """检查所有约束，返回是否满足和违反的约束列表"""
        violations = []
        for constraint in self.constraints:
            if not constraint.check(config):
                violations.append(constraint.name)
        return len(violations) == 0, violations
        
    def fix_config(self, config: Dict, 
                   strategy: str = 'sequential') -> Optional[Dict]:
        """尝试修正配置以满足所有约束"""
        if strategy == 'sequential':
            return self._fix_sequential(config)
        elif strategy == 'iterative':
            return self._fix_iterative(config)
        else:
            raise ValueError(f"未知的修正策略: {strategy}")
            
    def _fix_sequential(self, config: Dict) -> Optional[Dict]:
        """顺序修正：依次应用每个约束的修正"""
        fixed_config = config.copy()
        
        for constraint in self.constraints:
            if not constraint.check(fixed_config):
                result = constraint.fix(fixed_config)
                if result is None:
                    self.rejection_count += 1
                    return None
                fixed_config = result
                
        self.fix_count += 1
        return fixed_config
        
    def _fix_iterative(self, config: Dict, 
                      max_iterations: int = 10) -> Optional[Dict]:
        """迭代修正：反复应用修正直到满足所有约束"""
        fixed_config = config.copy()
        
        for _ in range(max_iterations):
            all_satisfied, violations = self.check_all(fixed_config)
            if all_satisfied:
                self.fix_count += 1
                return fixed_config
                
            # 尝试修正违反的约束
            made_progress = False
            for constraint in self.constraints:
                if constraint.name in violations:
                    result = constraint.fix(fixed_config)
                    if result is not None:
                        fixed_config = result
                        made_progress = True
                        
            if not made_progress:
                break
                
        # 最终检查
        all_satisfied, _ = self.check_all(fixed_config)
        if all_satisfied:
            self.fix_count += 1
            return fixed_config
            
        self.rejection_count += 1
        return None
        
    def filter_configs(self, configs: List[Dict]) -> List[Dict]:
        """过滤出满足约束的配置"""
        valid_configs = []
        for config in configs:
            satisfied, _ = self.check_all(config)
            if satisfied:
                valid_configs.append(config)
        return valid_configs
        
    def get_statistics(self) -> Dict:
        """获取约束统计信息"""
        return {
            'n_constraints': len(self.constraints),
            'rejection_count': self.rejection_count,
            'fix_count': self.fix_count,
            'rejection_rate': self.rejection_count / (self.rejection_count + self.fix_count)
            if (self.rejection_count + self.fix_count) > 0 else 0
        }
        
    def clear_statistics(self):
        """清除统计信息"""
        self.rejection_count = 0
        self.fix_count = 0