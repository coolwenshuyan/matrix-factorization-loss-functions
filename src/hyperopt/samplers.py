"""
采样策略实现模块

提供多种超参数采样策略
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
# 修改这一行，从space_impl导入ParameterSpace
from .space import ParameterSpace
import warnings

# 尝试导入可选依赖
try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy未安装，部分高级采样器不可用")


class Sampler(ABC):
    """采样器基类"""
    
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        self.space = space
        self.random_state = np.random.RandomState(seed)
        self.history: List[Dict] = []
        
    @abstractmethod
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """生成n个采样点"""
        pass
    
    def update(self, configs: List[Dict], scores: List[float]):
        """更新采样器状态（用于自适应采样）"""
        for config, score in zip(configs, scores):
            self.history.append({'config': config, 'score': score})


class RandomSampler(Sampler):
    """随机采样器"""
    
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """完全随机采样"""
        samples = []
        for _ in range(n_samples):
            config = self.space.sample(self.random_state)
            samples.append(config)
        return samples


class GridSampler(Sampler):
    """网格采样器"""
    
    def __init__(self, space: ParameterSpace, 
                 resolution: Union[int, Dict[str, int]] = 10,
                 seed: Optional[int] = None):
        super().__init__(space, seed)
        self.resolution = resolution
        self._grid_points = None
        self._current_idx = 0
        
    def _generate_grid(self):
        """生成网格点"""
        grid_coords = []
        
        for name, param in self.space.parameters.items():
            if isinstance(self.resolution, dict):
                res = self.resolution.get(name, 10)
            else:
                res = self.resolution
                
            # 生成该维度的网格点
            if hasattr(param, 'choices'):
                # 分类参数使用所有选项
                points = list(range(len(param.choices)))
            else:
                # 连续/离散参数均匀分布
                points = np.linspace(0, 1, res)
                
            grid_coords.append(points)
            
        # 生成所有组合
        import itertools
        self._grid_points = list(itertools.product(*grid_coords))
        
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """按网格顺序采样"""
        if self._grid_points is None:
            self._generate_grid()
            
        samples = []
        for _ in range(n_samples):
            if self._current_idx >= len(self._grid_points):
                # 网格用完后随机采样
                samples.append(self.space.sample(self.random_state))
            else:
                # 使用网格点
                point = self._grid_points[self._current_idx]
                vector = np.array(point)
                config = self.space.denormalize(vector)
                samples.append(config)
                self._current_idx += 1
                
        return samples


class LatinHypercubeSampler(Sampler):
    """拉丁超立方采样器"""
    
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """LHS采样确保每个维度均匀覆盖"""
        n_dims = len(self.space.parameters)
        
        # 生成LHS采样点
        samples_normalized = self._lhs_sample(n_samples, n_dims)
        
        # 转换为配置
        samples = []
        for i in range(n_samples):
            vector = samples_normalized[i]
            config = self.space.denormalize(vector)
            samples.append(config)
            
        return samples
        
    def _lhs_sample(self, n_samples: int, n_dims: int) -> np.ndarray:
        """生成LHS采样点"""
        # 每个维度分成n_samples个区间
        samples = np.zeros((n_samples, n_dims))
        
        for dim in range(n_dims):
            # 生成排列
            perm = self.random_state.permutation(n_samples)
            
            # 在每个区间内随机采样
            for i in range(n_samples):
                low = perm[i] / n_samples
                high = (perm[i] + 1) / n_samples
                samples[i, dim] = self.random_state.uniform(low, high)
                
        return samples


class SobolSampler(Sampler):
    """Sobol序列采样器（准随机）"""
    
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        super().__init__(space, seed)
        if not HAS_SCIPY:
            raise ImportError("SobolSampler需要scipy库")
            
        n_dims = len(self.space.parameters)
        self.sobol = qmc.Sobol(d=n_dims, scramble=True, seed=seed)
        
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """Sobol序列采样"""
        # 生成Sobol点
        samples_normalized = self.sobol.random(n_samples)
        
        # 转换为配置
        samples = []
        for i in range(n_samples):
            vector = samples_normalized[i]
            config = self.space.denormalize(vector)
            samples.append(config)
            
        return samples


class HaltonSampler(Sampler):
    """Halton序列采样器"""
    
    def __init__(self, space: ParameterSpace, seed: Optional[int] = None):
        super().__init__(space, seed)
        if not HAS_SCIPY:
            raise ImportError("HaltonSampler需要scipy库")
            
        n_dims = len(self.space.parameters)
        self.halton = qmc.Halton(d=n_dims, scramble=True, seed=seed)
        
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """Halton序列采样"""
        # 生成Halton点
        samples_normalized = self.halton.random(n_samples)
        
        # 转换为配置
        samples = []
        for i in range(n_samples):
            vector = samples_normalized[i]
            config = self.space.denormalize(vector)
            samples.append(config)
            
        return samples


class AdaptiveSampler(Sampler):
    """自适应采样器"""
    
    def __init__(self, space: ParameterSpace, 
                 base_sampler: Optional[Sampler] = None,
                 exploration_weight: float = 0.5,
                 seed: Optional[int] = None):
        super().__init__(space, seed)
        self.base_sampler = base_sampler or RandomSampler(space, seed)
        self.exploration_weight = exploration_weight
        self.best_configs: List[Tuple[Dict, float]] = []
        self.kde_bandwidth = 0.1
        
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """自适应采样，平衡探索和利用"""
        if len(self.history) < 10:
            # 初期使用基础采样器
            return self.base_sampler.sample(n_samples)
            
        samples = []
        for _ in range(n_samples):
            if self.random_state.random() < self.exploration_weight:
                # 探索：随机采样
                config = self.space.sample(self.random_state)
            else:
                # 利用：在好的区域附近采样
                config = self._exploit_sample()
                
            samples.append(config)
            
        return samples
        
    def _exploit_sample(self) -> Dict:
        """在表现好的配置附近采样"""
        # 选择一个好的历史配置
        sorted_history = sorted(self.history, 
                              key=lambda x: x['score'], 
                              reverse=True)
        
        # 从top-k中随机选择
        k = min(10, len(sorted_history))
        selected = self.random_state.choice(sorted_history[:k])
        base_config = selected['config']
        
        # 在附近添加噪声
        base_vector = self.space.normalize(base_config)
        noise = self.random_state.normal(0, self.kde_bandwidth, len(base_vector))
        new_vector = np.clip(base_vector + noise, 0, 1)
        
        return self.space.denormalize(new_vector)
        
    def update(self, configs: List[Dict], scores: List[float]):
        """更新采样器状态"""
        super().update(configs, scores)
        
        # 更新最佳配置列表
        for config, score in zip(configs, scores):
            self.best_configs.append((config, score))
            
        # 保持最佳配置数量限制
        self.best_configs = sorted(self.best_configs, 
                                 key=lambda x: x[1], 
                                 reverse=True)[:50]
                                 
        # 自适应调整带宽
        if len(self.history) % 20 == 0:
            self._update_bandwidth()
            
    def _update_bandwidth(self):
        """根据搜索进展调整带宽"""
        # 计算最近的改进率
        recent_scores = [h['score'] for h in self.history[-20:]]
        improvement = np.std(recent_scores)
        
        # 如果改进小，减小带宽以精细搜索
        if improvement < 0.01:
            self.kde_bandwidth *= 0.9
        else:
            self.kde_bandwidth *= 1.1
            
        # 限制带宽范围
        self.kde_bandwidth = np.clip(self.kde_bandwidth, 0.01, 0.5)


class ThompsonSampler(Sampler):
    """Thompson采样器（基于贝叶斯方法）"""
    
    def __init__(self, space: ParameterSpace, 
                 prior_mean: float = 0.0,
                 prior_std: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__(space, seed)
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.posterior_params = {}  # 存储每个区域的后验参数
        
    def sample(self, n_samples: int = 1) -> List[Dict]:
        """Thompson采样"""
        samples = []
        
        for _ in range(n_samples):
            # 为每个参数采样一个值
            config = {}
            for name, param in self.space.parameters.items():
                # 获取该参数的后验分布
                if name in self.posterior_params:
                    mean, std = self.posterior_params[name]
                else:
                    mean, std = self.prior_mean, self.prior_std
                    
                # 从后验分布采样
                sampled_score = self.random_state.normal(mean, std)
                
                # 根据采样的分数选择参数值
                # 这里简化为随机选择，实际可以更复杂
                config[name] = param.sample(self.random_state)
                
            samples.append(config)
            
        return samples
        
    def update(self, configs: List[Dict], scores: List[float]):
        """更新后验分布"""
        super().update(configs, scores)
        
        # 简化的后验更新（实际应该使用贝叶斯更新）
        for config, score in zip(configs, scores):
            for name, value in config.items():
                if name not in self.posterior_params:
                    self.posterior_params[name] = [score, self.prior_std]
                else:
                    # 简单的移动平均更新
                    old_mean, old_std = self.posterior_params[name]
                    new_mean = 0.9 * old_mean + 0.1 * score
                    new_std = 0.95 * old_std  # 逐渐减小不确定性
                    self.posterior_params[name] = [new_mean, new_std]
