"""
配置管理器模块

管理实验配置的创建、验证、版本控制和持久化
"""

import os
import json
import yaml
import copy
import hashlib
import git
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import shutil
from dataclasses import dataclass, asdict, field


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    # 基本信息
    name: str
    description: str = ""
    version: str = "1.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # 数据配置
    dataset: str = "ml-100k"
    data_path: str = "./data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # 模型配置
    model_type: str = "matrix_factorization"
    loss_type: str = "hpl"
    latent_factors: int = 50
    
    # HPL损失参数
    delta1: float = 0.5
    delta2: float = 1.5
    lambda_reg: float = 0.01
    
    # 训练配置
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip: float = 5.0
    
    # 评估配置
    eval_metrics: List[str] = field(default_factory=lambda: ["mae", "rmse", "hr@10", "ndcg@10"])
    eval_batch_size: int = 1024
    top_k: List[int] = field(default_factory=lambda: [5, 10, 20])
    
    # 实验设置
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    save_checkpoint: bool = True
    checkpoint_interval: int = 10
    
    # 日志设置
    log_level: str = "INFO"
    log_interval: int = 100
    tensorboard: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        """从字典创建"""
        return cls(**data)
    
    def get_hash(self) -> str:
        """获取配置哈希值"""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]


class ConfigTemplate:
    """配置模板管理"""
    
    def __init__(self, template_dir: str = "./configs/templates"):
        self.template_dir = Path(template_dir)
        self.template_dir.mkdir(parents=True, exist_ok=True)
        self._init_default_templates()
        
    def _init_default_templates(self):
        """初始化默认模板"""
        # 快速实验模板
        quick_template = ExperimentConfig(
            name="quick_experiment",
            description="快速验证实验模板",
            epochs=10,
            early_stopping_patience=3,
            checkpoint_interval=5
        )
        self.save_template("quick", quick_template)
        
        # 完整实验模板
        full_template = ExperimentConfig(
            name="full_experiment",
            description="完整评估实验模板",
            epochs=200,
            early_stopping_patience=20,
            checkpoint_interval=10
        )
        self.save_template("full", full_template)
        
        # 调试模板
        debug_template = ExperimentConfig(
            name="debug_experiment",
            description="调试实验模板",
            epochs=2,
            batch_size=32,
            log_interval=10,
            checkpoint_interval=1
        )
        self.save_template("debug", debug_template)
        
    def save_template(self, name: str, config: ExperimentConfig):
        """保存模板"""
        template_path = self.template_dir / f"{name}_template.yaml"
        with open(template_path, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
            
    def load_template(self, name: str) -> ExperimentConfig:
        """加载模板"""
        template_path = self.template_dir / f"{name}_template.yaml"
        if not template_path.exists():
            raise ValueError(f"模板不存在: {name}")
            
        with open(template_path, 'r') as f:
            data = yaml.safe_load(f)
            
        return ExperimentConfig.from_dict(data)
    
    def list_templates(self) -> List[str]:
        """列出所有模板"""
        templates = []
        for file in self.template_dir.glob("*_template.yaml"):
            name = file.stem.replace("_template", "")
            templates.append(name)
        return templates


class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate(config: ExperimentConfig) -> List[str]:
        """验证配置，返回错误列表"""
        errors = []
        
        # 数据配置验证
        if config.train_ratio + config.val_ratio + config.test_ratio != 1.0:
            errors.append("数据集划分比例之和必须为1.0")
            
        if not 0 < config.train_ratio < 1:
            errors.append("训练集比例必须在0到1之间")
            
        # 模型配置验证
        if config.latent_factors <= 0:
            errors.append("潜在因子数必须大于0")
            
        if config.loss_type == "hpl":
            if config.delta1 <= 0 or config.delta2 <= 0:
                errors.append("HPL损失的delta参数必须大于0")
            if config.delta1 >= config.delta2:
                errors.append("HPL损失要求delta1 < delta2")
                
        # 训练配置验证
        if config.learning_rate <= 0:
            errors.append("学习率必须大于0")
            
        if config.batch_size <= 0:
            errors.append("批量大小必须大于0")
            
        if config.epochs <= 0:
            errors.append("训练轮数必须大于0")
            
        # 设备验证
        if config.device not in ["cpu", "cuda", "mps"]:
            errors.append(f"不支持的设备类型: {config.device}")
            
        return errors
    
    @staticmethod
    def validate_compatibility(config1: ExperimentConfig, 
                             config2: ExperimentConfig) -> List[str]:
        """验证两个配置的兼容性"""
        warnings = []
        
        # 数据集兼容性
        if config1.dataset != config2.dataset:
            warnings.append("使用了不同的数据集")
            
        # 数据划分兼容性
        if (config1.train_ratio != config2.train_ratio or
            config1.val_ratio != config2.val_ratio):
            warnings.append("数据划分比例不同")
            
        # 评估指标兼容性
        if set(config1.eval_metrics) != set(config2.eval_metrics):
            warnings.append("评估指标不完全相同")
            
        return warnings


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, 
                 config_dir: str = "./configs",
                 use_git: bool = True):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiments_dir = self.config_dir / "experiments"
        self.experiments_dir.mkdir(exist_ok=True)
        
        self.use_git = use_git
        if use_git:
            self._init_git_repo()
            
        self.template_manager = ConfigTemplate(self.config_dir / "templates")
        self.validator = ConfigValidator()
        
    def _init_git_repo(self):
        """初始化Git仓库"""
        try:
            self.repo = git.Repo(self.config_dir)
        except git.InvalidGitRepositoryError:
            self.repo = git.Repo.init(self.config_dir)
            
    def create_config(self, 
                     name: str,
                     base_config: Optional[Union[ExperimentConfig, str]] = None,
                     **kwargs) -> ExperimentConfig:
        """创建新配置"""
        # 基础配置
        if base_config is None:
            config = ExperimentConfig(name=name, **kwargs)
        elif isinstance(base_config, str):
            # 从模板创建
            config = self.template_manager.load_template(base_config)
            config.name = name
            # 应用额外参数
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        else:
            # 从现有配置创建
            config_dict = base_config.to_dict()
            config_dict.update(kwargs)
            config_dict['name'] = name
            config = ExperimentConfig.from_dict(config_dict)
            
        # 更新创建时间和版本
        config.created_at = datetime.now().isoformat()
        
        # 验证配置
        errors = self.validator.validate(config)
        if errors:
            raise ValueError(f"配置验证失败:\n" + "\n".join(errors))
            
        return config
    
    def save_config(self, config: ExperimentConfig, 
                   tag: Optional[str] = None) -> str:
        """保存配置"""
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_hash = config.get_hash()
        filename = f"{config.name}_{timestamp}_{config_hash}.yaml"
        filepath = self.experiments_dir / filename
        
        # 保存配置
        with open(filepath, 'w') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False)
            
        # Git提交
        if self.use_git:
            self.repo.index.add([str(filepath.relative_to(self.config_dir))])
            commit_message = f"Add config: {config.name}"
            if tag:
                commit_message += f" (tag: {tag})"
            self.repo.index.commit(commit_message)
            
            if tag:
                self.repo.create_tag(tag, message=f"Config: {config.name}")
                
        return str(filepath)
    
    def load_config(self, 
                   name_or_path: str,
                   version: Optional[str] = None) -> ExperimentConfig:
        """加载配置"""
        if os.path.exists(name_or_path):
            # 直接从路径加载
            filepath = Path(name_or_path)
        else:
            # 按名称查找
            if version:
                # 加载特定版本
                filepath = self._find_config_by_version(name_or_path, version)
            else:
                # 加载最新版本
                filepath = self._find_latest_config(name_or_path)
                
        if not filepath or not filepath.exists():
            raise ValueError(f"找不到配置: {name_or_path}")
            
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            
        return ExperimentConfig.from_dict(data)
    
    def _find_latest_config(self, name: str) -> Optional[Path]:
        """查找最新的配置文件"""
        pattern = f"{name}_*.yaml"
        files = list(self.experiments_dir.glob(pattern))
        
        if not files:
            return None
            
        # 按修改时间排序
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files[0]
    
    def _find_config_by_version(self, name: str, version: str) -> Optional[Path]:
        """按版本查找配置"""
        if self.use_git:
            # 从Git历史查找
            try:
                # 检出特定版本
                self.repo.git.checkout(version)
                config_path = self._find_latest_config(name)
                # 恢复到当前版本
                self.repo.git.checkout('-')
                return config_path
            except git.GitCommandError:
                return None
        else:
            # 简单的版本匹配
            pattern = f"{name}_*_{version}*.yaml"
            files = list(self.experiments_dir.glob(pattern))
            return files[0] if files else None
    
    def list_configs(self, name_filter: Optional[str] = None) -> List[Dict]:
        """列出所有配置"""
        configs = []
        
        pattern = f"{name_filter}*.yaml" if name_filter else "*.yaml"
        for file in self.experiments_dir.glob(pattern):
            with open(file, 'r') as f:
                data = yaml.safe_load(f)
                
            configs.append({
                'name': data.get('name'),
                'file': file.name,
                'created_at': data.get('created_at'),
                'description': data.get('description', ''),
                'loss_type': data.get('loss_type'),
                'dataset': data.get('dataset')
            })
            
        # 按创建时间排序
        configs.sort(key=lambda x: x['created_at'], reverse=True)
        return configs
    
    def compare_configs(self, 
                       config1: Union[str, ExperimentConfig],
                       config2: Union[str, ExperimentConfig]) -> Dict:
        """比较两个配置"""
        # 加载配置
        if isinstance(config1, str):
            config1 = self.load_config(config1)
        if isinstance(config2, str):
            config2 = self.load_config(config2)
            
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        differences = {
            'only_in_first': {},
            'only_in_second': {},
            'different_values': {}
        }
        
        # 查找差异
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            if key not in dict2:
                differences['only_in_first'][key] = dict1[key]
            elif key not in dict1:
                differences['only_in_second'][key] = dict2[key]
            elif dict1[key] != dict2[key]:
                differences['different_values'][key] = {
                    'first': dict1[key],
                    'second': dict2[key]
                }
                
        return differences
    
    def merge_configs(self,
                     base_config: Union[str, ExperimentConfig],
                     update_config: Union[str, ExperimentConfig, Dict],
                     name: str) -> ExperimentConfig:
        """合并配置"""
        # 加载基础配置
        if isinstance(base_config, str):
            base_config = self.load_config(base_config)
            
        base_dict = base_config.to_dict()
        
        # 获取更新内容
        if isinstance(update_config, str):
            update_dict = self.load_config(update_config).to_dict()
        elif isinstance(update_config, ExperimentConfig):
            update_dict = update_config.to_dict()
        else:
            update_dict = update_config
            
        # 深度合并
        merged_dict = self._deep_merge(base_dict, update_dict)
        merged_dict['name'] = name
        merged_dict['created_at'] = datetime.now().isoformat()
        
        return ExperimentConfig.from_dict(merged_dict)
    
    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并字典"""
        result = copy.deepcopy(base)
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
                
        return result
    
    def export_configs(self, output_dir: str, names: Optional[List[str]] = None):
        """导出配置"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if names:
            # 导出指定配置
            for name in names:
                config = self.load_config(name)
                output_file = output_path / f"{name}_config.yaml"
                with open(output_file, 'w') as f:
                    yaml.dump(config.to_dict(), f, default_flow_style=False)
        else:
            # 导出所有配置
            shutil.copytree(self.experiments_dir, output_path / "experiments")
            
    def get_config_history(self, name: str) -> List[Dict]:
        """获取配置历史"""
        if not self.use_git:
            return []
            
        history = []
        
        # 获取包含该配置的提交
        pattern = f"{name}_*.yaml"
        for commit in self.repo.iter_commits(paths=pattern):
            history.append({
                'commit': commit.hexsha[:7],
                'date': commit.committed_datetime.isoformat(),
                'message': commit.message.strip(),
                'author': str(commit.author)
            })
            
        return history