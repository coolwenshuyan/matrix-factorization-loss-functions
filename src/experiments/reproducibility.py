"""
复现性保证模块

确保实验结果的可复现性
"""

import os
import sys
import json
import hashlib
import pickle
import random
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import subprocess
import platform
import pkg_resources
from datetime import datetime


@dataclass
class EnvironmentSnapshot:
    """环境快照"""
    timestamp: str
    platform: str
    python_version: str
    packages: Dict[str, str]
    env_variables: Dict[str, str]
    git_info: Optional[Dict[str, str]] = None
    hardware_info: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def save(self, filepath: str):
        """保存环境快照"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def load(cls, filepath: str) -> 'EnvironmentSnapshot':
        """加载环境快照"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_requirements(self) -> List[str]:
        """生成requirements.txt内容"""
        return [f"{pkg}=={ver}" for pkg, ver in self.packages.items()]


class RandomStateManager:
    """随机状态管理器"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random_states = {}
        
    def set_all_seeds(self, seed: Optional[int] = None):
        """设置所有随机种子"""
        if seed is None:
            seed = self.seed
            
        # Python随机数
        random.seed(seed)
        self.random_states['python'] = random.getstate()
        
        # NumPy随机数
        np.random.seed(seed)
        self.random_states['numpy'] = np.random.get_state()
        
        # PyTorch随机数
        try:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # 确定性算法
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
            self.random_states['torch'] = {
                'manual_seed': seed,
                'cuda_seed': seed,
                'cudnn_deterministic': True,
                'cudnn_benchmark': False
            }
        except ImportError:
            pass
            
        # TensorFlow随机数
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            self.random_states['tensorflow'] = seed
        except ImportError:
            pass
            
        return seed
        
    def save_state(self, filepath: str):
        """保存随机状态"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'seed': self.seed,
                'states': self.random_states
            }, f)
            
    def load_state(self, filepath: str):
        """加载随机状态"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            
        self.seed = data['seed']
        self.random_states = data['states']
        
        # 恢复Python随机状态
        if 'python' in self.random_states:
            random.setstate(self.random_states['python'])
            
        # 恢复NumPy随机状态
        if 'numpy' in self.random_states:
            np.random.set_state(self.random_states['numpy'])
            
        # 恢复PyTorch状态
        if 'torch' in self.random_states:
            try:
                import torch
                torch_state = self.random_states['torch']
                torch.manual_seed(torch_state['manual_seed'])
                torch.cuda.manual_seed(torch_state['cuda_seed'])
                torch.backends.cudnn.deterministic = torch_state['cudnn_deterministic']
                torch.backends.cudnn.benchmark = torch_state['cudnn_benchmark']
            except ImportError:
                pass
                
    def get_random_generator(self, name: str) -> np.random.RandomState:
        """获取独立的随机数生成器"""
        # 基于名称生成确定性种子
        hash_obj = hashlib.md5(f"{self.seed}_{name}".encode())
        sub_seed = int(hash_obj.hexdigest()[:8], 16) % (2**31)
        
        return np.random.RandomState(sub_seed)


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(self,
                       state: Dict[str, Any],
                       name: str,
                       metadata: Optional[Dict] = None):
        """保存检查点"""
        checkpoint = {
            'state': state,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'version': self._get_version()
        }
        
        # 计算检查点哈希
        state_str = json.dumps(state, sort_keys=True, default=str)
        checkpoint['hash'] = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        # 保存检查点
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        # 保存元数据（便于查看）
        metadata_file = self.checkpoint_dir / f"{name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                'name': name,
                'timestamp': checkpoint['timestamp'],
                'hash': checkpoint['hash'],
                'metadata': checkpoint['metadata']
            }, f, indent=2)
            
        return str(checkpoint_file)
        
    def load_checkpoint(self, name: str) -> Dict[str, Any]:
        """加载检查点"""
        checkpoint_file = self.checkpoint_dir / f"{name}.pkl"
        
        if not checkpoint_file.exists():
            raise FileNotFoundError(f"检查点不存在: {name}")
            
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            
        # 验证完整性
        state_str = json.dumps(checkpoint['state'], sort_keys=True, default=str)
        current_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        if current_hash != checkpoint['hash']:
            raise ValueError("检查点完整性验证失败")
            
        return checkpoint
        
    def list_checkpoints(self) -> List[Dict]:
        """列出所有检查点"""
        checkpoints = []
        
        for metadata_file in self.checkpoint_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                checkpoints.append(metadata)
                
        # 按时间排序
        checkpoints.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return checkpoints
        
    def clean_old_checkpoints(self, keep_n: int = 5):
        """清理旧检查点"""
        checkpoints = self.list_checkpoints()
        
        if len(checkpoints) <= keep_n:
            return
            
        # 删除旧检查点
        for checkpoint in checkpoints[keep_n:]:
            name = checkpoint['name']
            
            # 删除文件
            (self.checkpoint_dir / f"{name}.pkl").unlink(missing_ok=True)
            (self.checkpoint_dir / f"{name}_metadata.json").unlink(missing_ok=True)
            
    def _get_version(self) -> str:
        """获取版本信息"""
        try:
            # 尝试获取Git版本
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()[:8]
        except:
            return "unknown"


class ReproducibilityManager:
    """复现性管理器"""
    
    def __init__(self, base_dir: str = "./reproducibility"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_state_manager = RandomStateManager()
        self.checkpoint_manager = CheckpointManager(self.base_dir / "checkpoints")
        
    def capture_environment(self) -> EnvironmentSnapshot:
        """捕获当前环境"""
        # 基本信息
        snapshot = EnvironmentSnapshot(
            timestamp=datetime.now().isoformat(),
            platform=platform.platform(),
            python_version=sys.version,
            packages=self._get_installed_packages(),
            env_variables=self._get_relevant_env_vars(),
            git_info=self._get_git_info(),
            hardware_info=self._get_hardware_info()
        )
        
        return snapshot
        
    def _get_installed_packages(self) -> Dict[str, str]:
        """获取已安装的包"""
        packages = {}
        
        for pkg in pkg_resources.working_set:
            packages[pkg.key] = pkg.version
            
        return packages
        
    def _get_relevant_env_vars(self) -> Dict[str, str]:
        """获取相关环境变量"""
        relevant_vars = [
            'PYTHONPATH',
            'CUDA_VISIBLE_DEVICES',
            'OMP_NUM_THREADS',
            'MKL_NUM_THREADS',
            'PYTHONHASHSEED'
        ]
        
        env_vars = {}
        for var in relevant_vars:
            if var in os.environ:
                env_vars[var] = os.environ[var]
                
        return env_vars
        
    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """获取Git信息"""
        try:
            git_info = {}
            
            # 获取当前commit
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            git_info['commit'] = result.stdout.strip()
            
            # 获取分支
            result = subprocess.run(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                capture_output=True,
                text=True,
                check=True
            )
            git_info['branch'] = result.stdout.strip()
            
            # 检查是否有未提交的更改
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True,
                check=True
            )
            git_info['has_changes'] = bool(result.stdout.strip())
            
            # 获取远程仓库URL
            result = subprocess.run(
                ['git', 'remote', 'get-url', 'origin'],
                capture_output=True,
                text=True,
                check=True
            )
            git_info['remote'] = result.stdout.strip()
            
            return git_info
            
        except subprocess.CalledProcessError:
            return None
            
    def _get_hardware_info(self) -> Dict[str, Any]:
        """获取硬件信息"""
        info = {
            'cpu': platform.processor(),
            'cpu_count': os.cpu_count(),
            'machine': platform.machine()
        }
        
        # 内存信息
        try:
            import psutil
            mem = psutil.virtual_memory()
            info['memory_gb'] = mem.total / (1024**3)
        except ImportError:
            pass
            
        # GPU信息
        try:
            import torch
            if torch.cuda.is_available():
                info['cuda_available'] = True
                info['cuda_device_count'] = torch.cuda.device_count()
                info['cuda_devices'] = []
                
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    info['cuda_devices'].append({
                        'name': props.name,
                        'memory_gb': props.total_memory / (1024**3),
                        'capability': f"{props.major}.{props.minor}"
                    })
                    
                info['cudnn_version'] = torch.backends.cudnn.version()
                
        except ImportError:
            pass
            
        return info
        
    def create_reproducibility_report(self, 
                                    experiment_id: str,
                                    config: Dict,
                                    results: Dict) -> str:
        """创建复现性报告"""
        report_dir = self.base_dir / "reports" / experiment_id
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # 捕获环境
        env_snapshot = self.capture_environment()
        env_snapshot.save(report_dir / "environment.json")
        
        # 保存requirements.txt
        with open(report_dir / "requirements.txt", 'w') as f:
            f.write('\n'.join(env_snapshot.get_requirements()))
            
        # 创建Dockerfile
        self._create_dockerfile(report_dir, env_snapshot)
        
        # 保存配置
        with open(report_dir / "config.json", 'w') as f:
            json.dump(config, f, indent=2)
            
        # 保存结果摘要
        with open(report_dir / "results_summary.json", 'w') as f:
            json.dump(results, f, indent=2)
            
        # 创建README
        self._create_readme(report_dir, experiment_id, config, results, env_snapshot)
        
        # 创建运行脚本
        self._create_run_script(report_dir, config)
        
        print(f"复现性报告已创建: {report_dir}")
        
        return str(report_dir)
        
    def _create_dockerfile(self, output_dir: Path, env_snapshot: EnvironmentSnapshot):
        """创建Dockerfile"""
        python_version = env_snapshot.python_version.split()[0]
        
        dockerfile_content = f"""FROM python:{python_version}

# 设置工作目录
WORKDIR /app

# 复制requirements
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY . .

# 设置环境变量
"""
        
        # 添加环境变量
        for var, value in env_snapshot.env_variables.items():
            dockerfile_content += f"ENV {var}={value}\n"
            
        dockerfile_content += """
# 运行命令
CMD ["python", "run_experiment.py"]
"""
        
        with open(output_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile_content)
            
    def _create_readme(self, 
                      output_dir: Path,
                      experiment_id: str,
                      config: Dict,
                      results: Dict,
                      env_snapshot: EnvironmentSnapshot):
        """创建README文件"""
        readme_content = f"""# 实验复现指南

## 实验信息
- 实验ID: {experiment_id}
- 创建时间: {env_snapshot.timestamp}
- 平台: {env_snapshot.platform}
- Python版本: {env_snapshot.python_version.split()[0]}

## 主要结果
"""
        
        # 添加主要结果
        for key, value in results.items():
            if isinstance(value, (int, float)):
                readme_content += f"- {key}: {value}\n"
                
        readme_content += f"""
## 复现步骤

### 方法1: 使用Docker（推荐）

```bash
# 构建镜像
docker build -t {experiment_id} .

# 运行实验
docker run -it {experiment_id}
```

### 方法2: 本地环境

1. 安装Python {env_snapshot.python_version.split()[0]}

2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. 运行实验：
```bash
python run_experiment.py
```

## 配置参数
```json
{json.dumps(config, indent=2)}
```

## 注意事项
- 确保使用相同的随机种子
- GPU结果可能因硬件差异略有不同
- 检查环境变量设置是否正确

## 文件说明
- `environment.json`: 完整环境信息
- `config.json`: 实验配置
- `results_summary.json`: 结果摘要
- `requirements.txt`: Python依赖
- `Dockerfile`: Docker镜像定义
- `run_experiment.py`: 运行脚本
"""
        
        with open(output_dir / "README.md", 'w') as f:
            f.write(readme_content)
            
    def _create_run_script(self, output_dir: Path, config: Dict):
        """创建运行脚本"""
        script_content = """#!/usr/bin/env python
\"\"\"
自动生成的实验运行脚本
\"\"\"

import json
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.config_manager import ExperimentConfig
from experiments.experiment_runner import ExperimentRunner
from experiments.reproducibility import ReproducibilityManager

def main():
    # 加载配置
    with open('config.json', 'r') as f:
        config_dict = json.load(f)
    
    config = ExperimentConfig.from_dict(config_dict)
    
    # 设置随机种子
    repro_manager = ReproducibilityManager()
    repro_manager.set_all_seeds(config.seed)
    
    # 创建运行器
    # 注意：这里需要实现具体的train_fn和eval_fn
    def train_fn(**kwargs):
        # TODO: 实现训练逻辑
        raise NotImplementedError("请实现训练函数")
        
    def eval_fn(**kwargs):
        # TODO: 实现评估逻辑
        raise NotImplementedError("请实现评估函数")
    
    runner = ExperimentRunner(train_fn, eval_fn)
    
    # 运行实验
    result = runner.run_experiment(config)
    
    # 保存结果
    print(f"实验完成: {result.experiment_id}")
    print(f"测试结果: {result.test_results}")
    
if __name__ == '__main__':
    main()
"""
        
        script_path = output_dir / "run_experiment.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
            
        # 设置可执行权限
        os.chmod(script_path, 0o755)
        
    def verify_reproducibility(self,
                             original_result: Dict,
                             reproduced_result: Dict,
                             tolerance: float = 1e-6) -> Dict[str, bool]:
        """验证复现性"""
        verification = {
            'config_match': True,
            'results_match': True,
            'details': {}
        }
        
        # 比较配置
        original_config = original_result.get('config', {})
        reproduced_config = reproduced_result.get('config', {})
        
        for key in set(original_config.keys()) | set(reproduced_config.keys()):
            if key not in original_config or key not in reproduced_config:
                verification['config_match'] = False
                verification['details'][f'config_{key}'] = False
            elif original_config[key] != reproduced_config[key]:
                verification['config_match'] = False
                verification['details'][f'config_{key}'] = False
                
        # 比较结果
        original_metrics = original_result.get('test_results', {})
        reproduced_metrics = reproduced_result.get('test_results', {})
        
        for metric in set(original_metrics.keys()) | set(reproduced_metrics.keys()):
            if metric not in original_metrics or metric not in reproduced_metrics:
                verification['results_match'] = False
                verification['details'][f'result_{metric}'] = False
            else:
                orig_val = original_metrics[metric]
                repro_val = reproduced_metrics[metric]
                
                if isinstance(orig_val, (int, float)) and isinstance(repro_val, (int, float)):
                    # 数值比较
                    if abs(orig_val - repro_val) > tolerance:
                        verification['results_match'] = False
                        verification['details'][f'result_{metric}'] = {
                            'match': False,
                            'original': orig_val,
                            'reproduced': repro_val,
                            'difference': abs(orig_val - repro_val)
                        }
                    else:
                        verification['details'][f'result_{metric}'] = {
                            'match': True,
                            'difference': abs(orig_val - repro_val)
                        }
                else:
                    # 非数值比较
                    if orig_val != repro_val:
                        verification['results_match'] = False
                        verification['details'][f'result_{metric}'] = False
                        
        verification['fully_reproducible'] = verification['config_match'] and verification['results_match']
        
        return verification