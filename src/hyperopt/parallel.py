"""
并行执行管理模块

提供多种并行执行策略
"""

import os
import time
import queue
import threading
import multiprocessing as mp
from typing import List, Callable, Any, Optional, Dict, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from abc import ABC, abstractmethod
import traceback


class ParallelExecutor(ABC):
    """并行执行器基类"""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or self._get_default_workers()
        
    @abstractmethod
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """并行执行函数"""
        pass
    
    @abstractmethod
    def shutdown(self):
        """关闭执行器"""
        pass
    
    def _get_default_workers(self) -> int:
        """获取默认工作进程数"""
        return min(mp.cpu_count(), 4)
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class ThreadExecutor(ParallelExecutor):
    """线程池执行器（适合I/O密集型任务）"""
    
    def __init__(self, n_workers: int = None):
        super().__init__(n_workers)
        self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
        
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """使用线程池并行执行"""
        futures = []
        for inp in inputs:
            future = self.executor.submit(fn, inp)
            futures.append(future)
            
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
                
        return results
        
    def shutdown(self):
        """关闭线程池"""
        self.executor.shutdown(wait=True)


class ProcessExecutor(ParallelExecutor):
    """进程池执行器（适合CPU密集型任务）"""
    
    def __init__(self, n_workers: int = None):
        super().__init__(n_workers)
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """使用进程池并行执行"""
        futures = []
        for inp in inputs:
            future = self.executor.submit(fn, inp)
            futures.append(future)
            
        results = []
        for future in futures:
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(e)
                
        return results
        
    def shutdown(self):
        """关闭进程池"""
        self.executor.shutdown(wait=True)


class AsyncExecutor(ParallelExecutor):
    """异步执行器（支持更复杂的并行模式）"""
    
    def __init__(self, n_workers: int = None, use_process: bool = True):
        super().__init__(n_workers)
        self.use_process = use_process
        
        if use_process:
            self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.n_workers)
            
        self.futures = {}
        self.results = {}
        
    def submit(self, fn: Callable, inp: Any, task_id: Any = None) -> Any:
        """提交异步任务"""
        future = self.executor.submit(fn, inp)
        
        if task_id is None:
            task_id = id(future)
            
        self.futures[task_id] = future
        return task_id
        
    def get_result(self, task_id: Any, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        if task_id in self.results:
            return self.results[task_id]
            
        if task_id not in self.futures:
            raise KeyError(f"未知的任务ID: {task_id}")
            
        future = self.futures[task_id]
        
        try:
            result = future.result(timeout=timeout)
            self.results[task_id] = result
            del self.futures[task_id]
            return result
        except Exception as e:
            self.results[task_id] = e
            del self.futures[task_id]
            raise
            
    def get_completed(self) -> List[Any]:
        """获取已完成的任务ID"""
        completed = []
        
        for task_id, future in list(self.futures.items()):
            if future.done():
                completed.append(task_id)
                
        return completed
        
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """并行执行（阻塞直到完成）"""
        task_ids = []
        for i, inp in enumerate(inputs):
            task_id = self.submit(fn, inp, task_id=i)
            task_ids.append(task_id)
            
        results = []
        for task_id in task_ids:
            try:
                result = self.get_result(task_id)
                results.append(result)
            except Exception as e:
                results.append(e)
                
        return results
        
    def shutdown(self):
        """关闭执行器"""
        self.executor.shutdown(wait=True)


class DistributedExecutor(ParallelExecutor):
    """分布式执行器（使用任务队列）"""
    
    def __init__(self, n_workers: int = None):
        super().__init__(n_workers)
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.workers = []
        self.shutdown_event = mp.Event()
        
        # 启动工作进程
        for _ in range(self.n_workers):
            worker = mp.Process(
                target=self._worker_loop,
                args=(self.task_queue, self.result_queue, self.shutdown_event)
            )
            worker.start()
            self.workers.append(worker)
            
    @staticmethod
    def _worker_loop(task_queue: mp.Queue, result_queue: mp.Queue, 
                    shutdown_event: mp.Event):
        """工作进程主循环"""
        while not shutdown_event.is_set():
            try:
                # 获取任务
                task = task_queue.get(timeout=1)
                if task is None:  # 结束信号
                    break
                    
                task_id, fn, inp = task
                
                # 执行任务
                try:
                    result = fn(inp)
                    result_queue.put((task_id, 'success', result))
                except Exception as e:
                    result_queue.put((task_id, 'error', e))
                    
            except queue.Empty:
                continue
            except Exception as e:
                # 意外错误
                traceback.print_exc()
                
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """分布式执行"""
        # 提交任务
        task_ids = []
        for i, inp in enumerate(inputs):
            task_id = i
            self.task_queue.put((task_id, fn, inp))
            task_ids.append(task_id)
            
        # 收集结果
        results = {}
        while len(results) < len(task_ids):
            try:
                task_id, status, data = self.result_queue.get(timeout=300)
                
                if status == 'success':
                    results[task_id] = data
                else:
                    results[task_id] = data
                    
            except queue.Empty:
                raise TimeoutError("等待结果超时")
                
        # 按顺序返回结果
        ordered_results = []
        for task_id in task_ids:
            ordered_results.append(results[task_id])
            
        return ordered_results
        
    def shutdown(self):
        """关闭分布式执行器"""
        # 发送结束信号
        self.shutdown_event.set()
        
        # 清空队列
        for _ in range(self.n_workers):
            self.task_queue.put(None)
            
        # 等待工作进程结束
        for worker in self.workers:
            worker.join(timeout=10)
            if worker.is_alive():
                worker.terminate()


class AdaptiveExecutor(ParallelExecutor):
    """自适应执行器（根据任务特性选择执行策略）"""
    
    def __init__(self, n_workers: int = None):
        super().__init__(n_workers)
        self.thread_executor = ThreadExecutor(n_workers)
        self.process_executor = ProcessExecutor(n_workers)
        self.task_times = {'thread': [], 'process': []}
        self.current_executor = 'thread'
        
    def map(self, fn: Callable, inputs: List[Any]) -> List[Any]:
        """自适应选择执行器"""
        if len(inputs) <= 2:
            # 小批量直接用线程
            return self.thread_executor.map(fn, inputs)
            
        # 测试任务特性
        if len(self.task_times['thread']) < 5 or len(self.task_times['process']) < 5:
            # 收集性能数据
            executor_type = 'thread' if len(self.task_times['thread']) < 5 else 'process'
            executor = self.thread_executor if executor_type == 'thread' else self.process_executor
            
            start_time = time.time()
            results = executor.map(fn, inputs[:2])  # 测试前两个
            elapsed = time.time() - start_time
            
            self.task_times[executor_type].append(elapsed / 2)
            
            # 执行剩余任务
            if len(inputs) > 2:
                remaining_results = executor.map(fn, inputs[2:])
                results.extend(remaining_results)
                
            return results
            
        # 根据历史数据选择执行器
        avg_thread_time = np.mean(self.task_times['thread'])
        avg_process_time = np.mean(self.task_times['process'])
        
        if avg_thread_time < avg_process_time * 0.8:
            # 线程明显更快（考虑进程开销）
            return self.thread_executor.map(fn, inputs)
        else:
            # 进程更快或相当
            return self.process_executor.map(fn, inputs)
            
    def shutdown(self):
        """关闭执行器"""
        self.thread_executor.shutdown()
        self.process_executor.shutdown()


class ResourceManager:
    """资源管理器（管理计算资源）"""
    
    def __init__(self, 
                 max_cpu_percent: float = 80,
                 max_memory_gb: float = None,
                 max_workers: int = None):
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_gb = max_memory_gb
        self.max_workers = max_workers or mp.cpu_count()
        
        self.active_tasks = {}
        self.resource_lock = threading.Lock()
        
    def request_resources(self, task_id: str, 
                         cpu_cores: int = 1,
                         memory_gb: float = 1.0) -> bool:
        """请求资源"""
        with self.resource_lock:
            # 检查CPU限制
            current_cpu_usage = self._get_current_cpu_usage()
            if current_cpu_usage + (cpu_cores / mp.cpu_count() * 100) > self.max_cpu_percent:
                return False
                
            # 检查内存限制
            if self.max_memory_gb:
                current_memory_usage = self._get_current_memory_usage()
                if current_memory_usage + memory_gb > self.max_memory_gb:
                    return False
                    
            # 检查工作进程限制
            if len(self.active_tasks) >= self.max_workers:
                return False
                
            # 分配资源
            self.active_tasks[task_id] = {
                'cpu_cores': cpu_cores,
                'memory_gb': memory_gb,
                'start_time': time.time()
            }
            
            return True
            
    def release_resources(self, task_id: str):
        """释放资源"""
        with self.resource_lock:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
                
    def _get_current_cpu_usage(self) -> float:
        """获取当前CPU使用率"""
        # 简化实现
        return len(self.active_tasks) / mp.cpu_count() * 100
        
    def _get_current_memory_usage(self) -> float:
        """获取当前内存使用量"""
        # 简化实现
        return sum(task['memory_gb'] for task in self.active_tasks.values())
        
    def get_status(self) -> Dict:
        """获取资源状态"""
        with self.resource_lock:
            return {
                'active_tasks': len(self.active_tasks),
                'cpu_usage': self._get_current_cpu_usage(),
                'memory_usage': self._get_current_memory_usage(),
                'max_workers': self.max_workers
            }


class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, executor: ParallelExecutor, 
                 resource_manager: Optional[ResourceManager] = None):
        self.executor = executor
        self.resource_manager = resource_manager or ResourceManager()
        
        self.pending_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = {}
        
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self.scheduler_thread.daemon = True
        self.shutdown_event = threading.Event()
        
    def submit(self, fn: Callable, inp: Any, 
              priority: int = 0,
              task_id: Optional[str] = None,
              resources: Optional[Dict] = None) -> str:
        """提交任务"""
        if task_id is None:
            task_id = f"task_{time.time()}_{id(inp)}"
            
        task = {
            'id': task_id,
            'fn': fn,
            'input': inp,
            'priority': priority,
            'resources': resources or {'cpu_cores': 1, 'memory_gb': 1.0},
            'status': 'pending',
            'submit_time': time.time()
        }
        
        self.pending_queue.put((priority, task))
        return task_id
        
    def _scheduler_loop(self):
        """调度器主循环"""
        while not self.shutdown_event.is_set():
            try:
                # 获取待执行任务
                priority, task = self.pending_queue.get(timeout=1)
                
                # 请求资源
                if self.resource_manager.request_resources(
                    task['id'],
                    **task['resources']
                ):
                    # 执行任务
                    self._execute_task(task)
                else:
                    # 资源不足，放回队列
                    self.pending_queue.put((priority, task))
                    time.sleep(0.1)
                    
            except queue.Empty:
                continue
                
    def _execute_task(self, task: Dict):
        """执行任务"""
        task['status'] = 'running'
        task['start_time'] = time.time()
        self.running_tasks[task['id']] = task
        
        try:
            # 使用执行器执行
            result = self.executor.map(task['fn'], [task['input']])[0]
            
            task['status'] = 'completed'
            task['result'] = result
            
        except Exception as e:
            task['status'] = 'failed'
            task['error'] = str(e)
            
        finally:
            task['end_time'] = time.time()
            
            # 释放资源
            self.resource_manager.release_resources(task['id'])
            
            # 移动到完成列表
            del self.running_tasks[task['id']]
            self.completed_tasks[task['id']] = task
            
    def start(self):
        """启动调度器"""
        self.scheduler_thread.start()
        
    def stop(self):
        """停止调度器"""
        self.shutdown_event.set()
        self.scheduler_thread.join()
        
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        start_time = time.time()
        
        while True:
            if task_id in self.completed_tasks:
                task = self.completed_tasks[task_id]
                if task['status'] == 'completed':
                    return task['result']
                else:
                    raise Exception(f"任务失败: {task.get('error', 'Unknown error')}")
                    
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务 {task_id} 超时")
                
            time.sleep(0.1)