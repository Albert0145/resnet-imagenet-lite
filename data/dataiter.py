"""批次加载模块，这里暂时没有实现节省内存的iter方法，但也容易实现，利用多线程小批次 dataiter 分批读入处理同时处理后检查内存即可"""
from abc import ABC, abstractmethod
import numpy as np
import torch
import nibabel as nib
from typing import Union, Callable, List, Tuple
from concurrent.futures import ThreadPoolExecutor


class InputBackend(ABC):
    """
    决定输入文件处理模式
    """
    @abstractmethod
    def load_file(self, path):
        pass


class NumpyInputBackend(InputBackend):
    def load_file(self, path, dtype):
        return np.load(path).astype(dtype)
        

class NiiInputBackend(InputBackend):
    def load_file(self, path, dtype):
        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = nii.get_fdata(dtype=dtype)
        return data
        

class OutputBackend(ABC):
    """
    决定文件在函数内部和输出时的处理模式
    """
    @abstractmethod
    def to_output_backend(self, x):
        pass


class NumpyOutputBackend(OutputBackend):
    def to_output_backend(self, x, device=None):
        if isinstance(x, np.ndarray):
            return x
        elif isinstance(x, torch.Tensor):
            return x.numpy()
        else:
            raise ValueError("Unknown outputbackend")

    def stack_batch(self, batch_X:list):
        if isinstance(batch_X[0], torch.Tensor):
            return torch.stack(batch_X)
        elif isinstance(batch_X[0], np.ndarray):
            return np.stack(batch_X)
        else: return ValueError("Unknown stacktype")


class TorchOutputBackend(OutputBackend):
    def to_output_backend(self, x, device="cpu"):
        if isinstance(x, np.ndarray): 
            return torch.from_numpy(x).to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        else:
            raise ValueError("Unknown outputbackend")

    def stack_batch(self, batch_X:list):
        if isinstance(batch_X[0], torch.Tensor):
            return torch.stack(batch_X)
        elif isinstance(batch_X[0], np.ndarray):
            return np.stack(batch_X)
        else: return ValueError("Unknown stacktype")


class ListOutputBackend(OutputBackend):
    """
    为保证代码简洁性，这里list不改变元素数据类型，需外部实现数据类型转换
    """
    def to_output_backend(self, x, device="cpu"):
        return x

    def stack_batch(self, batch_X:list):
        return batch_X


class BaseDataManager:
    """
    单模态DataManager基类
    负责根据list返回数据
    """
    def __init__(self, input_backend="numpy", output_backend ="torch", device="cpu", dtype="float32"):
        self.device = device
        self.num_samples = 0
        self.dtype = np.dtype(dtype)
        
        if input_backend == "numpy":
            self.input_backend = NumpyInputBackend()
        elif input_backend == "nii":
            self.input_backend = NiiInputBackend()
        else:
            raise ValueError("Unknown backend")

        if output_backend == "torch":
            self.output_backend = TorchOutputBackend()
        elif output_backend == "numpy":
            self.output_backend = NumpyOutputBackend()
        elif output_backend == "list":
            self.output_backend = ListOutputBackend()
        else:
            raise ValueError("Unknown output backend")
        """
        - X: 数据集
        - y: 标签集
        - input_backend: 输入数据模式
        - output_backend: 输出数据模式
        - device: 设备
        - dtype: 元素数据类型
        """
        
    def get_data_by_indices(self, batch_idx):
        """
        子类必须实现此方法，按 batch_idx 获取一批数据
        """
        raise NotImplementedError


class InMemoryDataManager(BaseDataManager):
    """
    内存型 DataIter，为提高处理效率内部不提供数据类型转换功能，因此 input output backend 都置 None
    """
    def __init__(self, x: Union[np.ndarray, torch.Tensor], num_samples=None, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.num_samples = len(x)
        if isinstance(x, torch.Tensor):
            self.input_backend = None
            self.output_backend = None
            self.device = x.device
        elif isinstance(x, np.ndarray):
            self.input_backend = None
            self.output_backend = None
            self.device = 'cpu'
        else:
            raise ValueError("Unknown type of source")
        self.dtype = x.dtype


    def get_data_by_indices(self, batch_idx: np.ndarray):
        return self.x[batch_idx]


class LazyDataManager(BaseDataManager):
    """
    Lazy 型 DataIter，可通过output_backend指定输出类型
    """
    _shared_pool = None
    def __init__(self, data_source: Union[List[str], Callable[[List[int]], np.ndarray]], num_workers=4, **kwargs):
        """
        参数:
        - data_source:
            - 文件路径列表 [X_path, ...]
            - 自定义生成器 callable(batch_indices)
        - num_workers: 读取线程数
        - num_samples: 总样本数，如果 data_source 是生成器必须提供
        """
        super().__init__(**kwargs)
        self.data_source = data_source
        if LazyDataManager._shared_pool is None:
            LazyDataManager._shared_pool = ThreadPoolExecutor(max_workers=num_workers)

        self.pool = LazyDataManager._shared_pool

    def _load_single(self, idx:int):
        """
        获取单模态单样本
        """
        return self.input_backend.load_file(self.data_source[idx], self.dtype)
    
    def get_data_by_indices(self, batch_idx:np.ndarray):
        """
        获取单模态 batch，组织为 list
        """
        if callable(self.data_source):
            batch_X = self.data_source(batch_idx)
        else:
            batch_X = list(self.pool.map(self._load_single, batch_idx))

        batch_X = self.output_backend.stack_batch(batch_X)
        return self.output_backend.to_output_backend(batch_X)

    @classmethod
    def shutdown_pool(cls):
        """
        回收线程资源
        """
        if cls._shared_pool:
            cls._shared_pool.shutdown(wait=True)
            cls._shared_pool = None


class DataIter:
    """
    多模态迭代器，为保证灵活性返回list格式多模态数据
    """
    def __init__(self, X, y, num_samples = 0, num_workers=4, batch_size=32, shuffle=True, accum_size:int=None, 
                 input_backend="numpy", output_backend="numpy", manager='lazy', device="cpu", loop=True, dtype="float32"):
        if batch_size > num_samples:
            raise "batch_size > num_samples"
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.manager = manager
        self.shuffle = shuffle
        self.loop = loop
        self.current_idx = 0
        self._get_this_accum_flag = 1
        
        if accum_size == None:
            self.accum_size = batch_size
        else:
            self.accum_size = accum_size
        
        if loop:
            self.this_accum_len = self.accum_size
        else:
            self.this_accum_len = min(self.accum_size, self.num_samples)

        self.accum_idx = 0
        self._accum_flag = False
            
        if self.manager.lower() == "lazy":
            self.X_manager_list = [LazyDataManager(data_source=x, input_backend=input_backend, output_backend=output_backend, device=device
                                              , dtype=dtype, num_workers=num_workers) for x in X]
            self.y_manager = LazyDataManager(data_source=y, input_backend=input_backend, output_backend=output_backend, device=device
                                              , dtype=dtype, num_workers=num_workers)
            self.num_samples = num_samples
        elif self.manager.lower() == "inmemory":
            self.X_manager_list = [InMemoryDataManager(x=x) for x in X]
            self.y_manager = InMemoryDataManager(x=y)
            self.num_samples = len(X[0])
            print("InMemory 模式下需要在外部转换数据类型")
        else: 
            raise ValueError("Unknown type of source")
        """
        - X: 数据集，组织为多模态list
        - y: 标签集
        - input_backend: 输入数据模式
        - output_backend: 输出数据模式
        - device: 设备
        - dtype: 元素数据类型
        - batch_size: 批大小
        - shuffle: 是否随机idx
        - num_samples: 样本数量
        - indices: 索引
        - current_idx: 指针位置
        - loop: 是否循环采样
        - num_workers: 线程数
        - 
        """
        
    def __iter__(self):
        """
        用于迭代器初始化
        """
        self.current_idx = 0
        self.indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        """
        用于迭代器迭代，只实现丢弃小于batch批次的模式
        """
        if self.current_idx >= self.num_samples:
            if self.shuffle:
                np.random.shuffle(self.indices)
            if not self.loop:
                self._accum_flag = False
                self.current_idx = 0
                self.accum_idx = 0
                self._get_this_accum_flag = 1
                self.this_accum_len = min(self.accum_size, self.num_samples)
                
                raise StopIteration
            self.current_idx = 0
            
        if self._accum_flag:
            self.accum_idx = 0 
            self._get_this_accum_flag += 1
            self.this_accum_len = min(self.accum_size, self.num_samples-self.current_idx)
        
        self.accum_idx += self.batch_size
        start = self.current_idx
        next_batch_len = self.batch_size

        self._accum_flag = self.accum_idx >= self.accum_size or (self.batch_size+start) >= self.num_samples

        if self._accum_flag:
            if self.accum_size%self.batch_size !=0:
                next_batch_len = min(next_batch_len, self.accum_size%self.batch_size)

        end = start + next_batch_len
        batch_indices = self.indices[start:end]
        self.current_idx = end

        # ⚠ 多模态同步加载
        X = [manager.get_data_by_indices(batch_indices) for manager in self.X_manager_list]
        y = self.y_manager.get_data_by_indices(batch_indices)

        return X, y

    def get_accum_size_when_start(self, this_accum_len):
        if self.loop:
            return self.accum_size
        else:
            if self._get_this_accum_flag == 1:
                self._get_this_accum_flag -= 1
                return self.this_accum_len
            else:
                return this_accum_len

    def is_accum_complete(self):
        return self._accum_flag
        
    def __len__(self):
        """
        返回数据集长度
        """
        return (self.num_samples + self.batch_size - 1) // self.batch_size

    def shutdown_pools(self):
        """
        回收线程资源，这里是类内共用线程池，释放一个即可释放总线程池
        """
        self.y_manager.shutdown_pool()
        print("iter线程池已释放")