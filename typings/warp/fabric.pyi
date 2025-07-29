from warp.types import *
import ctypes
from _typeshed import Incomplete

class fabricbucket_t(ctypes.Structure):
    index_start: Incomplete
    index_end: Incomplete
    ptr: Incomplete
    lengths: Incomplete
    def __init__(self, index_start: int = 0, index_end: int = 0, ptr=None, lengths=None) -> None: ...

class fabricarray_t(ctypes.Structure):
    buckets: Incomplete
    nbuckets: Incomplete
    size: Incomplete
    def __init__(self, buckets=None, nbuckets: int = 0, size: int = 0) -> None: ...

class indexedfabricarray_t(ctypes.Structure):
    fa: Incomplete
    indices: Incomplete
    size: int
    def __init__(self, fa=None, indices=None) -> None: ...

def fabric_to_warp_dtype(type_info, attrib_name): ...

class fabricarray(noncontiguous_array_base[T]):
    def __new__(cls, *args, **kwargs): ...
    device: Incomplete
    dtype: Incomplete
    access: Incomplete
    ndim: int
    deleter: Incomplete
    buckets: Incomplete
    size: Incomplete
    shape: Incomplete
    ctype: Incomplete
    def __init__(self, data=None, attrib=None, dtype=..., ndim=None) -> None: ...
    def __del__(self) -> None: ...
    def __ctype__(self): ...
    def __len__(self) -> int: ...
    def __getitem__(self, key): ...
    @property
    def vars(self): ...
    def fill_(self, value) -> None: ...

def fabricarrayarray(**kwargs): ...

class indexedfabricarray(noncontiguous_array_base[T]):
    fa: Incomplete
    indices: Incomplete
    dtype: Incomplete
    ndim: Incomplete
    device: Incomplete
    size: Incomplete
    shape: Incomplete
    ctype: Incomplete
    def __init__(self, fa=None, indices=None, dtype=None, ndim=None) -> None: ...
    def __ctype__(self): ...
    def __len__(self) -> int: ...
    @property
    def vars(self): ...
    def fill_(self, value) -> None: ...

def indexedfabricarrayarray(**kwargs): ...
