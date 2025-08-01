import ctypes
import types
import warp as wp
import warp.context
from _typeshed import Incomplete
from typing import Any, Callable
from warp.context import Devicelike as Devicelike
from warp.types import Array as Array, DType as DType, type_repr as type_repr, types_equal as types_equal

warnings_seen: Incomplete

def warp_showwarning(message, category, filename, lineno, file=None, line=None) -> None: ...
def warn(message, category=None, stacklevel: int = 1) -> None: ...
def transform_expand(t): ...
@wp.func
def quat_between_vectors(a: wp.vec3, b: wp.vec3) -> wp.quat: ...
def array_scan(in_array, out_array, inclusive: bool = True) -> None: ...
def radix_sort_pairs(keys, values, count: int): ...
def segmented_sort_pairs(keys, values, count: int, segment_start_indices: None, segment_end_indices: None = None): ...
def runlength_encode(values, run_values, run_lengths, run_count=None, value_count=None): ...
def array_sum(values, out=None, value_count=None, axis=None): ...
def array_inner(a, b, out=None, count=None, axis=None): ...
def array_cast(in_array, out_array, count=None) -> None: ...
def create_warp_function(func: Callable) -> tuple[wp.Function, warp.context.Module]: ...
def broadcast_shapes(shapes: list[tuple[int]]) -> tuple[int]: ...
def map(func: Callable | wp.Function, *inputs: Array[DType] | Any, out: Array[DType] | list[Array[DType]] | None = None, return_kernel: bool = False, block_dim: int = 256, device: Devicelike = None) -> Array[DType] | list[Array[DType]] | wp.Kernel: ...
@wp.kernel
def copy_dense_volume_to_nano_vdb_v(volume: wp.uint64, values: None): ...
@wp.kernel
def copy_dense_volume_to_nano_vdb_f(volume: wp.uint64, values: None): ...
@wp.kernel
def copy_dense_volume_to_nano_vdb_i(volume: wp.uint64, values: None): ...

class MeshEdge:
    v0: Incomplete
    v1: Incomplete
    o0: Incomplete
    o1: Incomplete
    f0: Incomplete
    f1: Incomplete
    def __init__(self, v0, v1, o0, o1, f0, f1) -> None: ...

class MeshAdjacency:
    edges: Incomplete
    indices: Incomplete
    def __init__(self, indices, num_tris) -> None: ...
    def add_edge(self, i0, i1, o, f) -> None: ...

def mem_report() -> None: ...

class ScopedDevice:
    device: Incomplete
    def __init__(self, device: Devicelike) -> None: ...
    saved_device: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class ScopedStream:
    stream: Incomplete
    sync_enter: Incomplete
    sync_exit: Incomplete
    device: Incomplete
    device_scope: Incomplete
    def __init__(self, stream: wp.Stream | None, sync_enter: bool = True, sync_exit: bool = False) -> None: ...
    saved_stream: Incomplete
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

TIMING_KERNEL: int
TIMING_KERNEL_BUILTIN: int
TIMING_MEMCPY: int
TIMING_MEMSET: int
TIMING_GRAPH: int
TIMING_ALL: int

class ScopedTimer:
    indent: int
    enabled: bool
    name: Incomplete
    active: Incomplete
    print: Incomplete
    detailed: Incomplete
    dict: Incomplete
    use_nvtx: Incomplete
    color: Incomplete
    synchronize: Incomplete
    skip_tape: Incomplete
    elapsed: float
    cuda_filter: Incomplete
    report_func: Incomplete
    extra_msg: str
    def __init__(self, name: str, active: bool = True, print: bool = True, detailed: bool = False, dict: dict[str, list[float]] | None = None, use_nvtx: bool = False, color: int | str = 'rapids', synchronize: bool = False, cuda_filter: int = 0, report_func: Callable[[list[TimingResult], str], None] | None = None, skip_tape: bool = False) -> None: ...
    cp: Incomplete
    nvtx_range_id: Incomplete
    start: Incomplete
    def __enter__(self): ...
    timing_results: Incomplete
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class ScopedMempool:
    device: Incomplete
    enable: Incomplete
    def __init__(self, device: Devicelike, enable: bool) -> None: ...
    saved_setting: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class ScopedMempoolAccess:
    target_device: Incomplete
    peer_device: Incomplete
    enable: Incomplete
    def __init__(self, target_device: Devicelike, peer_device: Devicelike, enable: bool) -> None: ...
    saved_setting: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class ScopedPeerAccess:
    target_device: Incomplete
    peer_device: Incomplete
    enable: Incomplete
    def __init__(self, target_device: Devicelike, peer_device: Devicelike, enable: bool) -> None: ...
    saved_setting: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

class ScopedCapture:
    device: Incomplete
    stream: Incomplete
    force_module_load: Incomplete
    external: Incomplete
    active: bool
    graph: Incomplete
    def __init__(self, device: Devicelike = None, stream=None, force_module_load=None, external: bool = False) -> None: ...
    def __enter__(self): ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

def check_p2p(): ...

class timing_result_t(ctypes.Structure): ...

class TimingResult:
    device: warp.context.Device
    name: str
    filter: int
    elapsed: float
    def __init__(self, device, name, filter, elapsed) -> None: ...

def timing_begin(cuda_filter: int = ..., synchronize: bool = True) -> None: ...
def timing_end(synchronize: bool = True) -> list[TimingResult]: ...
def timing_print(results: list[TimingResult], indent: str = '') -> None: ...
