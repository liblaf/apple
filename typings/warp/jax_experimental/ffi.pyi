from .xla_ffi import *
from _typeshed import Incomplete
from typing import Callable
from warp.codegen import get_full_arg_spec as get_full_arg_spec, make_full_qualified_name as make_full_qualified_name
from warp.jax import get_jax_device as get_jax_device
from warp.types import array_t as array_t, launch_bounds_t as launch_bounds_t, strides_from_shape as strides_from_shape, type_to_warp as type_to_warp

class FfiArg:
    name: Incomplete
    type: Incomplete
    is_array: Incomplete
    dtype_shape: Incomplete
    dtype_ndim: Incomplete
    jax_scalar_type: Incomplete
    jax_ndim: Incomplete
    warp_ndim: Incomplete
    def __init__(self, name, type) -> None: ...

class FfiLaunchDesc:
    static_inputs: Incomplete
    launch_dims: Incomplete
    def __init__(self, static_inputs, launch_dims) -> None: ...

class FfiKernel:
    kernel: Incomplete
    name: Incomplete
    num_outputs: Incomplete
    vmap_method: Incomplete
    launch_dims: Incomplete
    output_dims: Incomplete
    first_array_arg: Incomplete
    launch_id: int
    launch_descriptors: Incomplete
    num_kernel_args: Incomplete
    num_inputs: Incomplete
    input_args: Incomplete
    output_args: Incomplete
    callback_func: Incomplete
    def __init__(self, kernel, num_outputs, vmap_method, launch_dims, output_dims) -> None: ...
    def __call__(self, *args, output_dims=None, launch_dims=None, vmap_method=None): ...
    def ffi_callback(self, call_frame): ...

class FfiCallDesc:
    static_inputs: Incomplete
    def __init__(self, static_inputs) -> None: ...

class FfiCallable:
    func: Incomplete
    name: Incomplete
    num_outputs: Incomplete
    vmap_method: Incomplete
    graph_compatible: Incomplete
    output_dims: Incomplete
    first_array_arg: Incomplete
    call_id: int
    call_descriptors: Incomplete
    num_inputs: Incomplete
    args: Incomplete
    input_args: Incomplete
    output_args: Incomplete
    callback_func: Incomplete
    def __init__(self, func, num_outputs, graph_compatible, vmap_method, output_dims) -> None: ...
    def __call__(self, *args, output_dims=None, vmap_method=None): ...
    def ffi_callback(self, call_frame): ...

def jax_kernel(kernel, num_outputs: int = 1, vmap_method: str = 'broadcast_all', launch_dims=None, output_dims=None): ...
def jax_callable(func: Callable, num_outputs: int = 1, graph_compatible: bool = True, vmap_method: str = 'broadcast_all', output_dims=None): ...
def register_ffi_callback(name: str, func: Callable, graph_compatible: bool = True) -> None: ...

ffi_name_counts: Incomplete

def generate_unique_name(func) -> str: ...
def get_warp_shape(arg, dims): ...
def get_jax_output_type(arg, dims): ...
