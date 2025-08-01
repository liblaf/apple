from warp.types import *
import ast
import builtins
import inspect
import warp
from _typeshed import Incomplete
from typing import Any, Callable, ClassVar, Mapping

options: Incomplete

class WarpCodegenError(RuntimeError):
    def __init__(self, message) -> None: ...

class WarpCodegenTypeError(TypeError):
    def __init__(self, message) -> None: ...

class WarpCodegenAttributeError(AttributeError):
    def __init__(self, message) -> None: ...

class WarpCodegenKeyError(KeyError):
    def __init__(self, message) -> None: ...

builtin_operators: dict[type[ast.AST], str]
comparison_chain_strings: Incomplete

def values_check_equal(a, b): ...
def op_str_is_chainable(op: str) -> builtins.bool: ...
def get_closure_cell_contents(obj): ...
def eval_annotations(annotations: Mapping[str, Any], obj: Any) -> Mapping[str, Any]: ...
def get_annotations(obj: Any) -> Mapping[str, Any]: ...
def get_full_arg_spec(func: Callable) -> inspect.FullArgSpec: ...
def struct_instance_repr_recursive(inst: StructInstance, depth: int, use_repr: bool) -> str: ...

class StructInstance:
    def __init__(self, cls: Struct, ctype) -> None: ...
    def __getattribute__(self, name): ...
    def __setattr__(self, name, value) -> None: ...
    def __ctype__(self): ...
    def to(self, device): ...
    def numpy_dtype(self): ...
    def numpy_value(self): ...

class Struct:
    hash: bytes
    key: Incomplete
    cls: Incomplete
    module: Incomplete
    vars: dict[str, Var]
    ctype: Incomplete
    native_name: Incomplete
    default_constructor: Incomplete
    value_constructor: Incomplete
    instance_type: Incomplete
    def __init__(self, key: str, cls: type, module: warp.context.Module) -> None: ...
    def __call__(self): ...
    def initializer(self): ...
    def numpy_dtype(self): ...
    def from_ptr(self, ptr): ...

class Reference:
    value_type: Incomplete
    def __init__(self, value_type) -> None: ...

def is_reference(type: Any) -> builtins.bool: ...
def strip_reference(arg: Any) -> Any: ...
def compute_type_str(base_name, template_params): ...

class Var:
    label: Incomplete
    type: Incomplete
    requires_grad: Incomplete
    constant: Incomplete
    prefix: Incomplete
    is_read: bool
    is_write: bool
    parent: Incomplete
    relative_lineno: Incomplete
    def __init__(self, label: str, type: type, requires_grad: builtins.bool = False, constant: builtins.bool | None = None, prefix: builtins.bool = True, relative_lineno: int | None = None) -> None: ...
    @staticmethod
    def dtype_to_ctype(t: type) -> str: ...
    @staticmethod
    def type_to_ctype(t: type, value_type: builtins.bool = False) -> str: ...
    def ctype(self, value_type: builtins.bool = False) -> str: ...
    def emit(self, prefix: str = 'var'): ...
    def emit_adj(self): ...
    def mark_read(self) -> None: ...
    def mark_write(self, **kwargs) -> None: ...

class Block:
    body_forward: Incomplete
    body_replay: Incomplete
    body_reverse: Incomplete
    vars: Incomplete
    def __init__(self) -> None: ...

def apply_defaults(bound_args: inspect.BoundArguments, values: Mapping[str, Any]): ...
def func_match_args(func, arg_types, kwarg_types): ...
def get_arg_type(arg: Var | Any) -> type: ...
def get_arg_value(arg: Any) -> Any: ...

class Adjoint:
    def __init__(adj, func: Callable[..., Any], overload_annotations=None, is_user_function: bool = False, skip_forward_codegen: bool = False, skip_reverse_codegen: bool = False, custom_reverse_mode: bool = False, custom_reverse_num_input_args: int = -1, transformers: list[ast.NodeTransformer] | None = None, source: str | None = None) -> None: ...
    def alloc_shared_extra(adj, num_bytes) -> None: ...
    def get_total_required_shared(adj): ...
    @staticmethod
    def extract_function_source(func: Callable) -> tuple[str, int]: ...
    def build(adj, builder, default_builder_options=None) -> None: ...
    def format_template(adj, template, input_vars, output_var): ...
    def format_args(adj, prefix, args): ...
    def format_forward_call_args(adj, args, use_initializer_list): ...
    def format_reverse_call_args(adj, args_var, args, args_out, use_initializer_list, has_output_args: bool = True, require_original_output_arg: bool = False): ...
    def indent(adj) -> None: ...
    def dedent(adj) -> None: ...
    def begin_block(adj, name: str = 'block'): ...
    def end_block(adj): ...
    def add_var(adj, type=None, constant=None): ...
    def register_var(adj, var): ...
    def get_line_directive(adj, statement: str, relative_lineno: int | None = None) -> str | None: ...
    def add_forward(adj, statement: str, replay: str | None = None, skip_replay: builtins.bool = False) -> None: ...
    def add_reverse(adj, statement: str) -> None: ...
    def add_constant(adj, n): ...
    def load(adj, var): ...
    def add_comp(adj, op_strings, left, comps): ...
    def add_bool_op(adj, op_string, exprs): ...
    def resolve_func(adj, func, arg_types, kwarg_types, min_outputs): ...
    def add_call(adj, func, args, kwargs, type_args, min_outputs=None): ...
    def add_builtin_call(adj, func_name, args, min_outputs=None): ...
    def add_return(adj, var) -> None: ...
    def begin_if(adj, cond) -> None: ...
    def end_if(adj, cond) -> None: ...
    def begin_else(adj, cond) -> None: ...
    def end_else(adj, cond) -> None: ...
    def begin_for(adj, iter): ...
    def end_for(adj, iter) -> None: ...
    def begin_while(adj, cond) -> None: ...
    def end_while(adj) -> None: ...
    def emit_FunctionDef(adj, node) -> None: ...
    def emit_If(adj, node) -> None: ...
    def emit_IfExp(adj, node): ...
    def emit_Compare(adj, node): ...
    def emit_BoolOp(adj, node): ...
    def emit_Name(adj, node): ...
    @staticmethod
    def resolve_type_attribute(var_type: type, attr: str): ...
    def vector_component_index(adj, component, vector_type): ...
    def transform_component(adj, component): ...
    @staticmethod
    def is_differentiable_value_type(var_type): ...
    def emit_Attribute(adj, node): ...
    def emit_Assert(adj, node) -> None: ...
    def emit_Constant(adj, node): ...
    def emit_BinOp(adj, node): ...
    def emit_UnaryOp(adj, node): ...
    def materialize_redefinitions(adj, symbols) -> None: ...
    def emit_While(adj, node) -> None: ...
    def eval_num(adj, a): ...
    def contains_break(adj, body): ...
    def get_unroll_range(adj, loop): ...
    def record_constant_iter_symbol(adj, sym) -> None: ...
    def is_constant_iter_symbol(adj, sym): ...
    def emit_For(adj, node) -> None: ...
    def emit_Break(adj, node) -> None: ...
    def emit_Continue(adj, node) -> None: ...
    def emit_Expr(adj, node): ...
    def check_tid_in_func_error(adj, node) -> None: ...
    def resolve_arg(adj, arg): ...
    def emit_Call(adj, node): ...
    def emit_Index(adj, node): ...
    def eval_subscript(adj, node): ...
    def emit_Subscript(adj, node): ...
    def emit_Assign(adj, node): ...
    def emit_Return(adj, node) -> None: ...
    def emit_AugAssign(adj, node) -> None: ...
    def emit_Tuple(adj, node): ...
    def emit_Pass(adj, node) -> None: ...
    node_visitors: ClassVar[dict[type[ast.AST], Callable]]
    def eval(adj, node): ...
    def resolve_path(adj, path): ...
    def get_static_evaluation_context(adj): ...
    def is_static_expression(adj, func): ...
    def verify_static_return_value(adj, value): ...
    @staticmethod
    def extract_node_source_from_lines(source_lines, node) -> str | None: ...
    @staticmethod
    def extract_lambda_source(func, only_body: bool = False) -> str | None: ...
    def extract_node_source(adj, node) -> str | None: ...
    def evaluate_static_expression(adj, node) -> tuple[Any, str]: ...
    def replace_static_expressions(adj): ...
    def resolve_static_expression(adj, root_node, eval_types: bool = True): ...
    def resolve_external_reference(adj, name: str): ...
    def set_lineno(adj, lineno) -> None: ...
    def get_node_source(adj, node): ...
    def get_references(adj) -> tuple[dict[str, Any], dict[Any, Any], dict[warp.context.Function, Any]]: ...

cpu_module_header: str
cuda_module_header: str
struct_template: str
cpu_forward_function_template: str
cpu_reverse_function_template: str
cuda_forward_function_template: str
cuda_reverse_function_template: str
cuda_kernel_template_forward: str
cuda_kernel_template_backward: str
cpu_kernel_template_forward: str
cpu_kernel_template_backward: str
cpu_module_template_forward: str
cpu_module_template_backward: str

def constant_str(value): ...
def indent(args, stops: int = 1): ...
def make_full_qualified_name(func: Union[str, Callable]) -> str: ...
def codegen_struct(struct, device: str = 'cpu', indent_size: int = 4): ...
def codegen_func_forward(adj, func_type: str = 'kernel', device: str = 'cpu'): ...
def codegen_func_reverse(adj, func_type: str = 'kernel', device: str = 'cpu'): ...
def codegen_func(adj, c_func_name: str, device: str = 'cpu', options=None): ...
def codegen_snippet(adj, name, snippet, adj_snippet, replay_snippet): ...
def codegen_kernel(kernel, device, options): ...
def codegen_module(kernel, device, options): ...
