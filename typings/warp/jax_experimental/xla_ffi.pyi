import ctypes
import enum
from _typeshed import Incomplete

class XLA_FFI_Extension_Type(enum.IntEnum):
    Metadata = 1

class XLA_FFI_Extension_Base(ctypes.Structure): ...

class XLA_FFI_ExecutionStage(enum.IntEnum):
    INSTANTIATE = 0
    PREPARE = 1
    INITIALIZE = 2
    EXECUTE = 3

class XLA_FFI_DataType(enum.IntEnum):
    INVALID = 0
    PRED = 1
    S8 = 2
    S16 = 3
    S32 = 4
    S64 = 5
    U8 = 6
    U16 = 7
    U32 = 8
    U64 = 9
    F16 = 10
    F32 = 11
    F64 = 12
    BF16 = 16
    C64 = 15
    C128 = 18
    TOKEN = 17
    F8E5M2 = 19
    F8E3M4 = 29
    F8E4M3 = 28
    F8E4M3FN = 20
    F8E4M3B11FNUZ = 23
    F8E5M2FNUZ = 24
    F8E4M3FNUZ = 25
    F4E2M1FN = 32
    F8E8M0FNU = 33

class XLA_FFI_Buffer(ctypes.Structure): ...

class XLA_FFI_ArgType(enum.IntEnum):
    BUFFER = 1

class XLA_FFI_RetType(enum.IntEnum):
    BUFFER = 1

class XLA_FFI_Args(ctypes.Structure): ...
class XLA_FFI_Rets(ctypes.Structure): ...
class XLA_FFI_ByteSpan(ctypes.Structure): ...
class XLA_FFI_Scalar(ctypes.Structure): ...
class XLA_FFI_Array(ctypes.Structure): ...

class XLA_FFI_AttrType(enum.IntEnum):
    ARRAY = 1
    DICTIONARY = 2
    SCALAR = 3
    STRING = 4

class XLA_FFI_Attrs(ctypes.Structure): ...
class XLA_FFI_Api_Version(ctypes.Structure): ...

class XLA_FFI_Handler_TraitsBits(enum.IntEnum):
    COMMAND_BUFFER_COMPATIBLE = ...

class XLA_FFI_Metadata(ctypes.Structure): ...
class XLA_FFI_Metadata_Extension(ctypes.Structure): ...

class XLA_FFI_Error_Code(enum.IntEnum):
    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16

class XLA_FFI_Error_Create_Args(ctypes.Structure): ...

XLA_FFI_Error_Create: Incomplete

class XLA_FFI_Stream_Get_Args(ctypes.Structure): ...

XLA_FFI_Stream_Get: Incomplete

class XLA_FFI_Api(ctypes.Structure): ...
class XLA_FFI_CallFrame(ctypes.Structure): ...

def decode_bytespan(span: XLA_FFI_ByteSpan): ...
def decode_scalar(scalar: XLA_FFI_Scalar): ...
def decode_array(array: XLA_FFI_Array): ...
def decode_attrs(attrs: XLA_FFI_Attrs): ...
def create_ffi_error(api, errc, message): ...
def create_invalid_argument_ffi_error(api, message): ...
def get_stream_from_callframe(call_frame): ...
def dtype_from_ffi(ffi_dtype): ...
def jax_dtype_from_ffi(ffi_dtype): ...

class ExecutionContext:
    stage: XLA_FFI_ExecutionStage
    stream: int
    def __init__(self, callframe: XLA_FFI_CallFrame) -> None: ...

class FfiBuffer:
    dtype: str
    data: int
    shape: tuple[int]
    def __init__(self, xla_buffer) -> None: ...
    @property
    def __cuda_array_interface__(self): ...
