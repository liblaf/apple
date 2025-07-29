import ctypes
from _typeshed import Incomplete

class DLDeviceType(ctypes.c_int):
    kDLCPU: int
    kDLCUDA: int
    kDLCUDAHost: int
    kDLOpenCL: int
    kDLVulkan: int
    kDLMetal: int
    kDLVPI: int
    kDLROCM: int
    kDLROCMHost: int
    kDLCUDAManaged: int
    kDLOneAPI: int

class DLDevice(ctypes.Structure): ...

class DLDataTypeCode(ctypes.c_uint8):
    kDLInt: int
    kDLUInt: int
    kDLFloat: int
    kDLOpaquePointer: int
    kDLBfloat: int
    kDLComplex: int
    kDLBool: int

class DLDataType(ctypes.Structure):
    TYPE_MAP: Incomplete

class DLTensor(ctypes.Structure): ...
class DLManagedTensor(ctypes.Structure): ...
