"""Minimal pynvml shim for WSL — wraps libnvidia-ml.so.1 via ctypes.

Provides just enough API for torch.cuda.utilization() and PufferLib GPU monitoring.
"""

import ctypes

_lib = ctypes.CDLL("libnvidia-ml.so.1")


class NVMLError(Exception):
    pass


class NVMLError_DriverNotLoaded(NVMLError):
    pass


def nvmlInit():
    rc = _lib.nvmlInit_v2()
    if rc != 0:
        raise NVMLError(f"nvmlInit failed: {rc}")


def nvmlDeviceGetHandleByIndex(index):
    handle = ctypes.c_void_p()
    rc = _lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(handle))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetHandleByIndex failed: {rc}")
    return handle


class _Utilization(ctypes.Structure):
    _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]


def nvmlDeviceGetUtilizationRates(handle):
    util = _Utilization()
    rc = _lib.nvmlDeviceGetUtilizationRates(handle, ctypes.byref(util))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetUtilizationRates failed: {rc}")
    return util


class _MemoryInfo(ctypes.Structure):
    _fields_ = [
        ("total", ctypes.c_ulonglong),
        ("free", ctypes.c_ulonglong),
        ("used", ctypes.c_ulonglong),
    ]


def nvmlDeviceGetMemoryInfo(handle):
    mem = _MemoryInfo()
    rc = _lib.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(mem))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetMemoryInfo failed: {rc}")
    return mem


def nvmlDeviceGetTemperature(handle, sensor):
    temp = ctypes.c_uint()
    rc = _lib.nvmlDeviceGetTemperature(handle, ctypes.c_uint(sensor), ctypes.byref(temp))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetTemperature failed: {rc}")
    return temp.value


def nvmlDeviceGetPowerUsage(handle):
    power = ctypes.c_uint()
    rc = _lib.nvmlDeviceGetPowerUsage(handle, ctypes.byref(power))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetPowerUsage failed: {rc}")
    return power.value


def nvmlDeviceGetClockInfo(handle, clock_type):
    clock = ctypes.c_uint()
    rc = _lib.nvmlDeviceGetClockInfo(handle, ctypes.c_uint(clock_type), ctypes.byref(clock))
    if rc != 0:
        raise NVMLError(f"nvmlDeviceGetClockInfo failed: {rc}")
    return clock.value
