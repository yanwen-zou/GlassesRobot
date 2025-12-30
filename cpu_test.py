import ctypes
import os
libc = ctypes.CDLL("libc.so.6")
SCHED_RR = 2  # real-time scheduling policy
class SchedParam(ctypes.Structure):
    _fields_ = [("sched_priority", ctypes.c_int)]
param = SchedParam()
param.sched_priority = 99  # highest priority

pid = os.getpid()
if libc.sched_setscheduler(pid, SCHED_RR, ctypes.byref(param)) != 0:
    raise OSError("Failed to set scheduler")