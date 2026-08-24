"""Minimal ctypes binding for the Digilent WaveForms SDK (dwf).

Only the subset of the C API needed for pulse generation + 2-channel scope
capture is exposed. Every wrapped call raises :class:`DwfError` on failure.
"""

from __future__ import annotations

import sys
from ctypes import (
    CDLL,
    byref,
    c_byte,
    c_double,
    c_int,
    cdll,
    create_string_buffer,
)

# --- constants (from dwf.h) -------------------------------------------------

HDWF_NONE = 0

# DwfState
STATE_READY = 0
STATE_ARMED = 1
STATE_DONE = 2
STATE_TRIGGERED = 3
STATE_RUNNING = 3
STATE_CONFIG = 4
STATE_PREFILL = 5
STATE_WAIT = 7

# TRIGSRC
TRIGSRC_NONE = 0
TRIGSRC_PC = 1
TRIGSRC_DETECTOR_ANALOG_IN = 2
TRIGSRC_DETECTOR_DIGITAL_IN = 3
TRIGSRC_ANALOG_IN = 4
TRIGSRC_DIGITAL_IN = 5
TRIGSRC_DIGITAL_OUT = 6
TRIGSRC_ANALOG_OUT1 = 7
TRIGSRC_ANALOG_OUT2 = 8
TRIGSRC_ANALOG_OUT3 = 9
TRIGSRC_ANALOG_OUT4 = 10
TRIGSRC_EXTERNAL1 = 11
TRIGSRC_EXTERNAL2 = 12

# TRIGTYPE
TRIGTYPE_EDGE = 0
TRIGTYPE_PULSE = 1
TRIGTYPE_TRANSITION = 2

# DwfTriggerSlope
SLOPE_RISE = 0
SLOPE_FALL = 1
SLOPE_EITHER = 2

# ACQMODE
ACQMODE_SINGLE = 0
ACQMODE_SCAN_SHIFT = 1
ACQMODE_SCAN_SCREEN = 2
ACQMODE_RECORD = 3

# FILTER
FILTER_DECIMATE = 0
FILTER_AVERAGE = 1
FILTER_MIN_MAX = 2

# AnalogOutNode
NODE_CARRIER = 0
NODE_FM = 1
NODE_AM = 2

# FUNC
FUNC_DC = 0
FUNC_SINE = 1
FUNC_SQUARE = 2
FUNC_TRIANGLE = 3
FUNC_PULSE = 7
FUNC_CUSTOM = 30

# DwfAnalogOutIdle
IDLE_DISCONNECT = 0
IDLE_OFFSET = 1
IDLE_INITIAL = 2


class DwfError(RuntimeError):
    """Raised when a WaveForms SDK call reports failure."""


def _load_library() -> CDLL:
    try:
        if sys.platform.startswith("win"):
            return cdll.dwf
        if sys.platform.startswith("darwin"):
            return cdll.LoadLibrary("/Library/Frameworks/dwf.framework/dwf")
        return cdll.LoadLibrary("libdwf.so")
    except OSError as exc:  # pragma: no cover - environment dependent
        raise DwfError(
            "Could not load the WaveForms runtime (dwf). Install Digilent "
            "WaveForms from https://digilent.com/shop/software/digilent-waveforms/"
        ) from exc


class _Dwf:
    """Attribute proxy that turns dwf C calls into checked Python calls."""

    def __init__(self) -> None:
        self._lib = _load_library()
        self._cache: dict[str, object] = {}

    def last_error(self) -> str:
        buf = create_string_buffer(512)
        self._lib.FDwfGetLastErrorMsg(buf)
        return buf.value.decode(errors="replace").strip()

    def __getattr__(self, name: str):
        if name in self._cache:
            return self._cache[name]
        fn = getattr(self._lib, name)

        def call(*args):
            if fn(*args) == 0:
                raise DwfError(f"{name} failed: {self.last_error()}")

        self._cache[name] = call
        return call

    def version(self) -> str:
        buf = create_string_buffer(32)
        self.FDwfGetVersion(buf)
        return buf.value.decode()

    def enumerate_devices(self) -> list[dict]:
        count = c_int()
        self.FDwfEnum(c_int(0), byref(count))
        devices = []
        for i in range(count.value):
            name = create_string_buffer(64)
            serial = create_string_buffer(64)
            in_use = c_int()
            self.FDwfEnumDeviceName(c_int(i), name)
            self.FDwfEnumSN(c_int(i), serial)
            self.FDwfEnumDeviceIsOpened(c_int(i), byref(in_use))
            devices.append(
                {
                    "index": i,
                    "name": name.value.decode(),
                    "serial": serial.value.decode(),
                    "in_use": bool(in_use.value),
                }
            )
        return devices


dwf = _Dwf()

__all__ = [
    "dwf",
    "DwfError",
    "byref",
    "c_byte",
    "c_double",
    "c_int",
]
