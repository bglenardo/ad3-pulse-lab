"""Pulse generation and 2-channel capture with a Digilent Analog Discovery 3."""

from .dwfapi import DwfError, dwf
from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec
from .sweep import run_sweep, summarize
from .repeat import capture_repeats

__all__ = [
    "AnalogDiscovery",
    "Capture",
    "PulseSpec",
    "ScopeSpec",
    "DwfError",
    "dwf",
    "run_sweep",
    "summarize",
    "capture_repeats",
]
