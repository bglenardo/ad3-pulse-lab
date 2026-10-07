"""AD3 pulse-generation tools plus external PMT measurement hardware wrappers.

The Analog Discovery 3 is treated as the pulse source only. The Rigol MHO954
scope and ISEG HV supply are the measurement/control devices for the PMT setup.
"""

from .dwfapi import DwfError, dwf
from .hardware import IsegHV, InstrumentError, RigolScope, RigolScopeCapture
from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec
from .repeat import capture_repeats
from .sweep import run_sweep, summarize

__all__ = [
    "AnalogDiscovery",
    "Capture",
    "PulseSpec",
    "ScopeSpec",
    "DwfError",
    "dwf",
    "IsegHV",
    "RigolScope",
    "RigolScopeCapture",
    "InstrumentError",
    "run_sweep",
    "summarize",
    "capture_repeats",
]
