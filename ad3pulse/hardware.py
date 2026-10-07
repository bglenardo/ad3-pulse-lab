"""Interfaces for the external capture chain: Rigol oscilloscope and ISEG HV supply.

The AD3 remains the pulse generator only; the oscilloscope and HV source are the
measurement/control devices used in the PMT experiment.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


class InstrumentError(RuntimeError):
    """Raised when the controlled instrument cannot accept a command."""


class _ResourceBase:
    """Small adapter around a fake or VISA instrument used in tests and runtime."""

    def __init__(self, instrument: Any | None = None) -> None:
        self._instrument = instrument

    def _write(self, cmd: str) -> None:
        if self._instrument is None:
            raise InstrumentError("No instrument connected")
        if hasattr(self._instrument, "write"):
            self._instrument.write(cmd)
            return
        raise InstrumentError("Instrument does not support write()")

    def _query(self, cmd: str) -> str:
        if self._instrument is None:
            raise InstrumentError("No instrument connected")
        if hasattr(self._instrument, "query"):
            return str(self._instrument.query(cmd))
        raise InstrumentError("Instrument does not support query()")

    def close(self) -> None:
        if self._instrument is not None and hasattr(self._instrument, "close"):
            self._instrument.close()


_ISEG_NUMBER = re.compile(r"\s*([+-]?\d+(?:\.\d*)?)(?:E([+-]?\d+)?)?")


def parse_iseg_value(text: str) -> float:
    """Parse an iseg reply such as ``1.23456E3V``, ``-4.0E3V`` or ``1.23456EA``.

    Replies carry a unit suffix, and the exponent digits may be missing
    (``1.23456EA`` means 1.23456 A; SCPI guide table 29).
    """
    m = _ISEG_NUMBER.match(text)
    if m is None:
        raise InstrumentError(f"Cannot parse iseg value {text!r}")
    return float(m.group(1)) * 10.0 ** int(m.group(2) or 0)


class IsegHV(_ResourceBase):
    """ISEG SHR 40 60 HV supply over Ethernet (raw TCP socket, port 10001).

    Commands follow ``Documentation/iseg_manual_iseg_SCPI_general_instruction_set.pdf``.
    Ethernet has no echo (section 3.1.2), but the SHR sends an empty line after
    order commands (section 2.1), so every order is sent with ``;*OPC?`` to get
    a definite answer and empty lines are skipped.
    """

    DEFAULT_HOST = "192.168.0.100"
    PORT = 10001  # fixed, SCPI guide table 10

    # Channel Status register bits (SCPI guide section 9.2).
    IS_POSITIVE = 1 << 0
    IS_ARC = 1 << 1
    IS_INPUT_ERROR = 1 << 2
    IS_ON = 1 << 3
    IS_VOLTAGE_RAMP = 1 << 4
    IS_EMERGENCY_OFF = 1 << 5
    IS_CONSTANT_CURRENT = 1 << 6
    IS_ARC_NUMBER_EXCEEDED = 1 << 9
    IS_EXTERNAL_INHIBIT = 1 << 12
    IS_CURRENT_TRIP = 1 << 13
    FAULT_BITS = {
        IS_INPUT_ERROR: "input error (set value out of range)",
        IS_EMERGENCY_OFF: "emergency off",
        IS_ARC_NUMBER_EXCEEDED: "arc number exceeded",
        IS_EXTERNAL_INHIBIT: "external inhibit",
        IS_CURRENT_TRIP: "current trip",
    }

    def __init__(self, resource_name: str | None = None, instrument: Any | None = None,
                 channel: int = 0, timeout_ms: int = 5000, visa_backend: str = "@py") -> None:
        super().__init__(instrument)
        self.resource_name = resource_name
        self.channel = channel
        self.timeout_ms = timeout_ms
        self.visa_backend = visa_backend
        self.idn = ""

    def connect(self, resource_name: str | None = None) -> "IsegHV":
        if self._instrument is not None:
            return self
        try:
            import pyvisa
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("pyvisa is required for live ISEG control") from exc
        rm = pyvisa.ResourceManager(self.visa_backend)
        name = self.visa_resource(resource_name or self.resource_name or self.DEFAULT_HOST)
        self.resource_name = name
        inst = rm.open_resource(name)
        inst.timeout = self.timeout_ms
        inst.write_termination = "\r\n"
        inst.read_termination = "\r\n"
        self._instrument = inst
        self.idn = self._query("*IDN?")
        return self

    @classmethod
    def visa_resource(cls, host_or_resource: str) -> str:
        """``192.168.0.100`` -> ``TCPIP0::192.168.0.100::10001::SOCKET``; full VISA names pass through."""
        if "::" in host_or_resource:
            return host_or_resource
        return f"TCPIP0::{host_or_resource}::{cls.PORT}::SOCKET"

    # -- line protocol -----------------------------------------------------
    def _readline(self) -> str:
        """Read the next non-empty line (skips the SHR's empty answers)."""
        while True:
            line = self._instrument.read().strip()
            if line:
                return line

    def _exchange(self, cmd: str) -> str:
        if self._instrument is None:
            raise InstrumentError("No instrument connected")
        self._instrument.write(cmd)
        try:
            return self._readline()
        except Exception as exc:
            # A command with an error gets no answer (SCPI guide section 2.2).
            raise InstrumentError(f"iseg did not answer {cmd!r} (rejected command?): {exc}") from exc

    def _write(self, cmd: str) -> None:
        answer = self._exchange(f"{cmd};*OPC?")
        if answer != "1":
            raise InstrumentError(f"iseg did not confirm {cmd!r}: got {answer!r}")

    def _query(self, cmd: str) -> str:
        return self._exchange(cmd)

    # -- control -----------------------------------------------------------
    @property
    def _ch(self) -> str:
        return f"(@{self.channel})"

    def set_voltage(self, volts: float) -> float:
        """Set VSET from a magnitude, signed to match the channel's configured polarity.

        The SHR reports VSET with its polarity sign (a negative channel reads
        back e.g. -1550 V), so the sign is taken from :CONF:OUTP:POL?.
        """
        if volts < 0:
            raise ValueError("Give the HV as a magnitude; polarity is set on the SHR itself")
        signed = -volts if self.read_polarity() == "n" else volts
        self._write(f":VOLT {signed:.3f},{self._ch}")
        self._check_faults()
        return float(signed)

    def set_current_limit(self, amps: float) -> float:
        self._write(f":CURR {amps:g},{self._ch}")
        self._check_faults()
        return float(amps)

    def set_ramp(self, volts_per_s: float) -> None:
        self._write(f":CONF:RAMP:VOLT {volts_per_s:g},{self._ch}")

    def output_on(self) -> None:
        self._write(f":VOLT ON,{self._ch}")

    def output_off(self) -> None:
        """Switch HV off with the configured ramp (the SHR ramps down on its own)."""
        self._write(f":VOLT OFF,{self._ch}")

    # -- readback ----------------------------------------------------------
    def read_voltage(self) -> float:
        return parse_iseg_value(self._query(f":MEAS:VOLT? {self._ch}"))

    def read_current(self) -> float:
        return parse_iseg_value(self._query(f":MEAS:CURR? {self._ch}"))

    def read_set_voltage(self) -> float:
        return parse_iseg_value(self._query(f":READ:VOLT? {self._ch}"))

    def read_current_limit(self) -> float:
        return parse_iseg_value(self._query(f":READ:CURR? {self._ch}"))

    def read_polarity(self) -> str:
        return self._query(f":CONF:OUTP:POL? {self._ch}").strip().lower()

    def read_status(self) -> int:
        return int(self._query(f":READ:CHAN:STAT? {self._ch}"))

    def _check_faults(self, status: int | None = None) -> int:
        status = self.read_status() if status is None else status
        faults = [text for bit, text in self.FAULT_BITS.items() if status & bit]
        if faults:
            raise InstrumentError(f"iseg channel {self.channel} fault: {', '.join(faults)} "
                                  f"(status {status})")
        return status

    @classmethod
    def describe_status(cls, status: int) -> list[str]:
        names = {cls.IS_POSITIVE: "positive", cls.IS_ARC: "arc", cls.IS_ON: "on",
                 cls.IS_VOLTAGE_RAMP: "ramping", cls.IS_CONSTANT_CURRENT: "constant current",
                 1 << 7: "constant voltage", **cls.FAULT_BITS}
        return [text for bit, text in names.items() if status & bit]

    def wait_stable(self, volts: float, tolerance_v: float = 1.0,
                    timeout_s: float = 120.0, poll_s: float = 0.5,
                    on_poll: Callable[[float, int], None] | None = None) -> float:
        """Wait until the ramp has finished and |VMEAS| is within tolerance of *volts*."""
        deadline = time.monotonic() + timeout_s
        while True:
            status = self._check_faults()
            measured = self.read_voltage()
            if on_poll is not None:
                on_poll(measured, status)
            if (status & self.IS_ON and not status & self.IS_VOLTAGE_RAMP
                    and abs(abs(measured) - volts) <= tolerance_v):
                return measured
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"iseg HV did not settle at {volts:g} V within {timeout_s:g} s "
                    f"(measured {measured:g} V, status {status})")
            time.sleep(poll_s)

    def wait_off(self, threshold_v: float = 5.0, timeout_s: float = 120.0,
                 poll_s: float = 0.5,
                 on_poll: Callable[[float, int], None] | None = None) -> float:
        """Wait until the channel is off, no longer ramping and |VMEAS| < *threshold_v*."""
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.read_status()
            measured = self.read_voltage()
            if on_poll is not None:
                on_poll(measured, status)
            if (not status & (self.IS_ON | self.IS_VOLTAGE_RAMP)
                    and abs(measured) < threshold_v):
                return measured
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"iseg channel {self.channel} still at {measured:g} V after "
                    f"{timeout_s:g} s (status {status})")
            time.sleep(poll_s)

    def __enter__(self) -> "IsegHV":
        self.connect(self.resource_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Deliberately leaves the HV as it is: closing the port does not change
        # the output, so the PMT stays biased between capture runs.
        self.close()


@dataclass
class RigolScopeCapture:
    """Ch1/Ch2 traces read from the Rigol, in volts and seconds (0 = trigger)."""

    time_s: np.ndarray
    ch1_V: np.ndarray
    ch2_V: np.ndarray
    sample_rate_Hz: float
    meta: dict = field(default_factory=dict)

    def to_csv(self, path) -> None:
        header = ",".join(f"{k}={v}" for k, v in self.meta.items())
        np.savetxt(
            path,
            np.column_stack((self.time_s, self.ch1_V, self.ch2_V)),
            delimiter=",",
            header=f"{header}\ntime_s,ch1_V,ch2_V",
            comments="# ",
        )


@dataclass
class Preamble:
    """Parsed ``:WAVeform:PREamble?`` reply (MHO900 programming guide p.477)."""

    format: int  # 0 = BYTE, 1 = WORD, 2 = ASCii
    type: int  # 0 = NORMal, 1 = MAXimum, 2 = RAW
    points: int
    count: int
    xincrement: float
    xorigin: float
    xreference: float
    yincrement: float
    yorigin: float
    yreference: float

    @classmethod
    def parse(cls, text: str) -> "Preamble":
        f = [float(v) for v in text.strip().split(",")[:10]]
        return cls(int(f[0]), int(f[1]), int(f[2]), int(f[3]), *f[4:])

    def volts(self, codes: np.ndarray) -> np.ndarray:
        return (np.asarray(codes, dtype=np.float64) - self.yorigin - self.yreference) * self.yincrement

    def times(self, n: int) -> np.ndarray:
        return self.xorigin + (np.arange(n) - self.xreference) * self.xincrement


class RigolScope(_ResourceBase):
    """Rigol MHO954 oscilloscope wrapper for single-shot Ch1/Ch2 capture.

    Commands follow ``Documentation/MHO900-ProgrammingGuide.pdf``. Readout is
    raw: codes are converted to volts with the scope's own preamble and no
    further calibration is applied.
    """

    DEFAULT_HOST = "192.168.0.101"
    CONNECT_ATTEMPTS = 5
    CONNECT_RETRY_S = 3.0
    # WORD byte order is not documented in the programming guide; confirm with
    # check_word_byte_order() on the real instrument.
    WORD_BIG_ENDIAN = False
    CHUNK_POINTS = 250_000

    def __init__(self, resource_name: str | None = None, instrument: Any | None = None,
                 timeout_ms: int = 20000, visa_backend: str = "@py") -> None:
        super().__init__(instrument)
        self.resource_name = resource_name
        self.timeout_ms = timeout_ms
        self.visa_backend = visa_backend
        self.idn = ""

    def connect(self, resource_name: str | None = None) -> "RigolScope":
        if self._instrument is not None:
            return self
        try:
            import pyvisa
        except ImportError as exc:  # pragma: no cover - environment-specific
            raise ImportError("pyvisa is required for live Rigol control") from exc
        rm = pyvisa.ResourceManager(self.visa_backend)
        name = self.visa_resource(resource_name or self.resource_name or self.DEFAULT_HOST)
        self.resource_name = name
        # The scope's VXI-11 server refuses a new link for a few seconds after
        # the previous one closes, so retry before giving up.
        for attempt in range(self.CONNECT_ATTEMPTS):
            try:
                self._instrument = rm.open_resource(name)
                break
            except pyvisa.errors.VisaIOError:
                if attempt == self.CONNECT_ATTEMPTS - 1:
                    raise
                time.sleep(self.CONNECT_RETRY_S)
        self._instrument.timeout = self.timeout_ms
        self.idn = self._query("*IDN?").strip()
        return self

    @staticmethod
    def visa_resource(host_or_resource: str) -> str:
        """``192.168.0.101`` -> ``TCPIP0::192.168.0.101::INSTR`` (LXI/VXI-11, the
        scope's default VISA type per :LAN:VISA?); full VISA names pass through."""
        if "::" in host_or_resource:
            return host_or_resource
        return f"TCPIP0::{host_or_resource}::INSTR"

    # -- configuration -----------------------------------------------------
    def configure_channel(self, channel: int, scale_v_div: float, offset_v: float = 0.0,
                          coupling: str = "DC", impedance: str = "OMEG") -> None:
        self._write(f":CHAN{channel}:DISP 1")
        self._write(f":CHAN{channel}:IMP {impedance}")
        self._write(f":CHAN{channel}:COUP {coupling}")
        self._write(f":CHAN{channel}:SCAL {scale_v_div:g}")
        self._write(f":CHAN{channel}:OFFS {offset_v:g}")

    def configure_timebase(self, scale_s_div: float, offset_s: float = 0.0,
                           memory_depth: str | None = None) -> None:
        self._write(f":TIM:MAIN:SCAL {scale_s_div:g}")
        self._write(f":TIM:MAIN:OFFS {offset_s:g}")
        if memory_depth is not None:
            self._write(f":ACQ:MDEP {memory_depth}")

    def configure_edge_trigger(self, source: int = 1, level_v: float = 0.0,
                               slope: str = "POS") -> None:
        self._write(":TRIG:MODE EDGE")
        self._write(f":TRIG:EDGE:SOUR CHAN{source}")
        self._write(f":TRIG:EDGE:SLOP {slope}")
        self._write(f":TRIG:EDGE:LEV {level_v:g}")

    # -- run control -------------------------------------------------------
    def trigger_status(self) -> str:
        return self._query(":TRIG:STAT?").strip().upper()

    def _wait_status(self, wanted: tuple[str, ...], timeout_s: float) -> str:
        deadline = time.monotonic() + timeout_s
        while True:
            status = self.trigger_status()
            if status in wanted:
                return status
            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Rigol trigger status stayed {status!r}, expected {'/'.join(wanted)}")
            time.sleep(0.02)

    def arm_single(self, timeout_s: float = 5.0, allow_early: bool = False) -> str:
        """Start a single acquisition and wait until the scope awaits a trigger.

        With *allow_early* a trigger that arrives before this returns is fine
        (forced or free-running captures); otherwise it is an error, because
        the capture would not contain the pulse fired afterwards.

        :SINGle takes about 1.4 s to take effect, so a STOP read right after it
        is usually the previous acquisition. Only WAIT or TD prove the new one
        is armed; STOP is accepted (with *allow_early*) only once the timeout
        has passed, i.e. when a noise trigger finished before the first poll.
        """
        self._write(":SINGle")
        try:
            status = self._wait_status(("WAIT", "TD"), timeout_s)
        except TimeoutError:
            status = self.trigger_status()
            if not (allow_early and status == "STOP"):
                raise
        if status != "WAIT" and not allow_early:
            raise InstrumentError(
                f"Scope triggered before the pulse was fired (status {status}); "
                "check the trigger level.")
        return status

    def force_trigger(self) -> None:
        """Trigger the armed single acquisition now (front-panel Force key)."""
        self._write(":TFORce")

    def wait_stopped(self, timeout_s: float = 5.0) -> None:
        self._wait_status(("STOP",), timeout_s)

    # -- readout -----------------------------------------------------------
    def read_preamble(self) -> Preamble:
        return Preamble.parse(self._query(":WAV:PRE?"))

    def read_channel(self, channel: int) -> tuple[np.ndarray, np.ndarray, Preamble]:
        """Read one channel's memory (scope must be stopped): (t_s, volts, preamble)."""
        self._write(f":WAV:SOUR CHAN{channel}")
        self._write(":WAV:MODE RAW")
        self._write(":WAV:FORM WORD")
        pre = self.read_preamble()
        chunks = []
        for start in range(1, pre.points + 1, self.CHUNK_POINTS):
            stop = min(start + self.CHUNK_POINTS - 1, pre.points)
            self._write(f":WAV:STAR {start}")
            self._write(f":WAV:STOP {stop}")
            chunks.append(self._query_codes())
        codes = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)
        return pre.times(len(codes)), pre.volts(codes), pre

    def _query_codes(self) -> np.ndarray:
        if not hasattr(self._instrument, "query_binary_values"):
            raise InstrumentError("Instrument does not support binary reads")
        values = self._instrument.query_binary_values(
            ":WAV:DATA?", datatype="H", is_big_endian=self.WORD_BIG_ENDIAN,
            container=np.array)
        return np.asarray(values, dtype=np.uint16)

    def check_word_byte_order(self, channel: int = 1) -> float:
        """Max |WORD - ASCii| difference in volts for the on-screen data.

        A result of about one Y increment means WORD_BIG_ENDIAN is right; a
        large value means the byte order must be flipped.
        """
        self._write(f":WAV:SOUR CHAN{channel}")
        self._write(":WAV:MODE NORM")
        self._write(":WAV:FORM ASC")
        raw = self._query(":WAV:DATA?").strip()
        if raw.startswith("#"):
            raw = raw[2 + int(raw[1]):]
        ascii_v = np.array([float(v) for v in raw.split(",") if v.strip()])
        self._write(":WAV:FORM WORD")
        pre = self.read_preamble()
        word_v = pre.volts(self._query_codes())
        n = min(len(ascii_v), len(word_v))
        return float(np.max(np.abs(ascii_v[:n] - word_v[:n]))) if n else float("nan")

    def capture_channels(self) -> RigolScopeCapture:
        """Read Ch1 and Ch2 from the stopped scope."""
        t1, v1, p1 = self.read_channel(1)
        _, v2, p2 = self.read_channel(2)
        n = min(len(v1), len(v2))
        if n == 0:
            raise InstrumentError(
                f"Rigol returned no data (Ch1 {len(v1)}, Ch2 {len(v2)} points); "
                "the acquisition was probably not complete when it was read.")
        rate = 1.0 / p1.xincrement
        meta = {
            "idn": self.idn.replace(",", " "),
            "sample_rate_Hz": rate,
            "ch1_yinc_V": p1.yincrement,
            "ch2_yinc_V": p2.yincrement,
            "points": n,
        }
        return RigolScopeCapture(time_s=t1[:n], ch1_V=v1[:n], ch2_V=v2[:n],
                                 sample_rate_Hz=rate, meta=meta)

    def __enter__(self) -> "RigolScope":
        self.connect(self.resource_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = [
    "IsegHV",
    "RigolScope",
    "RigolScopeCapture",
    "Preamble",
    "InstrumentError",
    "parse_iseg_value",
]
