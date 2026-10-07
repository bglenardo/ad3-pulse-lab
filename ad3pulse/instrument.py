"""High-level control of an Analog Discovery 3: square-pulse output + 2-ch capture."""

from __future__ import annotations

import time
from ctypes import byref, c_byte, c_double, c_int
from dataclasses import dataclass, field, replace

import numpy as np

from . import dwfapi as D
from .dwfapi import DwfError, dwf

# AD3 AWG sample rate (125 MS/s -> 8 ns steps).
AWG_MAX_RATE = 125e6


@dataclass
class PulseSpec:
    """A single rectangular pulse (or pulse train) played on one AWG channel."""

    amplitude: float  # pulse height in volts (may be negative)
    width: float  # pulse width in seconds
    pre_delay: float = 0.0  # baseline before the pulse, seconds
    post_delay: float | None = None  # baseline after the pulse; default = 4x width
    baseline: float = 0.0  # voltage between pulses
    repeat: int = 1  # number of pulses to play
    channel: int = 0  # AWG channel index (0 = W1, 1 = W2)
    samples: int = 4096  # points used to render one period

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be > 0")
        if self.pre_delay < 0:
            raise ValueError("pre_delay must be >= 0")
        if self.repeat < 1:
            raise ValueError("repeat must be >= 1")
        if self.post_delay is None:
            self.post_delay = 4.0 * self.width

    @property
    def period(self) -> float:
        return self.pre_delay + self.width + self.post_delay

    @property
    def total_time(self) -> float:
        return self.period * self.repeat

    def render(self, samples: int | None = None) -> tuple[np.ndarray, float]:
        """Normalised (-1..1) waveform for one period, plus the volt scale factor."""
        n = samples or self.samples
        t = (np.arange(n) + 0.5) / n * self.period
        high = (t >= self.pre_delay) & (t < self.pre_delay + self.width)
        span = max(abs(self.amplitude), abs(self.baseline), 1e-12)
        data = np.where(high, self.amplitude, self.baseline) / span
        return np.clip(data, -1.0, 1.0), span

    @property
    def timing_resolution(self) -> float:
        return self.period / self.samples

    def fit_to_awg(self, max_rate: float = AWG_MAX_RATE, max_samples: int | None = None) -> float:
        """Limit `samples` so the AWG plays them at <= max_rate; return the realised width.

        The AWG plays the custom buffer at samples / period, so short pulses
        with the default 4096 points would ask for far more than the device rate.
        """
        limit = max(2, int(self.period * max_rate + 1e-6))
        if max_samples:
            limit = min(limit, max_samples)
        self.samples = min(self.samples, limit)
        t = (np.arange(self.samples) + 0.5) / self.samples * self.period
        high = (t >= self.pre_delay) & (t < self.pre_delay + self.width)
        return np.count_nonzero(high) * self.timing_resolution


@dataclass
class ScopeSpec:
    """Analog-in configuration."""

    sample_rate: float | None = None  # Hz; None -> derived from the pulse timing
    buffer_size: int | None = None  # samples; None -> device maximum
    ranges: tuple[float, float] = (5.0, 5.0)  # V peak-to-peak per channel
    offsets: tuple[float, float] = (0.0, 0.0)
    pretrigger_fraction: float = 0.1  # fraction of the record kept before trigger
    window_margin: float = 1.2  # capture window = margin * pulse total time
    timeout: float = 10.0  # seconds to wait for the acquisition
    auto_range: bool = False  # pick range/offset per shot to avoid clipping
    headroom: float = 1.3  # required span multiple when auto-ranging
    shrink_threshold: float = 0.35  # re-range down if the signal fills less than this
    range_retries: int = 3  # extra acquisitions allowed while converging
    min_interval: float = 0.5  # minimum seconds between acquisition starts


@dataclass
class Capture:
    """Result of one acquisition."""

    t: np.ndarray
    ch1: np.ndarray
    ch2: np.ndarray
    sample_rate: float
    pulse: PulseSpec
    meta: dict = field(default_factory=dict)

    def to_csv(self, path) -> None:
        header = (
            f"amplitude_V={self.pulse.amplitude},width_s={self.pulse.width},"
            f"repeat={self.pulse.repeat},sample_rate_Hz={self.sample_rate},"
            f"ch1_range_V={self.meta.get('ch1_range_V')},"
            f"ch1_offset_V={self.meta.get('ch1_offset_V')},"
            f"ch2_range_V={self.meta.get('ch2_range_V')},"
            f"ch2_offset_V={self.meta.get('ch2_offset_V')}\n"
            "time_s,ch1_V,ch2_V"
        )
        np.savetxt(
            path,
            np.column_stack((self.t, self.ch1, self.ch2)),
            delimiter=",",
            header=header,
            comments="# ",
        )


class AnalogDiscovery:
    """Context manager wrapping one open device handle."""

    def __init__(self, device_index: int = -1, config_index: int | None = None) -> None:
        self._hdwf = c_int(D.HDWF_NONE)
        self._device_index = device_index
        self._config_index = config_index
        self._awg_configured: set[int] = set()
        self._next_capture_at = 0.0

    # -- lifecycle ---------------------------------------------------------
    def open(self) -> "AnalogDiscovery":
        if self._config_index is None:
            dwf.FDwfDeviceOpen(c_int(self._device_index), byref(self._hdwf))
        else:
            dwf.FDwfDeviceConfigOpen(
                c_int(self._device_index), c_int(self._config_index), byref(self._hdwf)
            )
        if self._hdwf.value == D.HDWF_NONE:
            raise DwfError(f"Could not open device: {dwf.last_error()}")
        # Apply settings immediately on each *Set call.
        dwf.FDwfDeviceAutoConfigureSet(self._hdwf, c_int(1))
        return self

    def close(self) -> None:
        if self._hdwf.value != D.HDWF_NONE:
            try:
                dwf.FDwfAnalogOutReset(self._hdwf, c_int(-1))
                dwf.FDwfAnalogInReset(self._hdwf)
            except DwfError:
                pass
            dwf.FDwfDeviceClose(self._hdwf)
            self._hdwf = c_int(D.HDWF_NONE)

    def __enter__(self) -> "AnalogDiscovery":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()

    # -- capability queries ------------------------------------------------
    def max_scope_buffer(self) -> int:
        lo, hi = c_int(), c_int()
        dwf.FDwfAnalogInBufferSizeInfo(self._hdwf, byref(lo), byref(hi))
        return hi.value

    def max_scope_rate(self) -> float:
        lo, hi = c_double(), c_double()
        dwf.FDwfAnalogInFrequencyInfo(self._hdwf, byref(lo), byref(hi))
        return hi.value

    def max_awg_samples(self, channel: int = 0) -> int:
        lo, hi = c_double(), c_double()
        dwf.FDwfAnalogOutNodeDataInfo(
            self._hdwf, c_int(channel), c_int(D.NODE_CARRIER), byref(lo), byref(hi)
        )
        return int(hi.value)

    def available_ranges(self, channel: int = 0) -> list[float]:
        """Selectable input spans (volts peak-to-peak), ascending."""
        steps = (c_double * 32)()
        count = c_int()
        dwf.FDwfAnalogInChannelRangeSteps(self._hdwf, steps, byref(count))
        values = sorted({round(steps[i], 6) for i in range(count.value)})
        return values or [5.0, 50.0]

    def offset_limits(self, channel: int = 0) -> tuple[float, float]:
        lo, hi, steps = c_double(), c_double(), c_double()
        dwf.FDwfAnalogInChannelOffsetInfo(
            self._hdwf, byref(lo), byref(hi), byref(steps)
        )
        return lo.value, hi.value

    def pick_range(self, span: float, channel: int = 0) -> float:
        """Smallest available input span that fits `span` volts."""
        options = self.available_ranges(channel)
        for value in options:
            if value >= span:
                return value
        return options[-1]

    # -- generator ---------------------------------------------------------
    def configure_pulse(self, pulse: PulseSpec) -> None:
        ch = c_int(pulse.channel)
        node = c_int(D.NODE_CARRIER)
        h = self._hdwf

        dwf.FDwfAnalogOutReset(h, ch)
        dwf.FDwfAnalogOutNodeEnableSet(h, ch, node, c_int(1))
        dwf.FDwfAnalogOutNodeFunctionSet(h, ch, node, c_byte(D.FUNC_CUSTOM))

        # The custom buffer depth is only reported once the function is selected.
        max_samples = self.max_awg_samples(pulse.channel)
        pulse.fit_to_awg(AWG_MAX_RATE, max_samples if max_samples > 0 else None)

        data, span = pulse.render()
        buf = (c_double * len(data))(*data.tolist())

        dwf.FDwfAnalogOutNodeDataSet(h, ch, node, buf, c_int(len(data)))
        dwf.FDwfAnalogOutNodeFrequencySet(h, ch, node, c_double(1.0 / pulse.period))
        dwf.FDwfAnalogOutNodeAmplitudeSet(h, ch, node, c_double(span))
        dwf.FDwfAnalogOutNodeOffsetSet(h, ch, node, c_double(0.0))
        dwf.FDwfAnalogOutIdleSet(h, ch, c_int(D.IDLE_OFFSET))
        dwf.FDwfAnalogOutRunSet(h, ch, c_double(pulse.total_time))
        dwf.FDwfAnalogOutRepeatSet(h, ch, c_int(1))
        dwf.FDwfAnalogOutTriggerSourceSet(h, ch, c_int(D.TRIGSRC_NONE))
        self._awg_configured.add(pulse.channel)

    def stop_output(self, channel: int = -1) -> None:
        dwf.FDwfAnalogOutConfigure(self._hdwf, c_int(channel), c_int(0))

    def fire(self, pulse: PulseSpec, timeout: float = 5.0) -> None:
        """Play a configured pulse once and wait for the AWG to finish (no AD3 capture)."""
        self._wait_awg_idle(pulse.channel, timeout)
        dwf.FDwfAnalogOutConfigure(self._hdwf, c_int(pulse.channel), c_int(1))
        sts = c_byte()
        deadline = time.monotonic() + timeout + pulse.total_time
        while True:
            dwf.FDwfAnalogOutStatus(self._hdwf, c_int(pulse.channel), byref(sts))
            if sts.value == D.STATE_DONE:
                return
            if time.monotonic() > deadline:
                self.stop_output(pulse.channel)
                raise TimeoutError("AWG did not finish playing the pulse.")
            time.sleep(0.001)

    # -- scope -------------------------------------------------------------
    def configure_scope(self, pulse: PulseSpec, scope: ScopeSpec) -> tuple[float, int]:
        h = self._hdwf
        max_buffer = self.max_scope_buffer()
        buffer_size = min(scope.buffer_size or max_buffer, max_buffer)

        window = pulse.total_time * scope.window_margin
        rate = min(scope.sample_rate or (buffer_size / window), self.max_scope_rate())
        if scope.buffer_size is None:
            # Rate clamped at the device maximum: shorten the record instead of
            # capturing a mostly-empty window.
            buffer_size = max(16, min(buffer_size, int(round(window * rate))))

        dwf.FDwfAnalogInReset(h)
        applied: dict = {}
        for i in (0, 1):
            dwf.FDwfAnalogInChannelEnableSet(h, c_int(i), c_int(1))
            dwf.FDwfAnalogInChannelRangeSet(h, c_int(i), c_double(scope.ranges[i]))
            dwf.FDwfAnalogInChannelOffsetSet(h, c_int(i), c_double(scope.offsets[i]))
            dwf.FDwfAnalogInChannelFilterSet(h, c_int(i), c_int(D.FILTER_DECIMATE))
            got_r, got_o = c_double(), c_double()
            dwf.FDwfAnalogInChannelRangeGet(h, c_int(i), byref(got_r))
            dwf.FDwfAnalogInChannelOffsetGet(h, c_int(i), byref(got_o))
            applied[f"ch{i + 1}_range_V"] = got_r.value
            applied[f"ch{i + 1}_offset_V"] = got_o.value
        self._applied = applied

        dwf.FDwfAnalogInAcquisitionModeSet(h, c_int(D.ACQMODE_SINGLE))
        dwf.FDwfAnalogInFrequencySet(h, c_double(rate))
        dwf.FDwfAnalogInBufferSizeSet(h, c_int(buffer_size))

        # Trigger on the AWG start so the pulse always lands at the same place.
        trigsrc = D.TRIGSRC_ANALOG_OUT1 + pulse.channel
        dwf.FDwfAnalogInTriggerSourceSet(h, c_byte(trigsrc))
        dwf.FDwfAnalogInTriggerAutoTimeoutSet(h, c_double(0.0))

        record = buffer_size / rate
        position = (0.5 - scope.pretrigger_fraction) * record
        dwf.FDwfAnalogInTriggerPositionSet(h, c_double(position))
        self._scope = (rate, buffer_size, position, record)
        return rate, buffer_size

    def configure_scope_edge_trigger(
        self, channel: int = 0, level: float = 0.1, slope: int = D.SLOPE_RISE
    ) -> None:
        """Optional: trigger on an analog edge instead of the AWG start."""
        h = self._hdwf
        dwf.FDwfAnalogInTriggerSourceSet(h, c_byte(D.TRIGSRC_DETECTOR_ANALOG_IN))
        dwf.FDwfAnalogInTriggerTypeSet(h, c_int(D.TRIGTYPE_EDGE))
        dwf.FDwfAnalogInTriggerChannelSet(h, c_int(channel))
        dwf.FDwfAnalogInTriggerLevelSet(h, c_double(level))
        dwf.FDwfAnalogInTriggerConditionSet(h, c_int(slope))

    # -- acquisition -------------------------------------------------------
    def capture(self, pulse: PulseSpec, scope: ScopeSpec) -> Capture:
        h = self._hdwf
        rate, buffer_size, position, record = self._scope

        self._throttle(scope.min_interval)

        # Make sure the previous run has fully stopped, otherwise its trailing
        # trigger can fire the newly armed acquisition before the pulse starts.
        self._wait_awg_idle(pulse.channel, scope.timeout)

        # Arm the scope first, then fire the generator.
        dwf.FDwfAnalogInConfigure(h, c_int(1), c_int(1))
        self._wait_for_state(D.STATE_ARMED, scope.timeout)
        dwf.FDwfAnalogOutConfigure(h, c_int(pulse.channel), c_int(1))

        sts = c_byte()
        deadline = time.monotonic() + scope.timeout + pulse.total_time
        while True:
            dwf.FDwfAnalogInStatus(h, c_int(1), byref(sts))
            if sts.value == D.STATE_DONE:
                break
            if time.monotonic() > deadline:
                self.stop_output(pulse.channel)
                raise TimeoutError(
                    "Acquisition did not complete; check the trigger configuration."
                )
            time.sleep(0.001)

        ch1 = (c_double * buffer_size)()
        ch2 = (c_double * buffer_size)()
        dwf.FDwfAnalogInStatusData(h, c_int(0), ch1, c_int(buffer_size))
        dwf.FDwfAnalogInStatusData(h, c_int(1), ch2, c_int(buffer_size))
        self.stop_output(pulse.channel)

        t0 = position - record / 2.0  # trigger position is the mid-buffer time
        t = t0 + np.arange(buffer_size) / rate
        return Capture(
            t=t,
            ch1=np.frombuffer(ch1, dtype=np.float64).copy(),
            ch2=np.frombuffer(ch2, dtype=np.float64).copy(),
            sample_rate=rate,
            pulse=pulse,
            meta={
                "buffer_size": buffer_size,
                "trigger_position_s": position,
                "record_s": record,
                "awg_timing_resolution_s": pulse.timing_resolution,
                **getattr(self, "_applied", {}),
            },
        )

    def capture_auto(self, pulse: PulseSpec, scope: ScopeSpec) -> Capture:
        """Capture, re-ranging the inputs until neither channel clips.

        Ch1 is ranged from the known drive amplitude; Ch2 is ranged from what
        the previous acquisition actually measured.
        """
        ranges = list(scope.ranges)
        offsets = list(scope.offsets)

        if scope.auto_range:
            lo = min(pulse.baseline, pulse.amplitude)
            hi = max(pulse.baseline, pulse.amplitude)
            ranges[0], offsets[0] = self._fit(lo, hi, scope, channel=0)

        cap = None
        for _ in range(max(1, scope.range_retries + 1)):
            spec = replace(scope, ranges=(ranges[0], ranges[1]),
                           offsets=(offsets[0], offsets[1]))
            self.configure_scope(pulse, spec)
            cap = self.capture(pulse, spec)
            if not scope.auto_range:
                return cap

            changed = False
            for i, y in ((0, cap.ch1), (1, cap.ch2)):
                r = cap.meta[f"ch{i + 1}_range_V"]
                o = cap.meta[f"ch{i + 1}_offset_V"]
                new_r, new_o = self._rerange(y, r, o, scope, channel=i)
                if abs(new_r - r) > 1e-9 or abs(new_o - o) > 1e-3:
                    ranges[i], offsets[i] = new_r, new_o
                    changed = True
            if not changed:
                return cap
        assert cap is not None
        return cap

    def _fit(self, lo: float, hi: float, scope: ScopeSpec, channel: int):
        span = max((hi - lo) * scope.headroom, 1e-3)
        rng = self.pick_range(span, channel)
        omin, omax = self.offset_limits(channel)
        offset = min(max((hi + lo) / 2.0, omin), omax)
        return rng, offset

    def _rerange(self, y: np.ndarray, rng: float, offset: float,
                 scope: ScopeSpec, channel: int):
        lo, hi = float(np.min(y)), float(np.max(y))
        options = self.available_ranges(channel)
        rail = rng / 2.0
        clipped = (hi >= offset + rail * 0.995) or (lo <= offset - rail * 0.995)
        if clipped:
            bigger = [v for v in options if v > rng]
            if not bigger:
                return rng, offset
            omin, omax = self.offset_limits(channel)
            return bigger[0], min(max(offset, omin), omax)
        if (hi - lo) < scope.shrink_threshold * rng:
            return self._fit(lo, hi, scope, channel)
        return rng, offset

    def _wait_awg_idle(self, channel: int, timeout: float) -> None:
        self.stop_output(channel)
        sts = c_byte()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            dwf.FDwfAnalogOutStatus(self._hdwf, c_int(channel), byref(sts))
            if sts.value in (D.STATE_READY, D.STATE_DONE):
                return
            time.sleep(0.001)

    def _throttle(self, min_interval: float) -> None:
        """Hold off until `min_interval` has elapsed since the last shot started."""
        if min_interval > 0:
            remaining = self._next_capture_at - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
        self._next_capture_at = time.monotonic() + max(min_interval, 0.0)

    def _wait_for_state(self, state: int, timeout: float) -> None:
        sts = c_byte()
        deadline = time.monotonic() + timeout
        while True:
            dwf.FDwfAnalogInStatus(self._hdwf, c_int(1), byref(sts))
            if sts.value == state:
                return
            if time.monotonic() > deadline:
                raise TimeoutError(f"Scope never reached state {state}")
            time.sleep(0.001)
