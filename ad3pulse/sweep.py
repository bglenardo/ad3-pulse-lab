"""Amplitude / width sweeps with per-run CSV export and summary metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec


def _fmt(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def summarize(cap: Capture) -> dict:
    """Baseline / peak / mean-in-pulse metrics for both channels."""
    p = cap.pulse
    t = cap.t
    # Pulse occupies [pre_delay, pre_delay + width) after the trigger.
    in_pulse = (t >= p.pre_delay) & (t < p.pre_delay + p.width)
    pre = t < p.pre_delay
    if in_pulse.sum() < 2:  # very short pulse: fall back to the extremum
        in_pulse = np.zeros_like(t, dtype=bool)
        in_pulse[np.argmax(np.abs(cap.ch1 - np.median(cap.ch1)))] = True
    if pre.sum() < 2:
        pre = t < t[0] + 0.05 * (t[-1] - t[0])

    row: dict = {
        "amplitude_set_V": p.amplitude,
        "width_set_s": p.width,
        "repeat": p.repeat,
        "sample_rate_Hz": cap.sample_rate,
        "ch1_range_V": cap.meta.get("ch1_range_V"),
        "ch1_offset_V": cap.meta.get("ch1_offset_V"),
        "ch2_range_V": cap.meta.get("ch2_range_V"),
        "ch2_offset_V": cap.meta.get("ch2_offset_V"),
    }
    dt = 1.0 / cap.sample_rate
    for name, y in (("ch1", cap.ch1), ("ch2", cap.ch2)):
        base = float(np.mean(y[pre]))
        seg = y[in_pulse] - base
        rng = cap.meta.get(f"{name}_range_V")
        off = cap.meta.get(f"{name}_offset_V")
        row[f"{name}_baseline_V"] = base
        row[f"{name}_mean_V"] = float(np.mean(seg))
        row[f"{name}_peak_V"] = float(seg[np.argmax(np.abs(seg))])
        row[f"{name}_min_V"] = float(np.min(y))
        row[f"{name}_max_V"] = float(np.max(y))
        row[f"{name}_area_Vs"] = float(np.sum(y - base) * dt)
        row[f"{name}_rms_noise_V"] = float(np.std(y[pre])) if pre.sum() > 1 else np.nan
        row[f"{name}_clipped"] = (
            bool(row[f"{name}_max_V"] >= off + rng / 2 * 0.995
                 or row[f"{name}_min_V"] <= off - rng / 2 * 0.995)
            if rng and off is not None else ""
        )
    return row


class LiveOverlay:
    """Interactive figure that accumulates Ch2 traces as the sweep progresses."""

    def __init__(self, amplitudes: Sequence[float], channel: str = "ch2") -> None:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        self._plt = plt
        self._channel = channel
        lo, hi = float(min(amplitudes)), float(max(amplitudes))
        self._norm = colors.Normalize(lo, hi if hi > lo else lo + 1e-9)
        self._cmap = plt.get_cmap("viridis")

        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(9, 5))
        self._ax.set_xlabel("time (us, 0 = AWG start)")
        self._ax.set_ylabel(f"{channel.upper()} (V)")
        self._ax.grid(alpha=0.3)
        self._fig.colorbar(
            cm.ScalarMappable(norm=self._norm, cmap=self._cmap),
            ax=self._ax,
            label="pulse height (V)",
        )
        self._fig.tight_layout()
        self._fig.show()
        self._fig.canvas.flush_events()

    def add(self, cap: Capture) -> None:
        y = cap.ch2 if self._channel == "ch2" else cap.ch1
        self._ax.plot(
            cap.t * 1e6, y, lw=0.8,
            color=self._cmap(self._norm(cap.pulse.amplitude)),
        )
        self._ax.set_title(
            f"{self._channel.upper()} overlay - "
            f"last: {cap.pulse.amplitude:g} V, {cap.pulse.width:g} s"
        )
        self._ax.relim()
        self._ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def finish(self, path: Path | None = None) -> None:
        if path:
            self._fig.savefig(path, dpi=150)
        self._plt.ioff()


def run_sweep(
    amplitudes: Sequence[float],
    widths: Sequence[float],
    outdir: str | Path,
    scope: ScopeSpec | None = None,
    averages: int = 1,
    save_waveforms: bool = True,
    device_index: int = -1,
    config_index: int | None = None,
    progress: bool = True,
    live: bool = False,
    **pulse_kwargs,
) -> list[dict]:
    """Sweep pulse height x width, capturing Ch1/Ch2 for every combination."""
    scope = scope or ScopeSpec()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    overlay = LiveOverlay(amplitudes) if live else None

    with AnalogDiscovery(device_index, config_index) as dev:
        for width in widths:
            for amp in amplitudes:
                pulse = PulseSpec(amplitude=amp, width=width, **pulse_kwargs)
                dev.configure_pulse(pulse)

                stack1, stack2, last = [], [], None
                for _ in range(max(1, averages)):
                    last = dev.capture_auto(pulse, scope)
                    stack1.append(last.ch1)
                    stack2.append(last.ch2)

                assert last is not None
                cap = Capture(
                    t=last.t,
                    ch1=np.mean(stack1, axis=0),
                    ch2=np.mean(stack2, axis=0),
                    sample_rate=last.sample_rate,
                    pulse=pulse,
                    meta={**last.meta, "averages": averages},
                )

                tag = f"A{_fmt(amp)}V_W{_fmt(width)}s"
                if save_waveforms:
                    cap.to_csv(outdir / f"wave_{tag}.csv")
                if overlay is not None:
                    overlay.add(cap)

                row = summarize(cap)
                row["file"] = f"wave_{tag}.csv" if save_waveforms else ""
                row["averages"] = averages
                rows.append(row)
                if progress:
                    flag = " CLIPPED" if row.get("ch2_clipped") else ""
                    print(
                        f"[{len(rows)}/{len(widths) * len(amplitudes)}] "
                        f"A={amp:g} V  W={width:g} s  ->  "
                        f"ch1 peak {row['ch1_peak_V']:+.4f} V, "
                        f"ch2 peak {row['ch2_peak_V']:+.4f} V "
                        f"[ch2 range {row['ch2_range_V']:g} V]{flag}"
                    )

    if overlay is not None:
        overlay.finish(outdir / "overlay_ch2.png")
    _write_summary(outdir, rows, amplitudes, widths, scope, averages, pulse_kwargs)
    return rows


def _write_summary(
    outdir: Path,
    rows: list[dict],
    amplitudes: Iterable[float],
    widths: Iterable[float],
    scope: ScopeSpec,
    averages: int,
    pulse_kwargs: dict,
) -> None:
    if rows:
        keys = [k for k in rows[0]]
        with open(outdir / "summary.csv", "w", encoding="utf-8", newline="") as fh:
            fh.write(",".join(keys) + "\n")
            for r in rows:
                fh.write(",".join(f"{r.get(k, '')}" for k in keys) + "\n")

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "amplitudes_V": list(amplitudes),
        "widths_s": list(widths),
        "averages": averages,
        "pulse_options": pulse_kwargs,
        "scope": asdict(scope),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
