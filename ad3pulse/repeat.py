"""Repeated captures at one fixed pulse setting: no analysis, just raw traces.

Fires the same pulse `count` times and saves every raw Ch1/Ch2 trace as its
own CSV, plus a bookkeeping `summary.csv` (filenames + applied scope ranges).
Any analysis (e.g. finding single-photoelectron pulses) is left to downstream
tools such as the Jupyter notebook.
"""

from __future__ import annotations

import csv
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec


class RepeatOverlay:
    """Interactive figure that accumulates Ch1/Ch2 traces as shots come in."""

    def __init__(self, count: int) -> None:
        import matplotlib.pyplot as plt
        from matplotlib import cm, colors

        self._plt = plt
        self._norm = colors.Normalize(0, max(count - 1, 1))
        self._cmap = plt.get_cmap("plasma")

        plt.ion()
        self._fig, (self._ax1, self._ax2) = plt.subplots(
            2, 1, sharex=True, figsize=(9, 6))
        self._ax1.set_ylabel("Ch1 (V)")
        self._ax2.set_ylabel("Ch2 (V)")
        self._ax2.set_xlabel("time (us, 0 = AWG start)")
        for ax in (self._ax1, self._ax2):
            ax.grid(alpha=0.3)
        self._fig.colorbar(
            cm.ScalarMappable(norm=self._norm, cmap=self._cmap),
            ax=(self._ax1, self._ax2), label="shot index",
        )
        self._fig.tight_layout()
        self._fig.show()
        self._fig.canvas.flush_events()

    def add(self, shot: int, cap: Capture) -> None:
        color = self._cmap(self._norm(shot))
        self._ax1.plot(cap.t * 1e6, cap.ch1, lw=0.7, color=color)
        self._ax2.plot(cap.t * 1e6, cap.ch2, lw=0.7, color=color)
        self._ax1.set_title(f"shot {shot + 1}")
        for ax in (self._ax1, self._ax2):
            ax.relim()
            ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()

    def finish(self, path: Path | None = None) -> None:
        if path:
            self._fig.savefig(path, dpi=150)
        self._plt.ioff()


def capture_repeats(
    amplitude: float,
    width: float,
    count: int,
    outdir: str | Path,
    scope: ScopeSpec | None = None,
    device_index: int = -1,
    config_index: int | None = None,
    lock_range: bool = True,
    progress: bool = True,
    live: bool = True,
    **pulse_kwargs,
) -> list[dict]:
    """Capture `count` identical shots; return one bookkeeping row per shot."""
    scope = scope or ScopeSpec()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pulse = PulseSpec(amplitude=amplitude, width=width, **pulse_kwargs)
    overlay = RepeatOverlay(count) if live else None

    rows: list[dict] = []
    with AnalogDiscovery(device_index, config_index) as dev:
        dev.configure_pulse(pulse)

        if scope.auto_range and lock_range:
            # One auto-ranging shot to settle on a range, then freeze it so
            # every shot lands on the same scale.
            probe = dev.capture_auto(pulse, scope)
            scope = replace(
                scope,
                auto_range=False,
                ranges=(probe.meta["ch1_range_V"], probe.meta["ch2_range_V"]),
                offsets=(probe.meta["ch1_offset_V"], probe.meta["ch2_offset_V"]),
            )
        dev.configure_scope(pulse, scope)

        for i in range(count):
            cap = dev.capture(pulse, scope)
            path = outdir / f"shot_{i:04d}.csv"
            cap.to_csv(path)
            if overlay is not None:
                overlay.add(i, cap)

            row = {
                "shot": i,
                "file": path.name,
                "ch1_range_V": cap.meta.get("ch1_range_V"),
                "ch1_offset_V": cap.meta.get("ch1_offset_V"),
                "ch2_range_V": cap.meta.get("ch2_range_V"),
                "ch2_offset_V": cap.meta.get("ch2_offset_V"),
            }
            rows.append(row)
            if progress:
                print(f"[{i + 1}/{count}] saved {path.name}")

    if overlay is not None:
        overlay.finish(outdir / "overlay.png")
    _write_summary(outdir, rows, pulse, scope)
    return rows


def _write_summary(outdir: Path, rows: list[dict], pulse: PulseSpec,
                    scope: ScopeSpec) -> None:
    if rows:
        keys = list(rows[0])
        with open(outdir / "summary.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)

    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "amplitude_V": pulse.amplitude,
        "width_s": pulse.width,
        "count": len(rows),
        "pulse": {
            "pre_delay": pulse.pre_delay,
            "post_delay": pulse.post_delay,
            "baseline": pulse.baseline,
            "repeat": pulse.repeat,
            "channel": pulse.channel,
        },
        "scope_ranges_V": list(scope.ranges),
        "scope_offsets_V": list(scope.offsets),
    }
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2), "utf-8")
