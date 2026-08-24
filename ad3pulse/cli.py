"""Command line interface: python -m ad3pulse ..."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .dwfapi import DwfError, dwf
from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec
from .sweep import run_sweep, summarize
from .repeat import capture_repeats


def parse_values(text: str) -> list[float]:
    """'1,2,5' | 'lin:0.1:5:10' | 'log:1e-6:1e-3:7' -> list of floats."""
    text = text.strip()
    if text.startswith(("lin:", "log:")):
        kind, start, stop, count = text.split(":")
        start, stop, count = float(start), float(stop), int(count)
        if kind == "lin":
            return np.linspace(start, stop, count).tolist()
        return np.logspace(np.log10(start), np.log10(stop), count).tolist()
    return [float(v) for v in text.replace(";", ",").split(",") if v.strip()]


def _plot(caps: list[Capture], path: Path | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.", file=sys.stderr)
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    for cap in caps:
        label = f"A={cap.pulse.amplitude:g} V, W={cap.pulse.width:g} s"
        axes[0].plot(cap.t * 1e6, cap.ch1, lw=1, label=label)
        axes[1].plot(cap.t * 1e6, cap.ch2, lw=1, label=label)
    axes[0].set_ylabel("Ch1 (V)")
    axes[1].set_ylabel("Ch2 (V)")
    axes[1].set_xlabel("time (us, 0 = pulse trigger)")
    for ax in axes:
        ax.grid(alpha=0.3)
    if len(caps) <= 12:
        axes[0].legend(fontsize=7)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        print(f"Plot saved to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _add_pulse_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--pre-delay", type=float, default=0.0,
                   help="baseline time before the pulse, seconds")
    p.add_argument("--post-delay", type=float, default=None,
                   help="baseline time after the pulse, seconds (default 4x width)")
    p.add_argument("--baseline", type=float, default=0.0,
                   help="output level between pulses, volts")
    p.add_argument("--repeat", type=int, default=1, help="number of pulses to emit")
    p.add_argument("--awg-channel", type=int, default=0, choices=(0, 1),
                   help="0 = W1, 1 = W2")
    p.add_argument("--awg-samples", type=int, default=4096,
                   help="points used to render one pulse period")


def _add_scope_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--sample-rate", type=float, default=None,
                   help="scope sample rate in Hz (default: fit the pulse window)")
    p.add_argument("--buffer", type=int, default=None,
                   help="samples per channel (default: device maximum)")
    p.add_argument("--range1", type=float, default=5.0, help="Ch1 range, volts")
    p.add_argument("--range2", type=float, default=5.0, help="Ch2 range, volts")
    p.add_argument("--offset1", type=float, default=0.0, help="Ch1 offset, volts")
    p.add_argument("--offset2", type=float, default=0.0, help="Ch2 offset, volts")
    p.add_argument("--pretrigger", type=float, default=0.1,
                   help="fraction of the record before the trigger")
    p.add_argument("--window-margin", type=float, default=1.2,
                   help="capture window as a multiple of the pulse duration")
    p.add_argument("--averages", type=int, default=1,
                   help="acquisitions averaged per setting")
    p.add_argument("--auto-range", action="store_true",
                   help="pick the input range/offset per step to avoid clipping")
    p.add_argument("--headroom", type=float, default=1.3,
                   help="span margin used when auto-ranging")
    p.add_argument("--min-interval", type=float, default=0.5,
                   help="minimum seconds between shots (0.5 = ~2 acquisitions/s)")
    p.add_argument("--device", type=int, default=-1, help="device enumeration index")
    p.add_argument("--config", type=int, default=None,
                   help="device configuration index (1 = larger scope buffer)")


def _scope_from_args(a: argparse.Namespace) -> ScopeSpec:
    return ScopeSpec(
        sample_rate=a.sample_rate,
        buffer_size=a.buffer,
        ranges=(a.range1, a.range2),
        offsets=(a.offset1, a.offset2),
        pretrigger_fraction=a.pretrigger,
        window_margin=a.window_margin,
        auto_range=a.auto_range,
        headroom=a.headroom,
        min_interval=a.min_interval,
    )


def _pulse_kwargs(a: argparse.Namespace) -> dict:
    return {
        "pre_delay": a.pre_delay,
        "post_delay": a.post_delay,
        "baseline": a.baseline,
        "repeat": a.repeat,
        "channel": a.awg_channel,
        "samples": a.awg_samples,
    }


def cmd_list(_: argparse.Namespace) -> int:
    print(f"WaveForms runtime version {dwf.version()}")
    devices = dwf.enumerate_devices()
    if not devices:
        print("No Digilent devices found.")
        return 1
    for d in devices:
        state = "in use" if d["in_use"] else "free"
        print(f"  [{d['index']}] {d['name']}  SN:{d['serial']}  ({state})")
    return 0


def cmd_single(a: argparse.Namespace) -> int:
    pulse = PulseSpec(amplitude=a.amplitude, width=a.width, **_pulse_kwargs(a))
    scope = _scope_from_args(a)
    caps = []
    with AnalogDiscovery(a.device, a.config) as dev:
        dev.configure_pulse(pulse)
        rate, n = dev.configure_scope(pulse, scope)
        print(f"Scope: {n} samples @ {rate:,.0f} S/s "
              f"({n / rate * 1e6:.2f} us window); "
              f"AWG step {pulse.timing_resolution * 1e9:.1f} ns")
        stack1, stack2, last = [], [], None
        for _ in range(max(1, a.averages)):
            last = dev.capture_auto(pulse, scope)
            stack1.append(last.ch1)
            stack2.append(last.ch2)
    assert last is not None
    cap = Capture(last.t, np.mean(stack1, 0), np.mean(stack2, 0),
                  last.sample_rate, pulse, {**last.meta, "averages": a.averages})
    caps.append(cap)

    row = summarize(cap)
    for k, v in row.items():
        print(f"  {k:22s} {v}")

    out = Path(a.out) if a.out else Path(
        f"capture_{datetime.now():%Y%m%d_%H%M%S}.csv")
    cap.to_csv(out)
    print(f"Waveform saved to {out}")
    if a.plot or a.plot_file:
        _plot(caps, Path(a.plot_file) if a.plot_file else None, a.plot)
    return 0


def cmd_sweep(a: argparse.Namespace) -> int:
    amplitudes = parse_values(a.amplitudes)
    widths = parse_values(a.widths)
    outdir = Path(a.out or f"sweep_{datetime.now():%Y%m%d_%H%M%S}")
    print(f"{len(amplitudes)} amplitudes x {len(widths)} widths -> {outdir}")
    run_sweep(
        amplitudes=amplitudes,
        widths=widths,
        outdir=outdir,
        scope=_scope_from_args(a),
        averages=a.averages,
        save_waveforms=not a.no_waveforms,
        device_index=a.device,
        config_index=a.config,
        live=a.live,
        **_pulse_kwargs(a),
    )
    print(f"Done. Summary: {outdir / 'summary.csv'}")
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    from .gui import launch

    return launch(a.device, a.config)


def cmd_repeat(a: argparse.Namespace) -> int:
    outdir = Path(a.out or f"repeat_{datetime.now():%Y%m%d_%H%M%S}")
    print(f"{a.count} shots @ {a.amplitude:g} V, {a.width:g} s -> {outdir}")
    rows = capture_repeats(
        amplitude=a.amplitude,
        width=a.width,
        count=a.count,
        outdir=outdir,
        scope=_scope_from_args(a),
        device_index=a.device,
        config_index=a.config,
        lock_range=not a.no_lock_range,
        live=not a.no_live,
        **_pulse_kwargs(a),
    )
    print(f"Done. {len(rows)} shots -> {outdir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ad3pulse",
        description="Square-pulse generation and 2-channel capture on an "
                    "Analog Discovery 3.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list connected Digilent devices").set_defaults(
        func=cmd_list)

    s = sub.add_parser("single", help="emit one pulse setting and capture it")
    s.add_argument("--amplitude", type=float, required=True, help="pulse height, volts")
    s.add_argument("--width", type=float, required=True, help="pulse width, seconds")
    s.add_argument("--out", help="CSV output path")
    s.add_argument("--plot", action="store_true", help="show a plot window")
    s.add_argument("--plot-file", help="save the plot to this image path")
    _add_pulse_args(s)
    _add_scope_args(s)
    s.set_defaults(func=cmd_single)

    w = sub.add_parser("sweep", help="sweep pulse heights and widths")
    w.add_argument("--amplitudes", required=True,
                   help="'0.5,1,2' or 'lin:0.1:5:10' or 'log:0.01:5:20' (volts)")
    w.add_argument("--widths", required=True,
                   help="'1e-6,1e-5' or 'log:1e-7:1e-3:9' (seconds)")
    w.add_argument("--out", help="output directory")
    w.add_argument("--no-waveforms", action="store_true",
                   help="store only summary.csv, not the raw traces")
    w.add_argument("--live", action="store_true",
                   help="overlay Ch2 traces in a live figure while sweeping")
    _add_pulse_args(w)
    _add_scope_args(w)
    w.set_defaults(func=cmd_sweep)

    g = sub.add_parser("gui", help="interactive window with live Ch1/Ch2 plots")
    g.add_argument("--device", type=int, default=-1, help="device enumeration index")
    g.add_argument("--config", type=int, default=None,
                   help="device configuration index (1 = larger scope buffer)")
    g.set_defaults(func=cmd_gui)

    c = sub.add_parser(
        "repeat",
        help="fire one fixed pulse setting N times and save each raw trace")
    c.add_argument("--amplitude", type=float, required=True, help="pulse height, volts")
    c.add_argument("--width", type=float, required=True, help="pulse width, seconds")
    c.add_argument("--count", type=int, default=100, help="number of shots to capture")
    c.add_argument("--out", help="output directory")
    c.add_argument("--no-lock-range", action="store_true",
                   help="re-run auto-ranging every shot instead of locking it "
                        "after the first (breaks scale comparability across shots)")
    c.add_argument("--no-live", action="store_true",
                   help="disable the live Ch1/Ch2 overlay plot while capturing")
    _add_pulse_args(c)
    _add_scope_args(c)
    c.set_defaults(func=cmd_repeat)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DwfError as exc:
        print(f"Device error: {exc}", file=sys.stderr)
        return 2
    except TimeoutError as exc:
        print(f"Timeout: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
