"""Command line interface: python -m ad3pulse ..."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .dwfapi import DwfError, dwf
from .hardware import InstrumentError, IsegHV, RigolScope, RigolScopeCapture
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


def _plot_rigol(cap: RigolScopeCapture, title: str, path: Path | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plot.", file=sys.stderr)
        return
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    axes[0].plot(cap.time_s * 1e9, cap.ch1_V, lw=1)
    axes[1].plot(cap.time_s * 1e9, cap.ch2_V, lw=1, color="C1")
    axes[0].set_ylabel("Ch1 (V)")
    axes[1].set_ylabel("Ch2 (V)")
    axes[1].set_xlabel("time (ns, 0 = Rigol trigger)")
    axes[0].set_title(title, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
        print(f"Plot saved to {path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def cmd_rigol_single(a: argparse.Namespace) -> int:
    post = a.post_delay if a.post_delay is not None else max(1e-6, 4 * a.width)
    pulse = PulseSpec(amplitude=a.amplitude, width=a.width,
                      **{**_pulse_kwargs(a), "post_delay": post})
    realised = pulse.fit_to_awg()
    print(f"AD3 pulse: {a.amplitude:g} V, requested {a.width * 1e9:.1f} ns, "
          f"realised {realised * 1e9:.1f} ns "
          f"({pulse.samples} AWG samples, {pulse.timing_resolution * 1e9:.1f} ns step)")
    if realised < 3 * pulse.timing_resolution:
        print("  warning: pulse is only a few AWG steps wide; its shape is "
              "limited by the AD3 output bandwidth.", file=sys.stderr)

    trig_level = a.trig_level if a.trig_level is not None else a.amplitude / 2.0
    time_offset = a.time_offset if a.time_offset is not None else 3.0 * a.timebase
    with RigolScope(a.resource) as scope:
        print(f"Rigol: {scope.idn} ({scope.resource_name})")
        scope.configure_channel(1, a.scale1, a.offset1, impedance=a.imp1)
        scope.configure_channel(2, a.scale2, a.offset2, impedance=a.imp2)
        scope.configure_timebase(a.timebase, time_offset, a.mdepth)
        scope.configure_edge_trigger(a.trig_source, trig_level, a.trig_slope)
        scope.arm_single(a.timeout)

        with AnalogDiscovery(a.device, a.config) as dev:
            dev.configure_pulse(pulse)
            dev.fire(pulse, a.timeout)

        try:
            scope.wait_stopped(a.timeout)
        except TimeoutError:
            scope._write(":STOP")
            print(f"Rigol did not trigger on Ch{a.trig_source} at {trig_level:g} V "
                  f"({a.trig_slope}); check --trig-level/--trig-slope/--scale"
                  f"{a.trig_source}.", file=sys.stderr)
            return 3
        cap = scope.capture_channels()

    cap.meta = {
        "amplitude_V": a.amplitude, "width_s": a.width, "realised_width_s": realised,
        "pre_delay_s": pulse.pre_delay, "post_delay_s": pulse.post_delay,
        "ch1_scale_V_div": a.scale1, "ch1_offset_V": a.offset1, "ch1_imp": a.imp1,
        "ch2_scale_V_div": a.scale2, "ch2_offset_V": a.offset2, "ch2_imp": a.imp2,
        "timebase_s_div": a.timebase, "trig_source": a.trig_source,
        "trig_level_V": trig_level, "trig_slope": a.trig_slope, **cap.meta,
    }
    print(f"Captured {len(cap.time_s)} points @ {cap.sample_rate_Hz:,.0f} S/s; "
          f"Ch1 {cap.ch1_V.min():.4g}..{cap.ch1_V.max():.4g} V, "
          f"Ch2 {cap.ch2_V.min():.4g}..{cap.ch2_V.max():.4g} V")

    out = Path(a.out) if a.out else Path(f"rigol_{datetime.now():%Y%m%d_%H%M%S}.csv")
    cap.to_csv(out)
    print(f"Waveform saved to {out}")
    if a.plot or a.plot_file:
        title = f"AD3 {a.amplitude:g} V, {realised * 1e9:.0f} ns pulse — {out.name}"
        _plot_rigol(cap, title, Path(a.plot_file) if a.plot_file else None, a.plot)
    return 0


def _print_hv_status(hv: IsegHV) -> None:
    status = hv.read_status()
    print(f"  polarity      {hv.read_polarity()}")
    print(f"  set voltage   {hv.read_set_voltage():.2f} V")
    print(f"  measured      {hv.read_voltage():.2f} V, {hv.read_current() * 1e6:.3f} uA")
    print(f"  current limit {hv.read_current_limit():.4g} A")
    print(f"  status        {status} ({', '.join(hv.describe_status(status)) or 'off'})")


def cmd_hv(a: argparse.Namespace) -> int:
    with IsegHV(a.resource, channel=a.channel) as hv:
        print(f"ISEG: {hv.idn} ({hv.resource_name}), channel {a.channel}")
        if a.action == "set":
            if a.current is not None:
                hv.set_current_limit(a.current)
            if a.ramp is not None:
                hv.set_ramp(a.ramp)
            hv.set_voltage(a.volts)
            hv.output_on()
            if not a.no_wait:
                print(f"Ramping to {a.volts:g} V...")
                measured = hv.wait_stable(
                    a.volts, a.tolerance, a.timeout,
                    on_poll=lambda v, _: print(f"\r  {v:9.2f} V", end="", flush=True))
                print(f"\rStable at {measured:.2f} V; HV stays on.")
        elif a.action == "off":
            hv.output_off()
            print("HV switched off (the SHR ramps down at its configured speed).")
        _print_hv_status(hv)
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

    r = sub.add_parser(
        "rigol-single",
        help="fire one AD3 pulse and capture Ch1/Ch2 on the Rigol MHO954")
    r.add_argument("--amplitude", type=float, required=True, help="pulse height, volts")
    r.add_argument("--width", type=float, default=100e-9, help="pulse width, seconds")
    r.add_argument("--out", help="CSV output path")
    r.add_argument("--plot", action="store_true", help="show a plot window")
    r.add_argument("--plot-file", help="save the plot to this image path")
    _add_pulse_args(r)
    r.set_defaults(pre_delay=100e-9)
    r.add_argument("--resource",
                   help="Rigol IP or VISA resource (default 192.168.0.101)")
    r.add_argument("--scale1", type=float, default=0.2, help="Ch1 V/div")
    r.add_argument("--scale2", type=float, default=0.2, help="Ch2 V/div")
    r.add_argument("--offset1", type=float, default=0.0, help="Ch1 offset, volts")
    r.add_argument("--offset2", type=float, default=0.0, help="Ch2 offset, volts")
    r.add_argument("--imp1", default="OMEG", choices=("OMEG", "FIFTy"),
                   help="Ch1 input impedance (1 MOhm or 50 Ohm)")
    r.add_argument("--imp2", default="OMEG", choices=("OMEG", "FIFTy"),
                   help="Ch2 input impedance (1 MOhm or 50 Ohm)")
    r.add_argument("--timebase", type=float, default=50e-9, help="Rigol s/div")
    r.add_argument("--time-offset", type=float, default=None,
                   help="Rigol horizontal offset, s (default 3 div: trigger near left)")
    r.add_argument("--mdepth", default=None,
                   help="Rigol memory depth, e.g. 10k (default: leave as set)")
    r.add_argument("--trig-source", type=int, default=2, choices=(1, 2, 3, 4),
                   help="Rigol edge-trigger channel")
    r.add_argument("--trig-level", type=float, default=None,
                   help="trigger level, volts (default amplitude/2)")
    r.add_argument("--trig-slope", default="POS", choices=("POS", "NEG", "RFAL"))
    r.add_argument("--timeout", type=float, default=5.0, help="seconds to wait for trigger")
    r.add_argument("--device", type=int, default=-1, help="AD3 enumeration index")
    r.add_argument("--config", type=int, default=None, help="AD3 configuration index")
    r.set_defaults(func=cmd_rigol_single)

    h = sub.add_parser(
        "hv", help="set, switch off or query the ISEG SHR HV (independent of captures)")
    hv_sub = h.add_subparsers(dest="action", required=True)
    hs = hv_sub.add_parser("set", help="set the HV, switch it on and wait until stable")
    hs.add_argument("volts", type=float,
                    help="HV magnitude, volts (polarity is the SHR's configured one)")
    hs.add_argument("--current", type=float, default=None,
                    help="current set/limit, amps (default: leave as set)")
    hs.add_argument("--ramp", type=float, default=None,
                    help="voltage ramp speed, V/s (default: leave as set)")
    hs.add_argument("--tolerance", type=float, default=1.0,
                    help="|measured - set| tolerance for 'stable', volts")
    hs.add_argument("--timeout", type=float, default=600.0,
                    help="seconds to wait for the ramp to finish")
    hs.add_argument("--no-wait", action="store_true",
                    help="return right after switching on instead of waiting")
    hv_sub.add_parser("off", help="switch the HV off (ramps down)")
    hv_sub.add_parser("status", help="print set/measured values and status flags")
    for sp in hv_sub.choices.values():
        sp.add_argument("--resource",
                        help="ISEG IP or VISA resource (default 192.168.0.100)")
        sp.add_argument("--channel", type=int, default=0, help="ISEG channel number")
    h.set_defaults(func=cmd_hv)
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
    except InstrumentError as exc:
        print(f"Instrument error: {exc}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
