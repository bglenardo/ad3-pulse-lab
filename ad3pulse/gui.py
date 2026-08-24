"""Tkinter front end: edit pulse settings, capture, and watch Ch1/Ch2 live."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from .dwfapi import DwfError
from .instrument import AnalogDiscovery, Capture, PulseSpec, ScopeSpec
from .sweep import summarize


class _Worker(threading.Thread):
    """Owns the device handle; all dwf calls happen on this thread."""

    def __init__(self, device_index: int, config_index: int | None) -> None:
        super().__init__(daemon=True)
        self.commands: queue.Queue = queue.Queue()
        self.results: queue.Queue = queue.Queue()
        self._device_index = device_index
        self._config_index = config_index

    def run(self) -> None:
        dev = AnalogDiscovery(self._device_index, self._config_index)
        try:
            dev.open()
        except DwfError as exc:
            self.results.put(("error", str(exc)))
            self.results.put(("closed", None))
            return
        self.results.put(("connected", None))
        try:
            while True:
                kind, payload = self.commands.get()
                if kind == "close":
                    break
                if kind != "capture":
                    continue
                pulse, scope, averages = payload
                try:
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
                    self.results.put(("capture", cap))
                except (DwfError, TimeoutError, ValueError) as exc:
                    self.results.put(("error", str(exc)))
        finally:
            dev.close()
            self.results.put(("closed", None))


class PulseLabApp(tk.Tk):
    FIELDS = [
        ("amplitude", "Pulse height (V)", "1.0"),
        ("width", "Pulse width (s)", "10e-6"),
        ("pre_delay", "Pre-delay (s)", "2e-6"),
        ("post_delay", "Post-delay (s)", "40e-6"),
        ("baseline", "Baseline (V)", "0.0"),
        ("repeat", "Pulses per shot", "1"),
        ("awg_samples", "AWG points/period", "4096"),
    ]
    SCOPE_FIELDS = [
        ("range1", "Ch1 range (V)", "5.0"),
        ("range2", "Ch2 range (V)", "5.0"),
        ("offset1", "Ch1 offset (V)", "0.0"),
        ("offset2", "Ch2 offset (V)", "0.0"),
        ("sample_rate", "Sample rate (Hz, blank=auto)", ""),
        ("buffer", "Buffer (samples, blank=max)", ""),
        ("pretrigger", "Pre-trigger fraction", "0.1"),
        ("window_margin", "Window margin", "1.2"),
        ("averages", "Averages", "1"),
        ("min_interval", "Min shot interval (s)", "0.5"),
    ]

    def __init__(self, device_index: int = -1, config_index: int | None = None) -> None:
        super().__init__()
        self.title("AD3 Pulse Lab")
        self.geometry("1150x720")
        self._device_index = device_index
        self._config_index = config_index
        self._worker: _Worker | None = None
        self._continuous = tk.BooleanVar(value=False)
        self._awg_channel = tk.IntVar(value=0)
        self._autoscale = tk.BooleanVar(value=True)
        self._auto_range = tk.BooleanVar(value=True)
        self._vars: dict[str, tk.StringVar] = {}
        self._last: Capture | None = None
        self._pending = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._poll_results)

    # -- UI ----------------------------------------------------------------
    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)

        side = ttk.Frame(root)
        side.pack(side="left", fill="y", padx=(0, 8))

        conn = ttk.LabelFrame(side, text="Device", padding=6)
        conn.pack(fill="x")
        self._connect_btn = ttk.Button(conn, text="Connect", command=self._toggle_conn)
        self._connect_btn.pack(fill="x")
        self._status = ttk.Label(conn, text="Disconnected", foreground="#a00")
        self._status.pack(fill="x", pady=(4, 0))

        pulse_box = ttk.LabelFrame(side, text="Pulse", padding=6)
        pulse_box.pack(fill="x", pady=(8, 0))
        self._grid_fields(pulse_box, self.FIELDS)
        row = ttk.Frame(pulse_box)
        row.grid(column=0, row=len(self.FIELDS), columnspan=2, sticky="w", pady=(4, 0))
        ttk.Label(row, text="AWG out:").pack(side="left")
        ttk.Radiobutton(row, text="W1", value=0, variable=self._awg_channel).pack(
            side="left")
        ttk.Radiobutton(row, text="W2", value=1, variable=self._awg_channel).pack(
            side="left")

        scope_box = ttk.LabelFrame(side, text="Scope", padding=6)
        scope_box.pack(fill="x", pady=(8, 0))
        self._grid_fields(scope_box, self.SCOPE_FIELDS)
        ttk.Checkbutton(scope_box, text="Auto-range inputs",
                        variable=self._auto_range).grid(
            column=0, row=len(self.SCOPE_FIELDS), columnspan=2, sticky="w",
            pady=(4, 0))

        actions = ttk.Frame(side)
        actions.pack(fill="x", pady=(8, 0))
        self._single_btn = ttk.Button(
            actions, text="Single shot", command=self._single, state="disabled")
        self._single_btn.pack(fill="x")
        self._run_btn = ttk.Checkbutton(
            actions, text="Run continuously", variable=self._continuous,
            command=self._on_continuous, state="disabled")
        self._run_btn.pack(fill="x", pady=(4, 0))
        ttk.Checkbutton(actions, text="Auto-scale Y", variable=self._autoscale,
                        command=self._redraw).pack(fill="x")
        self._save_btn = ttk.Button(
            actions, text="Save trace as CSV...", command=self._save, state="disabled")
        self._save_btn.pack(fill="x", pady=(4, 0))

        self._readout = tk.Text(side, height=13, width=38, font=("Consolas", 8))
        self._readout.pack(fill="both", expand=True, pady=(8, 0))

        fig = Figure(figsize=(8, 6), dpi=100)
        self._ax1 = fig.add_subplot(211)
        self._ax2 = fig.add_subplot(212, sharex=self._ax1)
        self._ax1.set_ylabel("Ch1 (V)")
        self._ax2.set_ylabel("Ch2 (V)")
        self._ax2.set_xlabel("time (µs, 0 = AWG start)")
        for ax in (self._ax1, self._ax2):
            ax.grid(alpha=0.3)
        (self._line1,) = self._ax1.plot([], [], lw=1, color="tab:blue")
        (self._line2,) = self._ax2.plot([], [], lw=1, color="tab:orange")
        fig.tight_layout()

        plot_frame = ttk.Frame(root)
        plot_frame.pack(side="left", fill="both", expand=True)
        self._canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self._canvas, plot_frame).update()

    def _grid_fields(self, parent, fields) -> None:
        for i, (key, label, default) in enumerate(fields):
            ttk.Label(parent, text=label).grid(column=0, row=i, sticky="w", pady=1)
            var = tk.StringVar(value=default)
            self._vars[key] = var
            ttk.Entry(parent, textvariable=var, width=12).grid(
                column=1, row=i, sticky="ew", pady=1)
        parent.columnconfigure(1, weight=1)

    # -- settings ----------------------------------------------------------
    def _num(self, key: str, cast=float, allow_blank: bool = False):
        text = self._vars[key].get().strip()
        if not text:
            if allow_blank:
                return None
            raise ValueError(f"{key} is required")
        try:
            return cast(text)
        except ValueError:
            raise ValueError(f"'{text}' is not a valid value for {key}") from None

    def _specs(self) -> tuple[PulseSpec, ScopeSpec, int]:
        pulse = PulseSpec(
            amplitude=self._num("amplitude"),
            width=self._num("width"),
            pre_delay=self._num("pre_delay"),
            post_delay=self._num("post_delay", allow_blank=True),
            baseline=self._num("baseline"),
            repeat=self._num("repeat", int),
            channel=self._awg_channel.get(),
            samples=self._num("awg_samples", int),
        )
        scope = ScopeSpec(
            sample_rate=self._num("sample_rate", allow_blank=True),
            buffer_size=self._num("buffer", int, allow_blank=True),
            ranges=(self._num("range1"), self._num("range2")),
            offsets=(self._num("offset1"), self._num("offset2")),
            pretrigger_fraction=self._num("pretrigger"),
            window_margin=self._num("window_margin"),
            auto_range=self._auto_range.get(),
            min_interval=self._num("min_interval"),
        )
        return pulse, scope, self._num("averages", int)

    # -- device ------------------------------------------------------------
    def _toggle_conn(self) -> None:
        if self._worker and self._worker.is_alive():
            self._continuous.set(False)
            self._worker.commands.put(("close", None))
            self._connect_btn.config(state="disabled")
        else:
            self._status.config(text="Connecting...", foreground="#a60")
            self._worker = _Worker(self._device_index, self._config_index)
            self._worker.start()

    def _request_capture(self) -> None:
        if not (self._worker and self._worker.is_alive()) or self._pending:
            return
        try:
            payload = self._specs()
        except ValueError as exc:
            messagebox.showerror("Invalid setting", str(exc))
            self._continuous.set(False)
            return
        self._pending = True
        self._worker.commands.put(("capture", payload))

    def _single(self) -> None:
        self._continuous.set(False)
        self._request_capture()

    def _on_continuous(self) -> None:
        if self._continuous.get():
            self._request_capture()

    def _poll_results(self) -> None:
        worker = self._worker
        if worker is not None:
            while True:
                try:
                    kind, payload = worker.results.get_nowait()
                except queue.Empty:
                    break
                if kind == "connected":
                    self._set_connected(True)
                elif kind == "closed":
                    self._set_connected(False)
                    self._worker = None
                elif kind == "error":
                    self._pending = False
                    self._continuous.set(False)
                    messagebox.showerror("Acquisition error", payload)
                elif kind == "capture":
                    self._pending = False
                    self._show(payload)
                    if self._continuous.get():
                        self.after(10, self._request_capture)
        self.after(30, self._poll_results)

    def _set_connected(self, connected: bool) -> None:
        state = "normal" if connected else "disabled"
        self._single_btn.config(state=state)
        self._run_btn.config(state=state)
        self._connect_btn.config(
            state="normal", text="Disconnect" if connected else "Connect")
        self._status.config(
            text="Connected" if connected else "Disconnected",
            foreground="#080" if connected else "#a00")
        if not connected:
            self._continuous.set(False)
            self._pending = False

    # -- display -----------------------------------------------------------
    def _show(self, cap: Capture) -> None:
        self._last = cap
        self._save_btn.config(state="normal")
        self._redraw()
        row = summarize(cap)
        lines = [
            f"{cap.sample_rate:,.0f} S/s, {len(cap.t)} pts, "
            f"{cap.meta.get('averages', 1)} avg",
            f"AWG step {cap.pulse.timing_resolution * 1e9:,.1f} ns",
            f"range ch1 {cap.meta.get('ch1_range_V', 0):g} V @ "
            f"{cap.meta.get('ch1_offset_V', 0):+g} V",
            f"range ch2 {cap.meta.get('ch2_range_V', 0):g} V @ "
            f"{cap.meta.get('ch2_offset_V', 0):+g} V",
            "",
        ]
        for key in (
            "ch1_baseline_V", "ch1_mean_V", "ch1_peak_V", "ch1_rms_noise_V",
            "ch2_baseline_V", "ch2_mean_V", "ch2_peak_V", "ch2_area_Vs",
            "ch2_rms_noise_V",
        ):
            lines.append(f"{key:18s} {row[key]:+.6g}")
        self._readout.delete("1.0", "end")
        self._readout.insert("1.0", "\n".join(lines))

    def _redraw(self) -> None:
        cap = self._last
        if cap is None:
            return
        t_us = cap.t * 1e6
        self._line1.set_data(t_us, cap.ch1)
        self._line2.set_data(t_us, cap.ch2)
        for ax, y in ((self._ax1, cap.ch1), (self._ax2, cap.ch2)):
            ax.set_xlim(t_us[0], t_us[-1])
            if self._autoscale.get():
                lo, hi = float(np.min(y)), float(np.max(y))
                pad = max((hi - lo) * 0.1, 1e-3)
                ax.set_ylim(lo - pad, hi + pad)
        self._canvas.draw_idle()

    def _save(self) -> None:
        if self._last is None:
            return
        p = self._last.pulse
        default = (f"wave_A{p.amplitude:g}V_W{p.width:g}s_"
                   f"{datetime.now():%H%M%S}.csv")
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")])
        if path:
            self._last.to_csv(Path(path))

    def _on_close(self) -> None:
        self._continuous.set(False)
        if self._worker and self._worker.is_alive():
            self._worker.commands.put(("close", None))
            self._worker.join(timeout=3)
        self.destroy()


def launch(device_index: int = -1, config_index: int | None = None) -> int:
    PulseLabApp(device_index, config_index).mainloop()
    return 0
