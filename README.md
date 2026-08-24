# AD3 Pulse Lab

Square-pulse generation and simultaneous 2-channel capture with a Digilent
Analog Discovery 3, built on the WaveForms SDK (`dwf.dll`).

## Requirements

- Digilent WaveForms installed (provides the `dwf` runtime). Detected: v3.25.1.
- Python 3.10+ with `numpy` and `matplotlib` (`pip install -r requirements.txt`).
- **Close the WaveForms GUI before running** — it holds an exclusive lock on the device.

## Wiring

| Signal | Pin |
| --- | --- |
| Pulse out | `W1` (AWG1), return to `GND` |
| Scope Ch1 | `1+` / `1-` — monitor the drive pulse |
| Scope Ch2 | `2+` / `2-` — PMT / detector output |

The scope is triggered by the AWG start, so `t = 0` in the saved data is the
beginning of the generated waveform; the pulse itself starts at `pre_delay`.

## Usage

```powershell
# Check the device is visible and free
python -m ad3pulse list

# Interactive window: edit settings, single shot or free-run, live Ch1/Ch2 plots
python -m ad3pulse gui

# One pulse: 1 V high, 10 us wide, capture both channels, plot and save CSV
python -m ad3pulse single --amplitude 1.0 --width 10e-6 --plot --out shot.csv

# Sweep heights and widths (saves one CSV per point plus summary.csv)
python -m ad3pulse sweep --amplitudes lin:0.1:5:10 --widths log:1e-7:1e-4:7 `
    --averages 8 --out runs\pmt_sat_01

# Sweep amplitude only, fixed width, with auto-ranging and a live Ch2 overlay
python -m ad3pulse sweep --amplitudes lin:1.3:2.5:121 --widths 10e-6 `
    --pre-delay 5e-6 --post-delay 25e-6 --auto-range --averages 4 `
    --min-interval 0.5 --live --out runs\w10us_amp1p3-2p5

# Repeat one fixed pulse setting N times (e.g. single-photoelectron
# calibration): saves every raw trace, no analysis. Prefer explicit
# --range1/--range2 over --auto-range here so the scale can't silently
# escalate mid-run (see "Notes on scope ranging" below).
python -m ad3pulse repeat --amplitude 1.59 --width 10e-6 `
    --pre-delay 5e-6 --post-delay 25e-6 --range1 5 --range2 5 `
    --count 100 --min-interval 0.3 --out runs\repeat_1p59V_100ct
```

`--amplitudes` / `--widths` accept:

- explicit lists: `0.1,0.5,1,2,5`
- linear ranges: `lin:<start>:<stop>:<count>`
- log ranges: `log:<start>:<stop>:<count>`

Useful options: `--repeat` (pulse train), `--pre-delay`, `--post-delay`,
`--baseline`, `--range1/--range2` (scope V range), `--sample-rate`, `--buffer`,
`--averages`, `--pretrigger`, `--config 1` (larger scope buffer on some devices).

`repeat`-specific options: `--count` (shots to capture), `--no-lock-range`
(re-run auto-ranging every shot instead of freezing it after the first —
only relevant with `--auto-range`), `--no-live` (disable the live Ch1/Ch2
overlay shown by default while capturing).

## Outputs

- `wave_A<amp>V_W<width>s.csv` (sweep) / `shot_<NNNN>.csv` (repeat) —
  `time_s, ch1_V, ch2_V` per capture.
- `summary.csv` — sweep: baseline, mean-in-pulse, peak, min/max, integrated
  area and pre-trigger RMS noise for both channels at every setting. repeat:
  shot index, filename, and applied Ch1/Ch2 range & offset per shot (no
  analysis — do that downstream, e.g. in the notebook).
- `run_metadata.json` — full sweep/repeat and scope configuration.
- `overlay.png` (repeat) / `overlay_ch2.png` (sweep, with `--live`) — snapshot
  of the live overlay plot at the end of the run.

## GUI

`python -m ad3pulse gui` opens a window with pulse and scope settings on the
left and stacked Ch1/Ch2 plots on the right. `Connect` opens the device (the
handle lives on a worker thread, so the UI stays responsive), `Single shot`
fires one acquisition, `Run continuously` free-runs, and `Save trace as CSV...`
writes the currently displayed trace. The readout panel shows baseline, mean
in-pulse, peak, integrated area and pre-trigger RMS noise after every shot.

## Library use

```python
from ad3pulse import AnalogDiscovery, PulseSpec, ScopeSpec, summarize

pulse = PulseSpec(amplitude=2.0, width=5e-6, pre_delay=1e-6, repeat=1)
scope = ScopeSpec(ranges=(5.0, 1.0), pretrigger_fraction=0.1)

with AnalogDiscovery() as dev:
    dev.configure_pulse(pulse)
    dev.configure_scope(pulse, scope)
    cap = dev.capture(pulse, scope)

print(summarize(cap))
cap.to_csv("shot.csv")
```

## Notes on timing

The pulse is rendered as a custom AWG waveform of `--awg-samples` points
(default 4096) spanning one period, so the edge placement resolution is
`period / samples`. Narrow pulses on long periods lose resolution — shrink
`--post-delay` for sub-microsecond pulses. The AD3 AWG bandwidth (~9 MHz on the
single-ended outputs) limits realistic rise times to roughly 50 ns.

## Notes on scope ranging

`--auto-range` picks the smallest input range that fits the signal, but this
particular AD3 only exposes two selectable ranges (~5.19 V and ~59.4 V, no
finer steps). Any single transient near the rail — an EMI pickup at the pulse
edge, stray light, etc. — flips it straight to the coarse 59.4 V range
(~3.6 mV/count), and `repeat`/`sweep` then lock or reuse that scale, silently
burying small signals (e.g. single-photoelectron pulses, a few mV) below the
quantization step. For small fixed-amplitude signals, prefer setting
`--range1`/`--range2` explicitly (e.g. `--range2 5`, which the hardware snaps
to ~5.19 V) instead of `--auto-range`.
