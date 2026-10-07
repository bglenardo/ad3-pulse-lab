# PMT saturation experiments — ad3pulse

Workspace: c:\Users\blenardo\Desktop\PMT saturatin experiments

## Project intent
- Build a square-pulse generation and PMT measurement workflow.
- Use the Analog Discovery 3 as a pulse generator only.
- Use the Rigol MHO954 for waveform capture and the ISEG SHR 40 60 for HV control.
- Keep calibration and scientific analysis in notebooks or downstream post-processing, not in the acquisition scripts.

## Current hardware roles
- AD3: pulse source only
- Rigol MHO954: Ch1/Ch2 waveform capture
- ISEG SHR 40 60: PMT bias / HV supply
- Notebook analysis: calibration, gain extraction, saturation analysis, single-photoelectron studies

## Environment
- Python 3.14.7 at C:\Users\blenardo\AppData\Local\Python\pythoncore-3.14-64
- Installed packages: numpy 2.5.2, matplotlib 3.11.1, pyvisa, pytest
- WaveForms runtime: 3.25.1; dwf.dll is loaded through the Digilent SDK
- AD3 serial/device previously confirmed with enumeration under the WaveForms runtime
- Important: the WaveForms GUI must be closed before running AD3 code, or the device stays locked

## Repository layout
- ad3pulse/dwfapi.py — ctypes wrapper over the Digilent WaveForms SDK
- ad3pulse/instrument.py — PulseSpec, ScopeSpec, Capture, AnalogDiscovery
- ad3pulse/sweep.py — sweep automation and summary generation
- ad3pulse/repeat.py — fixed-pulse repeated captures for repeated traces
- ad3pulse/cli.py — command-line interface (list, single, sweep, gui, repeat)
- ad3pulse/gui.py — Tkinter UI for interactive pulse generation / capture
- ad3pulse/hardware.py — Rigol + ISEG wrappers for the external measurement chain
- overlay_waveforms.ipynb — waveform overlays and analysis notebook
- README.md — project documentation and current hardware split

## Hard-won debugging notes from the AD3 phase
- The original AD3 implementation was built for both generation and capture.
- The data path was later refocused because AD3 capture was the wrong tool for low-level single-photoelectron measurements.
- Auto-ranging on the AD3 was a real problem: the device could silently jump to a much coarser range and bury small signals in quantization noise.
- For small-signal work, explicit --range1 / --range2 settings were preferred over --auto-range.
- FDwfAnalogOutNodeDataInfo returns 0 until FDwfAnalogOutNodeFunctionSet(..., FUNC_CUSTOM) is called; custom AWG setup must be complete before querying buffer info.
- FDwfAnalogInTriggerPositionSet sets the mid-buffer trigger offset, not a simple beginning-of-record offset; earlier sign mistakes shifted the time axis.
- Sample-rate clamping behavior required careful buffer/window tuning to avoid mostly-empty records.
- The AWG must be allowed to go idle before re-arming to prevent stale triggers from firing a new capture early.
- Pulse edges are limited by the custom AWG waveform generation; edge resolution is approximately period / samples.

## What was implemented
- A working AD3 pulse-generation + capture codebase was created and validated.
- The CLI supports list/single/sweep/repeat operations.
- Sweep functions can save individual waveforms plus summary files.
- There is a live overlay workflow for repeated traces.
- A Rigol MHO954 wrapper was added to represent the new measurement path.
- An ISEG HV wrapper was added to control the power supply over USB via VISA.
- A basic pytest suite was added for the external instrument wrappers.

## Network setup (2026-09-29)
- Both instruments are on Ethernet now (no USB): Rigol MHO954 at 192.168.0.101, ISEG SHR at 192.168.0.100. These are the defaults; --resource accepts another IP or a full VISA name.
- Rigol: LXI/VXI-11, TCPIP0::192.168.0.101::INSTR (the scope's default VISA type per :LAN:VISA?).
- ISEG: raw TCP socket, TCPIP0::192.168.0.100::10001::SOCKET (port 10001 is fixed).

## ISEG SHR control (2026-09-29)
- Reference: Documentation/iseg_manual_iseg_SCPI_general_instruction_set.pdf (v3.3). The NHQ RS-232 guide there is for a different product line and does not apply.
- Ethernet has no echo. The SHR sends an empty line after order commands, so IsegHV sends orders as "cmd;*OPC?" and skips empty lines. Rejected commands get no answer (timeout -> InstrumentError).
- Replies carry units and may omit exponent digits ("1.23456EA"): use parse_iseg_value().
- HV is controlled separately from captures (ramps are slow): `python -m ad3pulse hv set 1200 [--current 1e-4] [--ramp 20]`, `hv status`, `hv off`. Closing the connection leaves the HV as it is.
- Verified live 2026-09-29 (read-only `hv status`): SHR SR040060R4050000200, iCS 2.15.6 / N04C2 01.91, channel 0 polarity n; VSET reads back signed (-1550 V). set_voltage() therefore takes a magnitude and signs it from :CONF:OUTP:POL?.
- SHR has 4 channels, 6 kV nominal each; ch0/ch1 are negative (the two PMTs), ch2/ch3 positive. Ramp 25 V/s.
- Order commands verified live 2026-09-29 via pmt_bias_test.ipynb: ch0/ch1 to -100 V (stable at -100.1 V, ~1-1.5 uA readback), then off to ~0 V.

## Rigol over Ethernet: findings (2026-09-29)
- VXI-11 refuses a new link for a few seconds after the previous one closes; RigolScope.connect() retries (5 x 3 s).
- :SINGle takes ~1.4 s to take effect; a STOP read right after it is the previous acquisition. arm_single() now waits for WAIT/TD. Reading a stale STOP produced a 0-point capture once; capture_channels() now raises on 0 points.
- WORD byte order confirmed little-endian (WORD_BIG_ENDIAN = False): codes sit around 32768 midscale.
- :TFORce forces the armed single acquisition (RigolScope.force_trigger()).

## pmt_bias_test.ipynb
- Parameters cell: HV_VOLTAGES {channel: magnitude}, scope CHANNELS/timebase/trigger, TRIGGER_MODE "force" | "wait" | "ad3".
- Run cell biases up, acquires once, and biases down in a finally: block. Emergency cell switches off all 4 channels.
- No Jupyter kernel is installed in .venv; the notebook was exercised by exec'ing its cells (same code).

## Validation status
- Verified command: cd "c:\Users\blenardo\Desktop\PMT saturatin experiments"; .venv\Scripts\python -m pytest -q
- Result (2026-09-29): 22 passed

## Notes on acquisition philosophy
- The acquisition scripts should be simple and do only a few things:
  1. set pulse parameters,
  2. trigger the pulse,
  3. capture raw traces,
  4. save CSV data.
- Calibration and scientific interpretation are intentionally not embedded in the acquisition layer.
- The notebook or analysis code should handle:
  - baseline subtraction,
  - gain calibration,
  - response fitting,
  - single-photoelectron extraction,
  - histogram / area analysis,
  - saturation studies.

## Typical commands used during development
```powershell
python -m ad3pulse list
python -m ad3pulse gui
python -m ad3pulse single --amplitude 1.0 --width 10e-6 --plot --out shot.csv
python -m ad3pulse sweep --amplitudes lin:1.3:2.5:121 --widths 10e-6 --pre-delay 5e-6 --post-delay 25e-6 --auto-range --averages 4 --min-interval 0.5 --live --out runs\<name>
python -m ad3pulse repeat --amplitude 1.59 --width 10e-6 --pre-delay 5e-6 --post-delay 25e-6 --range1 5 --range2 5 --count 100 --min-interval 0.3 --out runs\repeat_1p59V_100ct
```

## Current working assumption for future sessions
- Do not reintroduce AD3 capture as the main PMT measurement path.
- Keep the AD3 as a pulse source and the Rigol scope as the measurement instrument.
- Keep HV control separated into the ISEG layer.
- Use notebooks for actual calibration / PMT response analysis.

## Immediate next useful task
- Try `hv status` and a low-voltage `hv set` on the real SHR to confirm the protocol notes above.
- Capture is `rigol-single` (AD3 pulse -> Rigol Ch1/Ch2 -> raw CSV); HV is set beforehand with `hv set`.
- Then do the calibration and photoelectron analysis in notebook code.
