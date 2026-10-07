from __future__ import annotations

import numpy as np
import pytest

from ad3pulse.hardware import InstrumentError, IsegHV, Preamble, RigolScope, parse_iseg_value
from ad3pulse.instrument import PulseSpec


class FakeInstrument:
    def __init__(self) -> None:
        self.commands: list[str] = []
        self.responses: dict[str, str] = {}
        self.binary: list[np.ndarray] = []

    def write(self, cmd: str) -> None:
        self.commands.append(cmd)

    def query(self, cmd: str) -> str:
        self.commands.append(cmd)
        key = cmd.strip().upper()
        if key in self.responses:
            return self.responses[key]
        return "0.0"

    def query_binary_values(self, cmd: str, **kwargs) -> np.ndarray:
        self.commands.append(cmd)
        return self.binary.pop(0)

    def close(self) -> None:
        self.commands.append("CLOSE")


class FakeIsegSocket:
    """Mimics the SHR TCP socket: an empty line, then the answer.

    Orders ending in ``;*OPC?`` answer ``1``; queries answer from ``responses``
    (a list is consumed one reply per query); anything else gets no answer,
    like a rejected command on the real device.
    """

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.responses: dict[str, str | list[str]] = {}
        self._lines: list[str] = []

    def write(self, cmd: str) -> None:
        self.commands.append(cmd)
        self._lines.append("")
        if cmd.endswith(";*OPC?"):
            self._lines.append("1")
        elif cmd in self.responses:
            reply = self.responses[cmd]
            self._lines.append(reply.pop(0) if isinstance(reply, list) else reply)

    def read(self) -> str:
        if not self._lines:
            raise TimeoutError("VI_ERROR_TMO")
        return self._lines.pop(0)

    def close(self) -> None:
        self.commands.append("CLOSE")


def test_iseg_hv_commands() -> None:
    fake = FakeIsegSocket()
    fake.responses[":READ:CHAN:STAT? (@0)"] = "1"
    fake.responses[":CONF:OUTP:POL? (@0)"] = "p"
    hv = IsegHV(instrument=fake)
    hv.set_current_limit(1e-4)
    hv.set_voltage(1234.5)
    hv.set_ramp(20)
    hv.output_on()

    orders = [c for c in fake.commands if c.endswith(";*OPC?")]
    assert orders == [
        ":CURR 0.0001,(@0);*OPC?",
        ":VOLT 1234.500,(@0);*OPC?",
        ":CONF:RAMP:VOLT 20,(@0);*OPC?",
        ":VOLT ON,(@0);*OPC?",
    ]


def test_iseg_negative_polarity_sends_signed_voltage() -> None:
    # A negative channel on the real SHR reads back VSET as -1550 V.
    fake = FakeIsegSocket()
    fake.responses[":READ:CHAN:STAT? (@0)"] = "0"
    fake.responses[":CONF:OUTP:POL? (@0)"] = "n"
    assert IsegHV(instrument=fake).set_voltage(1550) == -1550
    assert ":VOLT -1550.000,(@0);*OPC?" in fake.commands


@pytest.mark.parametrize("text, value", [
    ("1.23456E3V", 1234.56),
    ("-4.0E3V", -4000.0),
    ("12.3456E-6A", 12.3456e-6),
    ("1.23456EA", 1.23456),
    ("2.00002V", 2.00002),
    ("0.25000E3V/s", 250.0),
])
def test_parse_iseg_value(text: str, value: float) -> None:
    assert parse_iseg_value(text) == pytest.approx(value)


def test_iseg_wait_stable_waits_for_ramp_end() -> None:
    fake = FakeIsegSocket()
    # on + ramping + positive, then on + constant voltage + positive
    fake.responses[":READ:CHAN:STAT? (@1)"] = ["25", "25", "137"]
    fake.responses[":MEAS:VOLT? (@1)"] = ["0.40000E3V", "0.99000E3V", "0.99990E3V"]
    hv = IsegHV(instrument=fake, channel=1)
    assert hv.wait_stable(1000.0, tolerance_v=1.0, poll_s=0) == pytest.approx(999.9)


def test_iseg_wait_stable_raises_on_trip() -> None:
    fake = FakeIsegSocket()
    fake.responses[":READ:CHAN:STAT? (@0)"] = str(IsegHV.IS_CURRENT_TRIP)
    hv = IsegHV(instrument=fake)
    with pytest.raises(InstrumentError, match="current trip"):
        hv.wait_stable(1000.0, poll_s=0)


def test_iseg_rejected_command_raises() -> None:
    fake = FakeIsegSocket()  # no response configured -> device stays silent
    hv = IsegHV(instrument=fake)
    with pytest.raises(InstrumentError, match="did not answer"):
        hv.read_voltage()


def test_visa_resources_from_ip() -> None:
    assert IsegHV.visa_resource("192.168.0.100") == "TCPIP0::192.168.0.100::10001::SOCKET"
    assert RigolScope.visa_resource("192.168.0.101") == "TCPIP0::192.168.0.101::INSTR"
    assert RigolScope.visa_resource("TCPIP0::10.0.0.5::5555::SOCKET") == "TCPIP0::10.0.0.5::5555::SOCKET"


def test_iseg_rejects_negative_voltage() -> None:
    hv = IsegHV(instrument=FakeIsegSocket())
    with pytest.raises(ValueError):
        hv.set_voltage(-1000.0)


def test_iseg_close_leaves_hv_on() -> None:
    fake = FakeIsegSocket()
    with IsegHV(instrument=fake) as hv:
        hv.output_on()
    assert fake.commands[-1] == "CLOSE"
    assert not any("OFF" in c for c in fake.commands)


def test_rigol_scope_commands() -> None:
    fake = FakeInstrument()
    scope = RigolScope(instrument=fake)
    scope.configure_channel(1, 0.2, 0.1)
    scope.configure_timebase(50e-9, 150e-9)
    scope.configure_edge_trigger(2, level_v=0.25, slope="NEG")

    assert fake.commands == [
        ":CHAN1:DISP 1",
        ":CHAN1:IMP OMEG",
        ":CHAN1:COUP DC",
        ":CHAN1:SCAL 0.2",
        ":CHAN1:OFFS 0.1",
        ":TIM:MAIN:SCAL 5e-08",
        ":TIM:MAIN:OFFS 1.5e-07",
        ":TRIG:MODE EDGE",
        ":TRIG:EDGE:SOUR CHAN2",
        ":TRIG:EDGE:SLOP NEG",
        ":TRIG:EDGE:LEV 0.25",
    ]


def test_preamble_conversion() -> None:
    # Example reply from the MHO900 programming guide (p.478).
    pre = Preamble.parse("0,0,1000,1,1.000000E-8,-5.000000E-6,0.000000E-12,4.000000E-03,0,128")
    assert pre.points == 1000
    np.testing.assert_allclose(pre.volts(np.array([128, 0x8E])), [0.0, 14 * 4e-3])
    np.testing.assert_allclose(pre.times(3), [-5e-6, -5e-6 + 1e-8, -5e-6 + 2e-8])


def test_rigol_read_channel_chunks() -> None:
    fake = FakeInstrument()
    fake.responses[":WAV:PRE?"] = "1,2,5,1,1e-9,-1e-9,0,1e-3,0,100"
    fake.binary = [np.array([100, 101, 102], dtype=np.uint16),
                   np.array([103, 104], dtype=np.uint16)]
    scope = RigolScope(instrument=fake)
    scope.CHUNK_POINTS = 3
    t, v, _ = scope.read_channel(2)

    np.testing.assert_allclose(v, [0.0, 1e-3, 2e-3, 3e-3, 4e-3])
    np.testing.assert_allclose(t, -1e-9 + np.arange(5) * 1e-9)
    assert fake.commands[:4] == [":WAV:SOUR CHAN2", ":WAV:MODE RAW", ":WAV:FORM WORD", ":WAV:PRE?"]
    assert ":WAV:STAR 4" in fake.commands and ":WAV:STOP 5" in fake.commands


def test_rigol_arm_single_rejects_early_trigger() -> None:
    fake = FakeInstrument()
    fake.responses[":TRIG:STAT?"] = "TD"
    scope = RigolScope(instrument=fake)
    with pytest.raises(Exception, match="triggered before"):
        scope.arm_single(timeout_s=0.1)


def test_rigol_arm_single_ignores_stale_stop() -> None:
    # Right after :SINGle the scope can still report the previous STOP.
    fake = FakeInstrument()
    statuses = iter(["STOP", "STOP", "WAIT"])
    fake.query = lambda cmd: next(statuses) if cmd == ":TRIG:STAT?" else "0"
    scope = RigolScope(instrument=fake)
    assert scope.arm_single(timeout_s=1.0, allow_early=True) == "WAIT"


def test_rigol_capture_rejects_empty_readout() -> None:
    fake = FakeInstrument()
    fake.responses[":WAV:PRE?"] = "1,2,0,1,5e-10,-2.5e-6,0,6.6667e-06,0,32768"
    with pytest.raises(InstrumentError, match="no data"):
        RigolScope(instrument=fake).capture_channels()


def test_pulse_fit_to_awg_100ns() -> None:
    pulse = PulseSpec(amplitude=1.0, width=100e-9, pre_delay=100e-9, post_delay=1e-6)
    realised = pulse.fit_to_awg(max_rate=125e6)
    assert pulse.samples == 150  # 1.2 us period at 8 ns/sample
    assert pulse.timing_resolution == pytest.approx(8e-9)
    assert realised == pytest.approx(104e-9)


def test_pulse_fit_to_awg_keeps_long_pulses() -> None:
    pulse = PulseSpec(amplitude=1.0, width=10e-6, pre_delay=5e-6, post_delay=25e-6)
    pulse.fit_to_awg(max_rate=125e6, max_samples=32768)
    assert pulse.samples == 4096
