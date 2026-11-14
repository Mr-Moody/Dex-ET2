"""T-212: the speech queue, Gemini TTS, the system-voice fallback and mute (no audio device)."""

from __future__ import annotations

import threading
from collections.abc import Callable

import pytest

from teleop.copilot.gemini import GeminiHttpError
from teleop.copilot.speech import Speaker, Spoken, system_voice_command, wav_player_command

WAV = b"RIFF-fake-wav"


class FakeClock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns

    def advance(self, s: float) -> None:
        self.ns += round(s * 1e9)


class FakeTts:
    def __init__(self, behaviour: Callable[[str], bytes] | None = None) -> None:
        self.calls: list[tuple[str, str, str]] = []
        self._behaviour = behaviour or (lambda text: WAV + text.encode())

    def synthesize(self, text: str, model: str, voice: str, timeout_s: float) -> bytes:
        self.calls.append((text, model, voice))
        return self._behaviour(text)


class Recorder:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.system: list[str] = []

    def player(self, wav: bytes) -> None:
        self.played.append(wav)

    def voice(self, text: str) -> None:
        self.system.append(text)


def make(
    tts: FakeTts | None = None,
    clock: FakeClock | None = None,
    **kwargs: object,
) -> tuple[Speaker, Recorder]:
    rec = Recorder()
    speaker = Speaker(
        tts,
        tts_model="tts-model",
        voice="Kore",
        player=rec.player,
        system_voice=rec.voice,
        clock=clock or FakeClock(),
        start=False,
        **kwargs,  # type: ignore[arg-type]
    )
    return speaker, rec


def drain(speaker: Speaker) -> list[Spoken]:
    out: list[Spoken] = []
    while (item := speaker.process_next()) is not None:
        out.append(item)
    return out


def test_lines_are_spoken_in_order_with_gemini_tts() -> None:
    tts = FakeTts()
    speaker, rec = make(tts)
    speaker.say("one")
    speaker.say("two")
    spoken = drain(speaker)
    assert [s.text for s in spoken] == ["one", "two"]
    assert [s.route for s in spoken] == ["tts", "tts"]
    assert rec.played == [WAV + b"one", WAV + b"two"] and rec.system == []
    assert tts.calls[0] == ("one", "tts-model", "Kore")


def test_at_most_two_wait_and_the_oldest_is_dropped() -> None:
    speaker, _ = make(FakeTts())
    for text in ("a", "b", "c"):
        speaker.say(text)
    assert speaker.waiting() == ["b", "c"]


def test_alarms_jump_the_queue_and_push_out_info() -> None:
    speaker, _ = make(FakeTts())
    speaker.say("info one")
    speaker.say("info two")
    speaker.say("ALARM", severity="alarm")
    assert speaker.waiting() == ["ALARM", "info two"]
    speaker.say("ALARM 2", severity="alarm")
    assert speaker.waiting() == ["ALARM", "ALARM 2"]  # alarms keep their own order
    speaker.say("info three")
    assert speaker.waiting() == ["ALARM", "ALARM 2"]  # an info line never pushes out an alarm
    speaker.say("ALARM 3", severity="alarm")
    assert speaker.waiting() == ["ALARM 2", "ALARM 3"]  # only alarms: the oldest goes


def test_lines_older_than_ten_seconds_are_dropped() -> None:
    clock = FakeClock()
    speaker, rec = make(FakeTts(), clock)
    speaker.say("old news")
    clock.advance(10.5)
    speaker.say("fresh")
    spoken = drain(speaker)
    assert [(s.text, s.route) for s in spoken] == [("old news", "dropped_stale"), ("fresh", "tts")]
    assert rec.played == [WAV + b"fresh"]


def test_slow_tts_falls_back_to_the_system_voice_and_drops_late_audio() -> None:
    release = threading.Event()

    def slow(text: str) -> bytes:
        release.wait(5)
        return WAV

    speaker, rec = make(FakeTts(slow), tts_deadline_s=0.05)
    speaker.say("Bay 1 stalled.")
    spoken = drain(speaker)
    assert [s.route for s in spoken] == ["system"]
    assert "slower than" in spoken[0].note
    assert rec.system == ["Bay 1 stalled."] and rec.played == []
    release.set()  # the late WAV arrives after the fallback spoke: it is never played
    speaker.close()
    assert rec.played == []


def test_tts_errors_fall_back_to_the_system_voice() -> None:
    def broken(text: str) -> bytes:
        raise GeminiHttpError(503, "overloaded")

    speaker, rec = make(FakeTts(broken))
    speaker.say("Clutch engaged on bay 1.")
    assert [s.route for s in drain(speaker)] == ["system"]
    assert rec.system == ["Clutch engaged on bay 1."]


def test_a_player_failure_falls_back_to_the_system_voice() -> None:
    rec_voice: list[str] = []

    def no_player(wav: bytes) -> None:
        raise OSError("no WAV player found")

    speaker = Speaker(
        FakeTts(),
        tts_model="m",
        voice="v",
        player=no_player,
        system_voice=rec_voice.append,
        start=False,
    )
    speaker.say("hello")
    assert [s.route for s in drain(speaker)] == ["system"] and rec_voice == ["hello"]


def test_without_a_tts_client_the_system_voice_speaks() -> None:
    speaker, rec = make(None)
    speaker.say("templates only")
    assert [s.route for s in drain(speaker)] == ["system"] and rec.system == ["templates only"]


def test_repeated_lines_reuse_the_cached_audio() -> None:
    tts = FakeTts()
    speaker, rec = make(tts)
    speaker.say("Clutch engaged on bay 1.")
    drain(speaker)
    speaker.say("Clutch engaged on bay 1.")
    drain(speaker)
    assert len(tts.calls) == 1 and len(rec.played) == 2


def test_mute_speaks_nothing() -> None:
    tts = FakeTts()
    speaker, rec = make(tts, mute=True)
    speaker.say("quiet", severity="alarm")
    assert drain(speaker) == [] and tts.calls == [] and rec.played == rec.system == []


def test_replace_and_cancel_only_touch_waiting_lines() -> None:
    speaker, _ = make(FakeTts())
    t1 = speaker.say("template", bay=1)
    speaker.say("bay two", bay=2)
    assert speaker.replace(t1, "model line") is True
    assert speaker.waiting() == ["model line", "bay two"]
    assert speaker.cancel_bay(2) == 1
    drain(speaker)
    assert speaker.replace(t1, "too late") is False


def test_the_worker_thread_speaks_and_closes() -> None:
    done = threading.Event()
    spoken: list[Spoken] = []

    def on_spoken(s: Spoken) -> None:
        spoken.append(s)
        done.set()

    rec = Recorder()
    speaker = Speaker(
        FakeTts(),
        tts_model="m",
        voice="v",
        player=rec.player,
        system_voice=rec.voice,
        on_spoken=on_spoken,
    )
    speaker.say("threaded")
    assert done.wait(2)
    speaker.close()
    assert spoken[0].route == "tts"


@pytest.mark.parametrize("system", ["Windows", "Darwin"])
def test_player_and_voice_commands(system: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("teleop.copilot.speech.platform.system", lambda: system)
    voice = system_voice_command("it's bay 1")
    player = wav_player_command("C:/tmp/a.wav")
    assert voice is not None
    if system == "Windows":
        assert "it''s bay 1" in voice[-1] and "$s.Rate = 2" in voice[-1]
        assert player is not None and "PlaySync" in player[-1]
    else:
        assert voice == ["say", "-r", "230", "it's bay 1"]


def test_linux_without_players_has_no_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("teleop.copilot.speech.platform.system", lambda: "Linux")
    monkeypatch.setattr("teleop.copilot.speech.shutil.which", lambda tool: None)
    assert system_voice_command("x") is None and wav_player_command("a.wav") is None


class FakeVoice:
    def __init__(self) -> None:
        self.lines: list[str] = []
        self.warmed = threading.Event()
        self.closed = False

    def __call__(self, text: str) -> None:
        self.lines.append(text)

    def warm(self) -> None:
        self.warmed.set()

    def close(self) -> None:
        self.closed = True


def test_the_system_voice_is_warmed_at_start_and_closed_with_the_speaker() -> None:
    voice = FakeVoice()
    speaker = Speaker(None, tts_model="m", voice="v", system_voice=voice)
    assert voice.warmed.wait(2)
    speaker.close()
    assert voice.closed


def test_speech_timings_are_reported() -> None:
    clock = FakeClock()
    voice = FakeVoice()
    speaker = Speaker(None, tts_model="m", voice="v", system_voice=voice, clock=clock, start=False)
    speaker.say("Bay 1 stalled.")
    clock.advance(0.25)
    spoken = drain(speaker)
    assert spoken[0].route == "system" and spoken[0].start_ms == 250.0
    assert spoken[0].speak_ms == 0.0 and voice.lines == ["Bay 1 stalled."]


def test_run_speaks_with_the_system_voice_unless_gemini_tts_is_asked_for() -> None:
    from teleop.copilot.config import CopilotConfig
    from teleop.copilot.run import _speech_label  # pyright: ignore[reportPrivateUsage]

    cfg = CopilotConfig()
    assert _speech_label(cfg, gemini_tts=False) == "system voice"
    assert _speech_label(cfg, gemini_tts=True) == f"{cfg.tts_model}/{cfg.voice}"
    assert _speech_label(CopilotConfig(mute=True), gemini_tts=True) == "muted"
