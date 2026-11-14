"""The copilot's voice: one short queue, Gemini TTS, and the system voice as a fallback.

- **Queue.** At most ``max_waiting`` (2) lines wait. An alarm jumps ahead of every non-alarm
  line; when the queue is full, the oldest non-alarm line is dropped to make room. A line that
  has waited longer than ``stale_after_s`` (10 s) is dropped instead of spoken.
- **Gemini TTS** turns the line into a WAV, which is played with the first available of
  ``paplay``, ``aplay``, ``afplay`` or PowerShell ``SoundPlayer``.
- **Fallback.** If TTS fails or has not answered within ``tts_deadline_s`` (1.5 s), the same line
  is spoken with the system voice. A late WAV is dropped, never played out of order.
- **Mute.** With ``mute`` set nothing is spoken; the call-outs are still printed and logged.

Speech runs on its own worker thread, so it never blocks the console feed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Protocol

from teleop.copilot.gemini import GeminiError
from teleop.copilot.prompts import Severity

logger = logging.getLogger(__name__)

MAX_WAITING = 2
STALE_AFTER_S = 10.0
TTS_DEADLINE_S = 1.5  # Gemini TTS took 4+ s on every live line; see docs/copilot.md
TTS_REQUEST_TIMEOUT_S = 15.0  # per-socket limit for the late request; the deadline is above
WAV_CACHE_SIZE = 32  # template lines repeat ("Clutch engaged on bay 1."), so keep their audio
PLAY_TIMEOUT_S = 30.0

Clock = Callable[[], int]


class Synthesizer(Protocol):
    """What the speaker needs from :class:`~teleop.copilot.gemini.GeminiClient`."""

    def synthesize(self, text: str, model: str, voice: str, timeout_s: float) -> bytes: ...


@dataclass(slots=True)
class SpeechItem:
    token: int
    text: str
    severity: Severity
    bay: int | None
    queued_ns: int


@dataclass(frozen=True, slots=True)
class Spoken:
    """What happened to one line, for tests and logs."""

    token: int
    text: str
    route: str  # "tts", "system", "dropped_stale" or "muted"
    tts_ms: float | None = None
    note: str = ""
    start_ms: float | None = None  # from queueing the line to the voice or player starting
    speak_ms: float | None = None  # how long speaking it took


# -- players ------------------------------------------------------------------------------------


def system_voice_command(text: str) -> list[str] | None:
    """The OS speech command for ``text``, or ``None`` when there is no synthesizer."""
    system = platform.system()
    if system == "Windows":
        quoted = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
            f".Speak('{quoted}')"
        )
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if system == "Darwin":
        return ["say", text]
    for tool in ("spd-say", "espeak-ng", "espeak"):
        if shutil.which(tool):
            return [tool, text]
    return None


def wav_player_command(path: str) -> list[str] | None:
    """The first available command that plays the WAV at ``path`` to the end."""
    if platform.system() == "Windows":
        quoted = path.replace("'", "''")
        script = f"(New-Object System.Media.SoundPlayer '{quoted}').PlaySync()"
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    for tool in ("paplay", "aplay", "afplay"):
        if shutil.which(tool):
            return [tool, path]
    return None


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=PLAY_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("copilot %s failed: %s", what, exc)


def play_wav(wav: bytes) -> None:
    """Play a WAV through the system player (blocking until it ends)."""
    fd, path = tempfile.mkstemp(prefix="copilot-", suffix=".wav")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(wav)
        cmd = wav_player_command(path)
        if cmd is None:
            raise OSError("no WAV player found (paplay, aplay, afplay or PowerShell)")
        _run(cmd, "audio player")
    finally:
        with contextlib.suppress(OSError):
            os.unlink(path)


def speak_system(text: str) -> None:
    """Speak ``text`` with the OS voice (blocking); log it when there is no voice."""
    cmd = system_voice_command(text)
    if cmd is None:
        logger.info("copilot (no system voice): %s", text)
        return
    _run(cmd, "system voice")


_VOICE_SCRIPT = (
    "[Console]::InputEncoding = [Text.Encoding]::UTF8; "
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
    "[Console]::Out.WriteLine('ready'); [Console]::Out.Flush(); "
    "while (($l = [Console]::In.ReadLine()) -ne $null) { "
    "if ($l.Length -gt 0) { $s.Speak($l) }; "
    "[Console]::Out.WriteLine('done'); [Console]::Out.Flush() }"
)
VOICE_LINE_TIMEOUT_S = 30.0
NEWLINE = bytes([10])


class WindowsVoice:
    """One long-running PowerShell ``SpeechSynthesizer``, fed one line at a time on stdin.

    Starting PowerShell and loading System.Speech takes 1-2 s, so doing it once (at start-up, via
    :meth:`warm`) instead of per line takes that delay off every call-out. Calls block until the
    line has been spoken. If the process dies or hangs it is restarted for the next line, and the
    current line falls back to the one-shot :func:`speak_system`.
    """

    def __init__(self, line_timeout_s: float = VOICE_LINE_TIMEOUT_S) -> None:
        self._proc: subprocess.Popen[bytes] | None = None
        self._replies: queue.Queue[str] = queue.Queue()
        self._timeout_s = line_timeout_s
        self._lock = threading.Lock()

    def warm(self) -> None:
        """Start the process now so the first line does not pay for it."""
        with self._lock, contextlib.suppress(OSError):
            self._ensure()

    def _ensure(self) -> subprocess.Popen[bytes]:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        self._replies = queue.Queue()
        proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", _VOICE_SCRIPT],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        replies = self._replies

        def read() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                replies.put(raw.decode("utf-8", "replace").strip())
            replies.put("eof")

        threading.Thread(target=read, name="copilot-voice", daemon=True).start()
        self._proc = proc
        return proc

    def __call__(self, text: str) -> None:
        line = " ".join(text.split())  # one line: the voice reads up to the newline
        with self._lock:
            try:
                proc = self._ensure()
                assert proc.stdin is not None
                proc.stdin.write(line.encode("utf-8") + NEWLINE)
                proc.stdin.flush()
                while (reply := self._replies.get(timeout=self._timeout_s)) not in ("done", "eof"):
                    pass  # "ready" from start-up
                if reply == "done":
                    return
                raise OSError("voice process exited")
            except (OSError, queue.Empty) as exc:
                logger.warning("copilot voice process failed (%s); restarting it", exc)
                self._kill()
        speak_system(text)

    def _kill(self) -> None:
        if self._proc is not None:
            with contextlib.suppress(OSError):
                self._proc.kill()
            self._proc = None

    def close(self) -> None:
        with self._lock:
            if self._proc is not None and self._proc.stdin is not None:
                with contextlib.suppress(OSError):
                    self._proc.stdin.close()
            self._kill()


def default_system_voice() -> Callable[[str], None]:
    """A persistent voice on Windows; elsewhere ``say``/``spd-say`` start fast enough per line."""
    return WindowsVoice() if platform.system() == "Windows" else speak_system


# -- the speaker --------------------------------------------------------------------------------


class Speaker:
    """Speaks call-outs one at a time. ``say`` returns at once; a worker thread does the rest.

    Construct with ``start=False`` and call :meth:`process_next` to drive it by hand (tests).
    """

    def __init__(
        self,
        synthesizer: Synthesizer | None,
        *,
        tts_model: str,
        voice: str,
        mute: bool = False,
        player: Callable[[bytes], None] = play_wav,
        system_voice: Callable[[str], None] | None = None,
        clock: Clock = time.monotonic_ns,
        max_waiting: int = MAX_WAITING,
        stale_after_s: float = STALE_AFTER_S,
        tts_deadline_s: float = TTS_DEADLINE_S,
        on_spoken: Callable[[Spoken], None] | None = None,
        start: bool = True,
    ) -> None:
        self._synth = synthesizer
        self._tts_model = tts_model
        self._voice = voice
        self.mute = mute
        self._player = player
        self._system_voice = system_voice if system_voice is not None else default_system_voice()
        self._clock = clock
        self._max_waiting = max_waiting
        self._stale_ns = round(stale_after_s * 1e9)
        self._tts_deadline_s = tts_deadline_s
        self._on_spoken = on_spoken
        self._cond = threading.Condition()
        self._queue: list[SpeechItem] = []
        self._next_token = 1
        self._closed = False
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._tts_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="copilot-tts")
        self._thread: threading.Thread | None = None
        warm = getattr(self._system_voice, "warm", None)
        if start and not mute and callable(warm):
            threading.Thread(target=warm, name="copilot-voice-warm", daemon=True).start()
        if start:
            self._thread = threading.Thread(target=self._loop, name="copilot-speech", daemon=True)
            self._thread.start()

    # -- queue --------------------------------------------------------------------------------

    def say(self, text: str, *, severity: Severity = "info", bay: int | None = None) -> int:
        """Queue ``text``; returns a token for :meth:`replace`. Never blocks."""
        with self._cond:
            token = self._next_token
            self._next_token += 1
            if self.mute or self._closed or not text.strip():
                return token
            item = SpeechItem(token, text.strip(), severity, bay, self._clock())
            if severity == "alarm":
                at = next((i for i, q in enumerate(self._queue) if q.severity != "alarm"), None)
                self._queue.insert(len(self._queue) if at is None else at, item)
            else:
                self._queue.append(item)
            while len(self._queue) > self._max_waiting:
                # the oldest non-alarm line goes (the new one, if it is the only one); with only
                # alarms waiting, the oldest alarm goes
                victim = next((q for q in self._queue if q.severity != "alarm"), self._queue[0])
                self._queue.remove(victim)
                logger.debug("copilot speech: queue full, dropped %r", victim.text)
            self._cond.notify()
            return token

    def replace(self, token: int, text: str, severity: Severity | None = None) -> bool:
        """Swap the text of a line that is still waiting. ``False`` once it is being spoken."""
        with self._cond:
            for item in self._queue:
                if item.token == token:
                    item.text = text.strip() or item.text
                    if severity is not None:
                        item.severity = severity
                    return True
            return False

    def cancel_bay(self, bay: int) -> int:
        """Drop every waiting line about ``bay``; returns how many were dropped."""
        with self._cond:
            before = len(self._queue)
            self._queue = [q for q in self._queue if q.bay != bay]
            return before - len(self._queue)

    def waiting(self) -> list[str]:
        with self._cond:
            return [q.text for q in self._queue]

    # -- worker -------------------------------------------------------------------------------

    def _take(self, block: bool) -> SpeechItem | None:
        with self._cond:
            while not self._queue:
                if self._closed or not block:
                    return None
                self._cond.wait()
            return self._queue.pop(0)

    def process_next(self, block: bool = False) -> Spoken | None:
        """Speak the next waiting line (or drop it if stale). ``None`` when the queue is empty."""
        item = self._take(block)
        if item is None:
            return None
        if self._clock() - item.queued_ns > self._stale_ns:
            result = Spoken(item.token, item.text, "dropped_stale")
        elif self.mute:
            result = Spoken(item.token, item.text, "muted")
        else:
            result = self._speak(item)
        if self._on_spoken is not None:
            self._on_spoken(result)
        return result

    def _speak(self, item: SpeechItem) -> Spoken:
        wav, tts_ms, note = self._tts(item.text)
        started = self._clock()
        start_ms = (started - item.queued_ns) / 1e6
        if wav is not None:
            try:
                self._player(wav)
                speak_ms = (self._clock() - started) / 1e6
                return Spoken(item.token, item.text, "tts", tts_ms, note, start_ms, speak_ms)
            except OSError as exc:
                note = f"player failed: {exc}"
                started = self._clock()
        self._system_voice(item.text)
        speak_ms = (self._clock() - started) / 1e6
        return Spoken(item.token, item.text, "system", tts_ms, note, start_ms, speak_ms)

    def _tts(self, text: str) -> tuple[bytes | None, float | None, str]:
        """The WAV within the deadline, or ``None`` and why not."""
        if self._synth is None:
            return None, None, "no TTS client"
        with self._cond:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
        if cached is not None:
            return cached, 0.0, "cached"
        synth = self._synth
        t0 = self._clock()
        future: Future[bytes] = self._tts_pool.submit(
            synth.synthesize, text, self._tts_model, self._voice, TTS_REQUEST_TIMEOUT_S
        )
        try:
            wav = future.result(timeout=self._tts_deadline_s)
        except FutureTimeout:
            future.add_done_callback(lambda f: self._remember(text, f))  # keep it for next time
            return None, None, f"TTS slower than {self._tts_deadline_s:g} s"
        except (GeminiError, ValueError, OSError, EOFError) as exc:
            return None, None, f"TTS failed: {exc}"
        self._remember_wav(text, wav)
        return wav, (self._clock() - t0) / 1e6, ""

    def _remember(self, text: str, future: Future[bytes]) -> None:
        if not future.cancelled() and future.exception() is None:
            self._remember_wav(text, future.result())

    def _remember_wav(self, text: str, wav: bytes) -> None:
        with self._cond:
            self._cache[text] = wav
            self._cache.move_to_end(text)
            while len(self._cache) > WAV_CACHE_SIZE:
                self._cache.popitem(last=False)

    def _loop(self) -> None:
        while True:
            try:
                if self.process_next(block=True) is None:
                    return  # closed
            except Exception:  # never let one bad line kill the voice
                logger.exception("copilot speech failed")

    def close(self, timeout_s: float = 1.0) -> None:
        with self._cond:
            self._closed = True
            self._queue.clear()
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout_s)
        self._tts_pool.shutdown(wait=False, cancel_futures=True)
        close_voice = getattr(self._system_voice, "close", None)
        if callable(close_voice):
            close_voice()
