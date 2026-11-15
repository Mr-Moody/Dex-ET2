"""The copilot's voice: SSE-streamed Gemini TTS, raw PCM pipe playback, pre-cache & fallback.

- **Queue.** At most ``max_waiting`` (2) lines wait. An alarm jumps ahead of every non-alarm
  line; when the queue is full, the oldest non-alarm line is dropped to make room. A line that
  has waited longer than ``stale_after_s`` (10 s) is dropped instead of spoken.
- **Tier 1 Streaming Gemini TTS** streams 24 kHz 16-bit mono ``s16le`` PCM chunks over SSE and
  pipes them directly into ``paplay --raw`` / ``aplay`` (or wraps them via ``pcm_to_wav`` when
  raw players are absent). Unary ``Synthesizer.synthesize`` WAV playback remains supported.
- **Startup Pre-Cache.** :meth:`Speaker.precache_cues` warms the persistent TLS socket and caches
  fixed deterministic telemetry cues in memory so repeated cues play with zero network calls.
- **Fallback.** If chunk 1 exceeds ``first_chunk_timeout_s`` (~1.8 s) or TTS fails, the line
  immediately falls back to ``espeak-ng -s 230`` / the OS system voice.
- **Mute.** With ``mute`` set nothing is spoken and zero network or audio subprocess calls occur.

Speech runs on its own worker thread, so it never blocks the console feed or physics loop.
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
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Protocol, cast

from teleop.copilot.gemini import GeminiError, pcm_to_wav
from teleop.copilot.prompts import Severity
from teleop.copilot.tts_stream import DEFAULT_FIRST_CHUNK_TIMEOUT_S

logger = logging.getLogger(__name__)

MAX_WAITING = 2
STALE_AFTER_S = 10.0
TTS_DEADLINE_S = 1.5
TTS_REQUEST_TIMEOUT_S = 15.0
WAV_CACHE_SIZE = 64
PLAY_TIMEOUT_S = 30.0
WINDOWS_VOICE_RATE = 2
WORDS_PER_MINUTE = 230
SPD_SAY_RATE = 30

DEFAULT_FIXED_CUES: tuple[str, ...] = (
    "Bay 1 stalled on a flap jam. Rescue needed.",
    "Bay 1 refused. Jam is out of distribution. Inspect before rescue.",
    "Clutch engaged on bay 1.",
    "Bolt cleared on bay 1.",
    "Clutch released early on bay 1. Bolt not cleared.",
    "Bay 1 rescue captured. Validating the skill.",
    "Bay 1 skill passed. Shared with the fleet.",
    "Bay 1 skill failed the gate. Not shared.",
    "Bay 2 is reusing a fleet skill.",
)

Clock = Callable[[], int]


class Synthesizer(Protocol):
    """Unary WAV synthesizer protocol (:meth:`GeminiClient.synthesize`)."""

    def synthesize(self, text: str, model: str, voice: str, timeout_s: float) -> bytes: ...


class StreamingSynthesizer(Protocol):
    """Tier 1 SSE PCM streaming synthesizer protocol (:class:`GeminiStreamingTTS`)."""

    def stream_pcm(
        self,
        text: str,
        *,
        voice: str | None = None,
        first_chunk_timeout_s: float = DEFAULT_FIRST_CHUNK_TIMEOUT_S,
        total_timeout_s: float = 8.0,
    ) -> Iterator[bytes]: ...


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
    start_ms: float | None = None
    speak_ms: float | None = None


# -- players ------------------------------------------------------------------------------------


def raw_pcm_player_command() -> list[str] | None:
    """Command accepting raw 24 kHz 16-bit signed LE mono PCM on ``stdin``, or ``None``."""
    if platform.system() == "Windows":
        return None
    if shutil.which("paplay"):
        return [
            "paplay",
            "--raw",
            "--format=s16le",
            "--rate=24000",
            "--channels=1",
            "--latency-msec=40",
        ]
    if shutil.which("aplay"):
        return ["aplay", "-q", "-f", "S16_LE", "-r", "24000", "-c", "1", "-t", "raw", "-"]
    return None


def system_voice_command(text: str) -> list[str] | None:
    """The OS speech command for ``text``, preferring ``espeak-ng -s 230`` on Linux."""
    system = platform.system()
    if system == "Windows":
        quoted = text.replace("'", "''")
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.Rate = {WINDOWS_VOICE_RATE}; $s.Speak('{quoted}')"
        )
        return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
    if system == "Darwin":
        return ["say", "-r", str(WORDS_PER_MINUTE), text]
    for tool in ("espeak-ng", "espeak"):
        if shutil.which(tool):
            return [tool, "-s", str(WORDS_PER_MINUTE), text]
    if shutil.which("spd-say"):
        return ["spd-say", "-w", "-r", str(SPD_SAY_RATE), text]
    if shutil.which("say"):
        return ["say", "-r", str(WORDS_PER_MINUTE), text]
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


def stream_raw_pcm(
    chunks: Iterable[bytes],
    *,
    cmd: Sequence[str] | None = None,
    wav_player: Callable[[bytes], None] = play_wav,
) -> tuple[bytes, str]:
    """Stream 24 kHz s16le mono PCM chunks to ``paplay``/``aplay``, or fall back to WAV.

    Pulls chunk 1 BEFORE spawning any subprocess so a first-chunk TTS timeout never spawns an
    empty ``paplay`` pipe. Sets ``proc.stdin = None`` after closing ``stdin`` per Python 3.12
    ``subprocess`` rules.
    """
    it = iter(chunks)
    first_chunk = b""
    for candidate in it:
        if candidate:
            first_chunk = candidate
            break
    if not first_chunk:
        raise OSError("TTS stream yielded zero audio bytes")

    raw_cmd = list(cmd) if cmd is not None else raw_pcm_player_command()
    if raw_cmd is None:
        collected = [first_chunk]
        for chunk in it:
            if chunk:
                collected.append(chunk)
        pcm_all = b"".join(collected)
        wav_player(pcm_to_wav(pcm_all))
        return pcm_all, "wav_fallback"

    collected_pcm: list[bytes] = [first_chunk]
    proc = subprocess.Popen(
        raw_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert proc.stdin is not None
        try:
            proc.stdin.write(first_chunk)
            proc.stdin.flush()
            for chunk in it:
                if not chunk:
                    continue
                collected_pcm.append(chunk)
                proc.stdin.write(chunk)
                proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            logger.debug("copilot raw PCM pipe ended early: %s", exc)
    finally:
        if proc.stdin is not None:
            with contextlib.suppress(BrokenPipeError, OSError):
                proc.stdin.close()
            proc.stdin = None
        try:
            proc.wait(timeout=PLAY_TIMEOUT_S)
        except subprocess.SubprocessError:
            with contextlib.suppress(OSError):
                proc.kill()
                proc.wait(timeout=1.0)
    return b"".join(collected_pcm), "raw_pcm"


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
    f"$s.Rate = {WINDOWS_VOICE_RATE}; "
    "[Console]::Out.WriteLine('ready'); [Console]::Out.Flush(); "
    "while (($l = [Console]::In.ReadLine()) -ne $null) { "
    "if ($l.Length -gt 0) { $s.Speak($l) }; "
    "[Console]::Out.WriteLine('done'); [Console]::Out.Flush() }"
)
VOICE_LINE_TIMEOUT_S = 30.0
NEWLINE = bytes([10])


class WindowsVoice:
    """One long-running PowerShell ``SpeechSynthesizer``, fed one line at a time on stdin."""

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
        line = " ".join(text.split())
        with self._lock:
            try:
                proc = self._ensure()
                assert proc.stdin is not None
                proc.stdin.write(line.encode("utf-8") + NEWLINE)
                proc.stdin.flush()
                while (reply := self._replies.get(timeout=self._timeout_s)) not in ("done", "eof"):
                    pass
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
                self._proc.stdin = None
            self._kill()


def default_system_voice() -> Callable[[str], None]:
    """A persistent voice on Windows; elsewhere ``espeak-ng``/``say`` start fast per line."""
    return WindowsVoice() if platform.system() == "Windows" else speak_system


# -- the speaker --------------------------------------------------------------------------------


def _priority_to_severity(priority: int | None, default: Severity) -> Severity:
    if priority is None or default != "info":
        return default
    if priority <= 1:
        return "alarm"
    if priority == 2:
        return "attention"
    return default


class Speaker:
    """Speaks call-outs one at a time. ``say`` returns at once; a worker thread does the rest.

    Construct with ``start=False`` and call :meth:`process_next` to drive it by hand (tests).
    """

    def __init__(
        self,
        synthesizer: Synthesizer | StreamingSynthesizer | None = None,
        *,
        tts_model: str = "gemini-3.8-flash-lite-tts",
        voice: str = "Kore",
        mute: bool = False,
        streaming_tts: StreamingSynthesizer | None = None,
        first_chunk_timeout_s: float | None = None,
        player: Callable[[bytes], None] = play_wav,
        raw_player: Callable[[Iterator[bytes]], Any] | None = None,
        system_voice: Callable[[str], None] | None = None,
        clock: Clock = time.monotonic_ns,
        max_waiting: int = MAX_WAITING,
        stale_after_s: float = STALE_AFTER_S,
        tts_deadline_s: float = TTS_DEADLINE_S,
        on_spoken: Callable[[Spoken], None] | None = None,
        start: bool = True,
    ) -> None:
        resolved_streamer = streaming_tts
        resolved_synth: Synthesizer | None = None
        if synthesizer is not None:
            if resolved_streamer is None and hasattr(synthesizer, "stream_pcm"):
                resolved_streamer = cast(StreamingSynthesizer, synthesizer)
            if hasattr(synthesizer, "synthesize"):
                resolved_synth = cast(Synthesizer, synthesizer)
        self._synth: Synthesizer | None = resolved_synth
        self._streamer: StreamingSynthesizer | None = resolved_streamer
        self._tts_model = tts_model
        self._voice = voice
        self.mute = mute
        self._player = player
        self._raw_player = raw_player
        self._system_voice = system_voice if system_voice is not None else default_system_voice()
        self._clock = clock
        self._max_waiting = max_waiting
        self._stale_ns = round(stale_after_s * 1e9)
        self._tts_deadline_s = tts_deadline_s
        self._first_chunk_timeout_s = (
            first_chunk_timeout_s
            if first_chunk_timeout_s is not None
            else DEFAULT_FIRST_CHUNK_TIMEOUT_S
        )
        self._on_spoken = on_spoken
        self._cond = threading.Condition()
        self._queue: list[SpeechItem] = []
        self._next_token = 1
        self._closed = False
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._pcm_cache: OrderedDict[str, bytes] = OrderedDict()
        self._tts_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="copilot-tts")
        self._thread: threading.Thread | None = None
        warm = getattr(self._system_voice, "warm", None)
        if start and not mute and callable(warm):
            threading.Thread(target=warm, name="copilot-voice-warm", daemon=True).start()
        if start:
            self._thread = threading.Thread(target=self._loop, name="copilot-speech", daemon=True)
            self._thread.start()

    # -- startup pre-cache --------------------------------------------------------------------

    def precache_cues(self, cues: Sequence[str] | None = None) -> int:
        """Pre-warm persistent TLS connection and pre-cache fixed cues in memory."""
        if self.mute or (self._streamer is None and self._synth is None):
            return 0
        target_cues = DEFAULT_FIXED_CUES if cues is None else cues
        if not target_cues:
            return 0

        if self._streamer is not None:
            warm_fn = getattr(self._streamer, "warm", None)
            if callable(warm_fn):
                with contextlib.suppress(Exception):
                    warm_fn()

        seen: set[str] = set()
        cached_now = 0
        for raw in target_cues:
            line = raw.strip()
            if not line or line in seen:
                continue
            seen.add(line)
            with self._cond:
                if self._closed:
                    break
                already = line in self._pcm_cache or line in self._cache
            if already:
                continue
            try:
                if self._streamer is not None:
                    pcm = b"".join(
                        self._streamer.stream_pcm(
                            line,
                            voice=self._voice,
                            first_chunk_timeout_s=self._first_chunk_timeout_s,
                        )
                    )
                    if pcm:
                        self._remember_pcm(line, pcm)
                        cached_now += 1
                elif self._synth is not None:
                    wav = self._synth.synthesize(
                        line, self._tts_model, self._voice, TTS_REQUEST_TIMEOUT_S
                    )
                    if wav:
                        self._remember_wav(line, wav)
                        cached_now += 1
            except (GeminiError, TimeoutError, OSError, ValueError) as exc:
                logger.warning("copilot cue pre-cache skipped %r: %s", line, exc)
        return cached_now

    def precache_cues_async(self, cues: Sequence[str] | None = None) -> threading.Thread | None:
        """Warm the TLS socket and pre-cache fixed cues on a daemon thread (non-blocking)."""
        if self.mute or (self._streamer is None and self._synth is None):
            return None
        with self._cond:
            if self._closed:
                return None

        def _warm_worker() -> None:
            with contextlib.suppress(Exception):
                self.precache_cues(cues)

        thread = threading.Thread(target=_warm_worker, name="copilot-tts-precache", daemon=True)
        thread.start()
        return thread

    # -- queue --------------------------------------------------------------------------------

    def say(
        self,
        text: str,
        *,
        severity: Severity = "info",
        bay: int | None = None,
        priority: int | None = None,
    ) -> int:
        """Queue ``text``; returns a token for :meth:`replace`. Never blocks."""
        eff_severity = _priority_to_severity(priority, severity)
        muted_cb: Spoken | None = None
        with self._cond:
            token = self._next_token
            self._next_token += 1
            stripped = text.strip()
            if self._closed or not stripped:
                return token
            if self.mute:
                if self._on_spoken is not None:
                    muted_cb = Spoken(token, stripped, "muted")
            else:
                item = SpeechItem(token, stripped, eff_severity, bay, self._clock())
                if eff_severity == "alarm":
                    at = next((i for i, q in enumerate(self._queue) if q.severity != "alarm"), None)
                    self._queue.insert(len(self._queue) if at is None else at, item)
                else:
                    self._queue.append(item)
                while len(self._queue) > self._max_waiting:
                    victim = next((q for q in self._queue if q.severity != "alarm"), self._queue[0])
                    self._queue.remove(victim)
                    logger.debug("copilot speech: queue full, dropped %r", victim.text)
                self._cond.notify()
        if muted_cb is not None and self._on_spoken is not None:
            self._on_spoken(muted_cb)
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

    def _play_pcm_stream(self, chunks: Iterator[bytes]) -> tuple[bytes, str]:
        if self._raw_player is not None:
            res: object = self._raw_player(chunks)
            head = cast(tuple[object, ...], res)[0] if isinstance(res, tuple) and res else None
            if isinstance(head, bytes):
                return head, "raw_custom"
            return b"", "raw_custom"
        return stream_raw_pcm(chunks, wav_player=self._player)

    def _speak_streaming(self, item: SpeechItem) -> Spoken:
        assert self._streamer is not None
        with self._cond:
            cached_pcm = self._pcm_cache.get(item.text)
            if cached_pcm is not None:
                self._pcm_cache.move_to_end(item.text)
        if cached_pcm is not None:
            started = self._clock()
            start_ms = (started - item.queued_ns) / 1e6
            try:
                self._play_pcm_stream(iter([cached_pcm]))
                speak_ms = (self._clock() - started) / 1e6
                return Spoken(item.token, item.text, "tts", 0.0, "cached", start_ms, speak_ms)
            except OSError as exc:
                note = f"player failed: {exc}"
                started = self._clock()
                self._system_voice(item.text)
                speak_ms = (self._clock() - started) / 1e6
                return Spoken(item.token, item.text, "system", 0.0, note, start_ms, speak_ms)

        t0 = self._clock()
        first_chunk_ns: int | None = None

        def _timed_chunks() -> Iterator[bytes]:
            nonlocal first_chunk_ns
            assert self._streamer is not None
            for chunk in self._streamer.stream_pcm(
                item.text,
                voice=self._voice,
                first_chunk_timeout_s=self._first_chunk_timeout_s,
            ):
                if chunk and first_chunk_ns is None:
                    first_chunk_ns = self._clock()
                yield chunk

        try:
            pcm_all, play_note = self._play_pcm_stream(_timed_chunks())
            if pcm_all:
                self._remember_pcm(item.text, pcm_all)
            t_done = self._clock()
            ttfa_ns = first_chunk_ns if first_chunk_ns is not None else t_done
            tts_ms = (ttfa_ns - t0) / 1e6
            start_ms = (ttfa_ns - item.queued_ns) / 1e6
            speak_ms = (t_done - ttfa_ns) / 1e6
            return Spoken(item.token, item.text, "tts", tts_ms, play_note, start_ms, speak_ms)
        except (GeminiError, TimeoutError, OSError, ValueError) as exc:
            note = f"streaming TTS fallback: {exc}"
            started = self._clock()
            start_ms = (started - item.queued_ns) / 1e6
            self._system_voice(item.text)
            speak_ms = (self._clock() - started) / 1e6
            return Spoken(item.token, item.text, "system", None, note, start_ms, speak_ms)

    def _speak(self, item: SpeechItem) -> Spoken:
        if self._streamer is not None:
            return self._speak_streaming(item)
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
            future.add_done_callback(lambda f: self._remember(text, f))
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

    def _remember_pcm(self, text: str, pcm: bytes) -> None:
        with self._cond:
            self._pcm_cache[text] = pcm
            self._pcm_cache.move_to_end(text)
            while len(self._pcm_cache) > WAV_CACHE_SIZE:
                self._pcm_cache.popitem(last=False)

    def _loop(self) -> None:
        while True:
            try:
                if self.process_next(block=True) is None:
                    return
            except Exception:
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
