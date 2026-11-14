"""ACT runtime: one checkpoint, one batched forward pass for every bay, off the control thread.

The 60 Hz control loop never waits on the model. Each bay in ACT recovery posts its latest
observation with :meth:`InferenceWorker.submit` and reads the newest action chunk back with
:meth:`InferenceWorker.latest`. One daemon thread stacks every pending bay into one
:class:`ObsBatch` and calls the policy once (ACT recovery spec 5).

Health (spec 1, "automatic downgrade"): the worker turns unhealthy if the policy raises, returns
a chunk of the wrong shape, takes longer than ``timeout_ms`` on one call, or runs over
``budget_ms`` for ``over_budget_limit`` consecutive calls. The coordinator then downgrades to the
DMP at the next stall.

``torch`` and ``lerobot`` are imported only inside :func:`load_act_policy`, so the default
``dmp`` backend and the tests (which use :class:`ReplayPolicy`) never import them.
"""

from __future__ import annotations

import importlib
import json
import logging
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from teleop.policy.manifest import ManifestError
from teleop.policy.schema import ACTION_NAMES, CAMERAS, STATE_NAMES

logger = logging.getLogger(__name__)

STATE_KEY = "observation.state"
ACTION_KEY = "action"
IMAGE_KEY = "observation.images.{}"
WRIST_POS = slice(STATE_NAMES.index("wrist/x"), STATE_NAMES.index("wrist/z") + 1)

DEFAULT_BUDGET_MS = 30.0
DEFAULT_TIMEOUT_MS = 250.0
DEFAULT_OVER_BUDGET_LIMIT = 10
DEFAULT_GATHER_MS = 5.0  # max wait for the other active bays' observations of the same tick
LATENCY_WINDOW = 200  # inference calls kept for p95


@dataclass(frozen=True)
class ObsBatch:
    """``state`` is ``(B, 29)`` float32; each ``images[cam]`` is ``(B, H, W, 3)`` uint8."""

    state: NDArray[np.float32]
    images: dict[str, NDArray[np.uint8]]

    @property
    def size(self) -> int:
        return int(self.state.shape[0])


class PolicyBackend(Protocol):
    def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
        """Latch-frame action chunk, shape ``(B, chunk, 12)``."""
        ...


# --------------------------------------------------------------------------- replay stub


class ReplayPolicy:
    """Test and eval stub: replays a recorded latch-frame action sequence from where the bay is.

    Each batch row picks the action row whose wrist position is nearest the observed wrist
    position (``STATE_NAMES`` ``wrist/x..z``), then skips forward over the rows that follow it
    and are no farther than ``skip_m`` from the observation (a dwell, such as a finger press
    with the wrist still, or the lead a commanded target keeps over the measured wrist). It
    returns the next ``chunk`` rows from there, padded with the last row. Stateless, so one
    stub serves any number of bays in one batch.
    """

    def __init__(
        self, actions_latch: NDArray[np.float32], chunk: int = 30, skip_m: float = 0.002
    ) -> None:
        a = np.asarray(actions_latch, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != len(ACTION_NAMES) or a.shape[0] == 0:
            raise ValueError(f"actions_latch must be (T, {len(ACTION_NAMES)}), got {a.shape}")
        self.actions = a
        self.chunk = int(chunk)
        self.skip_m = float(skip_m)
        self.calls = 0

    def start_index(self, wrist_latch: NDArray[np.floating[Any]]) -> int:
        d = np.linalg.norm(self.actions[:, :3] - np.asarray(wrist_latch, dtype=np.float32), axis=1)
        i = int(np.argmin(d))
        limit = float(d[i]) + self.skip_m
        while i + 1 < len(d) and float(d[i + 1]) <= limit:
            i += 1
        return i

    def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
        self.calls += 1
        out = np.empty((batch.size, self.chunk, len(ACTION_NAMES)), dtype=np.float32)
        last = len(self.actions) - 1
        for b in range(batch.size):
            i = self.start_index(batch.state[b, WRIST_POS])
            rows = np.minimum(np.arange(i, i + self.chunk), last)
            out[b] = self.actions[rows]
        return out


# --------------------------------------------------------------------------- worker


@dataclass(frozen=True)
class _Obs:
    state: NDArray[np.float32]
    images: dict[str, NDArray[np.uint8]]
    tick: int


class InferenceWorker:
    """Batches every pending bay into one ``predict_chunk`` call on one daemon thread."""

    def __init__(
        self,
        policy: PolicyBackend,
        budget_ms: float = DEFAULT_BUDGET_MS,
        timeout_ms: float = DEFAULT_TIMEOUT_MS,
        over_budget_limit: int = DEFAULT_OVER_BUDGET_LIMIT,
        *,
        gather_ms: float = DEFAULT_GATHER_MS,
        autostart: bool = True,
    ) -> None:
        self.policy = policy
        self.gather_ms = gather_ms
        self.budget_ms = budget_ms
        self.timeout_ms = timeout_ms
        self.over_budget_limit = over_budget_limit
        self._cv = threading.Condition()
        self._pending: dict[int, _Obs] = {}
        self._results: dict[int, tuple[int, NDArray[np.float32]]] = {}
        self._latency_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._over = 0
        self._closed = False
        self._reason = ""
        self.calls = 0
        self.batch_sizes: list[int] = []
        self._thread = threading.Thread(target=self._run, name="act-inference", daemon=True)
        if autostart:
            self.start()

    def start(self) -> None:
        if not self._thread.is_alive() and not self._closed:
            self._thread.start()

    # ------------------------------------------------------------------ control-thread API

    def submit(
        self, bay_id: int, state: NDArray[Any], images: Mapping[str, NDArray[Any]], tick: int
    ) -> None:
        """Post a bay's observation (non-blocking); a newer one replaces an unserved older one."""
        obs = _Obs(
            np.asarray(state, dtype=np.float32).reshape(len(STATE_NAMES)).copy(),
            {c: np.asarray(images[c], dtype=np.uint8) for c in CAMERAS},
            int(tick),
        )
        with self._cv:
            if self._closed or self._reason:
                return
            self._pending[bay_id] = obs
            self._cv.notify()

    def latest(self, bay_id: int) -> tuple[int, NDArray[np.float32]] | None:
        """``(observation tick, chunk)`` of the newest chunk for the bay, or ``None``."""
        with self._cv:
            return self._results.get(bay_id)

    def forget(self, bay_id: int) -> None:
        """Drop the bay's pending observation and its last chunk (a recovery ended or starts)."""
        with self._cv:
            self._pending.pop(bay_id, None)
            self._results.pop(bay_id, None)

    def healthy(self) -> bool:
        with self._cv:
            return not self._closed and not self._reason

    @property
    def unhealthy_reason(self) -> str:
        with self._cv:
            return "inference worker closed" if self._closed and not self._reason else self._reason

    def p95_ms(self) -> float | None:
        with self._cv:
            samples = list(self._latency_ms)
        return float(np.percentile(samples, 95)) if samples else None

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._pending.clear()
            self._cv.notify_all()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------ worker thread

    def _fail(self, reason: str) -> None:
        with self._cv:
            if not self._reason:
                self._reason = reason
            self._pending.clear()
        logger.warning("act: inference unhealthy (%s)", reason)

    def _run(self) -> None:
        while True:
            with self._cv:
                while not self._pending and not self._closed:
                    self._cv.wait()
                # Gather: bays posting on the same control tick share one forward pass. Wait
                # briefly for every bay that already has a chunk (an active recovery) to post.
                deadline = time.perf_counter() + self.gather_ms * 1e-3
                while not self._closed and not set(self._results) <= set(self._pending):
                    left = deadline - time.perf_counter()
                    if left <= 0:
                        break
                    self._cv.wait(left)
                if self._closed or self._reason:
                    return
                pending, self._pending = self._pending, {}
            bays = sorted(pending)
            batch = ObsBatch(
                np.stack([pending[b].state for b in bays]),
                {c: np.stack([pending[b].images[c] for b in bays]) for c in CAMERAS},
            )
            # perf_counter: monotonic_ns ticks in 15.6 ms steps on Windows, too coarse for 30 ms
            t0 = time.perf_counter_ns()
            try:
                chunk = np.asarray(self.policy.predict_chunk(batch), dtype=np.float32)
            except Exception as exc:
                logger.exception("act: policy raised")
                self._fail(f"inference error: {exc}")
                return
            ms = (time.perf_counter_ns() - t0) * 1e-6
            if (
                chunk.ndim != 3
                or chunk.shape[0] != len(bays)
                or chunk.shape[2] != len(ACTION_NAMES)
            ):
                self._fail(f"inference error: chunk shape {chunk.shape}, batch of {len(bays)}")
                return
            with self._cv:
                self.calls += 1
                self.batch_sizes.append(len(bays))
                self._latency_ms.append(ms)
                self._over = self._over + 1 if ms > self.budget_ms else 0
                if not self._closed:
                    for k, b in enumerate(bays):
                        self._results[b] = (pending[b].tick, chunk[k])
                over = self._over
            if ms > self.timeout_ms:
                self._fail(f"inference timeout: {ms:.0f} ms > {self.timeout_ms:.0f} ms")
                return
            if over >= self.over_budget_limit:
                self._fail(
                    f"inference over budget: {over} consecutive calls > {self.budget_ms:.0f} ms"
                )
                return


# --------------------------------------------------------------------------- checkpoint


def pretrained_dir(checkpoint: Path) -> Path:
    """The directory holding the policy's ``config.json``.

    Accepts the pretrained model directory itself or a ``lerobot-train`` output directory
    (``checkpoints/last/pretrained_model``, else the highest numbered step).
    """
    checkpoint = Path(checkpoint)
    if (checkpoint / "config.json").is_file():
        return checkpoint
    root = checkpoint / "checkpoints"
    last = root / "last" / "pretrained_model"
    if (last / "config.json").is_file():
        return last
    steps = [
        p
        for p in root.glob("*/pretrained_model")
        if p.parent.name.isdigit() and (p / "config.json").is_file()
    ]
    if steps:
        return max(steps, key=lambda p: int(p.parent.name))
    raise ManifestError(f"{checkpoint}: no ACT checkpoint (config.json) found")


def _shape(features: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    ft = features.get(key)
    if not isinstance(ft, Mapping):
        return None
    shape = cast("Mapping[str, Any]", ft).get("shape")
    if not isinstance(shape, list | tuple):
        return None
    return tuple(int(v) for v in cast("list[Any]", shape))


def check_policy_config(config: Mapping[str, Any], where: str = "checkpoint") -> tuple[int, int]:
    """Refuse a checkpoint trained on another schema; return its camera image ``(H, W)``."""
    inputs = config.get("input_features")
    outputs = config.get("output_features")
    if not isinstance(inputs, Mapping) or not isinstance(outputs, Mapping):
        raise ManifestError(f"{where}: config.json has no input/output features")
    ins = cast("Mapping[str, Any]", inputs)
    outs = cast("Mapping[str, Any]", outputs)
    problems: list[str] = []
    state = _shape(ins, STATE_KEY)
    if state != (len(STATE_NAMES),):
        problems.append(f"{STATE_KEY} shape {state}, expected ({len(STATE_NAMES)},)")
    action = _shape(outs, ACTION_KEY)
    if action != (len(ACTION_NAMES),):
        problems.append(f"{ACTION_KEY} shape {action}, expected ({len(ACTION_NAMES)},)")
    hw: set[tuple[int, int]] = set()
    for cam in CAMERAS:
        key = IMAGE_KEY.format(cam)
        img = _shape(ins, key)
        if img is None or len(img) != 3 or img[0] != 3:
            problems.append(f"{key} missing or not a (3, H, W) image (got {img})")
        else:
            hw.add((img[1], img[2]))
    extra = sorted(k for k in ins if k.startswith("observation.images.") and k not in _image_keys())
    if extra:
        problems.append(f"unexpected cameras {extra}")
    if len(hw) > 1:
        problems.append(f"cameras disagree on image size {sorted(hw)}")
    if problems:
        raise ManifestError(f"{where} does not match the policy schema: " + "; ".join(problems))
    return next(iter(hw))


def _image_keys() -> tuple[str, ...]:
    return tuple(IMAGE_KEY.format(c) for c in CAMERAS)


def read_policy_config(checkpoint: Path) -> tuple[Path, tuple[int, int]]:
    """Locate and schema-check a checkpoint without importing torch: ``(dir, image (H, W))``."""
    path = pretrained_dir(checkpoint)
    try:
        raw: Any = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"{path / 'config.json'} unreadable: {exc}") from exc
    if not isinstance(raw, dict):
        raise ManifestError(f"{path / 'config.json'}: expected a JSON object")
    return path, check_policy_config(cast("dict[str, Any]", raw), str(path))


class _ActBackend:
    """A LeRobot ACT policy with the pre/post-processors saved alongside it."""

    def __init__(self, torch: Any, policy: Any, pre: Any, post: Any, device: str) -> None:
        self.torch, self.policy, self.pre, self.post = torch, policy, pre, post
        self.device = device

    def predict_chunk(self, batch: ObsBatch) -> NDArray[np.float32]:
        torch = self.torch
        obs: dict[str, Any] = {STATE_KEY: torch.from_numpy(np.ascontiguousarray(batch.state))}
        for cam in CAMERAS:
            img = torch.from_numpy(np.ascontiguousarray(batch.images[cam]))
            obs[IMAGE_KEY.format(cam)] = img.permute(0, 3, 1, 2).float().div_(255.0)
        cuda = self.device.startswith("cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
            chunk = self.policy.predict_action_chunk(self.pre(obs))
            chunk = self.post(chunk)
        return chunk.float().cpu().numpy().astype(np.float32, copy=False)


def load_act_policy(checkpoint: Path, device: str) -> PolicyBackend:
    """Load an ACT checkpoint for inference; refuses (``ManifestError``) another schema.

    The schema check reads ``config.json`` before torch is imported, so a mismatched checkpoint
    is refused even where the ``act`` dependency group is not installed.
    """
    path, _ = read_policy_config(checkpoint)
    torch: Any = importlib.import_module("torch")
    act_mod: Any = importlib.import_module("lerobot.policies.act.modeling_act")
    factory: Any = importlib.import_module("lerobot.policies.factory")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"ACT device {device!r} requested but CUDA is not available")
    if device == "cpu":
        logger.warning("act: running the policy on the CPU (tests and debugging only)")
    policy: Any = act_mod.ACTPolicy.from_pretrained(str(path))
    policy.config.device = device
    policy.to(device)
    policy.eval()
    pre, post = factory.make_pre_post_processors(
        policy.config,
        pretrained_path=str(path),
        preprocessor_overrides={"device_processor": {"device": device}},
    )
    return _ActBackend(torch, policy, pre, post, device)
