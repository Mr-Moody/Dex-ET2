"""Hot-swappable UNO Q client: board <-> in-process loopback node at runtime (booth health, §3).

The coordinator calls :class:`SwitchableUnoQ` synchronously from the 60 Hz loop, so the pointer
lock only covers the current client and the request in flight on it. Skill replay to a new node
runs on the swapping thread, against the new client, before it is published: a loop-thread
request never waits behind a replay or a reconnect.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar

from numpy.typing import ArrayLike

from teleop.atc.types import ConsoleInputState, SkillPrimitive, StallMatch, StallSignature
from teleop.types import Pose
from teleop.unoq.client import UnoQClient, UnoQTransportError
from teleop.unoq.server import UnoQServer

logger = logging.getLogger(__name__)

T = TypeVar("T")


def start_loopback() -> UnoQClient:
    """A fresh in-process loopback node on 127.0.0.1 (ephemeral ports), already connected."""
    server = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0, name="unoq-loopback").start()
    client = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port, local_server=server)
    try:
        return client.connect()
    except OSError:
        server.stop()
        raise


class SwitchableUnoQ:
    """Forwards to the current :class:`UnoQClient`; :meth:`swap` replaces it at runtime.

    Remembers every successful ``register_skill`` in order and replays them on each new node.
    A board request failing in transport (``OSError``/``UnoQTransportError``; an error *reply*
    propagates unchanged) swaps to a fresh loopback node and is
    retried there once.
    """

    def __init__(
        self,
        client: UnoQClient,
        loopback: Callable[[], UnoQClient] = start_loopback,
    ) -> None:
        self._client = client
        self._loopback = loopback
        self._lock = threading.Lock()  # the client pointer and the request in flight on it
        self._swap_lock = threading.Lock()  # one swap at a time
        self._intent_lock = threading.Lock()
        self._loopback_intents = 0  # threads inside _to_loopback (holding or awaiting the swap)
        self._skills: list[SkillPrimitive] = []
        self._closed = False
        self.on_swap: list[Callable[[SwitchableUnoQ], None]] = []

    # ------------------------------------------------------------------ state

    @property
    def current(self) -> UnoQClient:
        return self._client

    @property
    def host(self) -> str:
        return self._client.host

    @property
    def is_mock(self) -> bool:
        return self._client.is_mock

    # ------------------------------------------------------------------ swapping

    def swap(self, new: UnoQClient) -> None:
        """Replay the remembered skills on ``new``, publish it and close the old client."""
        with self._swap_lock:
            old = self._publish(new)
        if old is not None:
            self._retire(old)

    def swap_to_loopback(self) -> None:
        """Swap to a fresh loopback node unless already on one."""
        old = self._to_loopback(expected=None)
        if old is not None:
            self._retire(old)

    def _to_loopback(self, expected: UnoQClient | None) -> UnoQClient | None:
        """Swap to loopback (if current is still ``expected``); returns the old client.

        With ``expected`` (the request path) it waits only behind another swap *to loopback*
        (a local server start and a local replay, so cheap) and then sees ``expected`` replaced.
        It never waits behind a swap to a board (network replay; only an explicit board-to-board
        ``swap()`` can be in flight while ``expected`` is current): it returns at once and the
        caller retries on whatever is current.
        """
        if expected is not None and self._client is not expected:
            return None
        with self._intent_lock:
            self._loopback_intents += 1
        try:
            if expected is not None and not self._swap_lock.acquire(blocking=False):
                with self._intent_lock:
                    others = self._loopback_intents - 1
                if others == 0:
                    return None  # the swap in flight goes to a board and replaces ``expected``
                self._swap_lock.acquire()  # another detector is swapping to loopback
            elif expected is None:
                self._swap_lock.acquire()
            try:
                current = self._client
                if current.is_mock or (expected is not None and current is not expected):
                    return None
                logger.warning("unoq: board lost; switching to the local loopback node")
                return self._publish(self._loopback())
            finally:
                self._swap_lock.release()
        finally:
            with self._intent_lock:
                self._loopback_intents -= 1

    def _publish(self, new: UnoQClient) -> UnoQClient | None:
        """Replay skills on ``new`` outside the pointer lock, then swap; caller holds swap lock.

        Returns the replaced client, or None when the switch was closed meanwhile.
        """
        replayed = 0
        while True:
            with self._lock:
                pending = self._skills[replayed:]
                if not pending and not self._closed:
                    old, self._client = self._client, new
                    return old
            if self._closed:
                new.close()
                return None
            try:
                for skill in pending:
                    new.register_skill(skill)
            except Exception:
                new.close()
                raise
            replayed += len(pending)

    def _retire(self, old: UnoQClient) -> None:
        """Close the replaced client and notify listeners (on the swapping thread)."""
        try:
            old.close()
        except Exception:
            logger.exception("unoq: closing the replaced client failed")
        logger.info("unoq: now using %s", "local loopback" if self.is_mock else self.host)
        for callback in list(self.on_swap):
            try:
                callback(self)
            except Exception:
                logger.exception("unoq: on_swap callback failed")

    # ------------------------------------------------------------------ requests

    def _call(self, op: Callable[[UnoQClient], T]) -> T:
        """Run ``op`` on the current client; on a board failure swap to loopback and retry once."""
        with self._lock:
            client = self._client
            try:
                return op(client)
            except (OSError, UnoQTransportError) as exc:
                if client.is_mock:
                    raise
                logger.warning("unoq: board request failed (%s); retrying on loopback", exc)
        old = self._to_loopback(expected=client)
        if old is not None:
            # Listener work (closing pollers, rebuilding consoles) stays off the calling thread.
            notify = threading.Thread(target=self._retire, args=(old,), name="unoq-swap")
            notify.daemon = True
            notify.start()
        with self._lock:
            return op(self._client)

    def classify_stall(
        self, part_sku_id: str, z: StallSignature | Sequence[float]
    ) -> tuple[StallMatch, str | None]:
        return self._call(lambda c: c.classify_stall(part_sku_id, z))

    def fit_dmp(
        self,
        part_sku_id: str,
        z: StallSignature | Sequence[float],
        timestamps_s: ArrayLike,
        wrist_latch_se3: ArrayLike,
        fingers_q: ArrayLike,
        tau_ext_inf: ArrayLike,
        signature_id: str | None = None,
    ) -> SkillPrimitive:
        return self._call(
            lambda c: c.fit_dmp(
                part_sku_id, z, timestamps_s, wrist_latch_se3, fingers_q, tau_ext_inf, signature_id
            )
        )

    def register_skill(self, skill: SkillPrimitive) -> int:
        def op(c: UnoQClient) -> int:
            n = c.register_skill(skill)
            self._skills.append(skill)  # under the pointer lock: a concurrent swap replays it
            return n

        return self._call(op)

    def filter_step(
        self,
        wrist: Pose | None,
        joints: Mapping[str, float] | None = None,
        t_ns: int | None = None,
    ) -> tuple[Pose | None, dict[str, float]]:
        with self._lock:
            return self._client.filter_step(wrist, joints, t_ns)

    def console_state(self) -> ConsoleInputState:
        with self._lock:
            return self._client.console_state()

    def ping(self) -> str:
        with self._lock:
            return self._client.ping()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            client = self._client
        client.close()
