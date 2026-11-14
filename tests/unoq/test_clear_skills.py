"""T-230: ``clear_skills`` forgets a SKU's skills and stall prototypes on the board."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from teleop.atc.types import SkillPrimitive, StallSignature
from teleop.unoq import protocol as P
from teleop.unoq.client import UnoQClient, UnoQError
from teleop.unoq.server import UnoQServer
from teleop.unoq.switch import SwitchableUnoQ

from .test_edge_dmp import JAM_SIG_1, synthetic_rescue

SKU = "pack-a"
OTHER = "pack-b"


@pytest.fixture
def server() -> Iterator[UnoQServer]:
    srv = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    try:
        yield srv
    finally:
        srv.stop()


@pytest.fixture
def client(server: UnoQServer) -> Iterator[UnoQClient]:
    c = UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect()
    try:
        yield c
    finally:
        c.close()


def _fit(client: UnoQClient, sku: str) -> SkillPrimitive:
    demo = synthetic_rescue()
    return client.fit_dmp(
        sku, StallSignature(*JAM_SIG_1), demo["t"], demo["wrist"], demo["fingers"], demo["tau"]
    )


def test_clear_skills_forgets_skills_and_prototypes(client: UnoQClient) -> None:
    skill = _fit(client, SKU)
    client.register_skill(skill)
    assert not client.classify_stall(SKU, skill.prototype)[0].is_ood
    assert client.clear_skills(SKU) == 1
    match, skill_id = client.classify_stall(SKU, skill.prototype)
    assert match.is_ood and skill_id is None


def test_clear_skills_resets_signature_numbering(client: UnoQClient) -> None:
    client.register_skill(_fit(client, SKU))
    client.clear_skills(SKU)
    assert _fit(client, SKU).signature_id == "Jam_Sig_1"


def test_clear_skills_leaves_other_skus(client: UnoQClient, server: UnoQServer) -> None:
    keep = _fit(client, OTHER)
    client.register_skill(keep)
    client.register_skill(_fit(client, SKU))
    assert client.clear_skills(SKU) == 1
    assert set(server.skills) == {keep.skill_id}
    match, skill_id = client.classify_stall(OTHER, keep.prototype)
    assert not match.is_ood and skill_id == keep.skill_id


def test_clear_skills_with_nothing_registered_returns_zero(client: UnoQClient) -> None:
    assert client.clear_skills(SKU) == 0


def test_switch_clear_empties_replay_list(client: UnoQClient, server: UnoQServer) -> None:
    skill_a = _fit(client, SKU)
    skill_b = _fit(client, OTHER)
    sw = SwitchableUnoQ(UnoQClient("127.0.0.1", server.udp_port, server.tcp_port).connect())
    other = UnoQServer(host="127.0.0.1", udp_port=0, tcp_port=0).start()
    try:
        sw.register_skill(skill_a)
        sw.register_skill(skill_b)
        assert sw.clear_skills(SKU) == 1
        sw.swap(UnoQClient("127.0.0.1", other.udp_port, other.tcp_port).connect())
        assert set(other.skills) == {skill_b.skill_id}
    finally:
        sw.close()
        other.stop()


def test_old_firmware_raises_unoq_error(client: UnoQClient, server: UnoQServer) -> None:
    del server._handlers[P.OP_CLEAR_SKILLS]  # firmware that predates the op
    with pytest.raises(UnoQError, match="unknown op"):
        client.clear_skills(SKU)
