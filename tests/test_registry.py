import re

import pytest

from teleop import registry


def test_resolve_unknown_lists_valid() -> None:
    with pytest.raises(KeyError, match="synthetic"):
        registry.resolve(registry.SOURCES, "nope")


def test_registry_entries_are_well_formed() -> None:
    tables = [
        registry.SOURCES,
        registry.WRISTS,
        registry.RETARGETERS,
        registry.TASKS,
        registry.SINKS,
    ]
    for table in tables:
        for value in table.values():
            assert re.match(r"^[\w.]+:\w+$", value)
