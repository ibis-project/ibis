from __future__ import annotations

import pytest

pytest.importorskip("databricks.sql")

from ibis.backends.databricks import MemtableManager


class _FakeBackend:
    current_catalog = "cat"
    current_database = "db"


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        ("first.last", "first_last"),
        ("first last", "first_last"),
        ("plain-user_1", "plain-user_1"),
    ],
)
def test_volume_path_is_a_single_name(monkeypatch, user, expected):
    monkeypatch.setattr("getpass.getuser", lambda: user)

    path = MemtableManager(_FakeBackend(), None).get_volume_path()

    prefix, volume = path.rsplit("/", 1)
    assert prefix == "/Volumes/cat/db"
    assert volume.startswith(f"{expected}-py=")
    assert "." not in volume
