from __future__ import annotations

import getpass
import sys
from unittest.mock import MagicMock

import pytest

# Ensure databricks is mocked if not installed
if "databricks" not in sys.modules:
    sys.modules["databricks"] = MagicMock()
    sys.modules["databricks.sql"] = MagicMock()

from ibis.backends.databricks import MemtableManager


@pytest.mark.parametrize(
    ("username", "expected_user"),
    [
        ("first.last", "first_last"),
        ("john.doe.jr", "john_doe_jr"),
        ("user with space", "user_with_space"),
        ("user/slash", "user_slash"),
        ("valid-user_123", "valid-user_123"),
    ],
)
def test_generate_volume_path_sanitizes_username(monkeypatch, username, expected_user):
    backend = MagicMock()
    backend.current_catalog = "my_catalog"
    backend.current_database = "my_schema"

    monkeypatch.setattr(getpass, "getuser", lambda: username)

    manager = MemtableManager(backend, volume_path=None)
    volume_path = manager._generate_volume_path()

    assert volume_path.startswith("/Volumes/my_catalog/my_schema/")
    volume_name = volume_path.rsplit("/", 1)[-1]
    assert volume_name.startswith(expected_user)
    assert "." not in volume_name.split("-py=")[0]
