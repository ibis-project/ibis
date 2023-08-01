from __future__ import annotations

import pytest

from ibis.common.annotations import ValidationError
from ibis.config import options


def test_sql_config(monkeypatch):
    assert options.sql.default_limit is None

    with pytest.raises(ValidationError):
        options.sql.default_limit = -1

    monkeypatch.setattr(options.sql, "default_limit", 100)
    assert options.sql.default_limit == 100
