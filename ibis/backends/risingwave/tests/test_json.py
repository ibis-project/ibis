"""Tests for json data types."""
from __future__ import annotations

import json

import pytest
from pytest import param

import ibis


@pytest.mark.parametrize("data", [param({"status": True}, id="status")])
def test_json(data, alltypes):
    lit = ibis.literal(json.dumps(data), type="json").name("tmp")
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df["tmp"].iloc[0] == data
