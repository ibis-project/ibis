from __future__ import annotations

import uuid

import pytest
from pytest import param

import ibis


@pytest.mark.parametrize(
    "data, data_type",
    [param("123e4567-e89b-12d3-a456-426655440000", "uuid", id="uuid")],
)
@pytest.mark.usefixtures("con")
def test_special_strings(alltypes, data, data_type):
    lit = ibis.literal(data, type=data_type).name("tmp")
    expr = alltypes[[alltypes.id, lit]].head(1)
    df = expr.execute()
    assert df["tmp"].iloc[0] == uuid.UUID(data)
