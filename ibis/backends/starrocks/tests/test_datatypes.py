from __future__ import annotations

import ibis.expr.datatypes as dt
from ibis.backends.sql.datatypes import StarRocksType


def test_starrocks_type_mapper_uses_starrocks_dialect():
    assert StarRocksType.dialect == "starrocks"


def test_largeint_maps_to_decimal():
    assert StarRocksType.from_string("largeint") == dt.Decimal(38, 0)


def test_string_generates_starrocks_string_type():
    assert StarRocksType.to_string(dt.string) == "STRING"
