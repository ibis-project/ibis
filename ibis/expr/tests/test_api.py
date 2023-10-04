from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest
from dateutil.tz import tzoffset, tzutc
from pytest import param

import ibis
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch
from ibis import _
from ibis.common.exceptions import IntegrityError


def test_schema_from_names_types():
    s = ibis.schema(names=["a"], types=["array<float64>"])
    assert s == sch.Schema(dict(a="array<float64>"))


def test_schema_from_names_and_types_length_must_match():
    msg = "Schema names and types must have the same length"
    with pytest.raises(ValueError, match=msg):
        ibis.schema(names=["a", "b"], types=["int", "str", "float"])

    schema = ibis.schema(names=["a", "b"], types=["int", "str"])

    assert isinstance(schema, sch.Schema)
    assert schema.names == ("a", "b")
    assert schema.types == (dt.int64, dt.string)


def test_schema_from_names_and_typesield_names():
    msg = "Duplicate column name"
    with pytest.raises(IntegrityError, match=msg):
        ibis.schema(names=["a", "a"], types=["int", "str"])


@pytest.mark.parametrize(
    ("string", "expected_value", "expected_timezone"),
    [
        param(
            "2015-01-01 12:34:56.789",
            datetime(2015, 1, 1, 12, 34, 56, 789000),
            None,
            id="from_string_millis",
        ),
        param(
            "2015-01-01 12:34:56.789321",
            datetime(2015, 1, 1, 12, 34, 56, 789321),
            None,
            id="from_string_micros",
        ),
        param(
            "2015-01-01 12:34:56.789 UTC",
            datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzutc()),
            "UTC",
            id="from_string_millis_utc",
        ),
        param(
            "2015-01-01 12:34:56.789321 UTC",
            datetime(2015, 1, 1, 12, 34, 56, 789321, tzinfo=tzutc()),
            "UTC",
            id="from_string_micros_utc",
        ),
        param(
            "2015-01-01 12:34:56.789+00:00",
            datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzutc()),
            "UTC",
            id="from_string_millis_utc_offset",
        ),
        param(
            "2015-01-01 12:34:56.789+01:00",
            datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzoffset(None, 3600)),
            "UTC+01:00",
            id="from_string_millis_utc_+1_offset",
        ),
        param(
            pd.Timestamp("2015-01-01 12:34:56.789"),
            datetime(2015, 1, 1, 12, 34, 56, 789000),
            None,
            id="from_pandas_millis",
        ),
        param(
            pd.Timestamp("2015-01-01 12:34:56.789", tz="UTC"),
            datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzutc()),
            "UTC",
            id="from_pandas_millis_utc",
        ),
        param(
            pd.Timestamp("2015-01-01 12:34:56.789+03:00"),
            datetime(2015, 1, 1, 12, 34, 56, 789000, tzinfo=tzoffset(None, 10800)),
            "UTC+03:00",
            id="from_pandas_millis_+3_offset",
        ),
    ],
)
def test_timestamp(string, expected_value, expected_timezone):
    expr = ibis.timestamp(string)
    op = expr.op()
    assert isinstance(expr, ibis.expr.types.TimestampScalar)
    assert op.value == expected_value
    assert op.dtype == dt.Timestamp(timezone=expected_timezone)


@pytest.mark.parametrize(
    "f, sol",
    [
        (lambda t: _.x + t.a, "(_.x + <column[int64]>)"),
        (lambda t: _.x + t.a.sum(), "(_.x + <scalar[int64]>)"),
        (lambda t: ibis.date(_.x, 2, t.a), "date(_.x, 2, <column[int64]>)"),
    ],
)
def test_repr_deferred_with_exprs(f, sol):
    t = ibis.table({"a": "int64"})
    expr = f(t)
    res = repr(expr)
    assert res == sol
