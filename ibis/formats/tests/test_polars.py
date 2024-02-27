from __future__ import annotations

import pytest
from pytest import param

import ibis
import ibis.expr.datatypes as dt

pl = pytest.importorskip("polars")

from ibis.formats.polars import PolarsData, PolarsSchema, PolarsType  # noqa: E402


@pytest.mark.parametrize(
    ("ibis_dtype", "polars_type"),
    [
        param(dt.bool, pl.Boolean, id="bool"),
        param(dt.null, pl.Null, id="null"),
        param(dt.Array(dt.string), pl.List(pl.Utf8), id="array_string"),
        param(dt.string, pl.Utf8, id="string"),
        param(dt.binary, pl.Binary, id="binary"),
        param(dt.date, pl.Date, id="date"),
        param(dt.time, pl.Time, id="time"),
        param(dt.int8, pl.Int8, id="int8"),
        param(dt.int16, pl.Int16, id="int16"),
        param(dt.int32, pl.Int32, id="int32"),
        param(dt.int64, pl.Int64, id="int64"),
        param(dt.uint8, pl.UInt8, id="uint8"),
        param(dt.uint16, pl.UInt16, id="uint16"),
        param(dt.uint32, pl.UInt32, id="uint32"),
        param(dt.uint64, pl.UInt64, id="uint64"),
        param(dt.float32, pl.Float32, id="float32"),
        param(dt.float64, pl.Float64, id="float64"),
        param(dt.timestamp, pl.Datetime("ns", time_zone=None), id="timestamp"),
        param(
            dt.Timestamp("UTC"), pl.Datetime("ns", time_zone="UTC"), id="timestamp_tz"
        ),
        param(dt.Interval(unit="ms"), pl.Duration("ms"), id="interval_ms"),
        param(dt.Interval(unit="us"), pl.Duration("us"), id="interval_us"),
        param(dt.Interval(unit="ns"), pl.Duration("ns"), id="interval_ns"),
        param(
            dt.Struct(
                dict(a=dt.string, b=dt.Array(dt.Array(dt.Struct(dict(c=dt.float64)))))
            ),
            pl.Struct(
                [
                    pl.Field("a", pl.Utf8),
                    pl.Field(
                        "b", pl.List(pl.List(pl.Struct([pl.Field("c", pl.Float64)])))
                    ),
                ]
            ),
            id="struct",
        ),
    ],
)
def test_to_from_ibis_type(ibis_dtype, polars_type):
    assert PolarsType.from_ibis(ibis_dtype) == polars_type
    assert PolarsType.to_ibis(polars_type) == ibis_dtype
    assert PolarsType.to_ibis(polars_type, nullable=False) == ibis_dtype(nullable=False)


def test_decimal():
    assert PolarsType.to_ibis(pl.Decimal()) == dt.Decimal(precision=None, scale=0)
    assert PolarsType.to_ibis(pl.Decimal(precision=6, scale=3)) == dt.Decimal(
        precision=6, scale=3
    )
    assert PolarsType.from_ibis(dt.Decimal()) == pl.Decimal(precision=None, scale=9)
    assert PolarsType.from_ibis(dt.Decimal(precision=6, scale=3)) == pl.Decimal(
        precision=6, scale=3
    )


def test_categorical():
    assert PolarsType.to_ibis(pl.Categorical()) == dt.string


def test_interval_unsupported_unit():
    typ = dt.Interval(unit="s")
    with pytest.raises(ValueError, match="Unsupported polars duration unit"):
        PolarsType.from_ibis(typ)


def test_map_unsupported():
    typ = dt.Map(dt.String(), dt.Int64())
    with pytest.raises(NotImplementedError, match="to polars is not supported"):
        PolarsType.from_ibis(typ)


def test_schema_to_and_from_ibis():
    polars_schema = {"x": pl.Int64, "y": pl.List(pl.Utf8)}
    ibis_schema = ibis.schema({"x": "int64", "y": "array<string>"})

    s1 = PolarsSchema.to_ibis(polars_schema)
    assert s1.equals(ibis_schema)

    s2 = PolarsSchema.from_ibis(ibis_schema)
    assert s2 == polars_schema


def test_infer_scalar():
    assert PolarsData.infer_scalar(1).is_integer()
    nested = PolarsData.infer_scalar([1])
    assert nested.is_array()
    assert nested.value_type.is_integer()


def test_infer_column():
    assert PolarsData.infer_column([1, 2, None]).is_integer()
    assert PolarsData.infer_column(["a", "b"]).is_string()


def test_infer_table():
    schema = PolarsData.infer_table({"x": [1, 2, None], "y": ["a", "b", "c"]})
    assert schema.names == ("x", "y")
    assert schema["x"].is_integer()
    assert schema["y"].is_string()


def test_convert_scalar():
    df = pl.DataFrame({"x": ["1"]})
    res = PolarsData.convert_scalar(df, dt.int64)
    assert res == 1
    assert isinstance(res, int)


def test_convert_column():
    df = pl.DataFrame({"x": ["1", "2"]})
    res = PolarsData.convert_column(df, dt.int64)
    sol = pl.Series(name="x", values=[1, 2], dtype=pl.Int64)
    assert res.equals(sol)
    assert res.dtype == sol.dtype


def test_convert_table():
    df = pl.DataFrame({"x": ["1", "2"], "y": ["a", "b"]})
    schema = ibis.schema({"x": "int64", "z": "string"})
    df = PolarsData.convert_table(df, schema)
    sol = pl.DataFrame(
        {"x": [1, 2], "z": ["a", "b"]}, schema={"x": pl.Int64, "z": pl.Utf8}
    )
    assert df.equals(sol)
    assert df.schema == sol.schema
