import pytest
from multipledispatch.conflict import ambiguities
from pytest import param

import ibis.expr.datatypes as dt

from .datatypes import (
    TypeTranslationContext,
    UDFContext,
    ibis_type_to_bigquery_type,
)

pytestmark = pytest.mark.bigquery


def test_no_ambiguities():
    ambs = ambiguities(ibis_type_to_bigquery_type.funcs)
    assert not ambs


@pytest.mark.parametrize(
    ('datatype', 'expected'),
    [
        (dt.float32, 'FLOAT64'),
        (dt.float64, 'FLOAT64'),
        (dt.uint8, 'INT64'),
        (dt.uint16, 'INT64'),
        (dt.uint32, 'INT64'),
        (dt.int8, 'INT64'),
        (dt.int16, 'INT64'),
        (dt.int32, 'INT64'),
        (dt.int64, 'INT64'),
        (dt.string, 'STRING'),
        (dt.Array(dt.int64), 'ARRAY<INT64>'),
        (dt.Array(dt.string), 'ARRAY<STRING>'),
        (
            dt.Struct.from_tuples(
                [('a', dt.int64), ('b', dt.string), ('c', dt.Array(dt.string))]
            ),
            'STRUCT<a INT64, b STRING, c ARRAY<STRING>>',
        ),
        (dt.date, 'DATE'),
        (dt.timestamp, 'TIMESTAMP'),
        param(
            dt.Timestamp(timezone='US/Eastern'),
            'TIMESTAMP',
            marks=pytest.mark.xfail(
                raises=TypeError, reason='Not supported in BigQuery'
            ),
        ),
        ('array<struct<a: string>>', 'ARRAY<STRUCT<a STRING>>'),
        param(
            dt.Decimal(38, 9),
            'NUMERIC',
            marks=pytest.mark.xfail(
                raises=TypeError, reason='Not supported in BigQuery'
            ),
        ),
    ],
)
def test_simple(datatype, expected):
    context = TypeTranslationContext()
    assert ibis_type_to_bigquery_type(datatype, context) == expected


@pytest.mark.parametrize('datatype', [dt.uint64, dt.Decimal(8, 3)])
def test_simple_failure_mode(datatype):
    with pytest.raises(TypeError):
        ibis_type_to_bigquery_type(datatype)


@pytest.mark.parametrize(
    ('type', 'expected'),
    [
        param(dt.int64, 'INT64', marks=pytest.mark.xfail(raises=TypeError)),
        param(
            dt.Array(dt.int64),
            'ARRAY<INT64>',
            marks=pytest.mark.xfail(raises=TypeError),
        ),
        param(
            dt.Struct.from_tuples([('a', dt.Array(dt.int64))]),
            'STRUCT<a ARRAY<INT64>>',
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_ibis_type_to_bigquery_type_udf(type, expected):
    context = UDFContext()
    assert ibis_type_to_bigquery_type(type, context) == expected
