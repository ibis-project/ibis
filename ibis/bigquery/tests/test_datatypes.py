import pytest

from multipledispatch.conflict import ambiguities

import ibis.expr.datatypes as dt
from ibis.bigquery.datatypes import (
    ibis_type_to_bigquery_type, UDFContext, TypeTranslationContext
)


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
            dt.Struct.from_tuples([
                ('a', dt.int64),
                ('b', dt.string),
                ('c', dt.Array(dt.string)),
            ]),
            'STRUCT<a INT64, b STRING, c ARRAY<STRING>>'
        ),
        (dt.date, 'DATE'),
        (dt.timestamp, 'TIMESTAMP'),
        pytest.mark.xfail(
            (dt.timestamp(timezone='US/Eastern'), 'TIMESTAMP'),
            raises=TypeError,
            reason='Not supported in BigQuery'
        ),
        ('array<struct<a: string>>', 'ARRAY<STRUCT<a STRING>>'),
        pytest.mark.xfail(
            (dt.Decimal(38, 9), 'NUMERIC'),
            raises=TypeError,
            reason='Not supported in BigQuery'
        ),
    ]
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
        pytest.mark.xfail((dt.int64, 'INT64'), raises=TypeError),
        pytest.mark.xfail(
            (dt.Array(dt.int64), 'ARRAY<INT64>'),
            raises=TypeError
        ),
        pytest.mark.xfail(
            (
                dt.Struct.from_tuples([('a', dt.Array(dt.int64))]),
                'STRUCT<a ARRAY<INT64>>'
            ),
            raises=TypeError,
        )
    ]
)
def test_ibis_type_to_bigquery_type_udf(type, expected):
    context = UDFContext()
    assert ibis_type_to_bigquery_type(type, context) == expected
