# Copyright 2014 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

import pandas as pd
from datetime import datetime

import ibis
import ibis.expr.api as api
import ibis.expr.types as ir
import ibis.expr.rules as rlz
import ibis.expr.operations as ops

from ibis.compat import PY2


def test_field_select(alltypes):
    assert isinstance(alltypes.i, ir.TimestampColumn)


def test_string_cast_to_timestamp(alltypes):
    casted = alltypes.g.cast('timestamp')
    assert isinstance(casted, ir.TimestampColumn)

    string = api.literal('2000-01-01')
    casted = string.cast('timestamp')
    assert isinstance(casted, ir.TimestampScalar)


@pytest.mark.parametrize(
    ('field', 'expected_operation', 'expected_type'),
    [
        ('year', ops.ExtractYear, ir.IntegerColumn),
        ('month', ops.ExtractMonth, ir.IntegerColumn),
        ('day', ops.ExtractDay, ir.IntegerColumn),
        ('hour', ops.ExtractHour, ir.IntegerColumn),
        ('minute', ops.ExtractMinute, ir.IntegerColumn),
        ('second', ops.ExtractSecond, ir.IntegerColumn),
        ('millisecond', ops.ExtractMillisecond, ir.IntegerColumn),
    ]
)
def test_extract_fields(field, expected_operation, expected_type, alltypes):
    # type-size may be database specific
    result = getattr(alltypes.i, field)()
    assert result.get_name() == field
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


def test_now():
    result = api.now()
    assert isinstance(result, ir.TimestampScalar)
    assert isinstance(result.op(), ops.TimestampNow)


@pytest.mark.parametrize(
    ('function', 'value'),
    [
        (ibis.timestamp, '2015-01-01 00:00:00'),
        (ibis.literal, pd.Timestamp('2015-01-01 00:00:00')),
    ]
)
def test_timestamp_literals(function, value):
    expr = function(value)
    assert isinstance(expr, ir.TimestampScalar)


def test_invalid_timestamp_literal():
    with pytest.raises(ValueError):
        ibis.timestamp('2015-01-01 00:71')


@pytest.mark.xfail(raises=AssertionError, reason='NYT')
def test_integer_to_timestamp():
    # #246
    assert False


def test_comparison_timestamp(alltypes):
    expr = alltypes.i > (alltypes.i.min() + ibis.day(3))
    assert isinstance(expr, ir.BooleanColumn)


def test_comparisons_string(alltypes):
    val = '2015-01-01 00:00:00'
    expr = alltypes.i > val
    op = expr.op()
    assert isinstance(op.right, ir.TimestampScalar)

    expr2 = val < alltypes.i
    op = expr2.op()
    assert isinstance(op, ops.Greater)
    assert isinstance(op.right, ir.TimestampScalar)


def test_comparisons_pandas_timestamp(alltypes):
    val = pd.Timestamp('2015-01-01 00:00:00')
    expr = alltypes.i > val
    op = expr.op()
    assert isinstance(op.right, ir.TimestampScalar)


@pytest.mark.xfail(raises=TypeError, reason='Upstream pandas bug')
def test_greater_comparison_pandas_timestamp(alltypes):
    val = pd.Timestamp('2015-01-01 00:00:00')
    expr2 = val < alltypes.i
    op = expr2.op()
    assert isinstance(op, ops.Greater)
    assert isinstance(op.right, ir.TimestampScalar)


def test_timestamp_precedence():
    ts = ibis.literal(datetime.now())
    highest_type = rlz.highest_precedence_dtype([ibis.NA, ts])
    assert highest_type == 'timestamp'


@pytest.mark.parametrize(
    ('field', 'expected_operation', 'expected_type'),
    [
        ('year', ops.ExtractYear, ir.IntegerColumn),
        ('month', ops.ExtractMonth, ir.IntegerColumn),
        ('day', ops.ExtractDay, ir.IntegerColumn),
    ]
)
def test_timestamp_field_access_on_date(
    field, expected_operation, expected_type, alltypes
):
    date_col = alltypes.i.date()
    result = getattr(date_col, field)()
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


@pytest.mark.parametrize(
    ('field', 'expected_operation', 'expected_type'),
    [
        ('hour', ops.ExtractHour, ir.IntegerColumn),
        ('minute', ops.ExtractMinute, ir.IntegerColumn),
        ('second', ops.ExtractSecond, ir.IntegerColumn),
        ('millisecond', ops.ExtractMillisecond, ir.IntegerColumn),
    ]
)
def test_timestamp_field_access_on_date_failure(
    field, expected_operation, expected_type, alltypes
):
    time_col = alltypes.i.date()
    with pytest.raises(AttributeError):
        getattr(time_col, field)


@pytest.mark.parametrize(
    ('field', 'expected_operation', 'expected_type'),
    [
        ('hour', ops.ExtractHour, ir.IntegerColumn),
        ('minute', ops.ExtractMinute, ir.IntegerColumn),
        ('second', ops.ExtractSecond, ir.IntegerColumn),
        ('millisecond', ops.ExtractMillisecond, ir.IntegerColumn),
    ]
)
@pytest.mark.skipif(PY2, reason='Time type is not supported on python 2')
def test_timestamp_field_access_on_time(
    field, expected_operation, expected_type, alltypes
):
    time_col = alltypes.i.time()
    result = getattr(time_col, field)()
    assert isinstance(result, expected_type)
    assert isinstance(result.op(), expected_operation)


@pytest.mark.parametrize(
    ('field', 'expected_operation', 'expected_type'),
    [
        ('year', ops.ExtractYear, ir.IntegerColumn),
        ('month', ops.ExtractMonth, ir.IntegerColumn),
        ('day', ops.ExtractDay, ir.IntegerColumn),
    ]
)
@pytest.mark.skipif(PY2, reason='Time type is not supported on python 2')
def test_timestamp_field_access_on_time_failure(
    field, expected_operation, expected_type, alltypes
):
    date_col = alltypes.i.time()
    with pytest.raises(AttributeError):
        getattr(date_col, field)
