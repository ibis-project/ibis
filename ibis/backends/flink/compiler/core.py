"""Flink ibis expression to SQL string compiler."""

from __future__ import annotations

import datetime
import functools
import math

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import Compiler, Select, SelectBuilder
from ibis.backends.flink.translator import FlinkExprTranslator
from ibis.backends.flink.utils import (
    DaysToSecondsInterval,
    YearsToMonthsInterval,
    format_precision,
)


class FlinkSelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class FlinkSelect(Select):
    def format_group_by(self) -> str:
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            group_keys = map(self._translate, self.group_by)
            clause = 'GROUP BY {}'.format(', '.join(list(group_keys)))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)


class FlinkCompiler(Compiler):
    select_builder_class = FlinkSelectBuilder
    select_class = FlinkSelect
    cheap_in_memory_tables = True
    translator_class = FlinkExprTranslator


def translate(op: ops.TableNode) -> str:
    # TODO(chloeh13q): support translation of non-select exprs (e.g. literals)
    return translate_op(op)


@functools.singledispatch
def translate_op(op: ops.TableNode) -> str:
    raise com.OperationNotDefinedError(f'No translation rule for {type(op)}')


@translate_op.register(ops.Literal)
def _literal(op: ops.Literal) -> str:
    value = op.value
    dtype = op.dtype

    if dtype.is_boolean():
        # TODO(chloeh13q): Flink supports a third boolean called "UNKNOWN"
        return 'TRUE' if value else 'FALSE'
    elif dtype.is_string():
        quoted = value.replace("'", "''").replace("\\", "\\\\")
        return f"'{quoted}'"
    elif dtype.is_date():
        if isinstance(value, datetime.date):
            value = value.strftime('%Y-%m-%d')
        return repr(value)
    elif dtype.is_numeric():
        if math.isnan(value):
            raise ValueError("NaN is not supported in Flink SQL")
        elif math.isinf(value):
            raise ValueError("Infinity is not supported in Flink SQL")
        return repr(value)
    elif dtype.is_timestamp():
        # TODO(chloeh13q): support timestamp with local timezone
        if isinstance(value, datetime.datetime):
            fmt = '%Y-%m-%d %H:%M:%S'
            # datetime.datetime only supports resolution up to microseconds, even
            # though Flink supports fractional precision up to 9 digits. We will
            # need to use numpy or pandas datetime types for higher resolutions.
            if value.microsecond:
                fmt += '.%f'
            return 'TIMESTAMP ' + repr(value.strftime(fmt))
        raise NotImplementedError(f'No translation rule for timestamp {value}')
    elif dtype.is_time():
        return f"TIME '{value}'"
    elif dtype.is_interval():
        return f"INTERVAL {translate_interval(value, dtype)}"
    raise NotImplementedError(f'No translation rule for {dtype}')


@translate_op.register(ops.Selection)
@translate_op.register(ops.Aggregation)
@translate_op.register(ops.Limit)
def _(op: ops.Selection | ops.Aggregation | ops.Limit) -> str:
    return FlinkCompiler.to_sql(op)  # to_sql uses to_ast, which builds a select tree


def translate_interval(value, dtype):
    """Convert interval to Flink SQL type.

    Flink supports only two types of temporal intervals: day-time intervals with up to nanosecond
    granularity or year-month intervals with up to month granularity.

    An interval of year-month consists of +years-months with values ranging from -9999-11 to +9999-11.
    An interval of day-time consists of +days hours:minutes:seconds.fractional with values ranging from
    -999999 23:59:59.999999999 to +999999 23:59:59.999999999.

    The value representation is the same for all types of resolutions.

    For example, an interval of months of 50 is always represented in an interval-of-years-to-months
    format (with default year precision): +04-02; an interval of seconds of 70 is always represented in
    an interval-of-days-to-seconds format (with default precisions): +00 00:01:10.000000.
    """
    if dtype.unit in YearsToMonthsInterval.units:
        interval = YearsToMonthsInterval(value, dtype.unit.value)
    else:
        interval = DaysToSecondsInterval(value, dtype.unit.value)

    interval_segments = interval.interval_segments
    nonzero_interval_segments = {k: v for k, v in interval_segments.items() if v != 0}

    # YEAR, MONTH, DAY, HOUR, MINUTE, SECOND
    if len(nonzero_interval_segments) == 1:
        unit = next(iter(nonzero_interval_segments))
        value = nonzero_interval_segments[unit]
        return f"'{value}' {unit.value}{format_precision(value, unit)}"

    # YEAR TO MONTH, DAY TO SECOND
    return interval.format_as_string()
