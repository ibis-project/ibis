from __future__ import annotations

import contextlib
import functools
import operator
import string

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.base.sql.alchemy.geospatial import geospatial_supported
from ibis.backends.base.sql.alchemy.registry import (
    geospatial_functions,
)

operation_registry = sqlalchemy_operation_registry.copy()

# NOTE: window functions are available from MySQL 8 and MariaDB 10.2
operation_registry.update(sqlalchemy_window_functions_registry)

if geospatial_supported:
    operation_registry.update(geospatial_functions)

_truncate_formats = {
    "s": "%Y-%m-%d %H:%i:%s",
    "m": "%Y-%m-%d %H:%i:00",
    "h": "%Y-%m-%d %H:00:00",
    "D": "%Y-%m-%d",
    # 'W': 'week',
    "M": "%Y-%m-01",
    "Y": "%Y-01-01",
}


def _truncate(t, op):
    sa_arg = t.translate(op.arg)
    try:
        fmt = _truncate_formats[op.unit.short]
    except KeyError:
        raise com.UnsupportedOperationError(f"Unsupported truncate unit {op.unit}")
    return sa.func.date_format(sa_arg, fmt)


def _round(t, op):
    sa_arg = t.translate(op.arg)

    if op.digits is None:
        sa_digits = 0
    else:
        sa_digits = t.translate(op.digits)

    return sa.func.round(sa_arg, sa_digits)


def _interval_from_integer(t, op):
    if op.unit.short in {"ms", "ns"}:
        raise com.UnsupportedOperationError(
            f"MySQL does not allow operation with INTERVAL offset {op.unit}"
        )

    sa_arg = t.translate(op.arg)
    text_unit = op.dtype.resolution.upper()

    # XXX: Is there a better way to handle this? I.e. can we somehow use
    # the existing bind parameter produced by translate and reuse its name in
    # the string passed to sa.text?
    if isinstance(sa_arg, sa.sql.elements.BindParameter):
        return sa.text(f"INTERVAL :arg {text_unit}").bindparams(arg=sa_arg.value)
    return sa.text(f"INTERVAL {sa_arg} {text_unit}")


def _literal(_, op):
    dtype = op.dtype
    value = op.value
    if value is None:
        return sa.null()
    if dtype.is_interval():
        if dtype.unit.short in {"ms", "ns"}:
            raise com.UnsupportedOperationError(
                f"MySQL does not allow operation with INTERVAL offset {dtype.unit}"
            )
        text_unit = dtype.resolution.upper()
        sa_text = sa.text(f"INTERVAL :value {text_unit}")
        return sa_text.bindparams(value=value)
    elif dtype.is_binary():
        # the cast to BINARY is necessary here, otherwise the data come back as
        # Python strings
        #
        # This lets the database handle encoding rather than ibis
        return sa.cast(sa.literal(value), type_=sa.BINARY())
    elif dtype.is_time():
        return sa.func.maketime(
            value.hour, value.minute, value.second + value.microsecond / 1e6
        )
    else:
        with contextlib.suppress(AttributeError):
            value = value.to_pydatetime()

        return sa.literal(value)


def _group_concat(t, op):
    if op.where is not None:
        arg = t.translate(ops.IfElse(op.where, op.arg, ibis.NA))
    else:
        arg = t.translate(op.arg)
    sep = t.translate(op.sep)
    return sa.func.group_concat(arg.op("SEPARATOR")(sep))


def _json_get_item(t, op):
    arg = t.translate(op.arg)
    index = t.translate(op.index)
    if op.index.dtype.is_integer():
        path = "$[" + sa.cast(index, sa.TEXT) + "]"
    else:
        path = "$." + index
    return sa.func.json_extract(arg, path)


def _regex_extract(arg, pattern, index):
    return sa.func.IF(
        arg.op("REGEXP")(pattern),
        sa.func.IF(
            index == 0,
            sa.func.REGEXP_SUBSTR(arg, pattern),
            sa.func.REGEXP_REPLACE(
                sa.func.REGEXP_SUBSTR(arg, pattern), pattern, rf"\{index.value}"
            ),
        ),
        None,
    )


def _string_find(t, op):
    arg = t.translate(op.arg)
    substr = t.translate(op.substr)

    if op_start := op.start:
        start = t.translate(op_start)
        return sa.func.locate(substr, arg, start) - 1

    return sa.func.locate(substr, arg) - 1


class _mysql_trim(GenericFunction):
    inherit_cache = True

    def __init__(self, input, side: str) -> None:
        super().__init__(input)
        self.type = sa.VARCHAR()
        self.side = side


@compiles(_mysql_trim, "mysql")
def compiles_mysql_trim(element, compiler, **kw):
    arg = compiler.function_argspec(element, **kw)
    side = element.side.upper()
    # has to be called once for every whitespace character because mysql
    # interprets `char` literally, not as a set of characters like Python
    return functools.reduce(
        lambda arg, char: f"TRIM({side} '{char}' FROM {arg})", string.whitespace, arg
    )


def _temporal_delta(t, op):
    left = t.translate(op.left)
    right = t.translate(op.right)
    part = sa.literal_column(op.part.value.upper())
    return sa.func.timestampdiff(part, right, left)


operation_registry.update(
    {
        ops.Literal: _literal,
        # static checks are not happy with using "if" as a property
        ops.IfElse: fixed_arity(getattr(sa.func, "if"), 3),
        # strings
        ops.StringFind: _string_find,
        ops.FindInSet: (
            lambda t, op: (
                sa.func.find_in_set(
                    t.translate(op.needle),
                    sa.func.concat_ws(",", *map(t.translate, op.values)),
                )
                - 1
            )
        ),
        # LIKE in mysql is case insensitive
        ops.StartsWith: fixed_arity(
            lambda arg, start: sa.type_coerce(
                arg.op("LIKE BINARY")(sa.func.concat(start, "%")), sa.BOOLEAN()
            ),
            2,
        ),
        ops.EndsWith: fixed_arity(
            lambda arg, end: sa.type_coerce(
                arg.op("LIKE BINARY")(sa.func.concat("%", end)), sa.BOOLEAN()
            ),
            2,
        ),
        ops.RegexSearch: fixed_arity(
            lambda x, y: sa.type_coerce(x.op("REGEXP")(y), sa.BOOLEAN()), 2
        ),
        ops.RegexExtract: fixed_arity(_regex_extract, 3),
        # math
        ops.Log: fixed_arity(lambda arg, base: sa.func.log(base, arg), 2),
        ops.Log2: unary(sa.func.log2),
        ops.Log10: unary(sa.func.log10),
        ops.Round: _round,
        # dates and times
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(sa.func.datediff, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: fixed_arity(
            lambda left, right: sa.func.timestampdiff(sa.text("SECOND"), right, left), 2
        ),
        ops.StringToTimestamp: fixed_arity(
            lambda arg, format_str: sa.func.str_to_date(arg, format_str), 2
        ),
        ops.DateTruncate: _truncate,
        ops.TimestampTruncate: _truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.Strftime: fixed_arity(sa.func.date_format, 2),
        ops.ExtractDayOfYear: unary(sa.func.dayofyear),
        ops.ExtractEpochSeconds: unary(sa.func.UNIX_TIMESTAMP),
        ops.ExtractWeekOfYear: unary(sa.func.weekofyear),
        ops.ExtractMicrosecond: fixed_arity(
            lambda arg: sa.func.floor(sa.extract("microsecond", arg)), 1
        ),
        ops.ExtractMillisecond: fixed_arity(
            lambda arg: sa.func.floor(sa.extract("microsecond", arg) / 1000), 1
        ),
        ops.TimestampNow: fixed_arity(sa.func.now, 0),
        # others
        ops.GroupConcat: _group_concat,
        ops.DayOfWeekIndex: fixed_arity(
            lambda arg: (sa.func.dayofweek(arg) + 5) % 7, 1
        ),
        ops.DayOfWeekName: fixed_arity(lambda arg: sa.func.dayname(arg), 1),
        ops.JSONGetItem: _json_get_item,
        ops.Strip: unary(lambda arg: _mysql_trim(arg, "both")),
        ops.LStrip: unary(lambda arg: _mysql_trim(arg, "leading")),
        ops.RStrip: unary(lambda arg: _mysql_trim(arg, "trailing")),
        ops.TimeDelta: _temporal_delta,
        ops.DateDelta: _temporal_delta,
    }
)
