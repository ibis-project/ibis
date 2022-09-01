import functools
import itertools
import locale
import operator
import platform
import re
import string
import warnings

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

import ibis.common.exceptions as com
import ibis.common.geospatial as geo
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir

# used for literal translate
from ibis.backends.base.sql.alchemy import (
    fixed_arity,
    get_sqla_table,
    sqlalchemy_operation_registry,
    sqlalchemy_window_functions_registry,
    unary,
)
from ibis.backends.base.sql.alchemy.datatypes import to_sqla_type
from ibis.backends.base.sql.alchemy.registry import (
    _bitwise_op,
    get_col_or_deferred_col,
)

operation_registry = sqlalchemy_operation_registry.copy()
operation_registry.update(sqlalchemy_window_functions_registry)


def _extract(fmt, output_type=sa.SMALLINT):
    def translator(t, expr, output_type=output_type):
        (arg,) = expr.op().args
        sa_arg = t.translate(arg)
        return sa.cast(sa.extract(fmt, sa_arg), output_type)

    return translator


def _second(t, expr):
    # extracting the second gives us the fractional part as well, so smash that
    # with a cast to SMALLINT
    (sa_arg,) = map(t.translate, expr.op().args)
    return sa.cast(sa.func.FLOOR(sa.extract('second', sa_arg)), sa.SMALLINT)


def _millisecond(t, expr):
    # we get total number of milliseconds including seconds with extract so we
    # mod 1000
    (sa_arg,) = map(t.translate, expr.op().args)
    return sa.cast(
        sa.func.floor(sa.extract('millisecond', sa_arg)) % 1000,
        sa.SMALLINT,
    )


_truncate_precisions = {
    'us': 'microseconds',
    'ms': 'milliseconds',
    's': 'second',
    'm': 'minute',
    'h': 'hour',
    'D': 'day',
    'W': 'week',
    'M': 'month',
    'Q': 'quarter',
    'Y': 'year',
}


def _timestamp_truncate(t, expr):
    arg, unit = expr.op().args
    sa_arg = t.translate(arg)
    try:
        precision = _truncate_precisions[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'Unsupported truncate unit {unit!r}'
        )
    return sa.func.date_trunc(precision, sa_arg)


def _interval_from_integer(t, expr):
    op = expr.op()
    sa_arg = t.translate(op.arg)
    interval = sa.text(f"INTERVAL '1 {expr.type().resolution}'")
    return sa_arg * interval


def _is_nan(t, expr):
    (arg,) = expr.op().args
    sa_arg = t.translate(arg)
    return sa_arg == float('nan')


def _is_inf(t, expr):
    (arg,) = expr.op().args
    sa_arg = t.translate(arg)
    inf = float('inf')
    return sa.or_(sa_arg == inf, sa_arg == -inf)


def _typeof(t, expr):
    (arg,) = expr.op().args
    sa_arg = t.translate(arg)
    typ = sa.cast(sa.func.pg_typeof(sa_arg), sa.TEXT)

    # select pg_typeof('thing') returns unknown so we have to check the child's
    # type for nullness
    return sa.case(
        [
            ((typ == 'unknown') & (arg.type() != dt.null), 'text'),
            ((typ == 'unknown') & (arg.type() == dt.null), 'null'),
        ],
        else_=typ,
    )


_strftime_to_postgresql_rules = {
    '%a': 'TMDy',  # TM does it in a locale dependent way
    '%A': 'TMDay',
    '%w': 'D',  # 1-based day of week, see below for how we make this 0-based
    '%d': 'DD',  # day of month
    '%-d': 'FMDD',  # - is no leading zero for Python same for FM in postgres
    '%b': 'TMMon',  # Sep
    '%B': 'TMMonth',  # September
    '%m': 'MM',  # 01
    '%-m': 'FMMM',  # 1
    '%y': 'YY',  # 15
    '%Y': 'YYYY',  # 2015
    '%H': 'HH24',  # 09
    '%-H': 'FMHH24',  # 9
    '%I': 'HH12',  # 09
    '%-I': 'FMHH12',  # 9
    '%p': 'AM',  # AM or PM
    '%M': 'MI',  # zero padded minute
    '%-M': 'FMMI',  # Minute
    '%S': 'SS',  # zero padded second
    '%-S': 'FMSS',  # Second
    '%f': 'US',  # zero padded microsecond
    '%z': 'OF',  # utf offset
    '%Z': 'TZ',  # uppercase timezone name
    '%j': 'DDD',  # zero padded day of year
    '%-j': 'FMDDD',  # day of year
    '%U': 'WW',  # 1-based week of year
    # 'W': ?,  # meh
}

try:
    _strftime_to_postgresql_rules.update(
        {
            '%c': locale.nl_langinfo(locale.D_T_FMT),  # locale date and time
            '%x': locale.nl_langinfo(locale.D_FMT),  # locale date
            '%X': locale.nl_langinfo(locale.T_FMT),  # locale time
        }
    )
except AttributeError:
    warnings.warn(
        'locale specific date formats (%%c, %%x, %%X) are not yet implemented '
        'for %s' % platform.system()
    )


# translate strftime spec into mostly equivalent PostgreSQL spec
_scanner = re.Scanner(  # type: ignore # re does have a Scanner attribute
    # double quotes need to be escaped
    [('"', lambda *_: r'\"')]
    + [
        (
            '|'.join(
                map(
                    '(?:{})'.format,
                    itertools.chain(
                        _strftime_to_postgresql_rules.keys(),
                        [
                            # "%e" is in the C standard and Python actually
                            # generates this if your spec contains "%c" but we
                            # don't officially support it as a specifier so we
                            # need to special case it in the scanner
                            '%e',
                            r'\s+',
                            fr'[{re.escape(string.punctuation)}]',
                            fr'[^{re.escape(string.punctuation)}\s]+',
                        ],
                    ),
                )
            ),
            lambda _, token: token,
        )
    ]
)


_lexicon_values = frozenset(_strftime_to_postgresql_rules.values())

_strftime_blacklist = frozenset(['%w', '%U', '%c', '%x', '%X', '%e'])


def _reduce_tokens(tokens, arg):
    # current list of tokens
    curtokens = []

    # reduced list of tokens that accounts for blacklisted values
    reduced = []

    non_special_tokens = (
        frozenset(_strftime_to_postgresql_rules) - _strftime_blacklist
    )

    # TODO: how much of a hack is this?
    for token in tokens:
        # we are a non-special token %A, %d, etc.
        if token in non_special_tokens:
            curtokens.append(_strftime_to_postgresql_rules[token])

        # we have a string like DD, to escape this we
        # surround it with double quotes
        elif token in _lexicon_values:
            curtokens.append(f'"{token}"')

        # we have a token that needs special treatment
        elif token in _strftime_blacklist:
            if token == '%w':
                value = sa.extract('dow', arg)  # 0 based day of week
            elif token == '%U':
                value = sa.cast(sa.func.to_char(arg, 'WW'), sa.SMALLINT) - 1
            elif token == '%c' or token == '%x' or token == '%X':
                # re scan and tokenize this pattern
                try:
                    new_pattern = _strftime_to_postgresql_rules[token]
                except KeyError:
                    raise ValueError(
                        'locale specific date formats (%%c, %%x, %%X) are '
                        'not yet implemented for %s' % platform.system()
                    )

                new_tokens, _ = _scanner.scan(new_pattern)
                value = functools.reduce(
                    sa.sql.ColumnElement.concat,
                    _reduce_tokens(new_tokens, arg),
                )
            elif token == '%e':
                # pad with spaces instead of zeros
                value = sa.func.replace(sa.func.to_char(arg, 'DD'), '0', ' ')

            reduced += [
                sa.func.to_char(arg, ''.join(curtokens)),
                sa.cast(value, sa.TEXT),
            ]

            # empty current token list in case there are more tokens
            del curtokens[:]

        # uninteresting text
        else:
            curtokens.append(token)
    else:
        # append result to r if we had more tokens or if we have no
        # blacklisted tokens
        if curtokens:
            reduced.append(sa.func.to_char(arg, ''.join(curtokens)))
    return reduced


def _strftime(t, expr):
    arg, pattern = map(t.translate, expr.op().args)
    tokens, _ = _scanner.scan(pattern.value)
    reduced = _reduce_tokens(tokens, arg)
    return functools.reduce(sa.sql.ColumnElement.concat, reduced)


def _find_in_set(t, expr):
    # TODO
    # this operation works with any type, not just strings. should the
    # operation itself also have this property?
    op = expr.op()
    return (
        sa.func.coalesce(
            sa.func.array_position(
                postgresql.array(list(map(t.translate, op.values))),
                t.translate(op.needle),
            ),
            0,
        )
        - 1
    )


def _regex_replace(t, expr):
    string, pattern, replacement = map(t.translate, expr.op().args)

    # postgres defaults to replacing only the first occurrence
    return sa.func.regexp_replace(string, pattern, replacement, 'g')


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    if base is not None:
        sa_base = t.translate(base)
        return sa.cast(
            sa.func.log(
                sa.cast(sa_base, sa.NUMERIC), sa.cast(sa_arg, sa.NUMERIC)
            ),
            t.get_sqla_type(expr.type()),
        )
    return sa.func.ln(sa_arg)


def _regex_extract(t, expr):
    op = expr.op()
    arg = t.translate(op.arg)
    pattern = t.translate(op.pattern)
    return sa.case(
        [
            (
                sa.func.textregexeq(arg, pattern),
                sa.func.regexp_match(
                    arg,
                    pattern,
                    type_=postgresql.ARRAY(sa.TEXT),
                )[t.translate(op.index) + 1],
            )
        ],
        else_="",
    )


def _cardinality(array):
    return sa.case(
        [(array.is_(None), None)],  # noqa: E711
        else_=sa.func.coalesce(sa.func.array_length(array, 1), 0),
    )


def _array_repeat(t, expr):
    """Repeat an array."""
    op = expr.op()
    arg = t.translate(op.arg)
    times = t.translate(op.times)

    # SQLAlchemy uses our column's table in the FROM clause. We need a simpler
    # expression to workaround this.
    array = sa.column(arg.name, type_=arg.type)

    # We still need to prefix the table name to the column name in the final
    # query, so make sure the column knows its origin
    array.table = arg.table

    array_length = _cardinality(array)

    # sequence from 1 to the total number of elements desired in steps of 1.
    # the call to greatest isn't necessary, but it provides clearer intent
    # rather than depending on the implicit postgres generate_series behavior
    start = step = 1
    stop = sa.func.greatest(times, 0) * array_length
    series = sa.func.generate_series(start, stop, step).alias()
    series_column = sa.column(series.name, type_=sa.INTEGER)

    # if our current index modulo the array's length is a multiple of the
    # array's length, then the index is the array's length
    index_expression = series_column % array_length
    index = sa.func.coalesce(sa.func.nullif(index_expression, 0), array_length)

    # tie it all together in a scalar subquery and collapse that into an ARRAY
    return sa.func.array(
        sa.select(array[index]).select_from(series).scalar_subquery()
    )


def _table_column(t, expr):
    op = expr.op()
    ctx = t.context
    table = op.table

    sa_table = get_sqla_table(ctx, table)

    out_expr = get_col_or_deferred_col(sa_table, op.name)

    expr_type = expr.type()

    if isinstance(expr_type, dt.Timestamp):
        timezone = expr_type.timezone
        if timezone is not None:
            out_expr = out_expr.op('AT TIME ZONE')(timezone).label(op.name)

    # If the column does not originate from the table set in the current SELECT
    # context, we should format as a subquery
    if t.permit_subquery and ctx.is_foreign_expr(table):
        return sa.select(out_expr)

    return out_expr


def _round(t, expr):
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        return sa.func.round(sa_arg)

    # postgres doesn't allow rounding of double precision values to a specific
    # number of digits (though simple truncation on doubles is allowed) so
    # we cast to numeric and then cast back if necessary
    result = sa.func.round(sa.cast(sa_arg, sa.NUMERIC), t.translate(digits))
    if digits is not None and isinstance(arg.type(), dt.Decimal):
        return result
    result = sa.cast(result, sa.dialects.postgresql.DOUBLE_PRECISION())
    return result


def _mod(t, expr):
    left, right = map(t.translate, expr.op().args)

    # postgres doesn't allow modulus of double precision values, so upcast and
    # then downcast later if necessary
    if not isinstance(expr.type(), dt.Integer):
        left = sa.cast(left, sa.NUMERIC)
        right = sa.cast(right, sa.NUMERIC)

    result = left % right
    if expr.type().equals(dt.double):
        return sa.cast(result, sa.dialects.postgresql.DOUBLE_PRECISION())
    else:
        return result


def _neg_idx_to_pos(array, idx):
    return sa.case(
        [
            (array.is_(None), None),
            (idx < 0, sa.func.array_length(array, 1) + idx),
        ],
        else_=idx,
    )


def _array_slice(t, expr):
    arg, start, stop = expr.op().args
    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if stop is None:
        sa_stop = _cardinality(sa_arg)
    else:
        sa_stop = t.translate(stop)

    sa_start = _neg_idx_to_pos(sa_arg, sa_start)
    sa_stop = _neg_idx_to_pos(sa_arg, sa_stop)
    return sa_arg[sa_start + 1 : sa_stop]


def _literal(_, expr):
    dtype = expr.type()
    op = expr.op()
    value = op.value

    if isinstance(dtype, dt.Interval):
        return sa.text(f"INTERVAL '{value} {dtype.resolution}'")
    elif isinstance(dtype, dt.Set):
        return list(map(sa.literal, value))
    # geo spatial data type
    elif isinstance(expr, ir.GeoSpatialScalar):
        # inline_metadata ex: 'SRID=4326;POINT( ... )'
        return sa.literal_column(
            geo.translate_literal(expr, inline_metadata=True)
        )
    elif isinstance(value, tuple):
        return sa.literal(value, type_=to_sqla_type(dtype))
    else:
        return sa.literal(value)


def _day_of_week_index(t, expr):
    (sa_arg,) = map(t.translate, expr.op().args)
    return sa.cast(
        sa.cast(sa.extract('dow', sa_arg) + 6, sa.SMALLINT) % 7, sa.SMALLINT
    )


def _day_of_week_name(t, expr):
    (sa_arg,) = map(t.translate, expr.op().args)
    return sa.func.trim(sa.func.to_char(sa_arg, 'Day'))


def _array_column(t, expr):
    return postgresql.array(list(map(t.translate, expr.op().cols)))


def _string_agg(t, expr):
    op = expr.op()
    agg = sa.func.string_agg(t.translate(op.arg), t.translate(op.sep))
    if (where := op.where) is not None:
        return agg.filter(t.translate(where))
    return agg


def _corr(t, expr):
    if expr.op().how == "sample":
        raise ValueError(
            "PostgreSQL only implements population correlation coefficient"
        )
    return _binary_variance_reduction(sa.func.corr)(t, expr)


def _covar(t, expr):
    suffix = {"sample": "samp", "pop": "pop"}
    how = suffix.get(expr.op().how, "samp")
    func = getattr(sa.func, f"covar_{how}")
    return _binary_variance_reduction(func)(t, expr)


def _binary_variance_reduction(func):
    def variance_compiler(t, expr):
        op = expr.op()

        x = op.left
        if isinstance(x_type := x.type(), dt.Boolean):
            x = x.cast(dt.Int32(nullable=x_type.nullable))

        y = op.right
        if isinstance(y_type := y.type(), dt.Boolean):
            y = y.cast(dt.Int32(nullable=y_type.nullable))

        result = func(t.translate(x), t.translate(y))

        if (where := op.where) is not None:
            result = result.filter(t.translate(where))

        return result

    return variance_compiler


operation_registry.update(
    {
        ops.Literal: _literal,
        # We override this here to support time zones
        ops.TableColumn: _table_column,
        # types
        ops.TypeOf: _typeof,
        # Floating
        ops.IsNan: _is_nan,
        ops.IsInf: _is_inf,
        # null handling
        ops.IfNull: fixed_arity(sa.func.coalesce, 2),
        # boolean reductions
        ops.Any: unary(sa.func.bool_or),
        ops.All: unary(sa.func.bool_and),
        ops.NotAny: unary(lambda x: sa.not_(sa.func.bool_or(x))),
        ops.NotAll: unary(lambda x: sa.not_(sa.func.bool_and(x))),
        # strings
        ops.GroupConcat: _string_agg,
        ops.Capitalize: unary(sa.func.initcap),
        ops.RegexSearch: fixed_arity(lambda x, y: x.op("~")(y), 2),
        ops.RegexReplace: _regex_replace,
        ops.Translate: fixed_arity('translate', 3),
        ops.RegexExtract: _regex_extract,
        ops.StringSplit: fixed_arity(
            lambda col, sep: sa.func.string_to_array(
                col, sep, type_=sa.ARRAY(col.type)
            ),
            2,
        ),
        ops.FindInSet: _find_in_set,
        # math
        ops.Log: _log,
        ops.Log2: unary(lambda x: sa.func.log(2, x)),
        ops.Log10: unary(sa.func.log),
        ops.Round: _round,
        ops.Modulus: _mod,
        # dates and times
        ops.DateFromYMD: fixed_arity(sa.func.make_date, 3),
        ops.DateTruncate: _timestamp_truncate,
        ops.TimestampTruncate: _timestamp_truncate,
        ops.IntervalFromInteger: _interval_from_integer,
        ops.DateAdd: fixed_arity(operator.add, 2),
        ops.DateSub: fixed_arity(operator.sub, 2),
        ops.DateDiff: fixed_arity(operator.sub, 2),
        ops.TimestampAdd: fixed_arity(operator.add, 2),
        ops.TimestampSub: fixed_arity(operator.sub, 2),
        ops.TimestampDiff: fixed_arity(operator.sub, 2),
        ops.Strftime: _strftime,
        ops.ExtractYear: _extract('year'),
        ops.ExtractMonth: _extract('month'),
        ops.ExtractDay: _extract('day'),
        ops.ExtractDayOfYear: _extract('doy'),
        ops.ExtractQuarter: _extract('quarter'),
        ops.ExtractEpochSeconds: _extract('epoch', sa.Integer),
        ops.ExtractWeekOfYear: _extract('week'),
        ops.ExtractHour: _extract('hour'),
        ops.ExtractMinute: _extract('minute'),
        ops.ExtractSecond: _second,
        ops.ExtractMillisecond: _millisecond,
        ops.DayOfWeekIndex: _day_of_week_index,
        ops.DayOfWeekName: _day_of_week_name,
        # now is in the timezone of the server, but we want UTC
        ops.TimestampNow: lambda *_: sa.func.timezone('UTC', sa.func.now()),
        ops.TimeFromHMS: fixed_arity(sa.func.make_time, 3),
        ops.CumulativeAll: unary(sa.func.bool_and),
        ops.CumulativeAny: unary(sa.func.bool_or),
        # array operations
        ops.ArrayLength: unary(_cardinality),
        ops.ArrayCollect: unary(sa.func.array_agg),
        ops.ArrayColumn: _array_column,
        ops.ArraySlice: _array_slice,
        ops.ArrayIndex: fixed_arity(
            lambda array, index: array[_neg_idx_to_pos(array, index) + 1], 2
        ),
        ops.ArrayConcat: fixed_arity(
            sa.sql.expression.ColumnElement.concat, 2
        ),
        ops.ArrayRepeat: _array_repeat,
        ops.Unnest: unary(sa.func.unnest),
        ops.Covariance: _covar,
        ops.Correlation: _corr,
        ops.BitwiseXor: _bitwise_op("#"),
    }
)
