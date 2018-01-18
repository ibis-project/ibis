import sqlalchemy as sa

from ibis.sql.alchemy import (unary, varargs, fixed_arity, infix_op,
                              _variance_reduction)
import ibis.common as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.sql.alchemy as alch


_operation_registry = alch._operation_registry.copy()

# NOTE: window functions are available from MySQL 8 and MariaDB 10.2
_operation_registry.update(alch._window_functions)


def _substr(t, expr):
    f = sa.func.substr

    arg, start, length = expr.op().args

    sa_arg = t.translate(arg)
    sa_start = t.translate(start)

    if length is None:
        return f(sa_arg, sa_start + 1)
    else:
        sa_length = t.translate(length)
        return f(sa_arg, sa_start + 1, sa_length)


def _string_find(t, expr):
    arg, substr, start, _ = expr.op().args

    if start is not None:
        raise NotImplementedError

    sa_arg = t.translate(arg)
    sa_substr = t.translate(substr)

    return sa.func.locate(sa_arg, sa_substr) - 1


def _capitalize(t, expr):
    arg, = expr.op().args
    sa_arg = t.translate(arg)
    return sa.func.concat(
        sa.func.ucase(
            sa.func.left(sa_arg, 1)
        ),
        sa.func.substring(sa_arg, 2)
    )


def _extract(fmt):
    def translator(t, expr):
        arg, = expr.op().args
        sa_arg = t.translate(arg)
        return sa.extract(fmt, sa_arg)
    return translator


_truncate_formats = {
    's': '%Y-%m-%d %H:%i:%s',
    'm': '%Y-%m-%d %H:%i:00',
    'h': '%Y-%m-%d %H:00:00',
    'D': '%Y-%m-%d',
    # 'W': 'week',
    'M': '%Y-%m-01',
    'Y': '%Y-01-01'
}


def _truncate(t, expr):
    arg, unit = expr.op().args
    sa_arg = t.translate(arg)
    try:
        fmt = _truncate_formats[unit]
    except KeyError:
        raise com.TranslationError('Unsupported truncate unit '
                                   '{}'.format(unit))
    return sa.func.date_format(sa_arg, fmt)


def _cast(t, expr):
    arg, typ = expr.op().args

    sa_arg = t.translate(arg)
    sa_type = t.get_sqla_type(typ)

    # specialize going from an integer type to a timestamp
    if isinstance(arg.type(), dt.Integer) and isinstance(sa_type, sa.DateTime):
        return sa.func.timezone('UTC', sa.func.to_timestamp(sa_arg))

    if arg.type().equals(dt.binary) and typ.equals(dt.string):
        return sa.func.encode(sa_arg, 'escape')

    if typ.equals(dt.binary):
        #  decode yields a column of memoryview which is annoying to deal with
        # in pandas. CAST(expr AS BYTEA) is correct and returns byte strings.
        return sa.cast(sa_arg, sa.Binary())

    return sa.cast(sa_arg, sa_type)


def _log(t, expr):
    arg, base = expr.op().args
    sa_arg = t.translate(arg)
    sa_base = t.translate(base)
    return sa.func.log(sa_base, sa_arg)


def _power(t, expr):
    arg, exponent = expr.op().args
    sa_arg = t.translate(arg)
    sa_exponent = t.translate(exponent)
    return sa.func.pow(sa_exponent, sa_arg)


def _identical_to(t, expr):
    left, right = args = expr.op().args
    if left.equals(right):
        return True
    else:
        left, right = map(t.translate, args)
        return left.op('<=>')(right)


def _round(t, expr):
    arg, digits = expr.op().args
    sa_arg = t.translate(arg)

    if digits is None:
        sa_digits = 0
    else:
        sa_digits = t.translate(digits)

    return sa.func.round(sa_arg, sa_digits)


def _floor_divide(t, expr):
    left, right = map(t.translate, expr.op().args)
    return sa.func.floor(left / right)


def _string_join(t, expr):
    sep, elements = expr.op().args
    return sa.func.concat_ws(t.translate(sep), *map(t.translate, elements))


_operation_registry.update({
    # types
    # ops.Cast: _cast,

    # # miscellaneous varargs
    ops.Least: varargs(sa.func.least),
    ops.Greatest: varargs(sa.func.greatest),

    # null handling
    # ops.IfNull: _if_null,

    # boolean reductions
    # ops.Any: unary(sa.any_),
    # ops.All: unary(sa.all_),
    # ops.NotAny: unary(lambda x: sa.not_(sa.any_(x))),
    # ops.NotAll: unary(lambda x: sa.not_(sa.all_(x))),

    # strings
    # ops.Contains: fixed_arity(sa_contains, 2),
    ops.Substring: _substr,
    ops.StrRight: fixed_arity(sa.func.right, 2),
    ops.StringFind: _string_find,
    ops.StringLength: unary(sa.func.length),
    # ops.GroupConcat: fixed_arity('concat_ws', 2),
    ops.Lowercase: unary(sa.func.lower),
    ops.Uppercase: unary(sa.func.upper),
    ops.Strip: unary(sa.func.trim),
    ops.LStrip: unary(sa.func.ltrim),
    ops.RStrip: unary(sa.func.rtrim),
    ops.LPad: fixed_arity(sa.func.lpad, 3),
    ops.RPad: fixed_arity(sa.func.rpad, 3),
    ops.Reverse: unary(sa.func.reverse),
    ops.Capitalize: _capitalize,
    ops.Repeat: fixed_arity(sa.func.repeat, 2),
    ops.StringReplace: fixed_arity(sa.func.replace, 3),
    ops.RegexSearch: infix_op('REGEXP'),

    # ops.Translate: fixed_arity('translate', 3),
    ops.StringAscii: fixed_arity(sa.func.ascii, 1),
    # ops.StringJoin: _string_join,
    # ops.FindInSet: fixed_arity('find_in_set', 2),

    ops.Ceil: unary(sa.func.ceil),
    ops.Floor: unary(sa.func.floor),
    # ops.FloorDivide: _floor_divide,
    ops.Exp: unary(sa.func.exp),
    ops.Sign: unary(sa.func.sign),
    ops.Sqrt: unary(sa.func.sqrt),
    ops.Log: _log,
    ops.Ln: unary(sa.func.ln),
    ops.Log2: unary(sa.func.log2),
    ops.Log10: unary(sa.func.log10),
    ops.Power: _power,
    ops.Round: _round,

    # dates and times
    ops.Date: unary(sa.func.date),
    ops.DateTruncate: _truncate,
    ops.TimestampTruncate: _truncate,
    ops.Strftime: fixed_arity(sa.func.date_format, 2),
    ops.ExtractYear: _extract('year'),
    ops.ExtractMonth: _extract('month'),
    ops.ExtractDay: _extract('day'),
    ops.ExtractHour: _extract('hour'),
    ops.ExtractMinute: _extract('minute'),
    ops.ExtractSecond: _extract('second'),
    ops.ExtractMillisecond: _extract('millisecond'),

    # reductions
    ops.Variance: _variance_reduction('var'),
    ops.StandardDev: _variance_reduction('stddev'),

    # # now is in the timezone of the server, but we want UTC
    # ops.TimestampNow: lambda *args: sa.func.timezone('UTC', sa.func.now()),

    ops.IdenticalTo: _identical_to
})


def add_operation(op, translation_func):
    _operation_registry[op] = translation_func


class MySQLExprTranslator(alch.AlchemyExprTranslator):

    _registry = _operation_registry
    _rewrites = alch.AlchemyExprTranslator._rewrites.copy()
    _type_map = alch.AlchemyExprTranslator._type_map.copy()
    _type_map.update({
        dt.Double: sa.types.FLOAT,
        dt.Float: sa.types.REAL
    })


rewrites = MySQLExprTranslator.rewrites
compiles = MySQLExprTranslator.compiles


class MySQLDialect(alch.AlchemyDialect):

    translator = MySQLExprTranslator
