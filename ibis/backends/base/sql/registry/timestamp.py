import ibis.common.exceptions as com
import ibis.expr.types as ir
import ibis.util as util


def extract_field(sql_attr):
    def extract_field_formatter(translator, expr):
        op = expr.op()
        arg = translator.translate(op.args[0])

        # This is pre-2.0 Impala-style, which did not used to support the
        # SQL-99 format extract($FIELD from expr)
        return f"extract({arg}, '{sql_attr}')"

    return extract_field_formatter


def extract_epoch_seconds(t, expr):
    (arg,) = expr.op().args
    return f'unix_timestamp({t.translate(arg)})'


def truncate(translator, expr):
    base_unit_names = {
        'Y': 'Y',
        'Q': 'Q',
        'M': 'MONTH',
        'W': 'W',
        'D': 'J',
        'h': 'HH',
        'm': 'MI',
    }
    op = expr.op()
    arg, unit = op.args

    arg_formatted = translator.translate(arg)
    try:
        unit = base_unit_names[unit]
    except KeyError:
        raise com.UnsupportedOperationError(
            f'{unit!r} unit is not supported in timestamp truncate'
        )

    return f"trunc({arg_formatted}, '{unit}')"


def interval_from_integer(translator, expr):
    # interval cannot be selected from impala
    op = expr.op()
    arg, unit = op.args
    arg_formatted = translator.translate(arg)

    return 'INTERVAL {} {}'.format(
        arg_formatted, expr.type().resolution.upper()
    )


def timestamp_op(func):
    def _formatter(translator, expr):
        op = expr.op()
        left, right = op.args
        formatted_left = translator.translate(left)
        formatted_right = translator.translate(right)

        if isinstance(left, (ir.TimestampScalar, ir.DateValue)):
            formatted_left = f'cast({formatted_left} as timestamp)'

        if isinstance(right, (ir.TimestampScalar, ir.DateValue)):
            formatted_right = f'cast({formatted_right} as timestamp)'

        return f'{func}({formatted_left}, {formatted_right})'

    return _formatter


def timestamp_diff(translator, expr):
    op = expr.op()
    left, right = op.args

    return 'unix_timestamp({}) - unix_timestamp({})'.format(
        translator.translate(left), translator.translate(right)
    )


def _from_unixtime(translator, expr):
    arg = translator.translate(expr)
    return f'from_unixtime({arg}, "yyyy-MM-dd HH:mm:ss")'


def timestamp_from_unix(translator, expr):
    op = expr.op()

    val, unit = op.args
    val = util.convert_unit(val, unit, 's').cast('int32')

    arg = _from_unixtime(translator, val)
    return f'CAST({arg} AS timestamp)'


def day_of_week_index(t, expr):
    (arg,) = expr.op().args
    return f'pmod(dayofweek({t.translate(arg)}) - 2, 7)'


def day_of_week_name(t, expr):
    (arg,) = expr.op().args
    return f'dayname({t.translate(arg)})'
