import datetime
import math

import ibis.expr.types as ir


def _set_literal_format(translator, expr):
    value_type = expr.type().value_type

    formatted = [
        translator.translate(ir.literal(x, type=value_type))
        for x in expr.op().value
    ]

    return '(' + ', '.join(formatted) + ')'


def _boolean_literal_format(translator, expr):
    value = expr.op().value
    return 'TRUE' if value else 'FALSE'


def _string_literal_format(translator, expr):
    value = expr.op().value
    return "'{}'".format(value.replace("'", "\\'"))


def _number_literal_format(translator, expr):
    value = expr.op().value

    if math.isfinite(value):
        formatted = repr(value)
    else:
        if math.isnan(value):
            formatted_val = 'NaN'
        elif math.isinf(value):
            if value > 0:
                formatted_val = 'Infinity'
            else:
                formatted_val = '-Infinity'
        formatted = f"CAST({formatted_val!r} AS DOUBLE)"

    return formatted


def _interval_literal_format(translator, expr):
    return 'INTERVAL {} {}'.format(
        expr.op().value, expr.type().resolution.upper()
    )


def _date_literal_format(translator, expr):
    value = expr.op().value
    if isinstance(value, datetime.date):
        value = value.strftime('%Y-%m-%d')

    return repr(value)


def _timestamp_literal_format(translator, expr):
    value = expr.op().value
    if isinstance(value, datetime.datetime):
        value = value.strftime('%Y-%m-%d %H:%M:%S')

    return repr(value)


literal_formatters = {
    'boolean': _boolean_literal_format,
    'number': _number_literal_format,
    'string': _string_literal_format,
    'interval': _interval_literal_format,
    'timestamp': _timestamp_literal_format,
    'date': _date_literal_format,
    'set': _set_literal_format,
}


def literal(translator, expr):
    """Return the expression as its literal value."""
    if isinstance(expr, ir.BooleanValue):
        typeclass = 'boolean'
    elif isinstance(expr, ir.StringValue):
        typeclass = 'string'
    elif isinstance(expr, ir.NumericValue):
        typeclass = 'number'
    elif isinstance(expr, ir.DateValue):
        typeclass = 'date'
    elif isinstance(expr, ir.TimestampValue):
        typeclass = 'timestamp'
    elif isinstance(expr, ir.IntervalValue):
        typeclass = 'interval'
    elif isinstance(expr, ir.SetValue):
        typeclass = 'set'
    else:
        raise NotImplementedError

    return literal_formatters[typeclass](translator, expr)


def null_literal(translator, expr):
    return 'NULL'
