import ibis.expr.operations as ops

from . import helpers


def substring(translator, expr):
    op = expr.op()
    arg, start, length = op.args
    arg_formatted = translator.translate(arg)
    start_formatted = translator.translate(start)

    # Impala is 1-indexed
    if length is None or isinstance(length.op(), ops.Literal):
        lvalue = length.op().value if length is not None else None
        if lvalue:
            return 'substr({}, {} + 1, {})'.format(
                arg_formatted, start_formatted, lvalue
            )
        else:
            return f'substr({arg_formatted}, {start_formatted} + 1)'
    else:
        length_formatted = translator.translate(length)
        return 'substr({}, {} + 1, {})'.format(
            arg_formatted, start_formatted, length_formatted
        )


def string_find(translator, expr):
    op = expr.op()
    arg, substr, start, _ = op.args
    arg_formatted = translator.translate(arg)
    substr_formatted = translator.translate(substr)

    if start is not None and not isinstance(start.op(), ops.Literal):
        start_fmt = translator.translate(start)
        return 'locate({}, {}, {} + 1) - 1'.format(
            substr_formatted, arg_formatted, start_fmt
        )
    elif start is not None and start.op().value:
        sval = start.op().value
        return 'locate({}, {}, {}) - 1'.format(
            substr_formatted, arg_formatted, sval + 1
        )
    else:
        return f'locate({substr_formatted}, {arg_formatted}) - 1'


def find_in_set(translator, expr):
    op = expr.op()

    arg, str_list = op.args
    arg_formatted = translator.translate(arg)
    str_formatted = ','.join([x._arg.value for x in str_list])
    return f"find_in_set({arg_formatted}, '{str_formatted}') - 1"


def string_join(translator, expr):
    op = expr.op()
    arg, strings = op.args
    return helpers.format_call(translator, 'concat_ws', arg, *strings)


def string_like(translator, expr):
    arg, pattern, _ = expr.op().args
    return '{} LIKE {}'.format(
        translator.translate(arg), translator.translate(pattern)
    )


def parse_url(translator, expr):
    op = expr.op()

    arg, extract, key = op.args
    arg_formatted = translator.translate(arg)

    if key is None:
        return f"parse_url({arg_formatted}, '{extract}')"
    else:
        key_fmt = translator.translate(key)
        return "parse_url({}, '{}', {})".format(
            arg_formatted, extract, key_fmt
        )


def startswith(translator, expr):
    arg, start = expr.op().args

    arg_formatted = translator.translate(arg)
    start_formatted = translator.translate(start)

    return f"{arg_formatted} like concat({start_formatted}, '%')"


def endswith(translator, expr):
    arg, start = expr.op().args

    arg_formatted = translator.translate(arg)
    end_formatted = translator.translate(start)

    return f"{arg_formatted} like concat('%', {end_formatted})"
