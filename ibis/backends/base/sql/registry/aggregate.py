import itertools

import ibis


def _reduction_format(translator, func_name, where, arg, *args):
    if where is not None:
        arg = where.ifelse(arg, ibis.NA)

    return '{}({})'.format(
        func_name,
        ', '.join(map(translator.translate, itertools.chain([arg], args))),
    )


def reduction(func_name):
    def formatter(translator, expr):
        op = expr.op()
        *args, where = op.args
        return _reduction_format(translator, func_name, where, *args)

    return formatter


def variance_like(func_name):
    func_names = {
        'sample': f'{func_name}_samp',
        'pop': f'{func_name}_pop',
    }

    def formatter(translator, expr):
        arg, how, where = expr.op().args
        return _reduction_format(translator, func_names[how], where, arg)

    return formatter


def count_distinct(translator, expr):
    arg, where = expr.op().args

    if where is not None:
        arg_formatted = translator.translate(where.ifelse(arg, None))
    else:
        arg_formatted = translator.translate(arg)
    return f'count(DISTINCT {arg_formatted})'
