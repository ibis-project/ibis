from __future__ import annotations

from typing import Literal

import ibis.expr.analysis as an
from ibis.backends.base.sql.registry import helpers


def binary_infix_op(infix_sym):
    def formatter(translator, op):
        left, right = op.args

        left_arg = translator.translate(left)
        right_arg = translator.translate(right)
        if helpers.needs_parens(left):
            left_arg = helpers.parenthesize(left_arg)

        if helpers.needs_parens(right):
            right_arg = helpers.parenthesize(right_arg)

        return f'{left_arg} {infix_sym} {right_arg}'

    return formatter


def identical_to(translator, op):
    if op.args[0].equals(op.args[1]):
        return 'TRUE'

    left = translator.translate(op.left)
    right = translator.translate(op.right)

    if helpers.needs_parens(op.left):
        left = helpers.parenthesize(left)
    if helpers.needs_parens(op.right):
        right = helpers.parenthesize(right)
    return f'{left} IS NOT DISTINCT FROM {right}'


def xor(translator, op):
    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if helpers.needs_parens(op.left):
        left_arg = helpers.parenthesize(left_arg)

    if helpers.needs_parens(op.right):
        right_arg = helpers.parenthesize(right_arg)

    return '({0} OR {1}) AND NOT ({0} AND {1})'.format(left_arg, right_arg)


def contains(op_string: Literal["IN", "NOT IN"]) -> str:
    def translate(translator, op):
        from ibis.backends.base.sql.registry.main import table_array_view

        if isinstance(op.options, tuple) and not op.options:
            return {"NOT IN": "TRUE", "IN": "FALSE"}[op_string]

        left = translator.translate(op.value)
        if helpers.needs_parens(op.value):
            left = helpers.parenthesize(left)

        ctx = translator.context

        if isinstance(op.options, tuple):
            values = [translator.translate(x) for x in op.options]
            right = helpers.parenthesize(', '.join(values))
        elif op.options.output_shape.is_columnar():
            right = translator.translate(op.options)
            if not any(
                ctx.is_foreign_expr(leaf)
                for leaf in an.find_immediate_parent_tables(op.options)
            ):
                array = op.options.to_expr().as_table().to_array().op()
                right = table_array_view(translator, array)
            else:
                right = translator.translate(op.options)
        else:
            right = translator.translate(op.options)

        # we explicitly do NOT parenthesize the right side because it doesn't
        # make sense to do so for Sequence operations
        return f"{left} {op_string} {right}"

    return translate
