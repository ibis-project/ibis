from __future__ import annotations

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

        return f"{left_arg} {infix_sym} {right_arg}"

    return formatter


def identical_to(translator, op):
    if op.args[0].equals(op.args[1]):
        return "TRUE"

    left = translator.translate(op.left)
    right = translator.translate(op.right)

    if helpers.needs_parens(op.left):
        left = helpers.parenthesize(left)
    if helpers.needs_parens(op.right):
        right = helpers.parenthesize(right)
    return f"{left} IS NOT DISTINCT FROM {right}"


def xor(translator, op):
    left_arg = translator.translate(op.left)
    right_arg = translator.translate(op.right)

    if helpers.needs_parens(op.left):
        left_arg = helpers.parenthesize(left_arg)

    if helpers.needs_parens(op.right):
        right_arg = helpers.parenthesize(right_arg)

    return f"({left_arg} OR {right_arg}) AND NOT ({left_arg} AND {right_arg})"


def in_values(translator, op):
    if not op.options:
        return "FALSE"

    left = translator.translate(op.value)
    if helpers.needs_parens(op.value):
        left = helpers.parenthesize(left)

    values = [translator.translate(x) for x in op.options]
    right = helpers.parenthesize(", ".join(values))

    # we explicitly do NOT parenthesize the right side because it doesn't
    # make sense to do so for Sequence operations
    return f"{left} IN {right}"


def in_column(translator, op):
    from ibis.backends.base.sql.registry.main import table_array_view

    ctx = translator.context

    left = translator.translate(op.value)
    if helpers.needs_parens(op.value):
        left = helpers.parenthesize(left)

    right = translator.translate(op.options)
    if not any(
        ctx.is_foreign_expr(leaf)
        for leaf in an.find_immediate_parent_tables(op.options)
    ):
        array = op.options.to_expr().as_table().to_array().op()
        right = table_array_view(translator, array)
    else:
        right = translator.translate(op.options)

    # we explicitly do NOT parenthesize the right side because it doesn't
    # make sense to do so for Sequence operations
    return f"{left} IN {right}"
