from __future__ import annotations

from typing import Literal

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import helpers
from ibis.expr.rules import Shape


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

        left, right = op.args
        if isinstance(right, ops.ValueList) and not right.values:
            return {"NOT IN": "TRUE", "IN": "FALSE"}[op_string]

        left_arg = translator.translate(left)
        if helpers.needs_parens(left):
            left_arg = helpers.parenthesize(left_arg)

        ctx = translator.context

        # special case non-foreign isin/notin expressions
        if (
            not isinstance(right, ops.ValueList)
            and right.output_shape is Shape.COLUMNAR
            # foreign refs are already been compiled correctly during
            # TableColumn compilation
            and not any(
                ctx.is_foreign_expr(leaf)
                for leaf in an.find_immediate_parent_tables(right)
            )
        ):
            if not right.has_resolved_name():
                right = ops.Alias(right, name="tmp")  # .name("tmp")
            right_arg = table_array_view(
                translator,
                right.to_expr().to_projection().to_array().op(),
            )
        else:
            right_arg = translator.translate(right)

        # we explicitly do NOT parenthesize the right side because it doesn't
        # make sense to do so for ValueList operations

        return f"{left_arg} {op_string} {right_arg}"

    return translate
