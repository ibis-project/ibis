"""Flink ibis expression to SQL string compiler."""

from __future__ import annotations

import functools

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.compiler import (
    Compiler,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.flink.translator import FlinkExprTranslator


class FlinkTableSetFormatter(TableSetFormatter):
    def _format_in_memory_table(self, op):
        names = op.schema.names
        raw_rows = []
        for row in op.data.to_frame().itertuples(index=False):
            raw_row = []
            for val, name in zip(row, names):
                lit = ops.Literal(val, dtype=op.schema[name])
                raw_row.append(self._translate(lit))
            raw_rows.append(", ".join(raw_row))
        rows = ", ".join(f"({raw_row})" for raw_row in raw_rows)
        return f"(VALUES {rows})"

    def _format_table(self, op) -> str:
        result = super()._format_table(op)

        ref_op = op
        if isinstance(op, ops.SelfReference):
            ref_op = op.table

        if isinstance(ref_op, ops.InMemoryTable):
            names = op.schema.names
            result += f"({', '.join(self._quote_identifier(name) for name in names)})"

        return result


class FlinkSelectBuilder(SelectBuilder):
    def _convert_group_by(self, exprs):
        return exprs


class FlinkSelect(Select):
    def format_group_by(self) -> str:
        if not len(self.group_by):
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if len(self.group_by) > 0:
            group_keys = map(self._translate, self.group_by)
            clause = "GROUP BY {}".format(", ".join(list(group_keys)))
            lines.append(clause)

        if len(self.having) > 0:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append("HAVING {}".format(" AND ".join(trans_exprs)))

        return "\n".join(lines)


class FlinkCompiler(Compiler):
    translator_class = FlinkExprTranslator
    table_set_formatter_class = FlinkTableSetFormatter
    select_builder_class = FlinkSelectBuilder
    select_class = FlinkSelect

    cheap_in_memory_tables = True


def translate(op: ops.TableNode) -> str:
    return translate_op(op)


@functools.singledispatch
def translate_op(op: ops.TableNode) -> str:
    raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")


@translate_op.register(ops.Literal)
def _literal(op: ops.Literal) -> str:
    from ibis.backends.flink.utils import translate_literal

    return translate_literal(op)


@translate_op.register(ops.Selection)
@translate_op.register(ops.Aggregation)
@translate_op.register(ops.Limit)
def _(op: ops.Selection | ops.Aggregation | ops.Limit) -> str:
    return FlinkCompiler.to_sql(op)  # to_sql uses to_ast, which builds a select tree
