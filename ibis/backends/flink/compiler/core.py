"""Flink ibis expression to SQL string compiler."""

from __future__ import annotations

import functools

import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.compiler import (
    Compiler,
    Select,
    SelectBuilder,
    TableSetFormatter,
)
from ibis.backends.base.sql.registry import quote_identifier
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

    def _format_window_tvf(self, op) -> str:
        if isinstance(op, ops.TumbleWindowingTVF):
            function_type = "TUMBLE"
        elif isinstance(op, ops.HopWindowingTVF):
            function_type = "HOP"
        elif isinstance(op, ops.CumulateWindowingTVF):
            function_type = "CUMULATE"
        return f"TABLE({function_type}({format_windowing_tvf_params(op, self)}))"

    def _format_table(self, op) -> str:
        ctx = self.context
        if isinstance(op, ops.WindowingTVF):
            formatted_table = self._format_window_tvf(op)
            return f"{formatted_table} {ctx.get_ref(op)}"
        else:
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

    @classmethod
    def to_sql(cls, node, context=None, params=None):
        if isinstance(node, ir.Expr):
            node = node.op()

        if isinstance(node, ops.Literal):
            from ibis.backends.flink.utils import translate_literal

            return translate_literal(node)

        return super().to_sql(node, context, params)


@functools.singledispatch
def format_windowing_tvf_params(
    op: ops.WindowingTVF, formatter: TableSetFormatter
) -> str:
    raise com.OperationNotDefinedError(f"No formatting rule for {type(op)}")


@format_windowing_tvf_params.register(ops.TumbleWindowingTVF)
def _tumble_window_params(
    op: ops.TumbleWindowingTVF, formatter: TableSetFormatter
) -> str:
    return ", ".join(
        filter(
            None,
            [
                f"TABLE {quote_identifier(op.table.name)}",
                f"DESCRIPTOR({formatter._translate(op.time_col)})",
                formatter._translate(op.window_size),
                formatter._translate(op.offset) if op.offset else None,
            ],
        )
    )


@format_windowing_tvf_params.register(ops.HopWindowingTVF)
def _hop_window_params(op: ops.HopWindowingTVF, formatter: TableSetFormatter) -> str:
    return ", ".join(
        filter(
            None,
            [
                f"TABLE {quote_identifier(op.table.name)}",
                f"DESCRIPTOR({formatter._translate(op.time_col)})",
                formatter._translate(op.window_slide),
                formatter._translate(op.window_size),
                formatter._translate(op.offset) if op.offset else None,
            ],
        )
    )


@format_windowing_tvf_params.register(ops.CumulateWindowingTVF)
def _cumulate_window_params(
    op: ops.CumulateWindowingTVF, formatter: TableSetFormatter
) -> str:
    return ", ".join(
        filter(
            None,
            [
                f"TABLE {quote_identifier(op.table.name)}",
                f"DESCRIPTOR({formatter._translate(op.time_col)})",
                formatter._translate(op.window_step),
                formatter._translate(op.window_size),
                formatter._translate(op.offset) if op.offset else None,
            ],
        )
    )
