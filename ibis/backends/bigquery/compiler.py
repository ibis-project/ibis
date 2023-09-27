"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import re
from functools import partial

import sqlglot as sg
import toolz

import ibis.common.graph as lin
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql import compiler as sql_compiler
from ibis.backends.bigquery import operations, registry, rewrites
from ibis.backends.bigquery.datatypes import BigQueryType


class BigQueryUDFDefinition(sql_compiler.DDL):
    """Represents definition of a temporary UDF."""

    def __init__(self, expr, context):
        self.expr = expr
        self.context = context

    def compile(self):
        """Generate UDF string from definition."""
        op = expr.op() if isinstance(expr := self.expr, ir.Expr) else expr
        return op.sql


class BigQueryUnion(sql_compiler.Union):
    """Union of tables."""

    @classmethod
    def keyword(cls, distinct):
        """Use distinct UNION if distinct is True."""
        return "UNION DISTINCT" if distinct else "UNION ALL"


class BigQueryIntersection(sql_compiler.Intersection):
    """Intersection of tables."""

    @classmethod
    def keyword(cls, distinct):
        return "INTERSECT DISTINCT" if distinct else "INTERSECT ALL"


class BigQueryDifference(sql_compiler.Difference):
    """Difference of tables."""

    @classmethod
    def keyword(cls, distinct):
        return "EXCEPT DISTINCT" if distinct else "EXCEPT ALL"


def find_bigquery_udf(op):
    """Filter which includes only UDFs from expression tree."""
    if type(op) in BigQueryExprTranslator._rewrites:
        op = BigQueryExprTranslator._rewrites[type(op)](op)
    if isinstance(op, operations.BigQueryUDFNode):
        result = op
    else:
        result = None
    return lin.proceed, result


_NAME_REGEX = re.compile(r'[^!"$()*,./;?@[\\\]^`{}~\n]+')


class BigQueryExprTranslator(sql_compiler.ExprTranslator):
    """Translate expressions to strings."""

    _registry = registry.OPERATION_REGISTRY
    _rewrites = rewrites.REWRITES

    _forbids_frame_clause = (
        *sql_compiler.ExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
    )

    _unsupported_reductions = (ops.ApproxMedian, ops.ApproxCountDistinct)
    _dialect_name = "bigquery"

    @staticmethod
    def _gen_valid_name(name: str) -> str:
        name = "_".join(_NAME_REGEX.findall(name)) or "tmp"
        return f"`{name}`"

    def name(self, translated: str, name: str):
        # replace invalid characters in automatically generated names
        valid_name = self._gen_valid_name(name)
        if translated == valid_name:
            return translated
        return f"{translated} AS {valid_name}"

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator

    def _trans_param(self, op):
        if op not in self.context.params:
            raise KeyError(op)
        return f"@{op.name}"


compiles = BigQueryExprTranslator.compiles


class BigQueryTableSetFormatter(sql_compiler.TableSetFormatter):
    def _quote_identifier(self, name):
        return sg.to_identifier(name).sql("bigquery")

    def _format_in_memory_table(self, op):
        import ibis

        schema = op.schema
        names = schema.names
        types = schema.types

        raw_rows = []
        for row in op.data.to_frame().itertuples(index=False):
            raw_row = ", ".join(
                f"{self._translate(lit.op())} AS {name}"
                for lit, name in zip(
                    map(ibis.literal, row, types), map(self._quote_identifier, names)
                )
            )
            raw_rows.append(f"STRUCT({raw_row})")
        array_type = BigQueryType.from_ibis(dt.Array(op.schema.as_struct()))
        return f"UNNEST({array_type}[{', '.join(raw_rows)}])"


class BigQueryCompiler(sql_compiler.Compiler):
    translator_class = BigQueryExprTranslator
    table_set_formatter_class = BigQueryTableSetFormatter
    union_class = BigQueryUnion
    intersect_class = BigQueryIntersection
    difference_class = BigQueryDifference

    support_values_syntax_in_select = False
    null_limit = None

    @staticmethod
    def _generate_setup_queries(expr, context):
        """Generate DDL for temporary resources."""
        nodes = lin.traverse(find_bigquery_udf, expr)
        queries = map(partial(BigQueryUDFDefinition, context=context), nodes)

        # UDFs are uniquely identified by the name of the Node subclass we
        # generate.
        def key(x):
            expr = x.expr
            op = expr.op() if isinstance(expr, ir.Expr) else expr
            return op.__class__.__name__

        return list(toolz.unique(queries, key=key))


# Register custom UDFs
import ibis.backends.bigquery.custom_udfs  # noqa:  F401, E402
