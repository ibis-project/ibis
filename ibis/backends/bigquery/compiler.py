"""Module to convert from Ibis expression to SQL string."""

from __future__ import annotations

import re
from functools import partial

import toolz

import ibis.common.graph as lin
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql import compiler as sql_compiler
from ibis.backends.bigquery import operations, registry, rewrites


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
        """Use disctinct UNION if distinct is True."""
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


class BigQueryExprTranslator(sql_compiler.ExprTranslator):
    """Translate expressions to strings."""

    _registry = registry.OPERATION_REGISTRY
    _rewrites = rewrites.REWRITES
    _valid_name_pattern = re.compile(r"^[A-Za-z][A-Za-z_0-9]*$")

    _forbids_frame_clause = (
        *sql_compiler.ExprTranslator._forbids_frame_clause,
        ops.Lag,
        ops.Lead,
    )

    def name(self, translated: str, name: str):
        # replace invalid characters in automatically generated names
        if self._valid_name_pattern.match(name) is None:
            return f"{translated} AS `tmp`"
        return super().name(translated, name)

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
        if re.match(r"^[A-Za-z][A-Za-z_0-9]*$", name):
            return name
        return f"`{name}`"


class BigQueryCompiler(sql_compiler.Compiler):
    translator_class = BigQueryExprTranslator
    table_set_formatter_class = BigQueryTableSetFormatter
    union_class = BigQueryUnion
    intersect_class = BigQueryIntersection
    difference_class = BigQueryDifference

    support_values_syntax_in_select = False

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
