from __future__ import annotations

from itertools import chain

import toolz

import ibis.expr.analysis as an
import ibis.expr.operations as ops
from ibis import util


class DML:
    def compile(self):
        raise NotImplementedError()


class DDL:
    def compile(self):
        raise NotImplementedError()


class QueryAST:
    __slots__ = "context", "dml", "setup_queries", "teardown_queries"

    def __init__(self, context, dml, setup_queries=None, teardown_queries=None):
        self.context = context
        self.dml = dml
        self.setup_queries = setup_queries
        self.teardown_queries = teardown_queries

    @property
    def queries(self):
        return [self.dml]

    def compile(self):
        compiled_setup_queries = [q.compile() for q in self.setup_queries]
        compiled_queries = [q.compile() for q in self.queries]
        compiled_teardown_queries = [q.compile() for q in self.teardown_queries]
        return self.context.collapse(
            list(
                chain(
                    compiled_setup_queries,
                    compiled_queries,
                    compiled_teardown_queries,
                )
            )
        )


class SetOp(DML):
    def __init__(self, tables, node, context, distincts):
        assert isinstance(node, ops.Node)
        assert all(isinstance(table, ops.Node) for table in tables)
        self.context = context
        self.tables = tables
        self.table_set = node
        self.distincts = distincts
        self.filters = []

    @classmethod
    def keyword(cls, distinct):
        return cls._keyword + (not distinct) * " ALL"

    def _get_keyword_list(self):
        return map(self.keyword, self.distincts)

    def _extract_subqueries(self):
        # extract any subquery to avoid generating incorrect sql when at least
        # one of the set operands is invalid outside of being a subquery
        #
        # for example: SELECT * FROM t ORDER BY x UNION ...
        self.subqueries = an.find_subqueries(
            [self.table_set, *self.filters], min_dependents=1
        )
        for subquery in self.subqueries:
            self.context.set_extracted(subquery)

    def format_subqueries(self):
        context = self.context
        subqueries = self.subqueries

        return ",\n".join(
            "{} AS (\n{}\n)".format(
                context.get_ref(expr),
                util.indent(context.get_compiled_expr(expr), 2),
            )
            for expr in subqueries
        )

    def format_relation(self, expr):
        ref = self.context.get_ref(expr)
        if ref is not None:
            return f"SELECT *\nFROM {ref}"
        return self.context.get_compiled_expr(expr)

    def compile(self):
        self._extract_subqueries()

        extracted = self.format_subqueries()

        buf = []

        if extracted:
            buf.append(f"WITH {extracted}")

        buf.extend(
            toolz.interleave(
                (
                    map(self.format_relation, self.tables),
                    self._get_keyword_list(),
                )
            )
        )
        return "\n".join(buf)
