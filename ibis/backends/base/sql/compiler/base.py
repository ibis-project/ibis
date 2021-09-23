import abc
from itertools import chain

import toolz

import ibis.util as util

from .extract_subqueries import ExtractSubqueries


class DML(abc.ABC):
    @abc.abstractmethod
    def compile(self):
        pass


class DDL(abc.ABC):
    @abc.abstractmethod
    def compile(self):
        pass


class QueryAST:

    __slots__ = 'context', 'dml', 'setup_queries', 'teardown_queries'

    def __init__(
        self, context, dml, setup_queries=None, teardown_queries=None
    ):
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
        compiled_teardown_queries = [
            q.compile() for q in self.teardown_queries
        ]
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
    def __init__(self, tables, expr, context):
        self.context = context
        self.tables = tables
        self.table_set = expr
        self.filters = []

    def _extract_subqueries(self):
        self.subqueries = ExtractSubqueries.extract(self)
        for subquery in self.subqueries:
            self.context.set_extracted(subquery)

    def format_subqueries(self):
        context = self.context
        subqueries = self.subqueries

        return ',\n'.join(
            '{} AS (\n{}\n)'.format(
                context.get_ref(expr),
                util.indent(context.get_compiled_expr(expr), 2),
            )
            for expr in subqueries
        )

    def format_relation(self, expr):
        ref = self.context.get_ref(expr)
        if ref is not None:
            return f'SELECT *\nFROM {ref}'
        return self.context.get_compiled_expr(expr)

    def _get_keyword_list(self):
        raise NotImplementedError("Need objects to interleave")

    def compile(self):
        self._extract_subqueries()

        extracted = self.format_subqueries()

        buf = []

        if extracted:
            buf.append(f'WITH {extracted}')

        buf.extend(
            toolz.interleave(
                (
                    map(self.format_relation, self.tables),
                    self._get_keyword_list(),
                )
            )
        )
        return '\n'.join(buf)
