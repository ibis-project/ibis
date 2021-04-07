import itertools

import toolz

import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util

from . import base, extract_subqueries, select_builder


class _QueryAST:

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
                itertools.chain(
                    compiled_setup_queries,
                    compiled_queries,
                    compiled_teardown_queries,
                )
            )
        )


class _SetOp(base.DML):
    def __init__(self, tables, expr, context):
        self.context = context
        self.tables = tables
        self.table_set = expr
        self.filters = []

    def _extract_subqueries(self):
        self.subqueries = extract_subqueries.extract_subqueries(self)
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
            return 'SELECT *\nFROM {}'.format(ref)
        return self.context.get_compiled_expr(expr)

    def _get_keyword_list(self):
        raise NotImplementedError("Need objects to interleave")

    def compile(self):
        self._extract_subqueries()

        extracted = self.format_subqueries()

        buf = []

        if extracted:
            buf.append('WITH {}'.format(extracted))

        buf.extend(
            toolz.interleave(
                (
                    map(self.format_relation, self.tables),
                    self._get_keyword_list(),
                )
            )
        )
        return '\n'.join(buf)


class Union(_SetOp):
    def __init__(self, tables, expr, context, distincts):
        super().__init__(tables, expr, context)
        self.distincts = distincts

    @staticmethod
    def keyword(distinct):
        return 'UNION' if distinct else 'UNION ALL'

    def _get_keyword_list(self):
        return map(self.keyword, self.distincts)


class Intersection(_SetOp):
    def _get_keyword_list(self):
        return ["INTERSECT" for _ in range(len(self.tables) - 1)]


class Difference(_SetOp):
    def _get_keyword_list(self):
        return ["EXCEPT"] * (len(self.tables) - 1)


def _flatten_union(table: ir.TableExpr):
    """Extract all union queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr, bool]]
    """
    op = table.op()
    if isinstance(op, ops.Union):
        return toolz.concatv(
            _flatten_union(op.left), [op.distinct], _flatten_union(op.right)
        )
    return [table]


def _flatten_intersection(table: ir.TableExpr):
    """Extract all intersection queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr]]
    """
    op = table.op()
    if isinstance(op, ops.Intersection):
        return toolz.concatv(_flatten_union(op.left), _flatten_union(op.right))
    return [table]


def _flatten_difference(table: ir.TableExpr):
    """Extract all intersection queries from `table`.

    Parameters
    ----------
    table : TableExpr

    Returns
    -------
    Iterable[Union[TableExpr]]
    """
    op = table.op()
    if isinstance(op, ops.Difference):
        return toolz.concatv(_flatten_union(op.left), _flatten_union(op.right))
    return [table]


class QueryBuilder:

    select_builder = select_builder.SelectBuilder
    union_class = Union
    intersect_class = Intersection
    difference_class = Difference

    def __init__(self, expr, context):
        self.expr = expr
        self.context = context

    def generate_setup_queries(self):
        return []

    def generate_teardown_queries(self):
        return []

    def get_result(self):
        op = self.expr.op()

        # collect setup and teardown queries
        setup_queries = self.generate_setup_queries()
        teardown_queries = self.generate_teardown_queries()

        # TODO: any setup / teardown DDL statements will need to be done prior
        # to building the result set-generating statements.
        if isinstance(op, ops.Union):
            query = self._make_union()
        elif isinstance(op, ops.Intersection):
            query = self._make_intersect()
        elif isinstance(op, ops.Difference):
            query = self._make_difference()
        else:
            query = self._make_select()

        return _QueryAST(
            self.context,
            query,
            setup_queries=setup_queries,
            teardown_queries=teardown_queries,
        )

    def _make_union(self):
        # flatten unions so that we can codegen them all at once
        union_info = list(_flatten_union(self.expr))

        # since op is a union, we have at least 3 elements in union_info (left
        # distinct right) and if there is more than a single union we have an
        # additional two elements per union (distinct right) which means the
        # total number of elements is at least 3 + (2 * number of unions - 1)
        # and is therefore an odd number
        npieces = len(union_info)
        assert npieces >= 3 and npieces % 2 != 0, 'Invalid union expression'

        # 1. every other object starting from 0 is a TableExpr instance
        # 2. every other object starting from 1 is a bool indicating the type
        #    of union (distinct or not distinct)
        table_exprs, distincts = union_info[::2], union_info[1::2]
        return self.union_class(
            table_exprs, self.expr, distincts=distincts, context=self.context
        )

    def _make_intersect(self):
        # flatten intersections so that we can codegen them all at once
        table_exprs = list(_flatten_intersection(self.expr))
        return self.intersect_class(
            table_exprs, self.expr, context=self.context
        )

    def _make_difference(self):
        # flatten differences so that we can codegen them all at once
        table_exprs = list(_flatten_difference(self.expr))
        return self.difference_class(
            table_exprs, self.expr, context=self.context
        )

    def _make_select(self):
        builder = self.select_builder(self.expr, self.context)
        return builder.get_result()
