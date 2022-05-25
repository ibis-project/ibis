from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Sequence

import ibis.expr.operations as ops
import ibis.expr.types as ir

if TYPE_CHECKING:
    from ibis.backends.base.sql.compiler import SelectBuilder


class ExtractSubqueries:
    __slots__ = "query", "counts"

    def __init__(self, query) -> None:
        self.query = query
        self.counts = Counter()

    @classmethod
    def extract(cls, select_stmt: SelectBuilder):
        helper = cls(select_stmt)
        return helper.get_result()

    def get_result(self) -> Sequence[ir.Table]:
        query = self.query
        if query.table_set is not None:
            self.visit(query.table_set.op())

        for clause in query.filters:
            self.visit(clause.op())

        return [op.to_expr() for op, count in self.counts.items() if count > 1]

    def observe(self, node: ops.Node) -> None:
        self.counts[node] += 1

    def visit(self, node: ops.Node) -> None:
        if (
            method := getattr(self, f"visit_{type(node).__name__}", None)
        ) is not None:
            method(node)
        elif isinstance(node, ops.Join):
            self.visit_join(node)
        elif isinstance(node, ops.PhysicalTable):
            # do nothing, because physical tables can be referred to by name
            pass
        elif isinstance(node, ops.TableNode):
            for arg in node._flat_ops:
                if isinstance(arg, ops.TableNode):
                    self.visit(arg)
            self.observe(node)
        elif isinstance(node, ops.Value):
            for arg in node._flat_ops:
                self.visit(arg)
        else:
            raise NotImplementedError(type(node))

    def visit_join(self, node: ir.Join) -> None:
        self.visit(node.left.op())
        self.visit(node.right.op())

    def visit_ExistsSubquery(
        self, node: ops.ExistsSubquery | ops.NotExistsSubquery
    ) -> None:
        self.visit(node.foreign_table.op())
        for pred in node.predicates:
            self.visit(pred.op())

    visit_NotExistsSubquery = visit_ExistsSubquery

    def visit_Distinct(self, node: ops.Distinct) -> None:
        self.observe(node)

    def visit_Selection(self, node: ops.Selection) -> None:
        self.visit(node.table.op())
        self.observe(node)

    def visit_SQLQueryResult(self, node: ops.SQLQueryResult) -> None:
        self.observe(node)

    def visit_TableColumn(self, node: ops.TableColumn) -> None:
        table = node.table.op()
        if not self.counts[table]:
            self.visit(table)

    def visit_SelfReference(self, node):
        self.visit(node.table.op())
