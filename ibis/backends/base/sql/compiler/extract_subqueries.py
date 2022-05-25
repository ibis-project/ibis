from collections import Counter

import ibis.expr.operations as ops
import ibis.expr.types as ir


class ExtractSubqueries:
    __slots__ = "query", "counts"

    def __init__(self, query):
        self.query = query
        self.counts = Counter()

    @classmethod
    def extract(cls, select_stmt):
        helper = cls(select_stmt)
        return helper.get_result()

    def get_result(self):
        query = self.query
        if query.table_set is not None:
            self.visit(query.table_set)

        for clause in query.filters:
            self.visit(clause)

        return [op.to_expr() for op, count in self.counts.items() if count > 1]

    def observe(self, expr):
        self.counts[expr.op()] += 1

    def visit(self, expr):
        node = expr.op()

        if (
            method := getattr(self, f"visit_{type(node).__name__}", None)
        ) is not None:
            method(expr)
        elif isinstance(node, ops.Join):
            self.visit_join(expr)
        elif isinstance(node, ops.PhysicalTable):
            self.visit_physical_table(expr)
        elif isinstance(node, ops.TableNode):
            for arg in node.flat_args():
                if isinstance(arg, ir.Table):
                    self.visit(arg)
            self.observe(expr)
        elif isinstance(node, ops.Value):
            for arg in node.flat_args():
                if not isinstance(arg, ir.Expr):
                    continue
                self.visit(arg)
        else:
            raise NotImplementedError(type(node))

    def visit_join(self, expr):
        node = expr.op()
        self.visit(node.left)
        self.visit(node.right)

    def visit_physical_table(self, _):
        return

    def visit_Exists(self, expr):
        node = expr.op()
        self.visit(node.foreign_table)
        for pred in node.predicates:
            self.visit(pred)

    visit_NotExistsSubquery = visit_ExistsSubquery = visit_Exists

    def visit_Distinct(self, expr):
        self.observe(expr)

    def visit_Selection(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def visit_SQLQueryResult(self, expr):
        self.observe(expr)

    def visit_TableColumn(self, expr):
        table = expr.op().table
        if not self.counts[table.op()]:
            self.visit(table)

    def visit_SelfReference(self, expr):
        self.visit(expr.op().table)
