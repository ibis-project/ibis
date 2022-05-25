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

    def seen(self, expr) -> bool:
        return bool(self.counts[expr.op()])

    def visit(self, expr):
        node = expr.op()
        method = f"visit_{type(node).__name__}"

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr)
        elif isinstance(node, ops.Join):
            self.visit_join(expr)
        elif isinstance(node, ops.PhysicalTable):
            self.visit_physical_table(expr)
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

    def visit_Aggregation(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def visit_Distinct(self, expr):
        self.observe(expr)

    def visit_Limit(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def visit_Union(self, expr):
        op = expr.op()
        self.visit(op.left)
        self.visit(op.right)
        self.observe(expr)

    def visit_Intersection(self, expr):
        op = expr.op()
        self.visit(op.left)
        self.visit(op.right)
        self.observe(expr)

    def visit_Difference(self, expr):
        op = expr.op()
        self.visit(op.left)
        self.visit(op.right)
        self.observe(expr)

    def visit_Selection(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def visit_SQLQueryResult(self, expr):
        self.observe(expr)

    def visit_View(self, expr):
        self.visit(expr.op().child)
        self.observe(expr)

    def visit_SQLStringView(self, expr):
        self.visit(expr.op().child)
        self.observe(expr)

    def visit_TableColumn(self, expr):
        table = expr.op().table
        if not self.seen(table):
            self.visit(table)

    def visit_SelfReference(self, expr):
        self.visit(expr.op().table)
