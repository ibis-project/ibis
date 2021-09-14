from collections import OrderedDict

import ibis.expr.operations as ops
import ibis.expr.types as ir


class ExtractSubqueries:
    def __init__(self, query, greedy=False):
        self.query = query
        self.greedy = greedy
        self.expr_counts = OrderedDict()
        self.node_to_expr = {}

    @classmethod
    def extract(cls, select_stmt):
        helper = cls(select_stmt)
        return helper.get_result()

    def get_result(self):
        if self.query.table_set is not None:
            self.visit(self.query.table_set)

        for clause in self.query.filters:
            self.visit(clause)

        expr_counts = self.expr_counts

        if self.greedy:
            to_extract = list(expr_counts.keys())
        else:
            to_extract = [op for op, count in expr_counts.items() if count > 1]

        node_to_expr = self.node_to_expr
        return [node_to_expr[op] for op in to_extract]

    def observe(self, expr):
        key = expr.op()

        if key not in self.node_to_expr:
            self.node_to_expr[key] = expr

        assert self.node_to_expr[key].equals(expr)
        self.expr_counts[key] = self.expr_counts.setdefault(key, 0) + 1

    def seen(self, expr):
        return expr.op() in self.expr_counts

    def visit(self, expr):
        node = expr.op()
        method = f'visit_{type(node).__name__}'

        if hasattr(self, method):
            f = getattr(self, method)
            f(expr)
        elif isinstance(node, ops.Join):
            self.visit_join(expr)
        elif isinstance(node, ops.PhysicalTable):
            self.visit_physical_table(expr)
        elif isinstance(node, ops.ValueOp):
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

    def visit_MaterializedJoin(self, expr):
        self.visit(expr.op().join)
        self.observe(expr)

    def visit_Selection(self, expr):
        self.visit(expr.op().table)
        self.observe(expr)

    def visit_SQLQueryResult(self, expr):
        self.observe(expr)

    def visit_TableColumn(self, expr):
        table = expr.op().table
        if not self.seen(table):
            self.visit(table)

    def visit_SelfReference(self, expr):
        self.visit(expr.op().table)
