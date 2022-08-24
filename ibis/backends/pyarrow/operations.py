import ibis.expr.rules as rlz
import ibis.expr.operations as ops
import ibis.expr.types as ir


# perhaps needs a Computable trait too


class Result(ops.Node):
    pass


class TableResult(Result):
    pass


class TableResult(ir.Expr):
    pass


class ArrowTableResult(TableResult):
    table = rlz.table

    def to_expr(self):
        return TableResult(self)


def to_pyarrow(self):
    return ArrowTableResult(self).to_expr()


ir.Table.to_pyarrow = to_pyarrow
