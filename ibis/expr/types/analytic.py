from __future__ import annotations

from public import public

import ibis.common.exceptions as com
import ibis.expr.types as ir
from ibis.expr.types.core import Expr


@public
class AnalyticExpr(Expr):

    # TODO(kszucs): should be removed
    def type(self):
        return 'analytic'


@public
class ExistsExpr(AnalyticExpr):
    # TODO(kszucs): should be removed
    def type(self):
        return 'exists'


@public
class TopKExpr(AnalyticExpr):
    def type(self):
        return 'topk'

    def _table_getitem(self):
        return self.to_filter()

    def to_filter(self):
        # TODO: move to api.py
        import ibis.expr.operations as ops

        return ops.SummaryFilter(self).to_expr()

    def to_aggregation(
        self, metric_name=None, parent_table=None, backup_metric_name=None
    ):
        """
        Convert the TopK operation to a table aggregation
        """
        op = self.op()

        arg_table = ir.relations.find_base_table(op.arg)

        by = op.by
        if not isinstance(by, Expr):
            by = by(arg_table)
            by_table = arg_table
        else:
            by_table = ir.relations.find_base_table(op.by)

        if metric_name is None:
            if by.get_name() == op.arg.get_name():
                by = by.name(backup_metric_name)
        else:
            by = by.name(metric_name)

        if arg_table.equals(by_table):
            agg = arg_table.aggregate(by, by=[op.arg])
        elif parent_table is not None:
            agg = parent_table.aggregate(by, by=[op.arg])
        else:
            raise com.IbisError(
                'Cross-table TopK; must provide a parent ' 'joined table'
            )

        return agg.sort_by([(by.get_name(), False)]).limit(op.k)
