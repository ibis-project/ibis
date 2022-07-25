from __future__ import annotations

from public import public

import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.expr.types.core import Expr


@public
class Analytic(Expr):

    # TODO(kszucs): should be removed
    def type(self):
        return 'analytic'


@public
class TopK(Analytic):
    def type(self):
        return 'topk'

    def _semi_join_components(self):
        predicate = self._to_filter()
        right = predicate.op().right.table
        return predicate, right

    def _to_semi_join(self, left):
        predicate, right = self._semi_join_components()
        return left.semi_join(right, predicate)

    def _to_filter(self):
        op = self.op()

        rank_set = self.to_aggregation(backup_metric_name='__tmp__')

        # GH 1393: previously because of GH667 we were substituting parents,
        # but that introduced a bug when comparing reductions to columns on the
        # same relation, so we leave this alone.
        arg = op.arg.to_expr()
        return arg == getattr(rank_set, arg.get_name())

    def to_aggregation(
        self, metric_name=None, parent_table=None, backup_metric_name=None
    ):
        """
        Convert the TopK operation to a table aggregation
        """
        from ibis.expr.analysis import find_first_base_table

        op = self.op()

        arg_table = find_first_base_table(op.arg).to_expr()

        by = op.by
        if callable(by):
            by = by(arg_table)
            by_table = arg_table
        elif isinstance(by, ops.Value):
            by_table = find_first_base_table(op.by).to_expr()
        else:
            raise com.IbisTypeError(
                f"Invalid `by` argument with type {type(by)}"
            )

        if metric_name is None:
            if by.name == op.arg.name:
                by = by.name(backup_metric_name)
        else:
            by = by.name(metric_name)

        if arg_table.equals(by_table):
            agg = arg_table.aggregate(by.to_expr(), by=[op.arg.to_expr()])
        elif parent_table is not None:
            agg = parent_table.aggregate(by.to_expr(), by=[op.arg.to_expr()])
        else:
            raise com.IbisError(
                'Cross-table TopK; must provide a parent joined table'
            )

        return agg.sort_by([(by.name, False)]).limit(op.k)


public(AnalyticExpr=Analytic, TopKExpr=TopK)
