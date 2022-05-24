from __future__ import annotations

from public import public

import ibis.common.exceptions as com
from ibis.expr.types.core import Expr
from ibis.util import deprecated


@public
class Analytic(Expr):

    # TODO(kszucs): should be removed
    def type(self):
        return 'analytic'


@public
@deprecated(
    instead="remove usage of Exists/ExistsExpr, it will be removed",
    version="4.0.0",
)
class Exists(Analytic):
    # TODO(kszucs): should be removed
    def type(self):
        return 'exists'


@public
class TopK(Analytic):
    def type(self):
        return 'topk'

    def _semi_join_components(self):
        predicate = self._to_filter()
        right = predicate.op().right.op().table
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
        arg = op.arg
        return arg == getattr(rank_set, arg.get_name())

    @deprecated(
        instead=(
            "use __getitem__ on the relevant child table to produce a "
            "semi-join"
        ),
        version="4.0.0",
    )
    def to_filter(self):
        import ibis.expr.operations as ops

        return ops.SummaryFilter(self).to_expr()

    def to_aggregation(
        self, metric_name=None, parent_table=None, backup_metric_name=None
    ):
        """
        Convert the TopK operation to a table aggregation
        """
        from ibis.expr.analysis import find_first_base_table

        op = self.op()

        arg_table = find_first_base_table(op.arg)

        by = op.by
        if isinstance(by, Expr):
            by_table = find_first_base_table(op.by)
        else:
            by = by(arg_table)
            by_table = arg_table

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
                'Cross-table TopK; must provide a parent joined table'
            )

        return agg.sort_by([(by.get_name(), False)]).limit(op.k)


public(AnalyticExpr=Analytic, ExistsExpr=Exists, TopKExpr=TopK)
