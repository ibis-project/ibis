from __future__ import annotations

from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Value


@public
class Bucket(Value):
    arg = rlz.column(rlz.any)
    buckets = rlz.tuple_of(rlz.scalar(rlz.any))
    closed = rlz.optional(rlz.isin({'left', 'right'}), default='left')
    close_extreme = rlz.optional(rlz.instance_of(bool), default=True)
    include_under = rlz.optional(rlz.instance_of(bool), default=False)
    include_over = rlz.optional(rlz.instance_of(bool), default=False)
    output_shape = rlz.Shape.COLUMNAR

    @property
    def output_dtype(self):
        return dt.Category(self.nbuckets)

    def __init__(self, buckets, include_under, include_over, **kwargs):
        if not len(buckets):
            raise ValueError('Must be at least one bucket edge')
        elif len(buckets) == 1:
            if not include_under or not include_over:
                raise ValueError(
                    'If one bucket edge provided, must have '
                    'include_under=True and include_over=True'
                )
        super().__init__(
            buckets=buckets,
            include_under=include_under,
            include_over=include_over,
            **kwargs,
        )

    @property
    def nbuckets(self):
        return len(self.buckets) - 1 + self.include_over + self.include_under


@public
class CategoryLabel(Value):
    arg = rlz.category
    labels = rlz.tuple_of(rlz.instance_of(str))
    nulls = rlz.optional(rlz.instance_of(str))

    output_dtype = dt.string
    output_shape = rlz.shape_like("arg")

    def __init__(self, arg, labels, **kwargs):
        cardinality = arg.output_dtype.cardinality
        if len(labels) != cardinality:
            raise ValueError(
                f"Number of labels must match number of categories: {cardinality}"
            )
        super().__init__(arg=arg, labels=labels, **kwargs)
