from public import public

from ibis.expr import datatypes as dt
from ibis.expr import rules as rlz
from ibis.expr.operations.core import Value


@public
class BucketLike(Value):
    output_shape = rlz.Shape.COLUMNAR

    @property
    def nbuckets(self):
        return None

    @property
    def output_dtype(self):
        return dt.Category(self.nbuckets)


@public
class Bucket(BucketLike):
    arg = rlz.column(rlz.any)
    buckets = rlz.tuple_of(rlz.scalar(rlz.any))
    closed = rlz.optional(rlz.isin({'left', 'right'}), default='left')
    close_extreme = rlz.optional(rlz.instance_of(bool), default=True)
    include_under = rlz.optional(rlz.instance_of(bool), default=False)
    include_over = rlz.optional(rlz.instance_of(bool), default=False)

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
class Histogram(BucketLike):
    arg = rlz.numeric
    nbins = rlz.optional(rlz.instance_of(int))
    binwidth = rlz.optional(rlz.scalar(rlz.numeric))
    base = rlz.optional(rlz.scalar(rlz.numeric))
    closed = rlz.optional(rlz.isin({'left', 'right'}), default='left')
    aux_hash = rlz.optional(rlz.instance_of(str))

    def __init__(self, nbins, binwidth, **kwargs):
        if nbins is None:
            if binwidth is None:
                raise ValueError('Must indicate nbins or binwidth')
        elif binwidth is not None:
            raise ValueError('nbins and binwidth are mutually exclusive')
        super().__init__(nbins=nbins, binwidth=binwidth, **kwargs)

    @property
    def output_dtype(self):
        # always undefined cardinality (for now)
        return dt.category


@public
class CategoryLabel(Value):
    arg = rlz.category
    labels = rlz.tuple_of(rlz.instance_of(str))
    nulls = rlz.optional(rlz.instance_of(str))

    output_dtype = dt.string
    output_shape = rlz.shape_like("arg")

    def __init__(self, arg, labels, **kwargs):
        cardinality = arg.type().cardinality
        if len(labels) != cardinality:
            raise ValueError(
                'Number of labels must match number of '
                f'categories: {cardinality}'
            )
        super().__init__(arg=arg, labels=labels, **kwargs)
