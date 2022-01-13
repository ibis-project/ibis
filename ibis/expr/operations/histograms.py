from public import public

from .. import datatypes as dt
from .. import rules as rlz
from .core import ValueOp


@public
class BucketLike(ValueOp):
    @property
    def nbuckets(self):
        return None

    def output_type(self):
        dtype = dt.Category(self.nbuckets)
        return dtype.column_type()


@public
class Bucket(BucketLike):
    arg = rlz.column(rlz.any)
    buckets = rlz.list_of(rlz.scalar(rlz.any))
    closed = rlz.optional(rlz.isin({'left', 'right'}), default='left')
    close_extreme = rlz.optional(rlz.instance_of(bool), default=True)
    include_under = rlz.optional(rlz.instance_of(bool), default=False)
    include_over = rlz.optional(rlz.instance_of(bool), default=False)

    def _validate(self):
        if not len(self.buckets):
            raise ValueError('Must be at least one bucket edge')
        elif len(self.buckets) == 1:
            if not self.include_under or not self.include_over:
                raise ValueError(
                    'If one bucket edge provided, must have '
                    'include_under=True and include_over=True'
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

    def _validate(self):
        if self.nbins is None:
            if self.binwidth is None:
                raise ValueError('Must indicate nbins or binwidth')
        elif self.binwidth is not None:
            raise ValueError('nbins and binwidth are mutually exclusive')

    def output_type(self):
        # always undefined cardinality (for now)
        return dt.category.column_type()


@public
class CategoryLabel(ValueOp):
    arg = rlz.category
    labels = rlz.list_of(rlz.instance_of(str))
    nulls = rlz.optional(rlz.instance_of(str))
    output_type = rlz.shape_like('arg', dt.string)

    def _validate(self):
        cardinality = self.arg.type().cardinality
        if len(self.labels) != cardinality:
            raise ValueError(
                'Number of labels must match number of '
                f'categories: {cardinality}'
            )
