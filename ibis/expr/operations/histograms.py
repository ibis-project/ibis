from public import public

from .. import datatypes as dt
from .. import rules as rlz
from ..signature import Argument as Arg
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
    arg = Arg(rlz.column(rlz.any))
    buckets = Arg(rlz.list_of(rlz.scalar(rlz.any)))
    closed = Arg(rlz.isin({'left', 'right'}), default='left')
    close_extreme = Arg(bool, default=True)
    include_under = Arg(bool, default=False)
    include_over = Arg(bool, default=False)

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
    arg = Arg(rlz.numeric)
    nbins = Arg(int, default=None)
    binwidth = Arg(rlz.scalar(rlz.numeric), default=None)
    base = Arg(rlz.scalar(rlz.numeric), default=None)
    closed = Arg(rlz.isin({'left', 'right'}), default='left')
    aux_hash = Arg(str, default=None)

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
    arg = Arg(rlz.category)
    labels = Arg(rlz.list_of(rlz.instance_of(str)))
    nulls = Arg(str, default=None)
    output_type = rlz.shape_like('arg', dt.string)

    def _validate(self):
        cardinality = self.arg.type().cardinality
        if len(self.labels) != cardinality:
            raise ValueError(
                'Number of labels must match number of '
                f'categories: {cardinality}'
            )
