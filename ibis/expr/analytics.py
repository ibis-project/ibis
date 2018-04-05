# Copyright 2015 Cloudera Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import ibis.expr.rules as rlz
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops

from ibis.expr.signature import Argument as Arg


class BucketLike(ops.ValueOp):

    @property
    def nbuckets(self):
        return None

    def output_type(self):
        dtype = dt.Category(self.nbuckets)
        return dtype.array_type()


class Bucket(BucketLike):
    arg = Arg(rlz.noop)
    buckets = Arg(rlz.noop)
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


class Histogram(BucketLike):
    arg = Arg(rlz.noop)
    nbins = Arg(rlz.noop, default=None)
    binwidth = Arg(rlz.noop, default=None)
    base = Arg(rlz.noop, default=None)
    closed = Arg(rlz.isin({'left', 'right'}), default='left')
    aux_hash = Arg(rlz.noop, default=None)

    def _validate(self):
        if self.nbins is None:
            if self.binwidth is None:
                raise ValueError('Must indicate nbins or binwidth')
        elif self.binwidth is not None:
            raise ValueError('nbins and binwidth are mutually exclusive')

    def output_type(self):
        # always undefined cardinality (for now)
        return dt.category.array_type()


class CategoryLabel(ops.ValueOp):
    arg = Arg(rlz.category)
    labels = Arg(rlz.noop)
    nulls = Arg(rlz.noop, default=None)
    output_type = rlz.shape_like('arg', dt.string)

    def _validate(self):
        cardinality = self.arg.type().cardinality
        if len(self.labels) != cardinality:
            raise ValueError('Number of labels must match number of '
                             'categories: {}'.format(cardinality))


def bucket(arg, buckets, closed='left', close_extreme=True,
           include_under=False, include_over=False):
    """
    Compute a discrete binning of a numeric array

    Parameters
    ----------
    arg : numeric array expression
    buckets : list
    closed : {'left', 'right'}, default 'left'
      Which side of each interval is closed. For example
      buckets = [0, 100, 200]
      closed = 'left': 100 falls in 2nd bucket
      closed = 'right': 100 falls in 1st bucket
    close_extreme : boolean, default True

    Returns
    -------
    bucketed : coded value expression
    """
    op = Bucket(arg, buckets, closed=closed, close_extreme=close_extreme,
                include_under=include_under, include_over=include_over)
    return op.to_expr()


def histogram(arg, nbins=None, binwidth=None, base=None, closed='left',
              aux_hash=None):
    """
    Compute a histogram with fixed width bins

    Parameters
    ----------
    arg : numeric array expression
    nbins : int, default None
      If supplied, will be used to compute the binwidth
    binwidth : number, default None
      If not supplied, computed from the data (actual max and min values)
    base : number, default None
    closed : {'left', 'right'}, default 'left'
      Which side of each interval is closed

    Returns
    -------
    histogrammed : coded value expression
    """
    op = Histogram(arg, nbins, binwidth, base, closed=closed,
                   aux_hash=aux_hash)
    return op.to_expr()


def category_label(arg, labels, nulls=None):
    """
    Format a known number of categories as strings

    Parameters
    ----------
    labels : list of string
    nulls : string, optional
      How to label any null values among the categories

    Returns
    -------
    string_categories : string value expression
    """
    op = CategoryLabel(arg, labels, nulls)
    return op.to_expr()
