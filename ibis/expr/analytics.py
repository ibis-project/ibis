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


import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.expr.rules as rules
import ibis.expr.operations as ops


class BucketLike(ir.ValueNode):

    def _validate_closed(self, closed):
        closed = closed.lower()
        if closed not in ['left', 'right']:
            raise ValueError("closed must be 'left' or 'right'")
        return closed

    @property
    def nbuckets(self):
        return None

    def output_type(self):
        ctype = dt.Category(self.nbuckets)
        return ctype.array_type()


class Bucket(BucketLike):

    def __init__(self, arg, buckets, closed='left', close_extreme=True,
                 include_under=False, include_over=False):
        self.arg = arg
        self.buckets = buckets
        self.closed = self._validate_closed(closed)

        self.close_extreme = bool(close_extreme)
        self.include_over = bool(include_over)
        self.include_under = bool(include_under)

        if len(buckets) == 0:
            raise ValueError('Must be at least one bucket edge')
        elif len(buckets) == 1:
            if not self.include_under or not self.include_over:
                raise ValueError('If one bucket edge provided, must have'
                                 ' include_under=True and include_over=True')

        ir.ValueNode.__init__(self, self.arg, self.buckets, self.closed,
                              self.close_extreme, self.include_under,
                              self.include_over)

    @property
    def nbuckets(self):
        k = len(self.buckets) - 1
        k += int(self.include_over) + int(self.include_under)
        return k


class Histogram(BucketLike):

    def __init__(self, arg, nbins, binwidth, base, closed='left',
                 aux_hash=None):
        self.arg = arg

        self.nbins = nbins
        self.binwidth = binwidth
        self.base = base

        if self.nbins is None:
            if self.binwidth is None:
                raise ValueError('Must indicate nbins or binwidth')
        elif self.binwidth is not None:
            raise ValueError('nbins and binwidth are mutually exclusive')

        self.closed = self._validate_closed(closed)

        self.aux_hash = aux_hash
        ir.ValueNode.__init__(self, self.arg, self.nbins, self.binwidth,
                              self.base, self.closed, self.aux_hash)

    def output_type(self):
        # always undefined cardinality (for now)
        ctype = dt.Category()
        return ctype.array_type()


class CategoryLabel(ir.ValueNode):

    def __init__(self, arg, labels, nulls):
        self.arg = ops.as_value_expr(arg)
        self.labels = labels

        card = self.arg.type().cardinality
        if len(self.labels) != card:
            raise ValueError('Number of labels must match number of '
                             'categories: %d' % card)

        self.nulls = nulls
        ir.ValueNode.__init__(self, self.arg, self.labels, self.nulls)

    def output_type(self):
        return rules.shape_like(self.arg, 'string')


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
