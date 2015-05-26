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


import ibis.expr.types as ir
import ibis.expr.operations as ops


class Bucket(ir.ValueNode):

    def __init__(self, arg, buckets, closed='left', close_extreme=True,
                 include_under=False, include_over=False):
        self.arg = arg
        self.buckets = buckets

        self.closed = closed.lower()

        if self.closed not in ['left', 'right']:
            raise ValueError("closed must be 'left' or 'right'")

        self.close_extreme = bool(close_extreme)
        self.include_over = bool(include_over)
        self.include_under = bool(include_under)
        ir.ValueNode.__init__(self, [self.arg, self.buckets, self.closed,
                                     self.close_extreme,
                                     self.include_under,
                                     self.include_over])

    @property
    def nbuckets(self):
        k = len(self.buckets) - 1
        k += int(self.include_over) + int(self.include_under)
        return k

    def output_type(self):
        ctype = ir.CategoryType(self.nbuckets)
        return ctype.array_ctor()


class Histogram(ir.ValueNode):

    def __init__(self, arg, nbins, binwidth, base, closed='left',
                 close_extreme=True):
        self.arg = arg
        self.nbins = nbins
        self.binwidth = binwidth
        self.base = base
        self.closed = closed
        self.close_extreme = close_extreme
        self.include_over = include_over
        ir.ValueNode.__init__(self, [self.arg, self.nbins, self.binwidth,
                                     self.base, self.closed,
                                     self.close_extreme])

    def output_type(self):
        # always undefined cardinality (for now)
        ctype = ir.CategoryType()
        return ctype.array_ctor()


def bucket(arg, buckets, closed='left', close_extreme=True,
           include_under=False, include_over=False):
    """

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
              close_extreme=True, aux_hash=None):
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
                   close_extreme=close_extreme,
                   include_over=include_over)
    return op.to_expr()
