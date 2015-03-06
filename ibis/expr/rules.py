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

from collections import defaultdict
import operator

import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.util as util


class BinaryPromoter(object):
    # placeholder for type promotions for basic binary arithmetic

    def __init__(self, left, right, op):
        self.args = [left, right]
        self.left = left
        self.right = right
        self.op = op

        self._check_compatibility()

    def get_result(self):
        promoted_type = self._get_type()
        return ops._shape_like_args(self.args, promoted_type)

    def _get_type(self):
        if util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        else:
            raise NotImplementedError

    def _get_int_type(self):
        deps = [x.op() for x in self.args]

        if util.all_of(deps, ir.Literal):
            return _smallest_int_containing(
                [self.op(deps[0].value, deps[1].value)])
        elif util.any_of(deps, ir.Literal):
            if isinstance(deps[0], ir.Literal):
                val = deps[0].value
                atype = self.args[1].type()
            else:
                val = deps[1].value
                atype = self.args[0].type()
            return _int_one_literal_promotion(atype, val, self.op)
        else:
            return _int_bounds_promotion(self.left.type(),
                                         self.right.type(), self.op)

    def _check_compatibility(self):
        if (util.any_of(self.args, ir.StringValue) and
            not util.all_of(self.args, ir.StringValue)):
            raise TypeError('String and non-string incompatible')


class PowerPromoter(BinaryPromoter):

    def __init__(self, left, right):
        super(PowerPromoter, self).__init__(left, right, operator.pow)

    def _get_type(self):
        rval = self.args[1].op()

        if util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif isinstance(rval, ir.Literal) and rval.value < 0:
            return 'double'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        else:
            raise NotImplementedError



_nbytes = {
    'int8': 1,
    'int16': 2,
    'int32': 4,
    'int64': 8
}


_int_bounds = {
    'int8': (-128, 127),
    'int16': (-32768, 32767),
    'int32': (-2147483648, 2147483647),
    'int64': (-9223372036854775808, 9223372036854775807)
}



def highest_precedence_type(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    selector = _TypePrecedence(exprs)
    return selector.get_result()


class _TypePrecedence(object):
    # Impala type precedence (more complex in other database implementations)
    # timestamp
    # double
    # float
    # decimal
    # bigint
    # int
    # smallint
    # tinyint
    # boolean
    # string

    _precedence = ['double', 'float', 'decimal',
                   'int64', 'int32', 'int16', 'int8',
                   'boolean', 'string']

    def __init__(self, exprs):
        self.exprs = exprs

        if len(exprs) == 0:
            raise ValueError('Must pass at least one expression')

        self.type_counts = defaultdict(lambda: 0)
        self._count_types()

    def get_result(self):
        highest_type = self._get_highest_type()
        self._check_casts(highest_type)
        return highest_type

    def _count_types(self):
        for expr in self.exprs:
            self.type_counts[expr._base_type()] += 1

    def _get_highest_type(self):
        for typename in self._precedence:
            if self.type_counts[typename] > 0:
                return typename

    def _check_casts(self, typename):
        for expr in self.exprs:
            if not expr._can_cast_implicit(typename):
                raise ValueError('Expression with type {} cannot be '
                                 'implicitly casted to {}'
                                 .format(expr.type(), typename))


def _int_bounds_promotion(ltype, rtype, op):
    lmin, lmax = _int_bounds[ltype]
    rmin, rmax = _int_bounds[rtype]

    values = [op(lmin, rmin), op(lmin, rmax),
              op(lmax, rmin), op(lmax, rmax)]

    return _smallest_int_containing(values, allow_overflow=True)


def _int_one_literal_promotion(atype, lit_val, op):
    amin, amax = _int_bounds[atype]
    bound_type = _smallest_int_containing([op(amin, lit_val),
                                           op(amax, lit_val)],
                                          allow_overflow=True)
    # In some cases, the bounding type might be int8, even though neither of
    # the types are that small. We want to ensure the containing type is _at
    # least_ as large as the smallest type in the expression
    return _largest_int([bound_type, atype])


def _smallest_int_containing(values, allow_overflow=False):
    containing_types = [int_literal_class(x, allow_overflow=allow_overflow)
                        for x in values]
    return _largest_int(containing_types)


def int_literal_class(value, allow_overflow=False):
    if -128 <= value <= 127:
        scalar_type = 'int8'
    elif -32768 <= value <= 32767:
        scalar_type = 'int16'
    elif -2147483648 <= value <= 2147483647:
        scalar_type = 'int32'
    else:
        if value < -9223372036854775808 or value > 9223372036854775807:
            if not allow_overflow:
                raise OverflowError(value)
        scalar_type = 'int64'
    return scalar_type


def _largest_int(int_types):
    nbytes = max(_nbytes[t] for t in int_types)
    return 'int%d' % (8 * nbytes)
