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

import abc
import six
import sys
import operator

from collections import Counter

from ibis.common import IbisTypeError
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.common as com
import ibis.util as util


# TODO create a promoter decorator?


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
        return shape_like_args(self.args, promoted_type)

    def _get_type(self):
        if util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
        elif util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        elif self.left.type().equals(self.right.type()):
            return self.left.type()
        else:
            raise NotImplementedError(
                'Operands {}, {} not supported for binary operation {}'.format(
                    type(self.left).__name__, type(self.right).__name__,
                    self.op.__name__
                )
            )

    def _get_int_type(self):
        import ibis.expr.operations as ops

        deps = [x.op() for x in self.args]

        if util.all_of(deps, ops.Literal):
            return _smallest_int_containing(
                [self.op(deps[0].value, deps[1].value)])
        elif util.any_of(deps, ops.Literal):
            if isinstance(deps[0], ops.Literal):
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


class IntervalPromoter(BinaryPromoter):
    """Infers the output type of the binary interval operation

    This is a slightly modified version of BinaryPromoter, it converts
    back and forth between the interval and its inner value.

    This trick reuses the numeric type promotion logics.
    Any non-integer output type raises a TypeError.
    """

    def __init__(self, left, right, op):
        left_type = left.type()
        value_type = shape_like(left, left_type.value_type)
        self.unit = left_type.unit
        super(IntervalPromoter, self).__init__(value_type(left), right, op)

    def get_result(self):
        promoted_value_type = self._get_type()
        promoted_type = dt.Interval(self.unit, promoted_value_type)
        return shape_like_args(self.args, promoted_type)


# TODO: move to datatypes castable rule
def _decimal_promoted_type(args):
    max_precision = max_scale = ~sys.maxsize
    for arg in args:
        if isinstance(arg, ir.DecimalValue):
            max_precision = max(max_precision, arg.meta.precision)
            max_scale = max(max_scale, arg.meta.scale)
    return dt.Decimal(max_precision, max_precision)


class PowerPromoter(BinaryPromoter):

    def __init__(self, left, right):
        super(PowerPromoter, self).__init__(left, right, operator.pow)

    def _get_type(self):
        if util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
        elif util.all_of(self.args, ir.IntegerValue):
            return 'double'
        else:
            raise NotImplementedError(
                'Operands {}, {} not supported for binary operation {}'.format(
                    type(self.left).__name__, type(self.right).__name__,
                    self.op.__name__
                )
            )


def highest_precedence_type(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type
    if not exprs:
        raise ValueError('Must pass at least one expression')

    expr_dtypes = {expr.type() for expr in exprs}
    return dt.highest_precedence(expr_dtypes)


def _int_bounds_promotion(ltype, rtype, op):
    lmin, lmax = ltype.bounds
    rmin, rmax = rtype.bounds

    values = [op(lmin, rmin), op(lmin, rmax),
              op(lmax, rmin), op(lmax, rmax)]

    return _smallest_int_containing(values, allow_overflow=True)


def _int_one_literal_promotion(atype, lit_val, op):
    amin, amax = atype.bounds
    bound_type = _smallest_int_containing([op(amin, lit_val),
                                           op(amax, lit_val)],
                                          allow_overflow=True)
    # In some cases, the bounding type might be int8, even though neither of
    # the types are that small. We want to ensure the containing type is _at
    # least_ as large as the smallest type in the expression
    return _largest_int([bound_type, atype])


def _smallest_int_containing(values, allow_overflow=False):
    containing_types = [dt.infer(x, allow_overflow=allow_overflow)
                        for x in values]
    return _largest_int(containing_types)


def _largest_int(int_types):
    nbytes = max(t._nbytes for t in int_types)
    return dt.validate_type('int%d' % (8 * nbytes))


# ----------------------------------------------------------------------
# Input / output type rules and validation


def shape_like(arg, out_type):
    out_type = dt.validate_type(out_type)
    if isinstance(arg, ir.ScalarExpr):
        return out_type.scalar_type()
    else:
        return out_type.array_type()


def shape_like_args(args, out_type):
    out_type = dt.validate_type(out_type)
    if util.any_of(args, ir.ColumnExpr):
        return out_type.array_type()
    else:
        return out_type.scalar_type()



# class TypeSignature(object):

#     def __init__(self, type_specs):
#         types = []

#         for val in type_specs:
#             val = _to_argument(val)
#             types.append(val)

#         self.types = types

#     def __repr__(self):
#         types = '\n    '.join('arg {0}: {1}'.format(i, repr(x))
#                               for i, x in enumerate(self.types))
#         return '{0}\n    {1}'.format(type(self), types)

#     def validate(self, args):
#         n, k = len(args), len(self.types)
#         k_required = len([x for x in self.types if not x.optional])
#         if k != k_required:
#             if n < k_required:
#                 raise com.IbisError('Expected at least {0} args, got {1}'
#                                     .format(k, k_required))
#         elif n != k:
#             raise com.IbisError('Expected {0} args, got {1}'.format(k, n))

#         if n < k:
#             args = list(args) + [t.default for t in self.types[n:]]

#         return self._validate(args, self.types)

#     def _validate(self, args, types):
#         clean_args = list(args)
#         for i, validator in enumerate(types):
#             try:
#                 clean_args[i] = validator.validate(clean_args, i)
#             except IbisTypeError as e:
#                 exc = e.args[0]
#                 msg = ('Argument {0}: {1}'.format(i, exc) +
#                        '\nArgument was: {0}'.format(ir._safe_repr(args[i])))
#                 raise IbisTypeError(msg)

#         return clean_args


def scalar_output(rule):
    def f(self):
        if isinstance(rule, dt.DataType):
            t = rule
        else:
            t = dt.validate_type(rule(self))
        return t.scalar_type()
    return f


def array_output(rule):
    def f(self):
        if isinstance(rule, dt.DataType):
            t = rule
        else:
            t = dt.validate_type(rule(self))
        return t.array_type()
    return f


def shape_like_flatargs(out_type):

    def output_type(self):
        flattened = list(self.flat_args())
        return shape_like_args(flattened, out_type)

    return output_type


def shape_like_arg(i, out_type):

    def output_type(self):
        return shape_like(self.args[i], out_type)

    return output_type


def type_of_arg(i):

    def output_type(self):
        return self.args[i]._factory

    return output_type


def comparable(left, right):
    return ir.castable(left, right) or ir.castable(right, left)


@six.add_metaclass(abc.ABCMeta)
class TableColumnValidator(object):
    @abc.abstractmethod
    def validate(self):
        pass


class SubsetValidator(TableColumnValidator):
    def __init__(self, *rules):
        self.rules = self._validate_rules(rules)

    def _validate_rules(self, rules):
        for column_rule in rules:
            # Members of a schema are arguments with a name
            if not isinstance(column_rule, Argument):
                raise ValueError(
                    'Arguments of subset schema must be instances of the '
                    'Argument class (rules).')
            if column_rule.name is None:
                raise ValueError(
                    'Column rules must be named inside a table.')
        return rules

    def validate(self, arg):
        if isinstance(arg, ir.TableExpr):
            # Check that columns match the schema first
            for column_rule in self.rules:
                if column_rule.name not in arg:
                    if column_rule.optional:
                        continue
                    else:
                        raise IbisTypeError(
                            'No column with name {}.'.format(column_rule.name))

                column = arg[column_rule.name]
                try:
                    # Arguments must validate the column
                    column_rule.validate([column], 0)
                except IbisTypeError as e:
                    six.raise_from(
                        IbisTypeError('Could not satisfy rule: {}.'.format(
                            str(column_rule))), e)


class Table(Argument):
    """A table argument.

    Parameters
    ----------
    name : str
        The name of the table argument.
    optional : bool
        Whether this table argument is optional or not.
    schema : TableColumnValidator
        A validator for the table's columns. Only column subset validators are
        currently supported. One can be created through the class method
        ``Table.with_column_subset``. See the example for usage.
    doc : str
        A docstring to document this argument.
    validator : Argument
        Allows adding custom validation logic to this argument.

    Examples
    --------
    The following op will accept an argument named ``'table'``. Note that the
    ``schema`` argument specifies rules for columns that are required to be in
    the table: ``time``, ``group`` and ``value1``. These must match the types
    specified in the column rules. Column ``value2`` is optional, but if
    present it must be of the specified type. The table may have extra columns
    not specified in the schema.

    >>> import ibis.expr.datatypes as dt
    >>> import ibis.expr.rules as rules
    >>> import ibis.expr.operations as ops
    >>> class MyOp(ops.ValueOp):
    ...    input_type = [
    ...        rules.table(
    ...            name='table',
    ...            schema=rules.table.with_column_subset(
    ...                rules.column(name='time', value_type=rules.number),
    ...                rules.column(name='group', value_type=rules.number),
    ...                rules.column(name='value1', value_type=rules.number),
    ...                rules.column(name='value2', value_type=rules.number,
    ...                             optional=True)))]
    ...    output_type = rules.type_of_arg(0)
    """
    def __init__(self, name=None, optional=False, schema=None, doc=None,
                 validator=None, **arg_kwds):
        self.name = name
        self.optional = optional

        if not ((schema is None) or isinstance(schema, TableColumnValidator)):
            raise ValueError(
                'schema argument must be an instance of TableColumnValidator')

        self.schema = schema

        self.doc = doc
        self.validator = validator

    @classmethod
    def with_column_subset(cls, *col_rules):
        return SubsetValidator(*col_rules)

    def _validate(self, args, i):
        arg = args[i]

        if not isinstance(arg, ir.TableExpr):
            raise IbisTypeError('Argument must be a table.')

        if self.schema is not None:
            self.schema.validate(arg)

        return arg


table = Table


def _sum_output_type(self):
    arg = self.args[0]
    if isinstance(arg, (ir.IntegerValue, ir.BooleanValue)):
        t = 'int64'
    elif isinstance(arg, ir.FloatingValue):
        t = 'double'
    elif isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    else:
        raise TypeError(arg)
    return t


def _mean_output_type(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    elif isinstance(arg, ir.NumericValue):
        t = 'double'
    else:
        raise NotImplementedError
    return t


def _coerce_integer_to_double_type(self):
    first_arg = self.args[0]
    first_arg_type = first_arg.type()
    if isinstance(first_arg_type, dt.Integer):
        result_type = dt.double
    else:
        result_type = first_arg_type
    return result_type


def _decimal_scalar_ctor(precision, scale):
    out_type = dt.Decimal(precision, scale)
    return out_type.scalar_type()


def _min_max_output_rule(self):
    arg = self.args[0]
    if isinstance(arg, ir.DecimalValue):
        t = dt.Decimal(arg.meta.precision, 38)
    else:
        t = arg.type()

    return t

