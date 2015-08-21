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

from ibis.common import IbisTypeError
from ibis.compat import py_string
import ibis.expr.datatypes as dt
import ibis.expr.types as ir
import ibis.common as com
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
        return shape_like_args(self.args, promoted_type)

    def _get_type(self):
        if util.any_of(self.args, ir.FloatingValue):
            if util.any_of(self.args, ir.DoubleValue):
                return 'double'
            else:
                return 'float'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        elif util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
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


def _decimal_promoted_type(args):
    precisions = []
    scales = []
    for arg in args:
        if isinstance(arg, ir.DecimalValue):
            precisions.append(arg.meta.precision)
            scales.append(arg.meta.scale)
    return dt.Decimal(max(precisions), max(scales))


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
        elif util.any_of(self.args, ir.DecimalValue):
            return _decimal_promoted_type(self.args)
        elif isinstance(rval, ir.Literal) and rval.value < 0:
            return 'double'
        elif util.all_of(self.args, ir.IntegerValue):
            return self._get_int_type()
        else:
            raise NotImplementedError


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

    _precedence = {
        'double': 9,
        'float': 8,
        'decimal': 7,
        'int64': 6,
        'int32': 5,
        'int16': 4,
        'int8': 3,
        'boolean': 2,
        'string': 1,
        'null': 0
    }

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
            self.type_counts[expr.type()] += 1

    def _get_highest_type(self):
        scores = []
        for k, v in self.type_counts.items():
            if not v:
                continue
            score = self._precedence[k.name()]

            scores.append((score, k))

        scores.sort()
        return scores[-1][1]

    def _check_casts(self, typename):
        for expr in self.exprs:
            if not expr._can_cast_implicit(typename):
                raise ValueError('Expression with type {0} cannot be '
                                 'implicitly casted to {1}'
                                 .format(expr.type(), typename))


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
    containing_types = [int_literal_class(x, allow_overflow=allow_overflow)
                        for x in values]
    return _largest_int(containing_types)


def int_literal_class(value, allow_overflow=False):
    if -128 <= value <= 127:
        t = 'int8'
    elif -32768 <= value <= 32767:
        t = 'int16'
    elif -2147483648 <= value <= 2147483647:
        t = 'int32'
    else:
        if value < -9223372036854775808 or value > 9223372036854775807:
            if not allow_overflow:
                raise OverflowError(value)
        t = 'int64'
    return dt.validate_type(t)


def _largest_int(int_types):
    nbytes = max(t._nbytes for t in int_types)
    return dt.validate_type('int%d' % (8 * nbytes))


class ImplicitCast(object):

    def __init__(self, value_type, implicit_targets):
        self.value_type = value_type
        self.implicit_targets = implicit_targets

    def can_cast(self, target):
        base_type = target.name()
        return (base_type in self.implicit_targets or
                target == self.value_type)


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
    if util.any_of(args, ir.ArrayExpr):
        return out_type.array_type()
    else:
        return out_type.scalar_type()


def is_table(e):
    return isinstance(e, ir.TableExpr)


def is_array(e):
    return isinstance(e, ir.ArrayExpr)


def is_scalar(e):
    return isinstance(e, ir.ScalarExpr)


def is_collection(expr):
    return isinstance(expr, (ir.ArrayExpr, ir.TableExpr))


class Argument(object):

    """

    """

    def __init__(self, name=None, default=None, optional=False,
                 validator=None):
        self.name = name
        self.default = default
        self.optional = optional

        self.validator = validator

    def validate(self, args, i):
        arg = args[i]

        if self.validator is not None:
            arg = args[i] = self.validator(arg)

        if arg is None:
            if not self.optional:
                return ir.as_value_expr(self.default)
            elif self.optional:
                return arg

        return self._validate(args, i)

    def _validate(self, args, i):
        raise NotImplementedError


def _to_argument(val):
    if isinstance(val, dt.DataType):
        val = value_typed_as(val)
    elif not isinstance(val, Argument):
        val = val()
    return val


class TypeSignature(object):

    def __init__(self, type_specs):
        types = []

        for val in type_specs:
            val = _to_argument(val)
            types.append(val)

        self.types = types

    def __repr__(self):
        types = '\n    '.join('arg {0}: {1}'.format(i, repr(x))
                              for i, x in enumerate(self.types))
        return '{0}\n    {1}'.format(type(self), types)

    def validate(self, args):
        n, k = len(args), len(self.types)
        k_required = len([x for x in self.types if not x.optional])
        if k != k_required:
            if n < k_required:
                raise com.IbisError('Expected at least {0} args, got {1}'
                                    .format(k, k_required))
        elif n != k:
            raise com.IbisError('Expected {0} args, got {1}'.format(k, n))

        if n < k:
            args = list(args) + [t.default for t in self.types[n:]]

        return self._validate(args, self.types)

    def _validate(self, args, types):
        clean_args = list(args)
        for i, validator in enumerate(types):
            try:
                clean_args[i] = validator.validate(clean_args, i)
            except IbisTypeError as e:
                exc = e.args[0]
                msg = ('Argument {0}: {1}'.format(i, exc) +
                       '\nArgument was: {0}'.format(ir._safe_repr(args[i])))
                raise IbisTypeError(msg)

        return clean_args


class VarArgs(TypeSignature):

    def __init__(self, arg_type, min_length=1):
        self.arg_type = _to_argument(arg_type)
        self.min_length = min_length

    def __repr__(self):
        return '{0}\n    {1}'.format(type(self), repr(self.arg_type))

    def validate(self, args):
        n, k = len(args), self.min_length
        if n < k:
            raise com.IbisError('Expected at least {0} args, got {1}'
                                .format(k, n))

        return self._validate(args, [self.arg_type] * n)


varargs = VarArgs


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
        flattened = []
        for arg in self.args:
            if isinstance(arg, (list, tuple)):
                flattened.extend(arg)
            else:
                flattened.append(arg)
        return shape_like_args(flattened, out_type)

    return output_type


def shape_like_arg(i, out_type):

    def output_type(self):
        return shape_like(self.args[i], out_type)

    return output_type


def numeric_highest_promote(i):

    def output_type(self):
        arg = self.args[i]

        if isinstance(arg, ir.DecimalValue):
            return arg._factory
        elif isinstance(arg, ir.FloatingValue):
            # Impala upcasts float to double in this op
            return shape_like(arg, 'double')
        elif isinstance(arg, ir.IntegerValue):
            return shape_like(arg, 'int64')
        else:
            raise NotImplementedError

    return output_type


def type_of_arg(i):

    def output_type(self):
        return self.args[i]._factory

    return output_type


def signature(types):
    if isinstance(types, TypeSignature):
        return types

    return TypeSignature(types)


class ValueArgument(Argument):

    def _validate(self, args, i):
        arg = args[i]
        if not isinstance(arg, ir.Expr):
            arg = args[i] = ir.as_value_expr(arg)

        return arg


class AnyTyped(Argument):

    def __init__(self, types, fail_message, **arg_kwds):
        self.types = util.promote_list(types)
        self.fail_message = fail_message
        Argument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        arg = args[i]

        if not self._type_matches(arg):
            if isinstance(self.fail_message, py_string):
                exc = self.fail_message
            else:
                exc = self.fail_message(self.types, arg)
            raise IbisTypeError(exc)

        return arg

    def _type_matches(self, arg):
        for t in self.types:
            if (isinstance(t, dt.DataType) or
                    isinstance(t, type) and issubclass(t, dt.DataType)):
                if t.can_implicit_cast(arg.type()):
                    return True
            else:
                if isinstance(arg, t):
                    return True
        return False


class ValueTyped(AnyTyped, ValueArgument):

    def __repr__(self):
        return 'ValueTyped({0})'.format(repr(self.types))

    def _validate(self, args, i):
        ValueArgument._validate(self, args, i)
        return AnyTyped._validate(self, args, i)


class MultipleTypes(Argument):

    def __init__(self, types, **arg_kwds):
        self.types = [_to_argument(t) for t in types]
        Argument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        for t in self.types:
            arg = t.validate(args, i)
        return arg


class OneOf(Argument):

    def __init__(self, types, **arg_kwds):
        self.types = [_to_argument(t) for t in types]
        Argument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        validated = False
        for t in self.types:
            try:
                arg = t.validate(args, i)
                validated = True
            except:
                pass
            else:
                break

        if not validated:
            raise IbisTypeError('No type options validated')

        return arg


class CastIfDecimal(ValueArgument):

    def __init__(self, ref_j, **arg_kwds):
        self.ref_j = ref_j
        ValueArgument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        ValueArgument._validate(self, args, i)

        ref_arg = args[self.ref_j]
        if isinstance(ref_arg, ir.DecimalValue):
            return args[i].cast(ref_arg.type())

        return args[i]


cast_if_decimal = CastIfDecimal


def value_typed_as(types, **arg_kwds):
    fail_message = 'Arg was not in types {0}'.format(repr(types))
    return ValueTyped(types, fail_message, **arg_kwds)


def array(value_type=None, name=None, optional=False):
    array_checker = ValueTyped(ir.ArrayExpr, 'not an array expr',
                               name=name,
                               optional=optional)
    if value_type is None:
        return array_checker
    else:
        return MultipleTypes([array_checker, value_type],
                             name=name,
                             optional=optional)


def scalar(name=None, optional=False):
    return ValueTyped(ir.ScalarExpr, 'not a scalar expr', name=name,
                      optional=optional)


def collection(name=None, optional=False):
    return ValueTyped((ir.ArrayExpr, ir.TableExpr), 'not a collection',
                      name=name, optional=optional)


def value(name=None, optional=False):
    return ValueTyped(ir.ValueExpr, 'not a value expr',
                      name=name, optional=optional)


def table(name=None):
    pass


class Number(ValueTyped):

    def __init__(self, allow_boolean=True, **arg_kwds):
        self.allow_boolean = allow_boolean
        ValueTyped.__init__(self, ir.NumericValue, 'not numeric', **arg_kwds)

    def _validate(self, args, i):
        arg = ValueTyped._validate(self, args, i)

        if isinstance(arg, ir.BooleanValue) and not self.allow_boolean:
            raise IbisTypeError('not implemented for boolean values')

        return arg


number = Number


def integer(**arg_kwds):
    return ValueTyped(dt.int_, 'not integer', **arg_kwds)


def double(**arg_kwds):
    return ValueTyped(dt.double, 'not double', **arg_kwds)


def decimal(**arg_kwds):
    return ValueTyped(dt.Decimal, 'not decimal', **arg_kwds)


def timestamp(**arg_kwds):
    return ValueTyped(ir.TimestampValue, 'not decimal', **arg_kwds)


def timedelta(**arg_kwds):
    from ibis.expr.temporal import Timedelta
    return AnyTyped(Timedelta, 'not a timedelta', **arg_kwds)


def string(**arg_kwds):
    return ValueTyped(dt.string, 'not string', **arg_kwds)


def boolean(**arg_kwds):
    return ValueTyped(dt.boolean, 'not string', **arg_kwds)


def one_of(args, **arg_kwds):
    return OneOf(args, **arg_kwds)


class StringOptions(Argument):

    def __init__(self, options, **arg_kwds):
        self.options = options
        Argument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        arg = args[i]
        if arg not in self.options:
            raise IbisTypeError('{0} not among options {1}'
                                .format(arg, repr(self.options)))
        return arg


string_options = StringOptions


class ListOf(Argument):

    def __init__(self, value_type, min_length=0, **arg_kwds):
        self.value_type = _to_argument(value_type)
        self.min_length = min_length
        Argument.__init__(self, **arg_kwds)

    def _validate(self, args, i):
        arg = args[i]
        if isinstance(arg, tuple):
            arg = args[i] = list(arg)

        if not isinstance(arg, list):
            raise IbisTypeError('not a list')

        if len(arg) < self.min_length:
            raise IbisTypeError('list must have at least {} elements'
                                .format(self.min_length))

        checked_args = []
        for j in range(len(arg)):
            try:
                checked_arg = self.value_type.validate(arg, j)
            except IbisTypeError as e:
                exc = e.args[0]
                msg = ('List element {0} had a type error: {1}'
                       .format(j, exc))
                raise IbisTypeError(msg)
            checked_args.append(checked_arg)

        args[i] = checked_args

        return checked_args


list_of = ListOf


class DataTypeArgument(Argument):

    def _validate(self, args, i):
        arg = args[i]

        if isinstance(arg, py_string):
            arg = arg.lower()

        arg = args[i] = dt.validate_type(arg)
        return arg


data_type = DataTypeArgument
