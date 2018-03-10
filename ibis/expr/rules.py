import enum
from itertools import starmap, product

from ibis.compat import suppress
import ibis.util as util
import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.datatypes as dt

try:
    from cytoolz import curry, compose, identity
except ImportError:
    from toolz import curry, compose, identity


def highest_precedence_dtype(exprs):
    """Return the highest precedence type from the passed expressions

    Also verifies that there are valid implicit casts between any of the types
    and the selected highest precedence type.
    This is a thin wrapper around datatypes highest precedence check.

    Parameters
    ----------
    exprs : Iterable[ir.ValueExpr]
      A sequence of Expressions

    Returns
    -------
    dtype: DataType
      The highest precedence datatype
    """
    if not exprs:
        raise ValueError('Must pass at least one expression')

    return dt.highest_precedence(expr.type() for expr in exprs)


def castable(source, target):
    """Return whether source ir type is implicitly castable to target

    Based on the underlying datatypes and the value in case of Literals
    """
    op = source.op()
    value = getattr(op, 'value', None)
    return dt.castable(source.type(), target.type(), value=value)


def comparable(left, right):
    return castable(left, right) or castable(right, left)


def cast(source, target):
    """Currently Literal to *Scalar implicit casts are allowed"""
    import ibis.expr.operations as ops  # TODO: don't use ops here

    if not castable(source, target):
        raise com.IbisTypeError('Source is not castable to target type!')

    # currently it prevents column -> scalar implicit castings
    # however the datatypes are matching
    op = source.op()
    if not isinstance(op, ops.Literal):
        raise com.IbisTypeError('Only able to implicitly cast literals!')

    out_type = target.type().scalar_type()
    return out_type(op)


# ---------------------------------------------------------------------
# Input type validators / coercion functions


class validator(curry):

    def __repr__(self):
        return '{}({}{})'.format(
            self.func.__name__,
            repr(self.args)[1:-1],
            ', '.join('{}={!r}'.format(k, v) for k, v in self.keywords.items())
        )


noop = validator(identity)


@validator
def one_of(inners, arg):
    """At least one of the inner validators must pass"""
    for inner in inners:
        with suppress(com.IbisTypeError, ValueError):
            return inner(arg)
    # TODO: more verbose error
    raise com.IbisTypeError(
        'None of the {} are applicable on arg'.format(inners)
    )


@validator
def all_of(inners, arg):
    """All of the inner valudators must pass.

    The order of inner validators matters.

    Parameters
    ----------
    inners : List[validator]
      Functions are applied from right to left so allof([rule1, rule2], arg) is
      the same as rule1(rule2(arg)).
    arg : Any
      Value to be validated.

    Returns
    -------
    arg : Any
      Value maybe coerced by inner validators to the appropiate types
    """
    return compose(*inners)(arg)


@validator
def isin(values, arg):
    if arg not in values:
        raise ValueError(
            'Value with type {} is not in {!r}'.format(type(arg), values)
        )
    if isinstance(values, dict):  # TODO check for mapping instead
        return values[arg]
    else:
        return arg


@validator
def member_of(obj, arg):
    if isinstance(arg, enum.Enum):
        enum.unique(obj)  # check that enum has unique values
        arg = arg.name

    if not hasattr(obj, arg):
        raise com.IbisTypeError(
            'Value with type {} is not a member of {}'.format(type(arg), obj)
        )
    return getattr(obj, arg)


@validator
def list_of(inner, arg, min_length=0):
    if not isinstance(arg, (tuple, list, ir.ListExpr)):
        arg = [arg]

    if len(arg) < min_length:
        raise com.IbisTypeError(
            'Arg must have at least {} number of elements'.format(min_length)
        )
    return ir.sequence(list(map(inner, arg)))


@validator
def datatype(arg):
    return dt.dtype(arg)


@validator
def instance_of(klass, arg):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klass):
        raise com.IbisTypeError(
            'Given argument with type {} is not an instance of {}'.format(
                type(arg), klass
            )
        )
    return arg


@validator
def value(dtype, arg):
    """Validates that the given argument is a Value with a particular datatype

    Parameters
    ----------
    dtype : DataType subclass or DataType instance
    arg : python literal or an ibis expression
      If a python literal is given the validator tries to coerce it to an ibis
      literal.

    Returns
    -------
    arg : AnyValue
      An ibis value expression with the specified datatype
    """
    if not isinstance(arg, ir.Expr):
        # coerce python literal to ibis literal
        arg = ir.literal(arg)

    if not isinstance(arg, ir.AnyValue):
        raise com.IbisTypeError('Given argument with type {} is not a value '
                                'expression'.format(type(arg)))

    # retrieve literal values for implicit cast check
    value = getattr(arg.op(), 'value', None)

    if isinstance(dtype, type) and isinstance(arg.type(), dtype):
        # dtype class has been specified like dt.Interval or dt.Decimal
        return arg
    elif dt.castable(arg.type(), dt.dtype(dtype), value=value):
        # dtype instance or string has been specified and arg's dtype is
        # implicitly castable to it, like dt.int8 is castable to dt.int64
        return arg
    else:
        raise com.IbisTypeError('Given argument with datatype {} is not '
                                'subtype of {} nor implicitly castable to '
                                'it'.format(arg.type(), dtype))


@validator
def scalar(inner, arg):
    return instance_of(ir.ScalarExpr, inner(arg))


@validator
def column(inner, arg):
    return instance_of(ir.ColumnExpr, inner(arg))


any = value(dt.any)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.Decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.Timestamp)
category = value(dt.category)
temporal = one_of([timestamp, date, time])

strict_numeric = one_of([integer, floating, decimal])
soft_numeric = one_of([integer, floating, decimal, boolean])
numeric = soft_numeric

array = value(dt.Array(dt.any))
struct = value(dt.Struct)
mapping = value(dt.Map(dt.any, dt.any))


@validator
def interval(arg, units=None):
    arg = value(dt.Interval, arg)
    unit = arg.type().unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


@validator
def client(arg):
    from ibis.client import Client
    return instance_of(Client, arg)


# ---------------------------------------------------------------------
# Ouput type promoter functions


def promoter(fn):
    def wrapper(name_or_value, *args, **kwargs):
        if isinstance(name_or_value, str):
            return lambda self: fn(getattr(self, name_or_value),
                                   *args, **kwargs)
        else:
            return fn(name_or_value, *args, **kwargs)
    return wrapper


@promoter
def shape_like(arg, dtype=None):
    if isinstance(arg, (tuple, list, ir.ListExpr)):
        datatype = dtype or highest_precedence_dtype(arg)
        columnar = util.any_of(arg, ir.AnyColumn)
    else:
        datatype = dtype or arg.type()
        columnar = isinstance(arg, ir.AnyColumn)

    dtype = dt.dtype(datatype)

    if columnar:
        return dtype.array_type()
    else:
        return dtype.scalar_type()


@promoter
def scalar_like(arg):
    output_dtype = arg.type()
    return output_dtype.scalar_type()


@promoter
def array_like(arg):
    output_dtype = arg.type()
    return output_dtype.array_type()


column_like = array_like


@promoter
def typeof(arg):
    return arg._factory


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


def _coerce_integer_to_double_type(self):
    first_arg = self.args[0]
    first_arg_type = first_arg.type()
    if isinstance(first_arg_type, dt.Integer):
        result_type = dt.double
    else:
        result_type = first_arg_type
    return result_type


# TODO: might just use bounds instead of actual literal values
# that could simplify interval binop output_type methods
def _promote_numeric_binop(exprs, op):
    bounds, dtypes = [], []
    for arg in exprs:
        dtypes.append(arg.type())
        if hasattr(arg.op(), 'value'):
            # arg.op() is a literal
            bounds.append([arg.op().value])
        else:
            bounds.append(arg.type().bounds)

    # In some cases, the bounding type might be int8, even though neither
    # of the types are that small. We want to ensure the containing type is
    # _at least_ as large as the smallest type in the expression.
    values = starmap(op, product(*bounds))
    dtypes += [dt.infer(value, allow_overflow=True) for value in values]

    return dt.highest_precedence(dtypes)


@promoter
def numeric_like(args, op):
    if util.all_of(args, ir.IntegerValue):
        dtype = _promote_numeric_binop(args, op)
        return shape_like(args, dtype=dtype)
    else:
        return shape_like(args)


# TODO: create varargs marker for impala udfs
