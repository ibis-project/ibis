import pytest
import enum

from toolz import curry
from ibis.compat import suppress
import ibis.util as util
import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.schema as sch
import ibis.expr.datatypes as dt

from itertools import starmap, product
# TODO try to import cytoolz
from toolz import curry, compose, identity  # try to use cytoolz
from toolz import unique, curry


class validator(curry):
    pass


noop = validator(identity)


@validator
def oneof(inners, arg):
    """At least one of the inner valudators must pass"""
    for inner in inners:
        with suppress(com.IbisTypeError):
            return inner(arg)
    raise com.IbisTypeError('None of the {} are applicable on arg'.format(inners))


@validator
def allof(inners, arg):
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
      Value maybe transformed by inner validators.
    """
    return compose(*inners)(arg)


@validator
def optional(inner, arg, default=None):
    if arg is None:
        if default is None:
            return None
        elif callable(default):
            arg = default()  # required by genname
        else:
            arg = default
    return inner(arg)


@validator
def isin(values, arg):
    if arg not in values:
        raise ValueError('Value {!r} not in {!r}'.format(arg, values))
    if isinstance(values, dict):  # TODO check for mapping instead
        return values[arg]
    else:
        return arg


@validator
def memberof(obj, arg):
    if isinstance(arg, enum.Enum):
        enum.unique(obj)  # check that enum has unique values
        arg = arg.name

    if not hasattr(obj, arg):
        raise com.IbisTypeError('Value {!r} is not a member of '
                                '{!r}'.format(arg, obj))
    return getattr(obj, arg)


@validator
def listof(inner, arg, min_length=0):
    if not isinstance(arg, (tuple, list)):
        raise com.IbisTypeError('Arg is not an instance of list or tuple')
    if len(arg) < min_length:
        raise com.IbisTypeError(
            'Arg must have at least {} number of elements'.format(min_length)
        )
    return ir.sequence(list(map(inner, arg)))


@validator
def datatype(arg):
    return dt.dtype(arg)


@validator
def instanceof(klass, arg):
    """Require that a value has a particular Python type."""
    if not isinstance(arg, klass):
        raise com.IbisTypeError(
            '{!r} is not an instance of {!r}'.format(arg, klass)
        )
    return arg


@validator
def value(dtype, arg):
    # TODO support DataType classes, not just instances
    if arg is None:
        raise com.IbisTypeError('Passing value argument with datatype {} is '
                                'mandatory'.format(dtype))

    if not isinstance(arg, ir.Expr):
        # coerce python literal to ibis literal
        arg = ir.literal(arg)

    if not isinstance(arg, ir.ValueExpr):
        raise com.IbisTypeError('Given argument with type {} is not a value '
                                'expression'.format(type(arg)))

    if dt.issubtype(arg.type(), dtype):  # TODO: remove this, should not be required
        return arg  # subtype of expected
    if dt.castable(arg.type(), dtype):
        return arg  # implicitly castable
    else:
        raise com.IbisTypeError('Given argument with datatype {} is not '
                                'subtype of {} nor implicitly castable to '
                                'it'.format(arg.type(), dtype))


@validator
def scalar(inner, arg):
    return instanceof(ir.ScalarExpr, inner(arg))


@validator
def column(inner, arg):
    return instanceof(ir.ColumnExpr, inner(arg))


# TODO: change it to raise instead to locate all temporary noop validator
any = value(dt.any)
#null = instanceof(dt.Null)#value(dt.null)
double = value(dt.double)
string = value(dt.string)
boolean = value(dt.boolean)
integer = value(dt.int64)
decimal = value(dt.decimal)
floating = value(dt.float64)
date = value(dt.date)
time = value(dt.time)
timestamp = value(dt.timestamp)
category = value(dt.category)
# TODO: previouse number rules allowed booleans by default
temporal = oneof([timestamp, date, time])
numeric = oneof([integer, floating, decimal, boolean])
strict_numeric = oneof([integer, floating, decimal])  # without boolean



@validator
def interval(arg, units=None):
    arg = value(dt.interval, arg)
    unit = arg.type().unit
    if units is not None and unit not in units:
        msg = 'Interval unit `{}` is not among the allowed ones {}'
        raise com.IbisTypeError(msg.format(unit, units))
    return arg


table = instanceof(ir.TableExpr)
schema = instanceof(sch.Schema)


@validator
def szuper(klass, arg):
    # TODO
    return instanceof(klass, arg)


def promoter(fn):
    def wrapper(name_or_value, *args, **kwargs):
        if isinstance(name_or_value, str):
            return lambda self: fn(getattr(self, name_or_value),
                                   *args, **kwargs)
        else:
            return fn(name_or_value, *args, **kwargs)
    return wrapper


def highest_precedence_dtype(exprs):
    # Return the highest precedence type from the passed expressions. Also
    # verifies that there are valid implicit casts between any of the types and
    # the selected highest precedence type.
    # This is a thin wrapper around datatypes highest precedence check.
    if not exprs:
        raise ValueError('Must pass at least one expression')

    expr_dtypes = {expr.type() for expr in exprs}
    return dt.highest_precedence(expr_dtypes)


def comparable(left, right):
    return ir.castable(left, right) or ir.castable(right, left)


@promoter
def shapeof(arg, dtype=None):
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
def scalarof(arg):
    output_dtype = arg.type()
    return output_dtype.scalar_type()


@promoter
def arrayof(arg):
    output_dtype = arg.type()
    return output_dtype.array_type()


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
def bounded_binop_dtype(exprs, op):
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
def binopof(args, op):
    if util.all_of(args, ir.IntegerValue):
        dtype = bounded_binop_dtype(args, op)
        return shapeof(args, dtype=dtype)
    else:
        return shapeof(args)
