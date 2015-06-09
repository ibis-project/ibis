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


from ibis.expr.types import (Schema, Expr,  # noqa
                             ValueExpr, ScalarExpr, ArrayExpr,
                             TableExpr,
                             NumericValue, NumericArray,
                             IntegerValue,
                             Int8Value, Int8Scalar, Int8Array,
                             Int16Value, Int16Scalar, Int16Array,
                             Int32Value, Int32Scalar, Int32Array,
                             Int64Value, Int64Scalar, Int64Array,
                             NullScalar,
                             BooleanValue, BooleanScalar, BooleanArray,
                             FloatValue, FloatScalar, FloatArray,
                             DoubleValue, DoubleScalar, DoubleArray,
                             StringValue, StringScalar, StringArray,
                             DecimalValue, DecimalScalar, DecimalArray,
                             TimestampValue, TimestampScalar, TimestampArray,
                             CategoryValue, unnamed)

from ibis.expr.operations import (as_value_expr, table, literal, timestamp,
                                  null, sequence, desc)

# __all__ is defined
from ibis.expr.temporal import *  #  noqa

import ibis.common as _com

from ibis.expr.analytics import bucket, histogram
import ibis.expr.analytics as _analytics
import ibis.expr.analysis as _L
import ibis.expr.types as _ir
import ibis.expr.operations as _ops
import ibis.expr.temporal as _T


__all__ = ['schema', 'table', 'literal', 'timestamp',
           'case', 'where', 'sequence',
           'now', 'desc', 'null', 'NA',
           'cast', 'coalesce', 'greatest', 'least', 'join']
__all__ += _T.__all__


NA = null()


_data_type_docs = """\
Ibis uses its own type aliases that map onto database types. See, for
example, the correspondence between Ibis type names and Impala type names:

Ibis type      Impala Type
~~~~~~~~~      ~~~~~~~~~~~
int8           TINYINT
int16          SMALLINT
int32          INT
int64          BIGINT
float          FLOAT
double         DOUBLE
boolean        BOOLEAN
string         STRING
timestamp      TIMESTAMP
decimal(p, s)  DECIMAL(p,s)"""


def schema(pairs=None, names=None, types=None):
    if pairs is not None:
        return Schema.from_tuples(pairs)
    else:
        return Schema(names, types)


schema.__doc__ = """\
Validate and return an Ibis Schema object

{0}

Parameters
----------
pairs : list of (name, type) tuples
  Mutually exclusive with names/types
names : list of string
  Field names
types : list of string
  Field types

Examples
--------
sc = schema([('foo', 'string'),
             ('bar', 'int64'),
             ('baz', 'boolean')])

sc2 = schema(names=['foo', 'bar', 'baz'],
             types=['string', 'int64', 'boolean'])

Returns
-------
schema : Schema
""".format(_data_type_docs)


def case():
    """
    Similar to the .case method on array expressions, create a case builder
    that accepts self-contained boolean expressions (as opposed to expressions
    which are to be equality-compared with a fixed value expression)
    """
    return _ops.SearchedCaseBuilder()


def now():
    """
    Compute the current timestamp

    Returns
    -------
    now : Timestamp scalar
    """
    return _ops.TimestampNow().to_expr()


e = _ops.E().to_expr()


def _add_methods(klass, method_table):
    for k, v in method_table.items():
        setattr(klass, k, v)


def _unary_op(name, klass, doc=None):
    def f(arg):
        return klass(arg).to_expr()
    f.__name__ = name
    if doc is not None:
        f.__doc__ = doc
    else:
        f.__doc__ = klass.__doc__
    return f


def negate(arg):
    """
    Negate a numeric expression

    Parameters
    ----------
    arg : numeric value expression

    Returns
    -------
    negated : type of caller
    """
    op = arg.op()
    if hasattr(op, 'negate'):
        result = op.negate()
    else:
        result = _ops.Negate(arg)

    return result.to_expr()


def count(expr, where=None):
    """
    Compute cardinality / sequence size of expression. For array expressions,
    the count is excluding nulls. For tables, it's the size of the entire
    table.

    Returns
    -------
    counts : int64 type
    """
    op = expr.op()
    if isinstance(op, _ops.DistinctArray):
        if where is not None:
            raise NotImplementedError
        result = op.count().to_expr()
    else:
        result = _ops.Count(expr, where).to_expr()

    return result.name('count')


def group_concat(arg, sep=','):
    """
    Concatenate values using the indicated separator (comma by default) to
    produce a string

    Parameters
    ----------
    arg : array expression
    sep : string, default ','

    Returns
    -------
    concatenated : string scalar
    """
    return _ops.GroupConcat(arg, sep=sep).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        try:
            other = as_value_expr(other)
            op = klass(self, other)
            return op.to_expr()
        except _com.InputTypeError:
            return NotImplemented

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary _ops, like radd, etc.
    def f(self, other):
        other = as_value_expr(other)
        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _boolean_binary_op(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _boolean_binary_rop(name, klass):
    def f(self, other):
        other = as_value_expr(other)

        if not isinstance(other, BooleanValue):
            raise TypeError(other)

        op = klass(other, self)
        return op.to_expr()

    f.__name__ = name
    return f


def _agg_function(name, klass, assign_default_name=True):
    def f(self, where=None):
        expr = klass(self, where).to_expr()
        if assign_default_name:
            expr = expr.name(name)
        return expr
    f.__name__ = name
    return f


def _extract_field(name, klass):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    return f


# ---------------------------------------------------------------------
# Generic value API


def cast(arg, target_type):
    # validate
    op = _ops.Cast(arg, target_type)

    if op.target_type == arg.type():
        # noop case if passed type is the same
        return arg
    else:
        return op.to_expr()

cast.__doc__ = """
Cast value(s) to indicated data type. Values that cannot be
successfully casted

Parameters
----------
target_type : data type name

Notes
-----
{0}

Returns
-------
cast_expr : ValueExpr
""".format(_data_type_docs)


def hash(arg, how='fnv'):
    """
    Compute an integer hash value for the indicated value expression.

    Parameters
    ----------
    arg : value expression
    how : {'fnv'}, default 'fnv'
      Hash algorithm to use

    Returns
    -------
    hash_value : int64 expression
    """
    op = _ops.Hash(arg, how)
    return op.to_expr()


def fillna(arg, fill_value):
    """
    Replace any null values with the indicated fill value

    Parameters
    ----------
    fill_value : scalar / array value or expression

    Examples
    --------
    result = table.col.fillna(5)
    result2 = table.col.fillna(table.other_col * 3)

    Returns
    -------
    filled : type of caller
    """
    op = _ops.IfNull(arg, fill_value)
    return op.to_expr()


def coalesce(*args):
    """
    Compute the first non-null value(s) from the passed arguments in
    left-to-right order. This is also known as "combine_first" in pandas.

    Parameters
    ----------
    *args : variable-length value list

    Examples
    --------
    result = coalesce(expr1, expr2, 5)

    Returns
    -------
    coalesced : type of first provided argument
    """
    return _ops.Coalesce(args).to_expr()


def greatest(*args):
    """
    Compute the largest value (row-wise, if any arrays are present) among the
    supplied arguments.
    """
    return _ops.Greatest(args).to_expr()


def least(*args):
    """
    Compute the smallest value (row-wise, if any arrays are present) among the
    supplied arguments.
    """
    return _ops.Least(args).to_expr()


def where(boolean_expr, true_expr, false_null_expr):
    """

    Parameters
    ----------
    boolean_expr : BooleanValue (array or scalar)
    true_expr : value
      Values for each True value
    false_null_expr : value
      Values for False or NULL values

    Returns
    -------
    result : arity depending on inputs
      Type of true_expr used to determine output type
    """
    op = _ops.Where(boolean_expr, true_expr, false_null_expr)
    return op.to_expr()


def value_counts(arg, metric_name='count'):
    """
    Compute a frequency table for this value expression

    Parameters
    ----------

    Returns
    -------
    counts : TableExpr
      Aggregated table
    """
    base = _L.find_base_table(arg)
    metric = base.count().name(metric_name)

    try:
        arg.get_name()
    except _com.ExpressionError:
        arg = arg.name('unnamed')

    return base.group_by(arg).aggregate(metric)


def nullif(value, null_if_expr):
    """
    Set values to null if they match/equal a particular expression (scalar or
    array-valued).

    Common use to avoid divide-by-zero problems (get NULL instead of INF on
    divide-by-zero):
      5 / expr.nullif(0)

    Parameters
    ----------
    value : value expression
      Value to modify
    null_if_expr : value expression (array or scalar)

    Returns
    -------
    null_if : type of caller
    """
    return _ops.NullIf(value, null_if_expr).to_expr()


def between(arg, lower, upper):
    """
    Check if the input expr falls between the lower/upper bounds
    passed. Bounds are inclusive. All arguments must be comparable.

    Returns
    -------
    is_between : BooleanValue
    """
    lower = _ops.as_value_expr(lower)
    upper = _ops.as_value_expr(upper)
    op = _ops.Between(arg, lower, upper)
    return op.to_expr()


def isin(arg, values):
    """
    Check whether the value expression is contained within the indicated
    list of values.

    Parameters
    ----------
    values : list, tuple, or array expression
      The values can be scalar or array-like. Each of them must be
      comparable with the calling expression, or None (NULL).

    Examples
    --------
    expr = table.strings.isin(['foo', 'bar', 'baz'])

    expr2 = table.strings.isin(table2.other_string_col)

    Returns
    -------
    contains : BooleanValue
    """
    op = _ops.Contains(arg, values)
    return op.to_expr()


def notin(arg, values):
    """
    Like isin, but checks whether this expression's value(s) are not
    contained in the passed values. See isin docs for full usage.
    """
    op = _ops.NotContains(arg, values)
    return op.to_expr()


add = _binop_expr('__add__', _ops.Add)
sub = _binop_expr('__sub__', _ops.Subtract)
mul = _binop_expr('__mul__', _ops.Multiply)
div = _binop_expr('__div__', _ops.Divide)
pow = _binop_expr('__pow__', _ops.Power)
mod = _binop_expr('__mod__', _ops.Modulus)

rsub = _rbinop_expr('__rsub__', _ops.Subtract)
rdiv = _rbinop_expr('__rdiv__', _ops.Divide)


_generic_value_methods = dict(
    hash=hash,
    cast=cast,
    fillna=fillna,
    nullif=nullif,
    between=between,
    isin=isin,
    notin=notin,
    isnull=_unary_op('isnull', _ops.IsNull),
    notnull=_unary_op('notnull', _ops.NotNull),

    __add__=add,
    add=add,

    __sub__=sub,
    sub=sub,

    __mul__=mul,
    mul=mul,

    __div__=div,
    div=div,

    __rdiv__=rdiv,
    rdiv=rdiv,

    __pow__=pow,
    pow=pow,

    __radd__=add,

    __rsub__=rsub,
    rsub=rsub,

    __rmul__=_rbinop_expr('__rmul__', _ops.Multiply),
    __rpow__=_binop_expr('__rpow__', _ops.Power),

    __mod__=mod,
    __rmod__=_rbinop_expr('__rmod__', _ops.Modulus),

    __eq__=_binop_expr('__eq__', _ops.Equals),
    __ne__=_binop_expr('__ne__', _ops.NotEquals),
    __ge__=_binop_expr('__ge__', _ops.GreaterEqual),
    __gt__=_binop_expr('__gt__', _ops.Greater),
    __le__=_binop_expr('__le__', _ops.LessEqual),
    __lt__=_binop_expr('__lt__', _ops.Less)
)


approx_nunique = _agg_function('approx_nunique', _ops.HLLCardinality, True)
approx_median = _agg_function('approx_median', _ops.CMSMedian, True)
max = _agg_function('max', _ops.Max, True)
min = _agg_function('min', _ops.Min, True)



def distinct(arg):
    """
    Compute set of unique values occurring in this array. Can not be used
    in conjunction with other array expressions from the same context
    (because it's a cardinality-modifying pseudo-reduction).
    """
    op = _ops.DistinctArray(arg)
    return op.to_expr()


def nunique(arg):
    """
    Shorthand for foo.distinct().count(); computing the number of unique
    values in an array.
    """
    return _ops.CountDistinct(arg).to_expr()


def topk(arg, k, by=None):
    """
    Produces

    Returns
    -------
    topk : TopK filter expression
    """
    op = _ops.TopK(arg, k, by=by)
    return op.to_expr()


def bottomk(arg, k, by=None):
    raise NotImplementedError


def _case(arg):
    """
    Create a new SimpleCaseBuilder to chain multiple if-else
    statements. Add new search expressions with the .when method. These
    must be comparable with this array expression. Conclude by calling
    .end()

    Examples
    --------
    case_expr = (expr.case()
                 .when(case1, output1)
                 .when(case2, output2)
                 .default(default_output)
                 .end())

    Returns
    -------
    builder : CaseBuilder
    """
    return _ops.SimpleCaseBuilder(arg)


def cases(arg, case_result_pairs, default=None):
    """
    Create a case expression in one shot.

    Returns
    -------
    case_expr : SimpleCase
    """
    builder = arg.case()
    for case, result in case_result_pairs:
        builder = builder.when(case, result)
    if default is not None:
        builder = builder.else_(default)
    return builder.end()


def _generic_summary(arg, exact_nunique=False, prefix=None):
    """
    Compute a set of summary metrics from the input value expression

    Parameters
    ----------
    arg : value expression
    exact_nunique : boolean, default False
      Compute the exact number of distinct values (slower)
    prefix : string, default None
      String prefix for metric names

    Returns
    -------
    summary : (count, # nulls, nunique)
    """
    metrics = [
        arg.count(),
        arg.isnull().sum().name('nulls')
    ]

    if exact_nunique:
        unique_metric = arg.nunique().name('uniques')
    else:
        unique_metric = arg.approx_nunique().name('uniques')

    metrics.append(unique_metric)
    return _wrap_summary_metrics(metrics, prefix)


def _numeric_summary(arg, exact_nunique=False, prefix=None):
    """
    Compute a set of summary metrics from the input numeric value expression

    Parameters
    ----------
    arg : numeric value expression
    exact_nunique : boolean, default False
    prefix : string, default None
      String prefix for metric names

    Returns
    -------
    summary : (count, # nulls, min, max, sum, mean, nunique)
    """
    metrics = [
        arg.count(),
        arg.isnull().sum().name('nulls'),
        arg.min(),
        arg.max(),
        arg.sum(),
        arg.mean()
    ]

    if exact_nunique:
        unique_metric = arg.nunique().name('nunique')
    else:
        unique_metric = arg.approx_nunique().name('approx_nunique')

    metrics.append(unique_metric)
    return _wrap_summary_metrics(metrics, prefix)


def _wrap_summary_metrics(metrics, prefix):
    if prefix is not None:
        metrics = [x.name(prefix + x.get_name()) for x in metrics]

    return _ir.ExpressionList(metrics).to_expr()


_generic_array_methods = dict(
    case=_case,
    cases=cases,
    bottomk=bottomk,
    distinct=distinct,
    nunique=nunique,
    topk=topk,
    summary=_generic_summary,
    count=count,
    min=min,
    max=max,
    approx_median=approx_median,
    approx_nunique=approx_nunique,
    group_concat=group_concat,
    value_counts=value_counts
)


_add_methods(ValueExpr, _generic_value_methods)
_add_methods(ArrayExpr, _generic_array_methods)


# ---------------------------------------------------------------------
# Numeric API

def round(arg, digits=None):
    """
    Round values either to integer or indicated number of decimal places.

    Returns
    -------
    rounded : type depending on digits argument
      digits None or 0
        decimal types: decimal
        other numeric types: bigint
      digits nonzero
        decimal types: decimal
        other numeric types: double
    """
    op = _ops.Round(arg, digits)
    return op.to_expr()


def log(arg, base=None):
    """
    Perform the logarithm using a specified base

    Parameters
    ----------
    base : number, default None
      If None, base e is used

    Returns
    -------
    logarithm : double type
    """
    op = _ops.Log(arg, base)
    return op.to_expr()


def _integer_to_timestamp(arg, unit='s'):
    """
    Convert integer UNIX timestamp (at some resolution) to a timestamp type

    Parameters
    ----------
    unit : {'s', 'ms', 'us'}
      Second (s), millisecond (ms), or microsecond (us) resolution

    Returns
    -------
    timestamp : timestamp value expression
    """
    op = _ops.TimestampFromUNIX(arg, unit)
    return op.to_expr()


abs = _unary_op('abs', _ops.Abs)
ceil = _unary_op('ceil', _ops.Ceil)
exp = _unary_op('exp', _ops.Exp)
floor = _unary_op('floor', _ops.Floor)
log2 = _unary_op('log2', _ops.Log2)
log10 = _unary_op('log10', _ops.Log10)
ln = _unary_op('ln', _ops.Ln)
sign = _unary_op('sign', _ops.Sign)
sqrt = _unary_op('sqrt', _ops.Sqrt)


_numeric_value_methods = dict(
    __neg__=negate,
    abs=abs,
    ceil=ceil,
    floor=floor,
    sign=sign,
    exp=exp,
    sqrt=sqrt,
    log=log,
    ln=ln,
    log2=log2,
    log10=log10,
    round=round,
    zeroifnull=_unary_op('zeroifnull', _ops.ZeroIfNull),
)


_integer_value_methods = dict(
    to_timestamp=_integer_to_timestamp
)


mean = _agg_function('mean', _ops.Mean, True)
sum = _agg_function('sum', _ops.Sum, True)


_numeric_array_methods = dict(
    mean=mean,
    sum=sum,
    bucket=bucket,
    histogram=histogram,
    summary=_numeric_summary,
)

_add_methods(NumericValue, _numeric_value_methods)
_add_methods(IntegerValue, _integer_value_methods)

_add_methods(NumericArray, _numeric_array_methods)


# ----------------------------------------------------------------------
# Boolean API


# TODO: logical binary operators for BooleanValue


def ifelse(arg, true_expr, false_expr):
    """
    Shorthand for implementing ternary expressions

    bool_expr.ifelse(0, 1)
    e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END
    """
    # Result will be the result of promotion of true/false exprs. These
    # might be conflicting types; same type resolution as case expressions
    # must be used.
    case = _ops.SearchedCaseBuilder()
    return case.when(arg, true_expr).else_(false_expr).end()


_boolean_value_methods = dict(
    ifelse=ifelse,
    __and__=_boolean_binary_op('__and__', _ops.And),
    __or__=_boolean_binary_op('__or__', _ops.Or),
    __xor__=_boolean_binary_op('__xor__', _ops.Xor),
    __rand__=_boolean_binary_rop('__rand__', _ops.And),
    __ror__=_boolean_binary_rop('__ror__', _ops.Or),
    __rxor__=_boolean_binary_rop('__rxor__', _ops.Xor)
)


_boolean_array_methods = dict(
    any=_unary_op('any', _ops.Any)
)


_add_methods(BooleanValue, _boolean_value_methods)
_add_methods(BooleanArray, _boolean_array_methods)


# ---------------------------------------------------------------------
# String API

def _string_substr(self, start, length=None):
    """
    Pull substrings out of each string value by position and maximum
    length.

    Parameters
    ----------
    start : int
      First character to start splitting, indices starting at 0 (like
      Python)
    length : int, optional
      Maximum length of each substring. If not supplied, splits each string
      to the end

    Returns
    -------
    substrings : type of caller
    """
    op = _ops.Substring(self, start, length)
    return op.to_expr()


def _string_left(self, nchars):
    """
    Return left-most up to N characters from each string. Convenience
    use of substr.

    Returns
    -------
    substrings : type of caller
    """
    return self.substr(0, length=nchars)


def _string_right(self, nchars):
    """
    Split up to nchars starting from end of each string.

    Returns
    -------
    substrings : type of caller
    """
    return _ops.StrRight(self, nchars).to_expr()


def _string_like(self, pattern):
    """
    Wildcard fuzzy matching function equivalent to the SQL LIKE directive. Use
    % as a multiple-character wildcard or _ (underscore) as a single-character
    wildcard.

    Use re_search or rlike for regex-based matching.

    Parameters
    ----------
    pattern : string

    Returns
    -------
    matched : boolean value
    """
    return _ops.StringSQLLike(self, pattern).to_expr()


def re_search(arg, pattern):
    """
    Search string values using a regular expression. Returns True if the regex
    matches a string and False otherwise.

    Parameters
    ----------
    pattern : string (regular expression string)

    Returns
    -------
    searched : boolean value
    """
    return _ops.RegexSearch(arg, pattern).to_expr()


def _string_contains(arg, substr):
    return arg.like('%{}%'.format(substr))


def _string_dunder_contains(arg, substr):
    raise TypeError('Use val.contains(arg)')


_string_value_methods = dict(
    length=_unary_op('length', _ops.StringLength),
    lower=_unary_op('lower', _ops.Lowercase),
    upper=_unary_op('upper', _ops.Uppercase),

    __contains__=_string_dunder_contains,
    contains=_string_contains,
    like=_string_like,
    rlike=re_search,
    re_search=re_search,

    substr=_string_substr,
    left=_string_left,
    right=_string_right
)


_add_methods(StringValue, _string_value_methods)


# ---------------------------------------------------------------------
# Timestamp API


_timestamp_value_methods = dict(
    year=_extract_field('year', _ops.ExtractYear),
    month=_extract_field('month', _ops.ExtractMonth),
    day=_extract_field('day', _ops.ExtractDay),
    hour=_extract_field('hour', _ops.ExtractHour),
    minute=_extract_field('minute', _ops.ExtractMinute),
    second=_extract_field('second', _ops.ExtractSecond),
    millisecond=_extract_field('millisecond', _ops.ExtractMillisecond)
)


_add_methods(TimestampValue, _timestamp_value_methods)


# ---------------------------------------------------------------------
# Decimal API

_decimal_value_methods = dict(
    precision=_unary_op('precision', _ops.DecimalPrecision),
    scale=_unary_op('scale', _ops.DecimalScale),
)


_add_methods(DecimalValue, _decimal_value_methods)


# ----------------------------------------------------------------------
# Category API


_category_value_methods = dict(
    label=_analytics.category_label
)

_add_methods(CategoryValue, _category_value_methods)


# ---------------------------------------------------------------------
# Table API

_join_classes = {
    'inner': _ops.InnerJoin,
    'left': _ops.LeftJoin,
    'outer': _ops.OuterJoin,
    'left_semi': _ops.LeftSemiJoin,
    'semi': _ops.LeftSemiJoin,
    'anti': _ops.LeftAntiJoin,
    'cross': _ops.CrossJoin
}


def join(left, right, predicates=(), how='inner'):
    """
    Perform a relational join between two tables. Does not resolve resulting
    table schema.

    Parameters
    ----------
    left : TableExpr
    right : TableExpr
    predicates : join expression(s)
    how : string, default 'inner'
      - 'inner': inner join
      - 'left': left join
      - 'outer': full outer join
      - 'semi' or 'left_semi': left semi join
      - 'anti': anti join

    Returns
    -------
    joined : TableExpr
      Note, schema is not materialized yet
    """
    klass = _join_classes[how.lower()]
    if isinstance(predicates, Expr):
        predicates = _L.unwrap_ands(predicates)

    op = klass(left, right, predicates)
    return TableExpr(op)


def cross_join(left, right, prefixes=None):
    """

    """
    op = _ops.CrossJoin(left, right)
    return TableExpr(op)


def _table_set_column(table, name, expr):
    """
    Replace an existing column with a new expression

    Parameters
    ----------
    name : string
      Column name to replace
    expr : value expression
      New data for column

    Returns
    -------
    set_table : TableExpr
      New table expression
    """
    if expr._name != name:
        expr = expr.name(name)

    if name not in table:
        raise KeyError('{0} is not in the table'.format(name))

    # TODO: This assumes that projection is required; may be backend-dependent
    proj_exprs = []
    for key in table.columns:
        if key == name:
            proj_exprs.append(expr)
        else:
            proj_exprs.append(table[key])

    return table.projection(proj_exprs)


def _regular_join_method(name, how, doc=None):
    def f(self, other, predicates=()):
        return self.join(other, predicates, how=how)
    if doc:
        f.__doc__ = doc
    else:
        # XXX
        f.__doc__ = join.__doc__
    f.__name__ = name
    return f


_table_methods = dict(
    set_column=_table_set_column,
    join=join,
    cross_join=cross_join,
    inner_join=_regular_join_method('inner_join', 'inner'),
    left_join=_regular_join_method('left_join', 'left'),
    outer_join=_regular_join_method('outer_join', 'outer'),
    semi_join=_regular_join_method('semi_join', 'semi'),
    anti_join=_regular_join_method('anti_join', 'anti')
)


_add_methods(TableExpr, _table_methods)
