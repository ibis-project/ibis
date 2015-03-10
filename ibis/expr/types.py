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

import re

from ibis.common import RelationError
from ibis.expr.base import Expr, Literal, Schema
import ibis.expr.base as base
import ibis.expr.operations as ops


#----------------------------------------------------------------------
# Helper / factory functions

def _negate(expr):
    op = expr.op()
    if hasattr(op, 'negate'):
        return op.negate()
    else:
        return ops.Negate(expr)


def _count(expr):
    op = expr.op()
    if isinstance(op, ops.DistinctArray):
        return op.count()
    else:
        return ops.Count(expr).to_expr()


def _binop_expr(name, klass):
    def f(self, other):
        other = as_value_expr(other)
        op = klass(self, other)
        return op.to_expr()

    f.__name__ = name

    return f


def _rbinop_expr(name, klass):
    # For reflexive binary ops, like radd, etc.
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


def _agg_function(name, klass):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    return f


def _unary_op(name, klass, doc=None):
    def f(self):
        return klass(self).to_expr()
    f.__name__ = name
    f.__doc__ = doc
    return f


def as_value_expr(val):
    if not isinstance(val, Expr):
        if isinstance(val, (tuple, list)):
            val = value_list(val)
        else:
            val = literal(val)

    return val


def table(schema, name=None):
    if not isinstance(schema, base.Schema):
        if isinstance(schema, list):
            schema = base.Schema.from_tuples(schema)
        else:
            schema = base.Schema.from_dict(schema)

    node = ops.UnboundTable(schema, name=name)
    return TableExpr(node)


def literal(value):
    if value is None:
        return null()
    else:
        return base.Literal(value).to_expr()


def null():
    return _NULL


def desc(expr):
    return ops.SortKey(expr, ascending=False)


def value_list(values):
    return ops.ValueList(values).to_expr()


class ValueExpr(Expr):

    """
    Base class for a data generating expression having a fixed and known type,
    either a single value (scalar)
    """

    _implicit_casts = set()

    def __init__(self, arg, name=None):
        Expr.__init__(self, arg)
        self._name = name

    def type(self):
        return self._typename

    def _base_type(self):
        # Parametric types like "decimal"
        return self.type()

    def _can_cast_implicit(self, typename):
        return typename in self._implicit_casts or typename == self.type()

    def op(self):
        return self._arg

    def get_name(self):
        if self._name is not None:
            # This value has been explicitly named
            return self._name

        # In some but not all cases we can get a name from the node that
        # produces the value
        return self.op().resolve_name()

    def name(self, name):
        return self._factory(self._arg, name=name)

    def cast(self, target_type):
        """
        Cast value(s) to indicated data type. Values that cannot be
        successfully casted

        Parameters
        ----------
        target_type : data type name

        Returns
        -------
        cast_expr : ValueExpr
        """
        # validate
        op = ops.Cast(self, target_type)

        if op.target_type == self.type():
            # noop case if passed type is the same
            return self
        else:
            return op.to_expr()

    def between(self, lower, upper):
        """
        Check if the input expr falls between the lower/upper bounds
        passed. Bounds are inclusive. All arguments must be comparable.

        Returns
        -------
        is_between : BooleanValue
        """
        lower = as_value_expr(lower)
        upper = as_value_expr(upper)
        op = ops.Between(self, lower, upper)
        return op.to_expr()

    def isin(self, values):
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
        op = ops.Contains(self, values)
        return op.to_expr()

    def notin(self, values):
        """
        Like isin, but checks whether this expression's value(s) are not
        contained in the passed values. See isin docs for full usage.
        """
        op = ops.NotContains(self, values)
        return op.to_expr()

    isnull = _unary_op('isnull', ops.IsNull)
    notnull = _unary_op('notnull', ops.NotNull)

    def ifnull(self, sub_expr):
        pass

    __add__ = _binop_expr('__add__', ops.Add)
    __sub__ = _binop_expr('__sub__', ops.Subtract)
    __mul__ = _binop_expr('__mul__', ops.Multiply)
    __div__ = _binop_expr('__div__', ops.Divide)
    __pow__ = _binop_expr('__pow__', ops.Power)

    __radd__ = _rbinop_expr('__radd__', ops.Add)
    __rsub__ = _rbinop_expr('__rsub__', ops.Subtract)
    __rmul__ = _rbinop_expr('__rmul__', ops.Multiply)
    __rdiv__ = _rbinop_expr('__rdiv__', ops.Divide)
    __rpow__ = _binop_expr('__rpow__', ops.Power)

    __eq__ = _binop_expr('__eq__', ops.Equals)
    __ne__ = _binop_expr('__ne__', ops.NotEquals)
    __ge__ = _binop_expr('__ge__', ops.GreaterEqual)
    __gt__ = _binop_expr('__gt__', ops.Greater)
    __le__ = _binop_expr('__le__', ops.LessEqual)
    __lt__ = _binop_expr('__lt__', ops.Less)


class ScalarExpr(ValueExpr):

    def is_reduction(self):
        # Aggregations yield typed scalar expressions, since the result of an
        # aggregation is a single value. When creating an table expression
        # containing a GROUP BY equivalent, we need to be able to easily check
        # that we are looking at the result of an aggregation.
        #
        # As an example, the expression we are looking at might be something
        # like: foo.sum().log10() + bar.sum().log10()
        #
        # We examine the operator DAG in the expression to determine if there
        # are aggregations present.
        #
        # A bound aggregation referencing a separate table is a "false
        # aggregation" in a GROUP BY-type expression and should be treated a
        # literal, and must be computed as a separate query and stored in a
        # temporary variable (or joined, for bound aggregations with keys)
        def has_reduction(op):
            if isinstance(op, base.Reduction):
                return True

            for arg in op.args:
                if isinstance(arg, ScalarExpr) and has_reduction(arg.op()):
                    return True

            return False

        return has_reduction(self.op())


class ArrayExpr(ValueExpr):

    def parent(self):
        return self._arg

    def distinct(self):
        """
        Compute set of unique values occurring in this array. Can not be used
        in conjunction with other array expressions from the same context
        (because it's a cardinality-modifying pseudo-reduction).
        """
        op = ops.DistinctArray(self)
        return op.to_expr()

    def nunique(self):
        """
        Shorthand for foo.distinct().count(); computing the number of unique
        values in an array.
        """
        return ops.CountDistinct(self).to_expr()

    def topk(self, k, by=None):
        """
        Produces
        """
        op = ops.TopK(self, k, by=by)
        return op.to_expr()

    def count(self):
        return _count(self)

    def bottomk(self, k, by=None):
        raise NotImplementedError

    def case(self):
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
        return ops.SimpleCaseBuilder(self)

    def cases(self, case_result_pairs, default=None):
        """
        Create a case expression in one shot.

        Returns
        -------
        case_expr : SimpleCase
        """
        builder = self.case()
        for case, result in case_result_pairs:
            builder = builder.when(case, result)
        if default is not None:
            builder = builder.else_(default)
        return builder.end()

    def to_projection(self):
        """
        Promote this column expression to a table projection
        """
        roots = self._root_tables()
        if len(roots) > 1:
            raise RelationError('Cannot convert array expression involving '
                                'multiple base table references to a '
                                'projection')

        table = TableExpr(roots[0])
        return table.projection([self])



class TableExpr(Expr):

    def op(self):
        return self._arg

    def _assert_valid(self, exprs):
        from ibis.expr.analysis import ExprValidator
        ExprValidator([self]).validate_all(exprs)

    def __getitem__(self, what):
        if isinstance(what, basestring):
            return self.get_column(what)
        elif isinstance(what, (list, tuple)):
            # Projection case
            return self.projection(what)
        elif isinstance(what, BooleanArray):
            # Boolean predicate
            return self.filter([what])
        else:
            raise NotImplementedError

    def __getattr__(self, key):
        try:
            return object.__getattribute__(self, key)
        except AttributeError:
            if not self._is_materialized() or key not in self.schema():
                raise

            return self.get_column(key)

    def __dir__(self):
        attrs = dir(type(self))
        if self._is_materialized():
            attrs = list(sorted(set(attrs + self.schema().names)))
        return attrs

    def _resolve(self, exprs):
        # Stash this helper method here for now
        out_exprs = []
        for expr in exprs:
            expr = self._ensure_expr(expr)
            out_exprs.append(expr)
        return out_exprs

    def _get_type(self, name):
        return self._arg.get_type(name)

    def materialize(self):
        if self._is_materialized():
            return self
        else:
            op = ops.MaterializedJoin(self)
            return TableExpr(op)

    def get_columns(self, iterable):
        return [self.get_column(x) for x in iterable]

    def get_column(self, name):
        ref = ops.TableColumn(name, self)
        return ref.to_expr()

    def schema(self):
        if not self._is_materialized():
            raise Exception('Table operation is not yet materialized')
        return self.op().get_schema()

    def to_array(self):
        """
        Single column tables can be viewed as arrays.
        """
        op = ops.TableArrayView(self)
        return op.to_expr()

    def _is_materialized(self):
        # The operation produces a known schema
        op = self.op()
        return isinstance(op, base.HasSchema) or op.has_schema()

    def view(self):
        """
        Create a new table expression that is semantically equivalent to the
        current one, but is considered a distinct relation for evaluation
        purposes (e.g. in SQL).

        For doing any self-referencing operations, like a self-join, you will
        use this operation to create a reference to the current table
        expression.

        Returns
        -------
        expr : TableExpr
        """
        return TableExpr(ops.SelfReference(self))

    def add_column(self, expr, name=None):
        if not isinstance(expr, ArrayExpr):
            raise TypeError('Must pass array expression')

        if name is not None:
            expr = expr.name(name)

        # New column originates from this table expression if at all
        self._assert_valid([expr])
        return self.projection([self, expr])

    def add_columns(self, what):
        raise NotImplementedError

    def count(self):
        return ops.Count(self).to_expr()

    def distinct(self):
        """
        Compute set of unique rows/tuples occurring in this table
        """
        op = ops.Distinct(self)
        return op.to_expr()

    def cross_join(self, other, prefixes=None):
        """

        """
        op = ops.CrossJoin(self, other)
        return TableExpr(op)

    def inner_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = ops.InnerJoin(self, other, predicates)
        return TableExpr(op)

    def left_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = ops.LeftJoin(self, other, predicates)
        return TableExpr(op)

    def outer_join(self, other, predicates=(), prefixes=None):
        """

        """
        op = ops.OuterJoin(self, other, predicates)
        return TableExpr(op)

    def semi_join(self, other, predicates, prefixes=None):
        """

        """
        op = ops.LeftSemiJoin(self, other, predicates)
        return TableExpr(op)

    def anti_join(self, other, predicates, prefixes=None):
        """

        """
        op = ops.LeftAntiJoin(self, other, predicates)
        return TableExpr(op)

    def projection(self, exprs):
        """

        """
        import ibis.expr.analysis as L

        clean_exprs = []

        validator = L.ExprValidator([self])

        for expr in exprs:
            expr = self._ensure_expr(expr)

            # Perform substitution only if we share common roots
            if validator.shares_some_roots(expr):
                expr = L.substitute_parents(expr)
            clean_exprs.append(expr)

        op = L._maybe_fuse_projection(self, clean_exprs)
        return TableExpr(op)

    def _ensure_expr(self, expr):
        if isinstance(expr, basestring):
            expr = self[expr]

        return expr

    def filter(self, predicates):
        """

        Parameters
        ----------

        Returns
        -------
        filtered_expr : TableExpr
        """
        import ibis.expr.analysis as L
        op = L.apply_filter(self, predicates)
        return TableExpr(op)

    def group_by(self, by):
        """
        Create an intermediate grouped table expression, pending some group
        operation to be applied with it.

        Examples
        --------
        x.group_by([b1, b2]).aggregate(metrics)

        Returns
        -------
        grouped_expr : GroupedTableExpr
        """
        return GroupedTableExpr(self, by)

    def aggregate(self, agg_exprs, by=None, having=None):
        """
        Parameters
        ----------

        Returns
        -------
        agg_expr : TableExpr
        """
        op = ops.Aggregation(self, agg_exprs, by=by, having=having)
        return TableExpr(op)

    def limit(self, n, offset=None):
        """

        Parameters
        ----------

        Returns
        -------
        limited : TableExpr
        """
        op = ops.Limit(self, n, offset=offset)
        return TableExpr(op)

    def sort_by(self, what):
        if not isinstance(what, list):
            what = [what]

        op = ops.SortBy(self, what)
        return TableExpr(op)

    def union(self, other, distinct=False):
        """
        Form the table set union of two table expressions having identical
        schemas.

        Parameters
        ----------
        other : TableExpr
        distinct : boolean, default False
            Only union distinct rows not occurring in the calling table (this
            can be very expensive, be careful)

        Returns
        -------
        union : TableExpr
        """
        op = ops.Union(self, other, distinct=distinct)
        return TableExpr(op)


class GroupedTableExpr(object):
    """
    Helper intermediate construct
    """
    def __init__(self, table, by):
        if not isinstance(by, (list, tuple)):
            if not isinstance(by, Expr):
                by = table._resolve([by])
            else:
                by = [by]

        self.table = table
        self.by = by

    def aggregate(self, metrics, having=None):
        return self.table.aggregate(metrics, by=self.by, having=having)


#------------------------------------------------------------------------------
# Declare all typed ValueExprs. This is what the user will actually interact
# with: an instance of each is well-typed and includes all valid methods
# defined for each type.


class AnyValue(ValueExpr):

    _typename = 'any'


class NullValue(AnyValue):

    _typename = 'null'

    def _can_cast_implicit(self, typename):
        return True


class NumericValue(AnyValue):

    __neg__ = _unary_op('__neg__', _negate)

    abs = _unary_op('abs', ops.Abs, 'Absolute value')

    ceil = _unary_op('ceil', ops.Ceil)
    floor = _unary_op('floor', ops.Floor)

    sign = _unary_op('sign', ops.Sign)

    exp = _unary_op('exp', ops.Exp)
    sqrt = _unary_op('sqrt', ops.Sqrt)

    log = _unary_op('log', ops.Log, 'Natural logarithm')
    ln = log
    log2 = _unary_op('log2', ops.Log2, 'Logarithm base 2')
    log10 = _unary_op('log10', ops.Log10, 'Logarithm base 10')

    def round(self, digits=None):
        """

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
        op = ops.Round(self, digits)
        return op.to_expr()

    def _can_compare(self, other):
        return isinstance(other, NumericValue)


class IntegerValue(NumericValue):
    pass


class BooleanValue(NumericValue):

    _typename = 'boolean'

    # TODO: logical binary operators for BooleanValue
    __and__ = _boolean_binary_op('__and__', ops.And)
    __or__ = _boolean_binary_op('__or__', ops.Or)
    __xor__ = _boolean_binary_op('__xor__', ops.Xor)

    __rand__ = _boolean_binary_rop('__rand__', ops.And)
    __ror__ = _boolean_binary_rop('__ror__', ops.Or)
    __rxor__ = _boolean_binary_rop('__rxor__', ops.Xor)

    def ifelse(self, true_expr, false_expr):
        """
        Shorthand for implementing ternary expressions

        bool_expr.ifelse(0, 1)
        e.g., in SQL: CASE WHEN bool_expr THEN 0 else 1 END
        """
        # Result will be the result of promotion of true/false exprs. These
        # might be conflicting types; same type resolution as case expressions
        # must be used.
        case = ops.SearchedCaseBuilder()
        return case.when(self, true_expr).else_(false_expr).end()


class Int8Value(IntegerValue):

    _typename = 'int8'
    _implicit_casts = {'int16', 'int32', 'int64', 'float', 'double'}


class Int16Value(IntegerValue):

    _typename = 'int16'
    _implicit_casts = {'int32', 'int64', 'float', 'double'}

class Int32Value(IntegerValue):

    _typename = 'int32'
    _implicit_casts = {'int64', 'float', 'double'}


class Int64Value(IntegerValue):

    _typename = 'int64'
    _implicit_casts = {'float', 'double'}




class FloatingValue(NumericValue):
    pass


class FloatValue(FloatingValue):

    _typename = 'float'
    _implicit_casts = {'double'}


class DoubleValue(FloatingValue):

    _typename = 'double'


class StringValue(AnyValue):

    _typename = 'string'

    def _can_compare(self, other):
        return isinstance(other, StringValue)

    length = _unary_op('length', ops.StringLength)
    lower = _unary_op('lower', ops.Lowercase)
    upper = _unary_op('upper', ops.Uppercase)

    def substr(self, start, length=None):
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
        op = ops.Substring(self, start, length)
        return op.to_expr()

    def left(self, nchars):
        """
        Return left-most up to N characters from each string. Convenience
        use of substr.

        Returns
        -------
        substrings : type of caller
        """
        return self.substr(0, length=nchars)

    def right(self, nchars):
        """
        Split up to nchars starting from end of each string.

        Returns
        -------
        substrings : type of caller
        """
        return ops.StrRight(self, nchars).to_expr()


class DecimalType(base.DataType):
    # Decimal types are parametric, we store the parameters in this object

    def __init__(self, precision, scale):
        self.precision = precision
        self.scale = scale

    def __repr__(self):
        return ('decimal(precision=%s, scale=%s)'
                % (self.precision, self.scale))

    def __eq__(self, other):
        if not isinstance(other, DecimalType):
            return False

        return (self.precision == other.precision and
                self.scale == other.scale)

    def array_ctor(self):
        def constructor(op, name=None):
            return DecimalArray(op, self, name=name)
        return constructor

    def scalar_ctor(self):
        def constructor(op, name=None):
            return DecimalScalar(op, self, name=name)
        return constructor


class DecimalValue(NumericValue):

    _typename = 'decimal'
    _implicit_casts = {'float', 'double'}

    def __init__(self, meta):
        self.meta = meta
        self.precision = meta.precision
        self.scale = meta.scale

    @classmethod
    def _make_constructor(cls, meta):
        def constructor(arg, name=None):
            return cls(arg, meta, name=name)
        return constructor


def _extract_field(name, klass):
    def f(self):
        op = klass(self)
        return op.to_expr()
    f.__name__ = name
    return f


class TimestampValue(AnyValue):

    _typename = 'timestamp'

    year = _extract_field('year', ops.ExtractYear)
    month = _extract_field('month', ops.ExtractMonth)
    day = _extract_field('day', ops.ExtractDay)
    hour = _extract_field('hour', ops.ExtractHour)
    minute = _extract_field('minute', ops.ExtractMinute)
    second = _extract_field('second', ops.ExtractSecond)
    millisecond = _extract_field('millisecond', ops.ExtractMillisecond)


class NumericArray(ArrayExpr, NumericValue):

    def count(self):
        # TODO: should actually get the parent table expression here
        return ops.Count(self).to_expr()

    sum = _agg_function('sum', ops.Sum)
    mean = _agg_function('mean', ops.Mean)
    min = _agg_function('min', ops.Min)
    max = _agg_function('max', ops.Max)


class NullScalar(NullValue, ScalarExpr):
    pass


class ListExpr(ArrayExpr, AnyValue):
    pass


class BooleanScalar(ScalarExpr, BooleanValue):
    pass


class BooleanArray(NumericArray, BooleanValue):

    def any(self):
        op = ops.Any(self)
        return op.to_expr()

    def none(self):
        pass

    def all(self):
        raise NotImplementedError


class Int8Scalar(ScalarExpr, Int8Value):
    pass


class Int8Array(NumericArray, Int8Value):
    pass


class Int16Scalar(ScalarExpr, Int16Value):
    pass


class Int16Array(NumericArray, Int16Value):
    pass


class Int32Scalar(ScalarExpr, Int32Value):
    pass


class Int32Array(NumericArray, Int32Value):
    pass


class Int64Scalar(ScalarExpr, Int64Value):
    pass


class Int64Array(NumericArray, Int64Value):
    pass


class FloatScalar(ScalarExpr, FloatValue):
    pass


class FloatArray(NumericArray, FloatValue):
    pass


class DoubleScalar(ScalarExpr, DoubleValue):
    pass


class DoubleArray(NumericArray, DoubleValue):
    pass


class StringScalar(ScalarExpr, StringValue):
    pass


class StringArray(ArrayExpr, StringValue):
    pass


class TimestampScalar(ScalarExpr, TimestampValue):
    pass


class TimestampArray(ArrayExpr, TimestampValue):
    pass


class DecimalScalar(DecimalValue, ScalarExpr):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        ScalarExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return DecimalScalar(arg, self.meta, name=name)
        return factory


class DecimalArray(DecimalValue, NumericArray):

    def __init__(self, arg, meta, name=None):
        DecimalValue.__init__(self, meta)
        ArrayExpr.__init__(self, arg, name=name)

    @property
    def _factory(self):
        def factory(arg, name=None):
            return DecimalArray(arg, self.meta, name=name)
        return factory


def scalar_type(t):
    if isinstance(t, base.DataType):
        return t.scalar_ctor()
    else:
        return _scalar_types[t]


def array_type(t):
    if isinstance(t, base.DataType):
        return t.array_ctor()
    else:
        return _array_types[t]



_scalar_types = {
    'boolean': BooleanScalar,
    'int8': Int8Scalar,
    'int16': Int16Scalar,
    'int32': Int32Scalar,
    'int64': Int64Scalar,
    'float': FloatScalar,
    'double': DoubleScalar,
    'string': StringScalar,
    'timestamp': TimestampScalar
}


_array_types = {
    'boolean': BooleanArray,
    'int8': Int8Array,
    'int16': Int16Array,
    'int32': Int32Array,
    'int64': Int64Array,
    'float': FloatArray,
    'double': DoubleArray,
    'string': StringArray,
    'timestamp': TimestampArray
}

#----------------------------------------------------------------------

_NULL = NullScalar(base.NullLiteral())


def _validate_type(t):
    if isinstance(t, base.DataType):
        return t

    parsed_type = _parse_type(t)
    if parsed_type is not None:
        return parsed_type

    if t not in _array_types:
        raise ValueError('Invalid type: %s' % repr(t))
    return t



_DECIMAL_RE = re.compile('decimal\((\d+),[\s]*(\d+)\)')


def _parse_decimal(t):
    m = _DECIMAL_RE.match(t)
    if m:
        precision, scale = m.groups()
        return DecimalType(int(precision), int(scale))

    if t == 'decimal':
        # From the Impala documentation
        return DecimalType(9, 0)


_type_parsers = [
    _parse_decimal
]


def _parse_type(t):
    for parse_fn in _type_parsers:
        parsed = parse_fn(t)
        if parsed is not None:
            return parsed
    return None


class UnnamedMarker(object):
    pass


unnamed = UnnamedMarker()
