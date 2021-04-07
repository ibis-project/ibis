import operator

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql import (
    binary_infix_ops,
    operation_registry,
    quote_identifier,
)
from ibis.expr import analytics

from . import query_builder, query_context


class ExprTranslator:

    """Class that performs translation of ibis expressions into executable
    SQL.
    """

    _registry = {**operation_registry, **binary_infix_ops}
    _rewrites = {}

    context_class = query_context.QueryContext

    def __init__(self, expr, context, named=False, permit_subquery=False):
        self.expr = expr
        self.permit_subquery = permit_subquery

        assert context is not None, 'context is None in {}'.format(
            type(self).__name__
        )
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

    def get_result(self):
        """
        Build compiled SQL expression from the bottom up and return as a string
        """
        translated = self.translate(self.expr)
        if self._needs_name(self.expr):
            # TODO: this could fail in various ways
            name = self.expr.get_name()
            translated = self.name(translated, name)
        return translated

    @staticmethod
    def _name_expr(formatted_expr, quoted_name):
        return '{} AS {}'.format(formatted_expr, quoted_name)

    def name(self, translated, name, force=True):
        """Return expression with its identifier."""
        return self._name_expr(translated, quote_identifier(name, force=force))

    def _needs_name(self, expr):
        if not self.named:
            return False

        op = expr.op()
        if isinstance(op, ops.TableColumn):
            # This column has been given an explicitly different name
            if expr.get_name() != op.name:
                return True
            return False

        if expr.get_name() is ir.unnamed:
            return False

        return True

    def translate(self, expr):
        # The operation node type the typed expression wraps
        op = expr.op()

        if type(op) in self._rewrites:  # even if type(op) is in self._registry
            expr = self._rewrites[type(op)](expr)
            op = expr.op()

        # TODO: use op MRO for subclasses instead of this isinstance spaghetti
        if isinstance(op, ops.ScalarParameter):
            return self._trans_param(expr)
        elif isinstance(op, ops.TableNode):
            # HACK/TODO: revisit for more complex cases
            return '*'
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, expr)
        else:
            raise com.OperationNotDefinedError(
                'No translation rule for {}'.format(type(op))
            )

    def _trans_param(self, expr):
        raw_value = self.context.params[expr.op()]
        literal = ibis.literal(raw_value, type=expr.type())
        return self.translate(literal)

    @classmethod
    def rewrites(cls, klass):
        def decorator(f):
            cls._rewrites[klass] = f
            return f

        return decorator

    @classmethod
    def compiles(cls, klass):
        def decorator(f):
            cls._registry[klass] = f
            return f

        return decorator


rewrites = ExprTranslator.rewrites


@rewrites(analytics.Bucket)
def _bucket(expr):
    op = expr.op()
    stmt = ibis.case()

    if op.closed == 'left':
        l_cmp = operator.le
        r_cmp = operator.lt
    else:
        l_cmp = operator.lt
        r_cmp = operator.le

    user_num_buckets = len(op.buckets) - 1

    bucket_id = 0
    if op.include_under:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else r_cmp
        else:
            cmp = operator.le if op.closed == 'right' else operator.lt
        stmt = stmt.when(cmp(op.arg, op.buckets[0]), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if op.close_extreme and (
            (op.closed == 'right' and j == 0)
            or (op.closed == 'left' and j == (user_num_buckets - 1))
        ):
            stmt = stmt.when((lower <= op.arg) & (op.arg <= upper), bucket_id)
        else:
            stmt = stmt.when(
                l_cmp(lower, op.arg) & r_cmp(op.arg, upper), bucket_id
            )
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else l_cmp
        else:
            cmp = operator.lt if op.closed == 'right' else operator.le

        stmt = stmt.when(cmp(op.buckets[-1], op.arg), bucket_id)
        bucket_id += 1

    return stmt.end().name(expr._name)


@rewrites(analytics.CategoryLabel)
def _category_label(expr):
    op = expr.op()

    stmt = op.args[0].case()
    for i, label in enumerate(op.labels):
        stmt = stmt.when(i, label)

    if op.nulls is not None:
        stmt = stmt.else_(op.nulls)

    return stmt.end().name(expr._name)


@rewrites(ops.Any)
def _any_expand(expr):
    arg = expr.op().args[0]
    return arg.max()


@rewrites(ops.NotAny)
def _notany_expand(expr):
    arg = expr.op().args[0]
    return arg.max() == 0


@rewrites(ops.All)
def _all_expand(expr):
    arg = expr.op().args[0]
    return arg.min()


@rewrites(ops.NotAll)
def _notall_expand(expr):
    arg = expr.op().args[0]
    return arg.min() == 0


@rewrites(ops.Cast)
def _rewrite_cast(expr):
    arg, to = expr.op().args
    if isinstance(to, dt.Interval) and isinstance(arg.type(), dt.Integer):
        return arg.to_interval(unit=to.unit)
    return expr


@rewrites(ops.FloorDivide)
def _floor_divide(expr):
    left, right = expr.op().args
    return left.div(right).floor()


class Dialect:

    """Dialects encode the properties of a particular flavor of SQL.

    For example, quoting behavior is a property that should be encoded by
    ``Dialect``. Each backend has its own dialect.
    """

    translator = ExprTranslator

    @classmethod
    def make_context(cls, params=None):
        if params is None:
            params = {}
        params = {expr.op(): value for expr, value in params.items()}
        return cls.translator.context_class(dialect=cls(), params=params)


def build_ast(expr, context):
    assert context is not None, 'context is None'
    builder = query_builder.QueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    assert context is not None, 'context is None'
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    if context is None:
        context = Dialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()
