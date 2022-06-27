from __future__ import annotations

import itertools
import operator
from typing import Callable, Iterable, Iterator

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.registry import (
    operation_registry,
    quote_identifier,
)
from ibis.expr.types.core import unnamed


class QueryContext:
    """Records bits of information used during ibis AST to SQL translation.

    Notably, table aliases (for subquery construction) and scalar query
    parameters are tracked here.
    """

    def __init__(self, compiler, indent=2, parent=None, params=None):
        self.compiler = compiler
        self.table_refs = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent
        self.parent = parent
        self.always_alias = False
        self.query = None
        self.params = params if params is not None else {}

    def _compile_subquery(self, op):
        sub_ctx = self.subcontext()
        return self._to_sql(op, sub_ctx)

    def _to_sql(self, expr, ctx):
        return self.compiler.to_sql(expr, ctx)

    def collapse(self, queries: Iterable[str]) -> str:
        """Turn an iterable of queries into something executable.

        Parameters
        ----------
        queries
            Iterable of query strings

        Returns
        -------
        query
            A single query string
        """
        return '\n\n'.join(queries)

    @property
    def top_context(self):
        if self.parent is None:
            return self
        else:
            return self.parent.top_context

    def set_always_alias(self):
        self.always_alias = True

    def get_compiled_expr(self, op):
        this = self.top_context

        key = self._get_table_key(op)
        try:
            return this.subquery_memo[key]
        except KeyError:
            pass

        if isinstance(op, (ops.SQLQueryResult, ops.SQLStringView)):
            result = op.query
        else:
            result = self._compile_subquery(op)

        this.subquery_memo[key] = result
        return result

    def make_alias(self, expr):
        i = len(self.table_refs)

        key = self._get_table_key(expr)

        # Get total number of aliases up and down the tree at this point; if we
        # find the table prior-aliased along the way, however, we reuse that
        # alias
        for ctx in itertools.islice(self._contexts(), 1, None):
            try:
                alias = ctx.table_refs[key]
            except KeyError:
                pass
            else:
                self.set_ref(expr, alias)
                return

            i += len(ctx.table_refs)

        alias = f't{i:d}'
        self.set_ref(expr, alias)

    def need_aliases(self, expr=None):
        return self.always_alias or len(self.table_refs) > 1

    def _contexts(
        self,
        *,
        parents: bool = True,
    ) -> Iterator[QueryContext]:
        ctx = self
        yield ctx
        while parents and ctx.parent is not None:
            ctx = ctx.parent
            yield ctx

    def has_ref(self, expr, parent_contexts=False):
        key = self._get_table_key(expr)
        return any(
            key in ctx.table_refs
            for ctx in self._contexts(parents=parent_contexts)
        )

    def set_ref(self, expr, alias):
        key = self._get_table_key(expr)
        self.table_refs[key] = alias

    def get_ref(self, expr):
        """Return the alias used to refer to an expression."""
        key = self._get_table_key(expr)
        top = self.top_context

        if self.is_extracted(expr):
            return top.table_refs.get(key)

        return self.table_refs.get(key)

    def is_extracted(self, expr):
        key = self._get_table_key(expr)
        return key in self.top_context.extracted_subexprs

    def set_extracted(self, expr):
        key = self._get_table_key(expr)
        self.extracted_subexprs.add(key)
        self.make_alias(expr)

    def subcontext(self):
        return self.__class__(
            compiler=self.compiler,
            indent=self.indent,
            parent=self,
            params=self.params,
        )

    # Maybe temporary hacks for correlated / uncorrelated subqueries

    def set_query(self, query):
        self.query = query

    def is_foreign_expr(self, expr):
        from ibis.expr.analysis import shares_all_roots

        # The expression isn't foreign to us. For example, the parent table set
        # in a correlated WHERE subquery
        if self.has_ref(expr, parent_contexts=True):
            return False

        parents = [self.query.table_set] + self.query.select_set
        return not shares_all_roots(expr, parents)

    def _get_table_key(self, table):
        if isinstance(table, ir.Table):
            return table.op()
        elif isinstance(table, ops.TableNode):
            return table
        raise TypeError(f"invalid table expression: {type(table)}")


class ExprTranslator:
    """Translates ibis expressions into a compilation target."""

    _registry = operation_registry
    _rewrites: dict[ops.Node, Callable] = {}

    def __init__(self, node, context, named=False, permit_subquery=False):
        self.node = node
        self.permit_subquery = permit_subquery

        assert context is not None, 'context is None in {}'.format(
            type(self).__name__
        )
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

    def _needs_name(self, op):
        if not self.named:
            return False

        if isinstance(op, ops.TableColumn):
            # This column has been given an explicitly different name
            # return expr.get_name() != op.name
            return False

        return op.resolve_name() is not unnamed

    def name(self, translated, name, force=True):
        return '{} AS {}'.format(
            translated, quote_identifier(name, force=force)
        )

    def get_result(self):
        """Compile SQL expression into a string."""
        translated = self.translate(self.node)
        if self._needs_name(self.node):
            # TODO: this could fail in various ways
            name = self.node.resolve_name()
            translated = self.name(translated, name)
        return translated

    @classmethod
    def add_operation(cls, operation, translate_function):
        """Add an operation to the operation registry.

        In general, operations should be defined directly in the registry, in
        `registry.py`. There are couple of exceptions why this is needed.

        Operations defined by Ibis users (not Ibis or backend developers), and
        UDFs which are added dynamically.
        """
        cls._registry[operation] = translate_function

    def translate(self, op):
        assert isinstance(op, ops.Node), type(op)

        if type(op) in self._rewrites:  # even if type(op) is in self._registry
            op = self._rewrites[type(op)](op)

        # TODO: use op MRO for subclasses instead of this isinstance spaghetti
        if isinstance(op, ops.ScalarParameter):
            return self._trans_param(op)
        elif isinstance(op, ops.TableNode):
            # HACK/TODO: revisit for more complex cases
            return '*'
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, op)
        else:
            raise com.OperationNotDefinedError(
                f'No translation rule for {type(op)}'
            )

    def _trans_param(self, op):
        raw_value = self.context.params[op]
        dtype = op.output_dtype
        if isinstance(dtype, dt.Struct):
            literal = ibis.struct(raw_value, type=dtype)
        elif isinstance(dtype, dt.Map):
            literal = ibis.map(raw_value, type=dtype)
        else:
            literal = ibis.literal(raw_value, type=dtype)
        return self.translate(literal.op())

    @classmethod
    def rewrites(cls, klass):
        def decorator(f):
            cls._rewrites[klass] = f
            return f

        return decorator


rewrites = ExprTranslator.rewrites


# TODO(kszucs): use analysis.substitute() instead of a custom rewriter
@rewrites(ops.Bucket)
def _bucket(op):
    # TODO(kszucs): avoid doing the expr roundtrip
    expr = op.arg.to_expr()
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
        stmt = stmt.when(cmp(expr, op.buckets[0]), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if op.close_extreme and (
            (op.closed == 'right' and j == 0)
            or (op.closed == 'left' and j == (user_num_buckets - 1))
        ):
            stmt = stmt.when((lower <= expr) & (expr <= upper), bucket_id)
        else:
            stmt = stmt.when(
                l_cmp(lower, expr) & r_cmp(expr, upper), bucket_id
            )
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = operator.lt if op.close_extreme else l_cmp
        else:
            cmp = operator.lt if op.closed == 'right' else operator.le

        stmt = stmt.when(cmp(op.buckets[-1], expr), bucket_id)
        bucket_id += 1

    result = stmt.end()
    if expr.has_name():
        result = result.name(expr.get_name())

    return result.op()


@rewrites(ops.CategoryLabel)
def _category_label(op):
    # TODO(kszucs): avoid doing the expr roundtrip
    expr = op.to_expr()
    stmt = op.args[0].to_expr().case()
    for i, label in enumerate(op.labels):
        stmt = stmt.when(i, label)

    if op.nulls is not None:
        stmt = stmt.else_(op.nulls)

    result = stmt.end()
    if expr.has_name():
        result = result.name(expr.get_name())

    return result.op()


@rewrites(ops.Any)
def _any_expand(op):
    # TODO(kszucs): avoid doing the expr->op roundtrip
    arg = op.arg.to_expr()
    return arg.max().op()


@rewrites(ops.NotAny)
def _notany_expand(op):
    # TODO(kszucs): avoid doing the expr->op roundtrip
    arg = op.arg.to_expr()
    new_expr = arg.max() == ibis.literal(0, type=arg.type())
    return new_expr.op()


@rewrites(ops.All)
def _all_expand(op):
    # TODO(kszucs): avoid doing the expr->op roundtrip
    arg = op.arg.to_expr()
    return arg.min().op()


@rewrites(ops.NotAll)
def _notall_expand(op):
    # TODO(kszucs): avoid doing the expr->op roundtrip
    arg = op.arg.to_expr()
    dtype = op.arg.output_dtype
    new_expr = arg.min() == ibis.literal(0, type=dtype)
    return new_expr.op()


@rewrites(ops.Cast)
def _rewrite_cast(op):
    # TODO(kszucs): avoid doing the expr->op roundtrip
    if isinstance(op.to, dt.Interval) and isinstance(
        op.arg.output_dtype, dt.Integer
    ):
        return op.arg.to_expr().to_interval(unit=op.to.unit).op()
    return op


@rewrites(ops.StringContains)
def _rewrite_string_contains(op):
    return ops.GreaterEqual(ops.StringFind(op.haystack, op.needle), 0)
