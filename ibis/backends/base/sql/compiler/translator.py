from __future__ import annotations

import contextlib
import itertools
from typing import TYPE_CHECKING, Callable

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
from ibis.backends.base.sql.registry import operation_registry, quote_identifier

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator


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
        self.always_alias = True
        self.query = None
        self.params = params if params is not None else {}
        self._alias_counter = getattr(parent, "_alias_counter", 0)

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
        return "\n\n".join(queries)

    @property
    def top_context(self):
        if self.parent is None:
            return self
        else:
            return self.parent.top_context

    def set_always_alias(self):
        self.always_alias = True

    def get_compiled_expr(self, node):
        with contextlib.suppress(KeyError):
            return self.top_context.subquery_memo[node]

        if isinstance(node, (ops.SQLQueryResult, ops.SQLStringView)):
            result = node.query
        else:
            result = self._compile_subquery(node)

        self.top_context.subquery_memo[node] = result
        return result

    def _next_alias(self) -> str:
        alias = self._alias_counter
        self._alias_counter += 1
        return f"t{alias:d}"

    def make_alias(self, node):
        # check for existing tables that we're referencing from a parent context
        for ctx in itertools.islice(self._contexts(), 1, None):
            if (alias := ctx.table_refs.get(node)) is not None:
                self.set_ref(node, alias)
                return

        self.set_ref(node, self._next_alias())

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

    def has_ref(self, node, parent_contexts=False):
        return any(
            node in ctx.table_refs for ctx in self._contexts(parents=parent_contexts)
        )

    def set_ref(self, node, alias):
        self.table_refs[node] = alias

    def get_ref(self, node, search_parents=False):
        """Return the alias used to refer to an expression."""
        assert isinstance(node, ops.Node), type(node)

        if self.is_extracted(node):
            return self.top_context.table_refs.get(node)

        if (ref := self.table_refs.get(node)) is not None:
            return ref

        if search_parents and (parent := self.parent) is not None:
            return parent.get_ref(node, search_parents=search_parents)

        return None

    def is_extracted(self, node):
        return node in self.top_context.extracted_subexprs

    def set_extracted(self, node):
        self.extracted_subexprs.add(node)
        self.make_alias(node)

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

    def is_foreign_expr(self, op):
        from ibis.expr.analysis import shares_all_roots

        # The expression isn't foreign to us. For example, the parent table set
        # in a correlated WHERE subquery
        if self.has_ref(op, parent_contexts=True):
            return False

        parents = [self.query.table_set] + self.query.select_set
        return not shares_all_roots(op, parents)


class ExprTranslator:
    """Translates ibis expressions into a compilation target."""

    _registry = operation_registry
    _rewrites: dict[ops.Node, Callable] = {}
    _forbids_frame_clause = (
        ops.DenseRank,
        ops.MinRank,
        ops.NTile,
        ops.PercentRank,
        ops.CumeDist,
        ops.RowNumber,
    )
    _require_order_by = (
        ops.Lag,
        ops.Lead,
        ops.DenseRank,
        ops.MinRank,
        ops.FirstValue,
        ops.LastValue,
        ops.PercentRank,
        ops.CumeDist,
        ops.NTile,
    )
    _unsupported_reductions = (
        ops.ApproxMedian,
        ops.GroupConcat,
        ops.ApproxCountDistinct,
    )
    _dialect_name = "hive"
    _quote_identifiers = None

    def __init__(
        self, node, context, named=False, permit_subquery=False, within_where=False
    ):
        self.node = node
        self.permit_subquery = permit_subquery

        assert context is not None, f"context is None in {type(self).__name__}"
        self.context = context

        # For now, governing whether the result will have a name
        self.named = named

        # used to indicate whether the expression being rendered is within a
        # WHERE clause. This is used for MSSQL to determine whether to use
        # boolean expressions or not.
        self.within_where = within_where

    def _needs_name(self, op):
        if not self.named:
            return False

        if isinstance(op, ops.TableColumn):
            # This column has been given an explicitly different name
            return False

        return bool(op.name)

    def name(self, translated, name, force=True):
        return f"{translated} AS {quote_identifier(name, force=force)}"

    def get_result(self):
        """Compile SQL expression into a string."""
        translated = self.translate(self.node)
        if self._needs_name(self.node):
            # TODO: this could fail in various ways
            translated = self.name(translated, self.node.name)
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
            return "*"
        elif type(op) in self._registry:
            formatter = self._registry[type(op)]
            return formatter(self, op)
        else:
            raise com.OperationNotDefinedError(f"No translation rule for {type(op)}")

    def _trans_param(self, op):
        raw_value = self.context.params[op]
        dtype = op.dtype
        if dtype.is_struct():
            literal = ibis.struct(raw_value, type=dtype)
        elif dtype.is_map():
            literal = ibis.map(list(raw_value.keys()), list(raw_value.values()))
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


@rewrites(ops.Bucket)
def _bucket(op):
    # TODO(kszucs): avoid the expression roundtrip
    expr = op.arg.to_expr()
    stmt = ibis.case()

    if op.closed == "left":
        l_cmp = ops.LessEqual
        r_cmp = ops.Less
    else:
        l_cmp = ops.Less
        r_cmp = ops.LessEqual

    user_num_buckets = len(op.buckets) - 1

    bucket_id = 0
    if op.include_under:
        if user_num_buckets > 0:
            cmp = ops.Less if op.close_extreme else r_cmp
        else:
            cmp = ops.LessEqual if op.closed == "right" else ops.Less
        stmt = stmt.when(cmp(op.arg, op.buckets[0]).to_expr(), bucket_id)
        bucket_id += 1

    for j, (lower, upper) in enumerate(zip(op.buckets, op.buckets[1:])):
        if op.close_extreme and (
            (op.closed == "right" and j == 0)
            or (op.closed == "left" and j == (user_num_buckets - 1))
        ):
            stmt = stmt.when(
                ops.And(
                    ops.LessEqual(lower, op.arg), ops.LessEqual(op.arg, upper)
                ).to_expr(),
                bucket_id,
            )
        else:
            stmt = stmt.when(
                ops.And(l_cmp(lower, op.arg), r_cmp(op.arg, upper)).to_expr(),
                bucket_id,
            )
        bucket_id += 1

    if op.include_over:
        if user_num_buckets > 0:
            cmp = ops.Less if op.close_extreme else l_cmp
        else:
            cmp = ops.Less if op.closed == "right" else ops.LessEqual

        stmt = stmt.when(cmp(op.buckets[-1], op.arg).to_expr(), bucket_id)
        bucket_id += 1

    result = stmt.end()
    if expr.has_name():
        result = result.name(expr.get_name())

    return result.op()


@rewrites(ops.Any)
def _any_expand(op):
    return ops.Max(op.arg, where=op.where)


@rewrites(ops.All)
def _all_expand(op):
    return ops.Min(op.arg, where=op.where)


@rewrites(ops.Cast)
def _rewrite_cast(op):
    # TODO(kszucs): avoid the expression roundtrip
    if op.to.is_interval() and op.arg.dtype.is_integer():
        return op.arg.to_expr().to_interval(unit=op.to.unit).op()
    return op


@rewrites(ops.StringContains)
def _rewrite_string_contains(op):
    return ops.GreaterEqual(ops.StringFind(op.haystack, op.needle), 0)


@rewrites(ops.Clip)
def _rewrite_clip(op):
    dtype = op.dtype
    arg = ops.Cast(op.arg, dtype)

    arg_is_null = ops.IsNull(arg)

    if (upper := op.upper) is not None:
        clipped_lower = ops.Least((arg, ops.Cast(upper, dtype)))
        if dtype.nullable:
            arg = ops.IfElse(arg_is_null, arg, clipped_lower)
        else:
            arg = clipped_lower

    if (lower := op.lower) is not None:
        clipped_upper = ops.Greatest((arg, ops.Cast(lower, dtype)))
        if dtype.nullable:
            arg = ops.IfElse(arg_is_null, arg, clipped_upper)
        else:
            arg = clipped_upper

    return arg
