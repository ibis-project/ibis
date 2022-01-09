from __future__ import annotations

import itertools
import operator
from typing import Callable, Iterator

import ibis
import ibis.common.exceptions as com
import ibis.expr.datatypes as dt
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis.backends.base.sql.registry import (
    operation_registry,
    quote_identifier,
)


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

    def _compile_subquery(self, expr):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx)

    def _to_sql(self, expr, ctx):
        return self.compiler.to_sql(expr, ctx)

    def collapse(self, queries):
        """Turn a sequence of queries into something executable.

        Parameters
        ----------
        queries : List[str]

        Returns
        -------
        query : str
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

    def get_compiled_expr(self, expr):
        this = self.top_context

        key = self._get_table_key(expr)
        try:
            return this.subquery_memo[key]
        except KeyError:
            pass

        op = expr.op()
        if isinstance(op, ops.SQLQueryResult):
            result = op.query
        else:
            result = self._compile_subquery(expr)

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
        """
        Get the alias being used throughout a query to refer to a particular
        table or inline view
        """
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
        from ibis.expr.analysis import ExprValidator

        # The expression isn't foreign to us. For example, the parent table set
        # in a correlated WHERE subquery
        if self.has_ref(expr, parent_contexts=True):
            return False

        exprs = [self.query.table_set] + self.query.select_set
        validator = ExprValidator(exprs)
        return not validator.validate(expr)

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            return table.op()
        elif isinstance(table, ops.TableNode):
            return table
        raise TypeError(f"invalid table expression: {type(table)}")


class ExprTranslator:

    """Class that performs translation of ibis expressions into executable
    SQL.
    """

    _registry = operation_registry
    _rewrites: dict[ops.Node, Callable] = {}

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

    @classmethod
    def add_operation(cls, operation, translate_function):
        """
        Adds an operation to the operation registry. In general, operations
        should be defined directly in the registry, in `registry.py`. But
        there are couple of exceptions why this is needed. Operations defined
        by Ibis users (not Ibis or backend developers). and UDF, which are
        added dynamically.
        """
        cls._registry[operation] = translate_function

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

    def name(self, translated, name, force=True):
        return '{} AS {}'.format(
            translated, quote_identifier(name, force=force)
        )

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
                f'No translation rule for {type(op)}'
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


rewrites = ExprTranslator.rewrites


@rewrites(ops.Bucket)
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


@rewrites(ops.CategoryLabel)
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
