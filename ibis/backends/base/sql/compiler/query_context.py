import ibis.expr.format as fmt
import ibis.expr.operations as ops
import ibis.expr.types as ir

from . import translator


class QueryContext:

    """Records bits of information used during ibis AST to SQL translation.

    Notably, table aliases (for subquery construction) and scalar query
    parameters are tracked here.
    """

    def __init__(
        self, indent=2, parent=None, memo=None, dialect=None, params=None
    ):
        self._table_refs = {}
        self.extracted_subexprs = set()
        self.subquery_memo = {}
        self.indent = indent
        self.parent = parent

        self.always_alias = False

        self.query = None

        self._table_key_memo = {}
        self.memo = memo or fmt.FormatMemo()
        self.dialect = dialect
        self.params = params if params is not None else {}

    def _compile_subquery(self, expr):
        sub_ctx = self.subcontext()
        return self._to_sql(expr, sub_ctx)

    def _to_sql(self, expr, ctx):
        return translator.to_sql(expr, ctx)

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
        if key in this.subquery_memo:
            return this.subquery_memo[key]

        op = expr.op()
        if isinstance(op, ops.SQLQueryResult):
            result = op.query
        else:
            result = self._compile_subquery(expr)

        this.subquery_memo[key] = result
        return result

    def make_alias(self, expr):
        i = len(self._table_refs)

        key = self._get_table_key(expr)

        # Get total number of aliases up and down the tree at this point; if we
        # find the table prior-aliased along the way, however, we reuse that
        # alias
        ctx = self
        while ctx.parent is not None:
            ctx = ctx.parent

            if key in ctx._table_refs:
                alias = ctx._table_refs[key]
                self.set_ref(expr, alias)
                return

            i += len(ctx._table_refs)

        alias = 't{:d}'.format(i)
        self.set_ref(expr, alias)

    def need_aliases(self, expr=None):
        return self.always_alias or len(self._table_refs) > 1

    def has_ref(self, expr, parent_contexts=False):
        key = self._get_table_key(expr)
        return self._key_in(
            key, '_table_refs', parent_contexts=parent_contexts
        )

    def set_ref(self, expr, alias):
        key = self._get_table_key(expr)
        self._table_refs[key] = alias

    def get_ref(self, expr):
        """
        Get the alias being used throughout a query to refer to a particular
        table or inline view
        """
        return self._get_table_item('_table_refs', expr)

    def is_extracted(self, expr):
        key = self._get_table_key(expr)
        return key in self.top_context.extracted_subexprs

    def set_extracted(self, expr):
        key = self._get_table_key(expr)
        self.extracted_subexprs.add(key)
        self.make_alias(expr)

    def subcontext(self):
        return type(self)(indent=self.indent, parent=self, params=self.params)

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

    def _get_table_item(self, item, expr):
        key = self._get_table_key(expr)
        top = self.top_context

        if self.is_extracted(expr):
            return getattr(top, item).get(key)

        return getattr(self, item).get(key)

    def _get_table_key(self, table):
        if isinstance(table, ir.TableExpr):
            table = table.op()

        try:
            return self._table_key_memo[table]
        except KeyError:
            val = table._repr()
            self._table_key_memo[table] = val
            return val

    def _key_in(self, key, memo_attr, parent_contexts=False):
        if key in getattr(self, memo_attr):
            return True

        ctx = self
        while parent_contexts and ctx.parent is not None:
            ctx = ctx.parent
            if key in getattr(ctx, memo_attr):
                return True

        return False
