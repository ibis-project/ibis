"""OmniSciDB Compiler module."""
from io import StringIO

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
import ibis.util as util
from ibis.backends.base_sql.compiler import BaseExprTranslator
from ibis.backends.base_sqlalchemy import compiler
from ibis.expr.api import _add_methods, _binop_expr, _unary_op

from . import operations as omniscidb_ops
from .identifiers import quote_identifier  # noqa: F401
from .operations import _type_to_sql_string  # noqa: F401


def build_ast(
    expr: ibis.Expr, context: compiler.QueryContext,
) -> compiler.QueryAST:
    """Build AST from given expression.

    Parameters
    ----------
    expr : ibis.Expr
    context : compiler.QueryContext

    Returns
    -------
    compiler.QueryAST
    """
    assert context is not None, 'context is None'
    builder = OmniSciDBQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(
    expr: ibis.Expr, context: compiler.QueryContext,
):
    assert context is not None, 'context is None'
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr: ibis.Expr, context: compiler.QueryContext = None,) -> str:
    """Convert expression to SQL statement.

    Parameters
    ----------
    expr : ibis.Expr
    context : compiler.QueryContext, optional

    Returns
    -------
    str
    """
    if context is None:
        context = OmniSciDBDialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()


class OmniSciDBSelectBuilder(compiler.SelectBuilder):
    """OmniSciDB Select Builder class."""

    @property
    def _select_class(self):
        return OmniSciDBSelect

    def _convert_group_by(self, exprs):
        return exprs


class OmniSciDBQueryBuilder(compiler.QueryBuilder):
    """OmniSciDB Query Builder class."""

    select_builder = OmniSciDBSelectBuilder
    union_class = None

    def _make_union(self):
        raise com.UnsupportedOperationError(
            "OmniSciDB backend doesn't support Union operation"
        )


class OmniSciDBQueryContext(compiler.QueryContext):
    """OmniSciDB Query Context class."""

    always_alias = False

    def _to_sql(self, expr, ctx):
        ctx.always_alias = False
        return to_sql(expr, context=ctx)


class OmniSciDBSelect(compiler.Select):
    """OmniSciDB Select class."""

    @property
    def translator(self):
        """Return the translator class.

        Returns
        -------
        OmniSciDBExprTranslator
        """
        return OmniSciDBExprTranslator

    @property
    def table_set_formatter(self):
        """Return the Table Set Formatter.

        Returns
        -------
        OmniSciDBTableSetFormatter
        """
        return OmniSciDBTableSetFormatter

    def format_select_set(self) -> str:
        """Format the select clause.

        Returns
        -------
        string
        """
        return super().format_select_set()

    def format_group_by(self) -> str:
        """Format the group by clause.

        Returns
        -------
        string
        """
        if not self.group_by:
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if self.group_by:
            columns = ['{}'.format(expr.get_name()) for expr in self.group_by]
            clause = 'GROUP BY {}'.format(', '.join(columns))
            lines.append(clause)

        if self.having:
            trans_exprs = []
            for expr in self.having:
                translated = self._translate(expr)
                trans_exprs.append(translated)
            lines.append('HAVING {}'.format(' AND '.join(trans_exprs)))

        return '\n'.join(lines)

    def format_limit(self):
        """Format the limit clause.

        Returns
        -------
        string
        """
        if not self.limit:
            return None

        buf = StringIO()

        n, offset = self.limit['n'], self.limit['offset']
        buf.write('LIMIT {}'.format(n))
        if offset is not None and offset != 0:
            buf.write(', {}'.format(offset))

        return buf.getvalue()


class OmniSciDBTableSetFormatter(compiler.TableSetFormatter):
    """OmniSciDB Table Set Formatter class."""

    _join_names = {
        ops.InnerJoin: 'JOIN',
        ops.LeftJoin: 'LEFT JOIN',
        ops.LeftSemiJoin: 'JOIN',  # needed by topk as filter
        ops.CrossJoin: 'JOIN',
    }

    def get_result(self):
        """Get a formatted string for the expression.

        Got to unravel the join stack; the nesting order could be
        arbitrary, so we do a depth first search and push the join tokens
        and predicates onto a flat list, then format them

        Returns
        -------
        string
        """
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        buf = StringIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(
            self.join_types, self.join_tables[1:], self.join_predicates
        ):
            buf.write('\n')
            buf.write(util.indent('{} {}'.format(jtype, table), self.indent))

            fmt_preds = []
            npreds = len(preds)
            for pred in preds:
                new_pred = self._translate(pred)
                if npreds > 1:
                    new_pred = '({})'.format(new_pred)
                fmt_preds.append(new_pred)

            if len(fmt_preds):
                buf.write('\n')

                conj = ' AND\n{}'.format(' ' * 3)
                fmt_preds = util.indent(
                    'ON ' + conj.join(fmt_preds), self.indent * 2
                )
                buf.write(fmt_preds)
            else:
                buf.write(util.indent('ON TRUE', self.indent * 2))

        return buf.getvalue()

    _non_equijoin_supported = True

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()

            if (
                not isinstance(op, ops.Equals)
                and not self._non_equijoin_supported
            ):
                raise com.TranslationError(
                    'Non-equality join predicates, '
                    'i.e. non-equijoins, are not '
                    'supported'
                )

    def _format_predicate(self, predicate):
        column = predicate.op().args[0]
        return column.get_name()

    def _quote_identifier(self, name):
        return name


class OmniSciDBExprTranslator(compiler.ExprTranslator):
    """OmniSciDB Expr Translator class."""

    _registry = omniscidb_ops._operation_registry
    _rewrites = BaseExprTranslator._rewrites.copy()

    context_class = OmniSciDBQueryContext

    def name(self, translated: str, name: str, force=True):
        """Define name for the expression.

        Parameters
        ----------
        translated : str
            translated expresion
        name : str
        force : bool, optional
            if True force the new name, by default True

        Returns
        -------
        str
        """
        return omniscidb_ops._name_expr(translated, name)


class OmniSciDBDialect(compiler.Dialect):
    """OmniSciDB Dialect class."""

    translator = OmniSciDBExprTranslator


dialect = OmniSciDBDialect
compiles = OmniSciDBExprTranslator.compiles
rewrites = OmniSciDBExprTranslator.rewrites

omniscidb_reg = omniscidb_ops._operation_registry


@rewrites(ops.All)
def omniscidb_rewrite_all(expr: ibis.Expr) -> ibis.Expr:
    """Rewrite All operation.

    Parameters
    ----------
    expr : ibis.Expr

    Returns
    -------
    [type]
    """
    return omniscidb_ops._all(expr)


@rewrites(ops.Any)
def omniscidb_rewrite_any(expr: ibis.Expr) -> ibis.Expr:
    """Rewrite Any operation.

    Parameters
    ----------
    expr : ibis.Expr

    Returns
    -------
    ibis.Expr
    """
    return omniscidb_ops._any(expr)


@rewrites(ops.NotAll)
def omniscidb_rewrite_not_all(expr: ibis.Expr) -> ibis.Expr:
    """Rewrite Not All operation.

    Parameters
    ----------
    expr : ibis.Expr

    Returns
    -------
    ibis.Expr
    """
    return omniscidb_ops._not_all(expr)


@rewrites(ops.NotAny)
def omniscidb_rewrite_not_any(expr: ibis.Expr) -> ibis.Expr:
    """Rewrite Not Any operation.

    Parameters
    ----------
    expr : ibis.Expr

    Returns
    -------
    ibis.Expr
    """
    return omniscidb_ops._not_any(expr)


_add_methods(
    ir.NumericValue,
    dict(
        conv_4326_900913_x=_unary_op(
            'conv_4326_900913_x', omniscidb_ops.Conv_4326_900913_X
        ),
        conv_4326_900913_y=_unary_op(
            'conv_4326_900913_y', omniscidb_ops.Conv_4326_900913_Y
        ),
        truncate=_binop_expr('truncate', omniscidb_ops.NumericTruncate),
    ),
)

_add_methods(
    ir.StringValue,
    dict(byte_length=_unary_op('length', omniscidb_ops.ByteLength)),
)
