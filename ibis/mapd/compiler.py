from six import StringIO
from . import operations as mapd_ops
from .identifiers import quote_identifier  # noqa: F401
from .operations import _type_to_sql_string  # noqa: F401
from ibis.expr.api import _add_methods, _unary_op, _binop_expr

import ibis.common as com
import ibis.util as util
import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.sql.compiler as compiles

from ibis.impala import compiler as impala_compiler


def build_ast(expr, context):
    assert context is not None, 'context is None'
    builder = MapDQueryBuilder(expr, context=context)
    return builder.get_result()


def _get_query(expr, context):
    assert context is not None, 'context is None'
    ast = build_ast(expr, context)
    query = ast.queries[0]

    return query


def to_sql(expr, context=None):
    if context is None:
        context = MapDDialect.make_context()
    assert context is not None, 'context is None'
    query = _get_query(expr, context)
    return query.compile()


class MapDSelectBuilder(compiles.SelectBuilder):
    """

    """
    @property
    def _select_class(self):
        return MapDSelect

    def _convert_group_by(self, exprs):
        return exprs


class MapDQueryBuilder(compiles.QueryBuilder):
    """

    """
    select_builder = MapDSelectBuilder


class MapDQueryContext(compiles.QueryContext):
    """

    """
    always_alias = False

    def _to_sql(self, expr, ctx):
        ctx.always_alias = False
        return to_sql(expr, context=ctx)


class MapDSelect(compiles.Select):
    """

    """
    @property
    def translator(self):
        return MapDExprTranslator

    @property
    def table_set_formatter(self):
        return MapDTableSetFormatter

    def format_select_set(self):
        return super().format_select_set()

    def format_group_by(self):
        if not self.group_by:
            # There is no aggregation, nothing to see here
            return None

        lines = []
        if self.group_by:
            columns = [
                '{}'.format(expr.get_name())
                for expr in self.group_by
            ]
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
        if not self.limit:
            return None

        buf = StringIO()

        n, offset = self.limit['n'], self.limit['offset']
        buf.write('LIMIT {}'.format(n))
        if offset is not None and offset != 0:
            buf.write(', {}'.format(offset))

        return buf.getvalue()


class MapDTableSetFormatter(compiles.TableSetFormatter):
    """

    """
    _join_names = {
        ops.InnerJoin: 'JOIN',
        ops.LeftJoin: 'LEFT JOIN'
    }

    def get_result(self):
        # Got to unravel the join stack; the nesting order could be
        # arbitrary, so we do a depth first search and push the join tokens
        # and predicates onto a flat list, then format them
        op = self.expr.op()

        if isinstance(op, ops.Join):
            self._walk_join_tree(op)
        else:
            self.join_tables.append(self._format_table(self.expr))

        buf = StringIO()
        buf.write(self.join_tables[0])
        for jtype, table, preds in zip(self.join_types, self.join_tables[1:],
                                       self.join_predicates):
            buf.write('\n')
            buf.write(util.indent('{0} {1}'.format(jtype, table), self.indent))

            if len(preds):
                buf.write('\n')
                fmt_preds = map(self._format_predicate, preds)
                fmt_preds = util.indent('USING ' + ', '.join(fmt_preds),
                                        self.indent * 2)
                buf.write(fmt_preds)

        return buf.getvalue()

    def _validate_join_predicates(self, predicates):
        for pred in predicates:
            op = pred.op()
            if not isinstance(op, ops.Equals):
                raise com.TranslationError(
                    'Non-equality join predicates are not supported'
                )

            left_on, right_on = op.args
            if left_on.get_name() != right_on.get_name():
                raise com.TranslationError(
                    'Joining on different column names is not supported'
                )

    def _format_predicate(self, predicate):
        column = predicate.op().args[0]
        return column.get_name()

    def _quote_identifier(self, name):
        return name


class MapDExprTranslator(compiles.ExprTranslator):
    """

    """
    _registry = mapd_ops._operation_registry
    _rewrites = impala_compiler.ImpalaExprTranslator._rewrites.copy()

    context_class = MapDQueryContext

    def name(self, translated, name, force=True):
        return mapd_ops._name_expr(translated, name)


class MapDDialect(compiles.Dialect):
    """

    """
    translator = MapDExprTranslator


dialect = MapDDialect
compiles = MapDExprTranslator.compiles
rewrites = MapDExprTranslator.rewrites

mapd_reg = mapd_ops._operation_registry


@rewrites(ops.All)
def mapd_rewrite_all(expr):
    return mapd_ops._all(expr)


@rewrites(ops.Any)
def mapd_rewrite_any(expr):
    return mapd_ops._any(expr)


@rewrites(ops.NotAll)
def mapd_rewrite_not_all(expr):
    return mapd_ops._not_all(expr)


@rewrites(ops.NotAny)
def mapd_rewrite_not_any(expr):
    return mapd_ops._not_any(expr)


_add_methods(
    ir.NumericValue, dict(
        conv_4326_900913_x=_unary_op(
            'conv_4326_900913_x', mapd_ops.Conv_4326_900913_X
        ),
        conv_4326_900913_y=_unary_op(
            'conv_4326_900913_y', mapd_ops.Conv_4326_900913_Y
        ),
        truncate=_binop_expr(
            'truncate', mapd_ops.NumericTruncate
        )
    )
)

_add_methods(
    ir.StringValue, dict(
        byte_length=_unary_op('length', mapd_ops.ByteLength)
    )
)
