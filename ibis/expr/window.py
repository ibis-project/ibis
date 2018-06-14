import ibis.expr.types as ir
import ibis.expr.operations as ops
import ibis.util as util
import ibis.common as com


def _list_to_tuple(x):
    if isinstance(x, list):
        x = tuple(x)
    return x


class Window(object):

    """
    A generic window function clause, patterned after SQL window clauses for
    the time being. Can be expanded to cover more use cases as they arise.

    Using None for preceding or following currently indicates unbounded. Use 0
    for current_value
    """

    def __init__(self, group_by=None, order_by=None,
                 preceding=None, following=None, how='rows'):
        if group_by is None:
            group_by = []

        if order_by is None:
            order_by = []

        self._group_by = util.promote_list(group_by)

        self._order_by = []
        for x in util.promote_list(order_by):
            if isinstance(x, ir.SortExpr):
                pass
            elif isinstance(x, ir.Expr):
                x = ops.SortKey(x).to_expr()
            self._order_by.append(x)

        self.preceding = _list_to_tuple(preceding)
        self.following = _list_to_tuple(following)
        self.how = how

        self._validate_frame()

    def _validate_frame(self):
        preceding_tuple = has_preceding = False
        following_tuple = has_following = False
        if self.preceding is not None:
            preceding_tuple = isinstance(self.preceding, tuple)
            has_preceding = True

        if self.following is not None:
            following_tuple = isinstance(self.following, tuple)
            has_following = True

        if ((preceding_tuple and has_following) or
                (following_tuple and has_preceding)):
            raise com.IbisInputError(
                'Can only specify one window side when you want an '
                'off-center window'
            )
        elif preceding_tuple:
            start, end = self.preceding
            if start is None:
                assert end >= 0
            else:
                assert start > end
        elif following_tuple:
            start, end = self.following
            if end is None:
                assert start >= 0
            else:
                assert start < end
        else:
            if not isinstance(self.preceding, ir.Expr):
                if has_preceding and self.preceding < 0:
                    raise com.IbisInputError(
                        "'preceding' must be positive, got {}".format(
                            self.preceding
                        )
                    )

            if not isinstance(self.following, ir.Expr):
                if has_following and self.following < 0:
                    raise com.IbisInputError(
                        "'following' must be positive, got {}".format(
                            self.following
                        )
                    )
        if self.how not in {'rows', 'range'}:
            raise com.IbisInputError(
                "'how' must be 'rows' or 'range', got {}"
                .format(
                    self.how
                )
            )

    def bind(self, table):
        # Internal API, ensure that any unresolved expr references (as strings,
        # say) are bound to the table being windowed
        groups = table._resolve(self._group_by)
        sorts = [ops.to_sort_key(table, k) for k in self._order_by]
        return self._replace(group_by=groups, order_by=sorts)

    def combine(self, window):
        if self.how != window.how:
            raise com.IbisInputError(
                "Window types must match. Expecting '{}' Window, got '{}'"
                .format(
                    self.how.upper(), window.how.upper()
                )
            )
        kwds = dict(
            preceding=self.preceding or window.preceding,
            following=self.following or window.following,
            group_by=self._group_by + window._group_by,
            order_by=self._order_by + window._order_by
        )
        return Window(**kwds)

    def group_by(self, expr):
        new_groups = self._group_by + util.promote_list(expr)
        return self._replace(group_by=new_groups)

    def _replace(self, **kwds):
        new_kwds = dict(
            group_by=kwds.get('group_by', self._group_by),
            order_by=kwds.get('order_by', self._order_by),
            preceding=kwds.get('preceding', self.preceding),
            following=kwds.get('following', self.following),
            how=kwds.get('how', self.how)
        )
        return Window(**new_kwds)

    def order_by(self, expr):
        new_sorts = self._order_by + util.promote_list(expr)
        return self._replace(order_by=new_sorts)

    def equals(self, other, cache=None):
        if cache is None:
            cache = {}

        if self is other:
            cache[self, other] = True
            return True

        if not isinstance(other, Window):
            cache[self, other] = False
            return False

        try:
            return cache[self, other]
        except KeyError:
            pass

        if (len(self._group_by) != len(other._group_by) or
                not ops.all_equal(self._group_by, other._group_by,
                                  cache=cache)):
            cache[self, other] = False
            return False

        if (len(self._order_by) != len(other._order_by) or
                not ops.all_equal(self._order_by, other._order_by,
                                  cache=cache)):
            cache[self, other] = False
            return False

        equal = (
            ops.all_equal(self.preceding, other.preceding, cache=cache) and
            ops.all_equal(self.following, other.following, cache=cache)
        )
        cache[self, other] = equal
        return equal


def window(preceding=None, following=None, group_by=None, order_by=None):
    """
    Create a window clause for use with window (analytic and aggregate)
    functions. This ROW window clause aggregates adjacent rows based
    on differences in row number.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    preceding : int, tuple, or None, default None
      Specify None for unbounded, 0 to include current row
      tuple for off-center window
    following : int, tuple, or None, default None
      Specify None for unbounded, 0 to include current row
      tuple for off-center window
    group_by : expressions, default None
      Either specify here or with TableExpr.group_by
    order_by : expressions, default None
      For analytic functions requiring an ordering, specify here, or let Ibis
      determine the default ordering (for functions like rank)

    Returns
    -------
    win : ibis Window
    """
    return Window(preceding=preceding, following=following,
                  group_by=group_by, order_by=order_by, how='rows')


def range_window(preceding=None, following=None, group_by=None, order_by=None):
    """
    Create a window clause for use with window (analytic and aggregate)
    functions. This RANGE window clause aggregates rows based upon differences
    in the value of the order-by expression.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    preceding : int, tuple, or None, default None
      Specify None for unbounded, 0 to include current row
      tuple for off-center window
    following : int, tuple, or None, default None
      Specify None for unbounded, 0 to include current row
      tuple for off-center window
    group_by : expressions, default None
      Either specify here or with TableExpr.group_by
    order_by : expressions, default None
      For analytic functions requiring an ordering, specify here, or let Ibis
      determine the default ordering (for functions like rank)

    Returns
    -------
    win : ibis Window
    """
    return Window(preceding=preceding, following=following,
                  group_by=group_by, order_by=order_by, how='range')


def cumulative_window(group_by=None, order_by=None):
    """
    Create a cumulative window clause for use with aggregate window functions.

    All window frames / ranges are inclusive.

    Parameters
    ----------
    group_by : expressions, default None
      Either specify here or with TableExpr.group_by
    order_by : expressions, default None
      For analytic functions requiring an ordering, specify here, or let Ibis
      determine the default ordering (for functions like rank)

    Returns
    -------
    win : ibis Window
    """
    return Window(preceding=None, following=0,
                  group_by=group_by, order_by=order_by)


def trailing_window(rows, group_by=None, order_by=None):
    """
    Create a trailing window for use with aggregate window functions.

    Parameters
    ----------
    rows : int
      Number of trailing rows to include. 0 includes only the current row
    group_by : expressions, default None
      Either specify here or with TableExpr.group_by
    order_by : expressions, default None
      For analytic functions requiring an ordering, specify here, or let Ibis
      determine the default ordering (for functions like rank)

    Returns
    -------
    win : ibis Window
    """
    return Window(preceding=rows, following=0,
                  group_by=group_by, order_by=order_by)


def trailing_range_window(preceding, order_by, group_by=None):
    """
    Create a trailing time window for use with aggregate window functions.

    Parameters
    ----------
    preceding : float or expression of intervals, i.e.
      1 * ibis.day() + 5 * ibis.hour()
    order_by : expressions, default None
      For analytic functions requiring an ordering, specify here, or let Ibis
      determine the default ordering (for functions like rank)
    group_by : expressions, default None
      Either specify here or with TableExpr.group_by

    Returns
    -------
    win: ibis Window
    """
    return Window(preceding=preceding, following=0,
                  group_by=group_by, order_by=order_by, how='range')


def propagate_down_window(expr, window):
    op = expr.op()

    clean_args = []
    unchanged = True
    for arg in op.args:
        if (isinstance(arg, ir.Expr) and
                not isinstance(op, ops.WindowOp)):
            new_arg = propagate_down_window(arg, window)
            if isinstance(new_arg.op(), ops.AnalyticOp):
                new_arg = ops.WindowOp(new_arg, window).to_expr()
            if arg is not new_arg:
                unchanged = False
            arg = new_arg

        clean_args.append(arg)

    if unchanged:
        return expr
    else:
        return type(op)(*clean_args).to_expr()
