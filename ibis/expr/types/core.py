from __future__ import annotations

import os
import webbrowser
from typing import TYPE_CHECKING, Any, Hashable, Mapping, MutableMapping

from public import public

import ibis
import ibis.common.exceptions as com
import ibis.config as config
import ibis.util as util
from ibis.config import options
from ibis.expr.typing import TimeContext

if TYPE_CHECKING:
    from ...backends.base import BaseBackend
    from .. import operations as ops
    from .. import types as ir
    from ..format import FormatMemo
    from .generic import ValueExpr


@public
class Expr(util.EqMixin):
    """Base expression class"""

    def _type_display(self) -> str:
        return type(self).__name__

    def __init__(self, arg: ops.Node) -> None:
        # TODO: all inputs must inherit from a common table API
        self._arg = arg

    def __repr__(self) -> str:
        from ibis.expr.format import FormatMemo

        if not config.options.interactive:
            return self._repr(memo=FormatMemo(get_text_repr=True))

        try:
            result = self.execute()
        except com.TranslationError as e:
            output = (
                'Translation to backend failed\n'
                'Error message: {}\n'
                'Expression repr follows:\n{}'.format(e.args[0], self._repr())
            )
            return output
        else:
            return repr(result)

    def __hash__(self) -> int:
        return hash(self._key)

    def __bool__(self) -> bool:
        raise ValueError(
            "The truth value of an Ibis expression is not defined"
        )

    __nonzero__ = __bool__

    def _repr(self, memo: FormatMemo | None = None) -> str:
        from ibis.expr.format import ExprFormatter

        return ExprFormatter(self, memo=memo).get_result()

    @property
    def _safe_name(self) -> str | None:
        """Get the name of an expression `expr` if one exists

        Returns
        -------
        str | None
            `str` if the Expr has a name, otherwise `None`
        """
        try:
            return self.get_name()
        except (com.ExpressionError, AttributeError):
            return None

    @property
    def _key(self) -> tuple[Hashable, ...]:
        """Key suitable for hashing an expression.

        Returns
        -------
        tuple[Hashable, ...]
            A tuple of hashable objects uniquely identifying this expression.
        """
        return type(self), self._safe_name, self.op()

    def _repr_png_(self) -> bytes | None:
        if config.options.interactive or not ibis.options.graphviz_repr:
            return None
        try:
            import ibis.expr.visualize as viz
        except ImportError:
            return None
        else:
            try:
                return viz.to_graph(self).pipe(format='png')
            except Exception:
                # Something may go wrong, and we can't error in the notebook
                # so fallback to the default text representation.
                return None

    def visualize(self, format: str = 'svg') -> None:
        """Visualize an expression in the browser as an SVG image.

        Parameters
        ----------
        format
            Image output format. These are specified by the ``graphviz`` Python
            library.

        Notes
        -----
        This method opens a web browser tab showing the image of the expression
        graph created by the code in [ibis.expr.visualize][].

        Raises
        ------
        ImportError
            If ``graphviz`` is not installed.
        """
        import ibis.expr.visualize as viz

        path = viz.draw(viz.to_graph(self), format=format)
        webbrowser.open(f'file://{os.path.abspath(path)}')

    def pipe(self, f, *args: Any, **kwargs: Any) -> Expr:
        """Compose `f` with `self`.

        Parameters
        ----------
        f
            If the expression needs to be passed as anything other than the
            first argument to the function, pass a tuple with the argument
            name. For example, (f, 'data') if the function f expects a 'data'
            keyword
        args
            Positional arguments to `f`
        kwargs
            Keyword arguments to `f`

        Examples
        --------
        >>> import ibis
        >>> t = ibis.table([('a', 'int64'), ('b', 'string')], name='t')
        >>> f = lambda a: (a + 1).name('a')
        >>> g = lambda a: (a * 2).name('a')
        >>> result1 = t.a.pipe(f).pipe(g)
        >>> result1  # doctest: +NORMALIZE_WHITESPACE
        ref_0
        UnboundTable[table]
          name: t
          schema:
            a : int64
            b : string
        a = Multiply[int64*]
          left:
            a = Add[int64*]
              left:
                a = Column[int64*] 'a' from table
                  ref_0
              right:
                Literal[int8]
                  1
          right:
            Literal[int8]
              2
        >>> result2 = g(f(t.a))  # equivalent to the above
        >>> result1.equals(result2)
        True

        Returns
        -------
        Expr
            Result type of passed function
        """
        if isinstance(f, tuple):
            f, data_keyword = f
            kwargs = kwargs.copy()
            kwargs[data_keyword] = self
            return f(*args, **kwargs)
        else:
            return f(self, *args, **kwargs)

    def op(self) -> ops.Node:
        return self._arg

    @property
    def _factory(self) -> type[Expr]:
        return type(self)

    def _find_backends(self) -> list[BaseBackend]:
        """Return the possible backends for an expression.

        Returns
        -------
        list[BaseBackend]
            A list of the backends found.
        """
        from ibis.backends.base import BaseBackend

        seen_backends: dict[
            str, BaseBackend
        ] = {}  # key is backend.db_identity

        stack = [self.op()]
        seen = set()

        while stack:
            node = stack.pop()

            if node not in seen:
                seen.add(node)

                for arg in node.flat_args():
                    if isinstance(arg, BaseBackend):
                        if arg.db_identity not in seen_backends:
                            seen_backends[arg.db_identity] = arg
                    elif isinstance(arg, Expr):
                        stack.append(arg.op())

        return list(seen_backends.values())

    def _find_backend(self) -> BaseBackend:
        backends = self._find_backends()

        if not backends:
            default = options.default_backend
            if default is None:
                raise com.IbisError(
                    'Expression depends on no backends, and found no default'
                )
            return default

        if len(backends) > 1:
            raise ValueError('Multiple backends found')

        return backends[0]

    def execute(
        self,
        limit: int | str | None = 'default',
        timecontext: TimeContext | None = None,
        params: Mapping[ValueExpr, Any] | None = None,
        **kwargs: Any,
    ):
        """Execute an expression against its backend if one exists.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        timecontext
            Defines a time range of `(begin, end)`. When defined, the execution
            will only compute result for data inside the time range. The time
            range is inclusive of both endpoints. This is conceptually same as
            a time filter.
            The time column must be named `'time'` and should preserve
            across the expression. For example, if that column is dropped then
            execute will result in an error.
        params
            Mapping of scalar parameter expressions to value
        """
        return self._find_backend().execute(
            self, limit=limit, timecontext=timecontext, params=params, **kwargs
        )

    def compile(
        self,
        limit: int | None = None,
        timecontext: TimeContext | None = None,
        params: Mapping[ValueExpr, Any] | None = None,
    ):
        """Compile to an execution target.

        Parameters
        ----------
        limit
            An integer to effect a specific row limit. A value of `None` means
            "no limit". The default is in `ibis/config.py`.
        timecontext
            Defines a time range of `(begin, end)`. When defined, the execution
            will only compute result for data inside the time range. The time
            range is inclusive of both endpoints. This is conceptually same as
            a time filter.
            The time column must be named `'time'` and should preserve
            across the expression. For example, if that column is dropped then
            execute will result in an error.
        params
            Mapping of scalar parameter expressions to value
        """
        return self._find_backend().compile(
            self, limit=limit, timecontext=timecontext, params=params
        )

    @util.deprecated(
        version='2.0',
        instead=(
            "[`Expr.compile`][ibis.expr.types.core.Expr.compile] and "
            "catch TranslationError"
        ),
    )
    def verify(self):
        """Return True if expression can be compiled to its attached client."""
        try:
            self.compile()
        except Exception:
            return False
        else:
            return True

    def _type_check(self, other: Any) -> None:
        if not isinstance(other, Expr):
            raise TypeError(
                f"Cannot compare non-Expr object {type(other)} with Expr"
            )

    def __component_eq__(
        self,
        other: ir.Expr,
        cache: MutableMapping[Hashable, bool] | None = None,
    ) -> bool:
        return self._arg.equals(other._arg, cache=cache)


class UnnamedMarker:
    pass


unnamed = UnnamedMarker()


def _binop(
    op_class: type[ops.BinaryOp],
    left: ir.ValueExpr,
    right: ir.ValueExpr,
) -> ir.ValueExpr | NotImplemented:
    """Try to construct a binary operation.

    Parameters
    ----------
    op_class
        The [`BinaryOp`][ibis.expr.operations.BinaryOp] subclass for the
        operation
    left
        Left operand
    right
        Right operand

    Returns
    -------
    ValueExpr
        A value expression

    Examples
    --------
    >>> import ibis.expr.operations as ops
    >>> expr = _binop(ops.TimeAdd, ibis.time("01:00"), ibis.interval(hours=1))
    >>> expr
    time = TimeAdd
      left:
        value: time = datetime.time(1, 0)
      right:
        value: interval<int8>(unit='h') = 1
    >>> _binop(ops.TimeAdd, 1, ibis.interval(hours=1))
    NotImplemented
    """
    try:
        node = op_class(left, right)
    except (com.IbisTypeError, NotImplementedError):
        return NotImplemented
    else:
        return node.to_expr()
