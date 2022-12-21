import functools
import inspect


class bound_table_method:
    """A bound method on a table."""

    __slots__ = ("_func", "_obj")

    def __init__(self, func, obj):
        self._func = func
        self._obj = obj

    @property
    def __doc__(self):
        return self._func.__doc__

    @property
    def __name__(self):
        return self._func.__name__

    @property
    def __signature__(self):
        sig = inspect.signature(self._func)
        return sig.replace(parameters=list(sig.parameters.values())[1:])

    def __call__(self, *args, **kwargs):
        return self._func(self._obj, *args, **kwargs)

    def _errmsg(self, prefix="Got method `Table.{name}`.", **kwargs):
        return (
            prefix
            + (
                " If you meant to access a column named {name!r} use "
                "`table[{name!r}]` syntax instead."
            )
        ).format(name=self._func.__name__, **kwargs)

    def __getattr__(self, key):
        try:
            return getattr(self._func, key)
        except AttributeError:
            pass
        raise AttributeError(
            self._errmsg("Method `Table.{name}` has no attribute {key!r}.", key=key)
        )

    def _compare(self, other, sym):
        raise TypeError(
            self._errmsg("{sym!r} not supported for method `Table.{name}`.", sym=sym)
        )

    __eq__ = functools.partialmethod(_compare, sym="=")
    __ne__ = functools.partialmethod(_compare, sym="!=")
    __lt__ = functools.partialmethod(_compare, sym="<")
    __le__ = functools.partialmethod(_compare, sym="<=")
    __gt__ = functools.partialmethod(_compare, sym=">")
    __ge__ = functools.partialmethod(_compare, sym=">=")


class table_method:
    """A method on an `ibis.expr.types.Table`.

    This decorator adds extra features for erroring nicely if a Table
    method is used somewhere a column is expected.
    """

    def __init__(self, func):
        self._func = func

    def __getattr__(self, key):
        return getattr(self._func, key)

    def __get__(self, obj, objtype=None):
        return bound_table_method(self._func, obj)
