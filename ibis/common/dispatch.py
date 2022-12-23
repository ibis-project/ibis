import abc
import functools
import re
from collections import defaultdict
from typing import Any

from ibis.util import import_object


def normalize(r: str | re.Pattern):
    """Normalize a expression by wrapping it with `'^'` and `'$'`.

    Parameters
    ----------
    r
        The pattern to normalize.

    Returns
    -------
    Pattern
        The compiled regex.
    """
    r = getattr(r, "pattern", r)
    return re.compile('^' + r.lstrip('^').rstrip('$') + '$')


class RegexDispatcher:
    r"""Regular Expression Dispatcher.

    >>> f = RegexDispatcher('f')

    >>> f.register('\d*')
    ... def parse_int(s):
    ...     return int(s)

    >>> f.register('\d*\.\d*')
    ... def parse_float(s):
    ...     return float(s)

    Set priorities to break ties between multiple matches.
    Default priority is set to 10

    >>> f.register('\w*', priority=9)
    ... def parse_str(s):
    ...     return s

    >>> type(f('123'))
    int

    >>> type(f('123.456'))
    float
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.funcs = {}
        self.priorities = {}

    def add(self, regex: str, func: Any, priority: int = 10) -> None:
        self.funcs[normalize(regex)] = func
        self.priorities[func] = priority

    def register(self, regex: str, priority: int = 10) -> Any:
        """Register a new handler in this regex dispatcher.

        Parameters
        ----------
        regex
            The pattern to match against.
        priority
            The priority for this pattern. This is used to resolve ambigious
            matches. The highest priority match wins.

        Returns
        -------
        decorator : callable
            A decorator that registers the function with this RegexDispatcher
            but otherwise returns the function unchanged.
        """

        def _(func):
            self.add(regex, func, priority)
            return func

        return _

    def dispatch(self, s: str) -> Any:
        funcs = (
            (func, match)
            for r, func in self.funcs.items()
            if (match := r.match(s)) is not None
        )
        priorities = self.priorities
        value = max(
            funcs,
            key=lambda pair: priorities.get(pair[0]),
            default=None,
        )
        if value is None:
            raise NotImplementedError(
                f"no pattern for `{self.name}` matches input string: {s!r}"
            )
        return value

    def __call__(self, s: str, *args: Any, **kwargs: Any) -> Any:
        func, match = self.dispatch(s)
        return func(s, *args, **kwargs, **match.groupdict())

    @property
    def __doc__(self) -> Any:
        # take the min to give the docstring of the last fallback function
        return min(self.priorities.items(), key=lambda x: x[1])[0].__doc__


def lazy_singledispatch(func):
    """A `singledispatch` implementation that supports lazily registering implementations."""

    lookup = {object: func}
    abc_lookup = {}
    lazy_lookup = defaultdict(dict)

    def register(cls, func=None):
        """Registers a new implementation for arguments of type `cls`."""

        def inner(func):
            if isinstance(cls, tuple):
                for t in cls:
                    register(t, func)
            elif isinstance(cls, abc.ABCMeta):
                abc_lookup[cls] = func
            elif isinstance(cls, str):
                module = cls.split(".", 1)[0]
                lazy_lookup[module][cls] = func
            else:
                lookup[cls] = func
            return func

        return inner if func is None else inner(func)

    def dispatch(cls):
        """Return the implementation for the given `cls`."""
        for cls2 in cls.__mro__:
            # 1. Check for a concrete implementation
            try:
                impl = lookup[cls2]
            except KeyError:
                pass
            else:
                if cls is not cls2:
                    # Cache implementation
                    lookup[cls] = impl
                return impl
            # 2. Check lazy implementations
            module = cls2.__module__.split(".", 1)[0]
            if lazy := lazy_lookup.get(module):
                # Import all lazy implementations first before registering
                # (which should never fail), to ensure an error anywhere
                # doesn't result in a half-registered state.
                new = {import_object(name): func for name, func in lazy.items()}
                lookup.update(new)
                # drop lazy implementations, idempotent for thread safety
                lazy_lookup.pop(module, None)
                return dispatch(cls)
            # 3. Check for abcs
            for abc_cls, impl in abc_lookup.items():
                if issubclass(cls, abc_cls):
                    lookup[cls] = impl
                    return impl
        # Can never get here, since a base `object` implementation is
        # always registered
        assert False, "should never get here"  # pragma: no cover

    @functools.wraps(func)
    def call(arg, *args, **kwargs):
        return dispatch(type(arg))(arg, *args, **kwargs)

    call.dispatch = dispatch
    call.register = register

    return call
