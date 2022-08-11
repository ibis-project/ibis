import re
from typing import Any


def normalize(r):
    """Normalize a regular expression by ensuring that it is wrapped with:
    '^' and '$'

    Parameters
    ----------
    r : str or Pattern
        The pattern to normalize.

    Returns
    -------
    p : Pattern
        The compiled regex.
    """
    r = getattr(r, "pattern", r)
    return re.compile('^' + r.lstrip('^').rstrip('$') + '$')


class RegexDispatcher:
    """
    Regular Expression Dispatcher

    >>> f = RegexDispatcher('f')

    >>> f.register('\\d*')
    ... def parse_int(s):
    ...     return int(s)

    >>> f.register('\\d*\\.\\d*')
    ... def parse_float(s):
    ...     return float(s)

    Set priorities to break ties between multiple matches.
    Default priority is set to 10

    >>> f.register('\\w*', priority=9)
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
