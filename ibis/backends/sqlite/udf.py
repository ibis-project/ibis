from __future__ import annotations

import abc
import functools
import inspect
import math
import operator
from collections import defaultdict
from typing import Any, Callable, NamedTuple
from urllib.parse import parse_qs, urlsplit

try:
    import regex as re
except ImportError:
    import re


class _UDF(NamedTuple):
    """An internal record holding info about a registered UDF."""

    name: str
    impl: Any
    nargs: int
    skip_if_exists: bool = False
    deterministic: bool = True


_SQLITE_UDF_REGISTRY = {}
_SQLITE_UDAF_REGISTRY = {}


def ignore_nulls(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if any(arg is None for arg in args):
            return None
        return f(*args, **kwargs)

    return wrapper


def _number_of_arguments(callable):
    signature = inspect.signature(callable)
    parameters = signature.parameters.values()
    kinds = [param.kind for param in parameters]
    valid_kinds = (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_ONLY,
    )
    if any(kind not in valid_kinds for kind in kinds) or any(
        param.default is not inspect.Parameter.empty for param in parameters
    ):
        raise TypeError(
            "Only positional arguments without defaults are supported in Ibis "
            "SQLite function registration"
        )
    return len(parameters)


def udf(func=None, *, skip_if_exists=False, deterministic=True):
    """Create a SQLite scalar UDF from `func`.

    Parameters
    ----------
    func
        A callable object
    skip_if_exists
        If true, the UDF will only be registered if an existing function
        with that name doesn't already exist.
    deterministic
        Whether the UDF is deterministic, defaults to True.

    Returns
    -------
    callable
        A callable object that returns ``None`` if any of its inputs are
        ``None``.

    """
    if func is None:
        return lambda func: udf(
            func, skip_if_exists=skip_if_exists, deterministic=deterministic
        )

    name = func.__name__
    nargs = _number_of_arguments(func)
    wrapper = ignore_nulls(func)

    _SQLITE_UDF_REGISTRY[name] = _UDF(
        name, wrapper, nargs, skip_if_exists, deterministic
    )
    return wrapper


def udaf(cls):
    """Register a UDAF class with any SQLite connection."""
    name = cls.__name__
    nargs = _number_of_arguments(cls.step) - 1
    _SQLITE_UDAF_REGISTRY[name] = _UDF(name, cls, nargs)
    return cls


# Optional builtin functions
#
# These functions may exist as builtins depending on the SQLite versions.
# They're only registered if they don't exist already in the connection.


@udf(skip_if_exists=True)
def unhex(string):
    return bytes.fromhex(string)


@udf(skip_if_exists=True)
def exp(arg):
    return math.exp(arg)


@udf(skip_if_exists=True)
def ln(arg):
    if arg < 0:
        return None
    return math.log(arg)


@udf(skip_if_exists=True)
def log2(arg):
    if arg < 0:
        return None
    return math.log(arg, 2)


@udf(skip_if_exists=True)
def log10(arg):
    if arg < 0:
        return None
    return math.log(arg, 10)


@udf(skip_if_exists=True)
def floor(arg):
    return math.floor(arg)


@udf(skip_if_exists=True)
def ceil(arg):
    return math.ceil(arg)


@udf(skip_if_exists=True)
def sign(arg):
    if not arg:
        return 0
    return math.copysign(1, arg)


@udf(skip_if_exists=True)
def mod(left, right):
    return None if right == 0 else (left % right)


@udf(skip_if_exists=True)
def power(arg, power):
    # mirroring sqlite - return NULL if negative or non-integral
    if arg < 0.0 and not power.is_integer():
        return None
    return arg**power


@udf(skip_if_exists=True)
def sqrt(arg):
    return None if arg < 0.0 else math.sqrt(arg)


@udf(skip_if_exists=True)
def sin(arg):
    return math.sin(arg)


@udf(skip_if_exists=True)
def cos(arg):
    return math.cos(arg)


@udf(skip_if_exists=True)
def tan(arg):
    return math.tan(arg)


@udf(skip_if_exists=True)
def asin(arg):
    return math.asin(arg)


@udf(skip_if_exists=True)
def acos(arg):
    return math.acos(arg)


@udf(skip_if_exists=True)
def atan(arg):
    return math.atan(arg)


@udf(skip_if_exists=True)
def atan2(y, x):
    return math.atan2(y, x)


@udf(skip_if_exists=True)
def degrees(x):
    return math.degrees(x)


@udf(skip_if_exists=True)
def radians(x):
    return math.radians(x)


@udf(skip_if_exists=True)
def pi():
    return math.pi


# Additional UDFS


@udf
def _ibis_reverse(string):
    return string[::-1]


@udf
def _ibis_string_ascii(string):
    return ord(string[0])


@udf
def _ibis_rpad(string, width, pad):
    return string.ljust(width, pad)[:width]


@udf
def _ibis_lpad(string, width, pad):
    return string.rjust(width, pad)[:width]


@udf
def _ibis_repeat(string, n):
    return string * n


@udf
def _ibis_translate(string, from_string, to_string):
    table = str.maketrans(from_string, to_string)
    return string.translate(table)


@udf
def _ibis_regex_search(string, regex):
    """Return whether `regex` exists in `string`."""
    return re.search(regex, string) is not None


@udf
def _ibis_regex_replace(string, pattern, replacement):
    """Replace occurrences of `pattern` in `string` with `replacement`."""
    return re.sub(pattern, replacement, string)


@udf
def _ibis_regex_extract(string, pattern, index):
    """Extract match of regular expression `pattern` from `string` at `index`."""
    result = re.search(pattern, string)
    if result is not None and 0 <= index <= (result.lastindex or -1):
        return result.group(index)
    return None


@udf
def _ibis_xor(x, y):
    return x ^ y


@udf
def _ibis_inv(x):
    return ~x


@udf
def _ibis_extract_fragment(url):
    return _extract_url_field(url, "fragment")


@udf
def _ibis_extract_protocol(url):
    return _extract_url_field(url, "scheme")


@udf
def _ibis_extract_authority(url):
    return _extract_url_field(url, "netloc")


@udf
def _ibis_extract_path(url):
    return _extract_url_field(url, "path")


@udf
def _ibis_extract_host(url):
    return _extract_url_field(url, "hostname")


def _extract_url_field(data, field_name):
    return getattr(urlsplit(data), field_name, "")


@udf
def _ibis_extract_full_query(url):
    return urlsplit(url).query


@udf
def _ibis_extract_query(url, param_name):
    query = urlsplit(url).query
    value = parse_qs(query)[param_name]
    return value if len(value) > 1 else value[0]


@udf
def _ibis_extract_user_info(url):
    url_parts = urlsplit(url)
    username = url_parts.username or ""
    password = url_parts.password or ""

    return f"{username}:{password}"


class _ibis_var:
    def __init__(self, offset):
        self.mean = 0.0
        self.sum_of_squares_of_differences = 0.0
        self.count = 0
        self.offset = offset

    def step(self, value):
        if value is not None:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            self.sum_of_squares_of_differences += delta * (value - self.mean)

    def finalize(self):
        count = self.count
        if count:
            return self.sum_of_squares_of_differences / (count - self.offset)
        return None


@udaf
class _ibis_mode:
    def __init__(self):
        self.counter = defaultdict(int)

    def step(self, value):
        if value is not None:
            self.counter[value] += 1

    def finalize(self):
        if self.counter:
            return max(self.counter, key=self.counter.get)
        return None


@udaf
class _ibis_var_pop(_ibis_var):
    def __init__(self):
        super().__init__(0)


@udaf
class _ibis_var_sample(_ibis_var):
    def __init__(self):
        super().__init__(1)


class _ibis_bit_agg:
    def __init__(self, op):
        self.value: int | None = None
        self.count: int = 0
        self.op: Callable[[int, int], int] = op

    def step(self, value):
        if value is not None:
            if not self.count:
                self.value = value
            else:
                self.value = self.op(self.value, value)
            self.count += 1

    def finalize(self) -> int | None:
        return self.value


@udaf
class _ibis_bit_or(_ibis_bit_agg):
    def __init__(self):
        super().__init__(operator.or_)


@udaf
class _ibis_bit_and(_ibis_bit_agg):
    def __init__(self):
        super().__init__(operator.and_)


@udaf
class _ibis_bit_xor(_ibis_bit_agg):
    def __init__(self):
        super().__init__(operator.xor)


class _ibis_arbitrary(abc.ABC):
    def __init__(self) -> None:
        self.value = None

    @abc.abstractmethod
    def step(self, value):
        ...

    def finalize(self) -> int | None:
        return self.value


@udaf
class _ibis_arbitrary_first(_ibis_arbitrary):
    def step(self, value):
        if self.value is None:
            self.value = value


@udaf
class _ibis_arbitrary_last(_ibis_arbitrary):
    def step(self, value):
        if value is not None:
            self.value = value


def register_all(con):
    """Register all udf and udaf with the connection.

    All udf and udaf are defined in this file with the `udf` and `udaf`
    decorators.
    """
    existing = {
        name for (name,) in con.execute("SELECT name FROM pragma_function_list()")
    }

    for udf in _SQLITE_UDF_REGISTRY.values():
        if udf.skip_if_exists and udf.name in existing:
            continue
        con.create_function(
            udf.name, udf.nargs, udf.impl, deterministic=udf.deterministic
        )

    for udf in _SQLITE_UDAF_REGISTRY.values():
        con.create_aggregate(udf.name, udf.nargs, udf.impl)
