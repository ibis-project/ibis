from __future__ import annotations

import functools
import inspect
import math
import operator
from typing import Callable

try:
    import regex as re
except ImportError:
    import re

_SQLITE_UDF_REGISTRY = set()
_SQLITE_UDAF_REGISTRY = set()


def udf(f):
    """Create a SQLite scalar UDF from `f`.

    Parameters
    ----------
    f
        A callable object

    Returns
    -------
    callable
        A callable object that returns ``None`` if any of its inputs are
        ``None``.
    """

    @functools.wraps(f)
    def wrapper(*args):
        if any(arg is None for arg in args):
            return None
        return f(*args)

    _SQLITE_UDF_REGISTRY.add(wrapper)
    return wrapper


def udaf(cls):
    """Register a UDAF class with any SQLite connection."""
    _SQLITE_UDAF_REGISTRY.add(cls)
    return cls


@udf
def _ibis_sqlite_reverse(string):
    return string[::-1]


@udf
def _ibis_sqlite_string_ascii(string):
    return ord(string[0])


@udf
def _ibis_sqlite_capitalize(string):
    return string.capitalize()


@udf
def _ibis_sqlite_translate(string, from_string, to_string):
    table = str.maketrans(from_string, to_string)
    return string.translate(table)


@udf
def _ibis_sqlite_regex_search(string, regex):
    """Return whether `regex` exists in `string`."""
    return re.search(regex, string) is not None


@udf
def _ibis_sqlite_regex_replace(string, pattern, replacement):
    """Replace occurences of `pattern` in `string` with `replacement`."""
    return re.sub(pattern, replacement, string)


@udf
def _ibis_sqlite_regex_extract(string, pattern, index):
    """Extract match of regular expression `pattern` from `string` at `index`."""
    result = re.search(pattern, string)
    if result is not None and 0 <= index <= (result.lastindex or -1):
        return result.group(index)
    return None


@udf
def _ibis_sqlite_exp(arg):
    """Exponentiate `arg`.

    Parameters
    ----------
    arg : number
        Number to raise to `e`.

    Returns
    -------
    result : Optional[number]
        None If the input is None
    """
    return math.exp(arg)


@udf
def _ibis_sqlite_log(arg, base):
    if arg < 0 or base < 0:
        return None
    return math.log(arg, base)


@udf
def _ibis_sqlite_ln(arg):
    if arg < 0:
        return None
    return math.log(arg)


@udf
def _ibis_sqlite_log2(arg):
    return _ibis_sqlite_log(arg, 2)


@udf
def _ibis_sqlite_log10(arg):
    return _ibis_sqlite_log(arg, 10)


@udf
def _ibis_sqlite_floor(arg):
    return math.floor(arg)


@udf
def _ibis_sqlite_ceil(arg):
    return math.ceil(arg)


@udf
def _ibis_sqlite_sign(arg):
    if not arg:
        return 0
    return math.copysign(1, arg)


@udf
def _ibis_sqlite_floordiv(left, right):
    return left // right


@udf
def _ibis_sqlite_mod(left, right):
    return left % right


@udf
def _ibis_sqlite_power(arg, power):
    """Raise `arg` to the `power` power.

    Parameters
    ----------
    arg : number
        Number to raise to `power`.
    power : number
        Number to raise `arg` to.

    Returns
    -------
    result : Optional[number]
        None If either argument is None or we're trying to take a fractional
        power or a negative number
    """
    if arg < 0.0 and not power.is_integer():
        return None
    return arg**power


@udf
def _ibis_sqlite_sqrt(arg):
    """Square root of `arg`.

    Parameters
    ----------
    arg : Optional[number]
        Number to take the square root of

    Returns
    -------
    result : Optional[number]
        None if `arg` is None or less than 0 otherwise the square root
    """
    return None if arg is None or arg < 0.0 else math.sqrt(arg)


def _trig_func_unary(func, arg):
    if arg is None:
        return None

    return func(float(arg))


def _trig_func_binary(func, arg1, arg2):
    if arg1 is None or arg2 is None:
        return None

    return func(float(arg1), float(arg2))


@udf
def _ibis_sqlite_cot(arg):
    return _trig_func_unary(
        lambda arg: float("inf") if not arg else 1.0 / math.tan(arg), arg
    )


@udf
def _ibis_sqlite_sin(arg):
    return _trig_func_unary(math.sin, arg)


@udf
def _ibis_sqlite_cos(arg):
    return _trig_func_unary(math.cos, arg)


@udf
def _ibis_sqlite_tan(arg):
    return _trig_func_unary(math.tan, arg)


@udf
def _ibis_sqlite_asin(arg):
    return _trig_func_unary(math.asin, arg)


@udf
def _ibis_sqlite_acos(arg):
    return _trig_func_unary(math.acos, arg)


@udf
def _ibis_sqlite_atan(arg):
    return _trig_func_unary(math.atan, arg)


@udf
def _ibis_sqlite_atan2(y, x):
    return _trig_func_binary(math.atan2, y, x)


@udf
def _ibis_sqlite_degrees(x):
    return None if x is None else math.degrees(x)


@udf
def _ibis_sqlite_radians(x):
    return None if x is None else math.radians(x)


@udf
def _ibis_sqlite_xor(x, y):
    return None if x is None or y is None else x ^ y


@udf
def _ibis_sqlite_inv(x):
    return None if x is None else ~x


@udf
def _ibis_sqlite_pi():
    return math.pi


@udf
def _ibis_sqlite_e():
    return math.e


class _ibis_sqlite_var:
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
class _ibis_sqlite_var_pop(_ibis_sqlite_var):
    def __init__(self):
        super().__init__(0)


@udaf
class _ibis_sqlite_var_samp(_ibis_sqlite_var):
    def __init__(self):
        super().__init__(1)


class _ibis_sqlite_bit_agg:
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
class _ibis_sqlite_bit_or(_ibis_sqlite_bit_agg):
    def __init__(self):
        super().__init__(operator.or_)


@udaf
class _ibis_sqlite_bit_and(_ibis_sqlite_bit_agg):
    def __init__(self):
        super().__init__(operator.and_)


@udaf
class _ibis_sqlite_bit_xor(_ibis_sqlite_bit_agg):
    def __init__(self):
        super().__init__(operator.xor)


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
            'Only positional arguments without defaults are supported in Ibis '
            'SQLite function registration'
        )
    return len(parameters)


def register_all(dbapi_connection):
    """Register all udf and udaf with the connection.

    All udf and udaf are defined in this file with the `udf` and `udaf`
    decorators.

    Parameters
    ----------
    dbapi_connection
        sqlalchemy.Connection object
    """
    for func in _SQLITE_UDF_REGISTRY:
        dbapi_connection.create_function(
            func.__name__, _number_of_arguments(func), func
        )

    for agg in _SQLITE_UDAF_REGISTRY:
        dbapi_connection.create_aggregate(
            agg.__name__,
            # substract one to ignore the `self` argument of the step method
            _number_of_arguments(agg.step) - 1,
            agg,
        )
