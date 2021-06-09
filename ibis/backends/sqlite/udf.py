import functools
import inspect
import math

import regex as re

_SQLITE_UDF_REGISTRY = set()
_SQLITE_UDAF_REGISTRY = set()


def udf(f):
    """Create a SQLite scalar UDF from `f`

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
    """Return whether `regex` exists in `string`.

    Parameters
    ----------
    string : str
    regex : str

    Returns
    -------
    found : bool
    """
    return re.search(regex, string) is not None


@udf
def _ibis_sqlite_regex_replace(string, pattern, replacement):
    """Replace occurences of `pattern` in `string` with `replacement`.

    Parameters
    ----------
    string : str
    pattern : str
    replacement : str

    Returns
    -------
    result : str
    """
    return re.sub(pattern, replacement, string)


@udf
def _ibis_sqlite_regex_extract(string, pattern, index):
    """Extract match of regular expression `pattern` from `string` at `index`.

    Parameters
    ----------
    string : str
    pattern : str
    index : int

    Returns
    -------
    result : str or None
    """
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
    return arg ** power


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


def _register_function(func, con):
    """Register a Python callable with a SQLite connection `con`.

    Parameters
    ----------
    func : callable
    con : sqlalchemy.Connection
    """
    nargs = _number_of_arguments(func)
    con.connection.connection.create_function(func.__name__, nargs, func)


def _register_aggregate(agg, con):
    """Register a Python class that performs aggregation in SQLite.

    Parameters
    ----------
    agg : type
    con : sqlalchemy.Connection
    """
    nargs = _number_of_arguments(agg.step) - 1  # because self
    con.connection.connection.create_aggregate(agg.__name__, nargs, agg)


def register_all(con):
    """Register all udf and udaf with the connection.

    All udf and udaf are defined in this file with the `udf` and `udaf`
    decorators.

    Parameters
    ----------
    con : sqlalchemy.Connection
    """
    for func in _SQLITE_UDF_REGISTRY:
        con.run_callable(functools.partial(_register_function, func))

    for agg in _SQLITE_UDAF_REGISTRY:
        con.run_callable(functools.partial(_register_aggregate, agg))
