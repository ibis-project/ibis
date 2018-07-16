import os
import regex as re
import math
import inspect

import sqlalchemy as sa

from ibis.client import Database
from ibis.compat import maketrans, functools
from ibis.sql.sqlite.compiler import SQLiteDialect

import ibis.sql.alchemy as alch
import ibis.common as com


class SQLiteTable(alch.AlchemyTable):
    pass


class SQLiteDatabase(Database):
    pass


_SQLITE_UDF_REGISTRY = set()
_SQLITE_UDAF_REGISTRY = set()


def udf(f):
    _SQLITE_UDF_REGISTRY.add(f)
    return f


def udaf(f):
    _SQLITE_UDAF_REGISTRY.add(f)
    return f


@udf
def _ibis_sqlite_reverse(string):
    if string is not None:
        return string[::-1]
    return None


@udf
def _ibis_sqlite_string_ascii(string):
    if string is not None:
        return ord(string[0])
    return None


@udf
def _ibis_sqlite_capitalize(string):
    if string is not None:
        return string.capitalize()
    return None


@udf
def _ibis_sqlite_translate(string, from_string, to_string):
    if (string is not None and
            from_string is not None and to_string is not None):
        table = maketrans(from_string, to_string)
        return string.translate(table)
    return None


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
    if string is None or regex is None:
        return None
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
    if string is None or pattern is None or replacement is None:
        return None
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
    if string is None or pattern is None or index is None:
        return None

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
    return math.exp(arg) if arg is not None else None


@udf
def _ibis_sqlite_log(arg, base):
    if arg is None or base is None or arg < 0 or base < 0:
        return None
    return math.log(arg, base)


@udf
def _ibis_sqlite_ln(arg):
    if arg is None or arg < 0:
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
    return math.floor(arg) if arg is not None else None


@udf
def _ibis_sqlite_ceil(arg):
    return math.ceil(arg) if arg is not None else None


@udf
def _ibis_sqlite_sign(arg):
    if arg is None:
        return None
    elif arg == 0:
        return 0
    else:
        return math.copysign(1, arg)


@udf
def _ibis_sqlite_floordiv(left, right):
    if left is None or right is None:
        return None
    return left // right


@udf
def _ibis_sqlite_mod(left, right):
    if left is None or right is None:
        return None
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
    if arg is None or power is None or (arg < 0.0 and not power.is_integer()):
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


class _ibis_sqlite_var(object):

    def __init__(self, offset):
        self.mean = 0.0
        self.sum_of_squares_of_differences = 0.0
        self.count = 0
        self.offset = offset

    def step(self, value):
        if value is None:
            return

        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.sum_of_squares_of_differences += delta * (value - self.mean)

    def finalize(self):
        if not self.count:
            return None
        return self.sum_of_squares_of_differences / (self.count - self.offset)


@udaf
class _ibis_sqlite_var_pop(_ibis_sqlite_var):

    def __init__(self):
        super(_ibis_sqlite_var_pop, self).__init__(0)


@udaf
class _ibis_sqlite_var_samp(_ibis_sqlite_var):

    def __init__(self):
        super(_ibis_sqlite_var_samp, self).__init__(1)


def number_of_arguments(callable):
    argspec = inspect.getargspec(callable)

    if argspec.varargs is not None:
        raise TypeError(
            'Variable length arguments not supported in Ibis SQLite function '
            'registration'
        )

    if argspec.keywords is not None:
        raise NotImplementedError(
            'Keyword arguments not implemented for Ibis SQLite function '
            'registration'
        )

    if argspec.defaults is not None:
        raise NotImplementedError(
            'Keyword arguments not implemented for Ibis SQLite function '
            'registration'
        )
    return len(argspec.args)


def _register_function(func, con):
    """Register a Python callable with a SQLite connection `con`.

    Parameters
    ----------
    func : callable
    con : sqlalchemy.Connection
    """
    nargs = number_of_arguments(func)
    con.connection.connection.create_function(func.__name__, nargs, func)


def _register_aggregate(agg, con):
    """Register a Python class that performs aggregation in SQLite.

    Parameters
    ----------
    agg : type
    con : sqlalchemy.Connection
    """
    nargs = number_of_arguments(agg.step) - 1  # because self
    con.connection.connection.create_aggregate(agg.__name__, nargs, agg)


class SQLiteClient(alch.AlchemyClient):

    """
    The Ibis SQLite client class
    """

    dialect = SQLiteDialect
    database_class = SQLiteDatabase
    table_class = SQLiteTable

    def __init__(self, path=None, create=False):
        super(SQLiteClient, self).__init__(sa.create_engine('sqlite://'))
        self.name = path
        self.database_name = 'base'

        if path is not None:
            self.attach(self.database_name, path, create=create)

        for func in _SQLITE_UDF_REGISTRY:
            self.con.run_callable(functools.partial(_register_function, func))

        for agg in _SQLITE_UDAF_REGISTRY:
            self.con.run_callable(functools.partial(_register_aggregate, agg))

    @property
    def current_database(self):
        return self.database_name

    def list_databases(self):
        raise NotImplementedError(
            'Listing databases in SQLite is not implemented'
        )

    def set_database(self, name):
        raise NotImplementedError('set_database is not implemented for SQLite')

    def attach(self, name, path, create=False):
        """Connect another SQLite database file

        Parameters
        ----------
        name : string
            Database name within SQLite
        path : string
            Path to sqlite3 file
        create : boolean, optional
            If file does not exist, create file if True otherwise raise an
            Exception
        """
        if not os.path.exists(path) and not create:
            raise com.IbisError('File {!r} does not exist'.format(path))

        self.raw_sql(
            "ATTACH DATABASE {path!r} AS {name}".format(
                path=path,
                name=self.con.dialect.identifier_preparer.quote(name),
            )
        )

    @property
    def client(self):
        return self

    def table(self, name, database=None):
        """
        Create a table expression that references a particular table in the
        SQLite database

        Parameters
        ----------
        name : string
        database : string, optional
          name of the attached database that the table is located in.

        Returns
        -------
        table : TableExpr
        """
        alch_table = self._get_sqla_table(name, schema=database)
        node = self.table_class(alch_table, self)
        return self.table_expr_class(node)

    def list_tables(self, like=None, database=None, schema=None):
        if database is None:
            database = self.database_name
        return super(SQLiteClient, self).list_tables(like, schema=database)
