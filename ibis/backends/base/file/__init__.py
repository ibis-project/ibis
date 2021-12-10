import abc
from pathlib import Path

import pandas as pd

import ibis
import ibis.expr.types as ir
from ibis.backends.base import BaseBackend, Database
from ibis.backends.pandas.core import execute_and_reset
from ibis.util import warn_deprecated

# Load options of pandas backend
ibis.pandas


class FileDatabase(Database):
    def __init__(self, name, client):
        super().__init__(name, client)
        self.path = client.path

    def __str__(self):
        return '{0.__class__.__name__}({0.name})'.format(self)

    def __dir__(self):
        dbs = self.list_databases(path=self.path)
        tables = self.list_tables(path=self.path)
        return sorted(set(dbs).union(set(tables)))

    def __getattr__(self, name):
        try:
            return self.table(name, path=self.path)
        except AttributeError:
            return self.database(name, path=self.path)

    def table(self, name, path):
        return self.client.table(name, path=path)

    def database(self, name=None, path=None):
        return self.client.database(name=name, path=path)

    def list_databases(self, path=None):
        if path is None:
            path = self.path
        return sorted(self.client.list_databases(path=path))

    def list_tables(self, path=None, database=None):
        if path is None:
            path = self.path
        return sorted(self.client.list_tables(path=path, database=database))


class BaseFileBackend(BaseBackend):
    """
    Base backend class for pandas pseudo-backends for file formats.
    """

    database_class = FileDatabase

    def do_connect(self, path):
        """Create a Client for use with Ibis

        Parameters
        ----------
        path : str or pathlib.Path

        Returns
        -------
        Backend
        """
        self.path = self.root = Path(path)
        self.dictionary = {}

    @property
    def version(self) -> str:
        return pd.__version__

    def list_tables(
        self, path: Path = None, like: str = None, database: str = None
    ):
        # For file backends, we return files in the `path` directory.

        def is_valid(path):
            return path.is_file() and path.suffix == '.' + self.extension

        path = path or self.path

        if path.is_dir():
            tables = [f.stem for f in path.iterdir() if is_valid(f)]
        elif is_valid(path):
            tables = [path.stem]
        else:
            tables = []

        return self._filter_with_like(tables, like)

    @property
    def current_database(self):
        # Databases for the file backend are a bit confusing
        # `list_databases()` will return the directories in the current path
        # The  `current_database` is not in that list. Probably we want to
        # rethink this eventually.  For now we just return `None` here, as if
        # databases were not supported
        return '.'

    def compile(self, expr, *args, **kwargs):
        return expr

    def _list_databases_dirs(self, path=None):
        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
        return tables

    def _list_tables_files(self, path=None):
        # tables are files in a dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_file():
                    if str(d).endswith(self.extension):
                        tables.append(d.stem)
        elif path.is_file():
            if str(path).endswith(self.extension):
                tables.append(path.stem)
        return tables

    def list_databases(self, path=None, like=None):
        if path is None:
            path = self.path
        else:
            warn_deprecated(
                'The `path` argument of `list_databases`',
                version='2.0',
                instead='`connect()` with a different path',
            )
        databases = ['.'] + self._list_databases_dirs(path)
        return self._filter_with_like(databases, like)

    @abc.abstractmethod
    def insert(self, path, expr, **kwargs):
        pass

    @abc.abstractmethod
    def table(self, name, path):
        pass

    def database(self, name=None, path=None):
        if name is None:
            self.path = path or self.path
            return super().database(name)

        if path is None:
            path = self.root
        if name not in self.list_databases(path):
            raise AttributeError(name)

        new_name = f"{name}.{self.extension}"
        if (self.root / name).is_dir():
            path /= name
        elif not str(path).endswith(new_name):
            path /= new_name

        self.path = path
        return super().database(name)

    def execute(self, expr, params=None, **kwargs):  # noqa
        assert isinstance(expr, ir.Expr)
        return execute_and_reset(expr, params=params, **kwargs)
