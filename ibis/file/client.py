import ibis
import ibis.expr.types as ir
from ibis.pandas.core import execute_and_reset
from ibis.compat import Path


class FileClient(ibis.client.Client):

    def __init__(self, root):
        self.root = Path(str(root))
        self.dictionary = {}

    def insert(self, path, expr, **kwargs):
        raise NotImplementedError

    def table(self, name, path):
        raise NotImplementedError

    def database(self, name=None, path=None):
        if name is None:
            return FileDatabase('root', self, path=path)

        if name not in self.list_databases(path):
            raise AttributeError(name)
        if path is None:
            path = self.root

        new_name = "{}.{}".format(name, self.extension)
        if (self.root / name).is_dir():
            path /= name
        elif not str(path).endswith(new_name):
            path /= new_name

        return FileDatabase(name, self, path=path)

    def execute(self, expr, params=None, **kwargs):  # noqa
        assert isinstance(expr, ir.Expr)
        scope = kwargs.pop('scope', {})
        return execute_and_reset(expr, params=params, scope=scope, **kwargs)

    def list_tables(self, path=None):
        raise NotImplementedError

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

    def list_databases(self, path=None):
        raise NotImplementedError

    def _list_databases_dirs(self, path=None):
        # databases are dir
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
        return tables

    def _list_databases_dirs_or_files(self, path=None):
        # databases are dir & file
        if path is None:
            path = self.root

        tables = []
        if path.is_dir():
            for d in path.iterdir():
                if d.is_dir():
                    tables.append(d.name)
                elif d.is_file():
                    if str(d).endswith(self.extension):
                        tables.append(d.stem)
        elif path.is_file():
            # by definition we are at the db level at this point
            pass

        return tables


class FileDatabase(ibis.client.Database):

    def __init__(self, name, client, path=None):
        super(FileDatabase, self).__init__(name, client)
        self.path = path

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

    def list_tables(self, path=None):
        if path is None:
            path = self.path
        return sorted(self.client.list_tables(path=path))
