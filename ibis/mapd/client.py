from ibis.compat import parse_version
from ibis.client import Database, SQLClient
from ibis.mapd import compiler as comp
from ibis.util import log

import regex as re
import pymapd

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")

_mapd_dtypes = {
    'BIGINT': dt.int64,
    'BOOLEAN': dt.Boolean,
    'CHAR': dt.string,
    'DATE': dt.date,
    'DECIMAL': dt.float,
    'DOUBLE': dt.float,
    'INT': dt.int32,
    'FLOAT': dt.float,
    'NULL': dt.Null,
    'NUMERIC': dt.float,
    'REAL': dt.float,
    'SMALLINT': dt.int8,
    'STR': dt.string,
    'TEXT': dt.string,
    'TIME': dt.time,
    'TIMESTAMP': dt.timestamp,
    'VAR': dt.string,
}

_ibis_dtypes = {v: k for k, v in _mapd_dtypes.items()}
_ibis_dtypes[dt.String] = 'String'


class MapDDataType(object):

    __slots__ = 'typename', 'nullable'

    def __init__(self, typename, nullable=False):
        if typename not in _mapd_dtypes:
            raise com.UnsupportedBackendType(typename)
        self.typename = typename
        self.nullable = nullable

    def __str__(self):
        if self.nullable:
            return 'Nullable({})'.format(self.typename)
        else:
            return self.typename

    def __repr__(self):
        return '<MapD {}>'.format(str(self))

    @classmethod
    def parse(cls, spec):
        if spec.startswith('Nullable'):
            return cls(spec[9:-1], nullable=True)
        else:
            return cls(spec)

    def to_ibis(self):
        return _mapd_dtypes[self.typename](nullable=self.nullable)

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        typename = _ibis_dtypes[type(dtype)]
        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


class MapDClient(SQLClient):
    """

    """
    database_class = Database
    dialect = comp.MapDDialect

    def __init__(
        self, uri: str=None, user: str=None, password: str=None,
        host: str=None, port: int=9091, dbname: str=None,
        protocol: str='binary', execution_type: int=3
    ):
        """

        :param uri:
        :param user:
        :param password:
        :param host:
        :param port:
        :param dbname:
        :param protocol:
        :param execution_type:
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.protocol = protocol
        self.execution_type = execution_type

        self.con = pymapd.connect(
            uri=uri, user=user, password=password, host=host,
            port=port, dbname=dbname, protocol=protocol
        )

    @property
    def _table_expr_klass(self):
        return ir.TableExpr

    def log(self, msg):
        log(msg)

    def close(self):
        """Close MapD connection and drop any temporary objects"""
        self.con.close()

    def _build_ast(self, expr, context):
        result = comp.build_ast(expr, context)
        return result

    def _fully_qualified_name(self, name, database):
        if bool(fully_qualified_re.search(name)):
            return name

        database = database or self.current_database
        return '{0}.{1}'.format(database, name)

    def _get_table_schema(self, table_name):
        """

        :param table_name:
        :return:
        """
        database = None
        table_name_ = table_name.split('.')
        if len(table_name_) == 2:
            database, table_name = table_name_
        return self.get_schema(table_name, database)

    def _execute(self, query):
        with self.con as conn:
            if self.execution_type == 1:
                stmt_exec = conn.select_ipc_gpu
            elif self.execution_type == 2:
                self.stmt_exec = conn.select_ipc
            else:
                self.stmt_exec = conn.execute

            return stmt_exec(query)

    def database(self, name=None):
        """Connect to a database called `name`.

        Parameters
        ----------
        name : str, optional
            The name of the database to connect to. If ``None``, return
            the database named ``self.current_database``.

        Returns
        -------
        db : Database
            An :class:`ibis.client.Database` instance.

        Notes
        -----
        This creates a new connection if `name` is both not ``None`` and not
        equal to the current database.
        """
        if name == self.current_database or (
            name is None and name != self.current_database
        ):
            return self.database_class(self.current_database, self)
        else:
            client_class = type(self)
            new_client = client_class(
                uri=self.uri, user=self.user, password=self.password,
                host=self.host, port=self.port, dbname=name,
                protocol=self.protocol, execution_type=self.execution_type
            )
            return self.database_class(name, new_client)

    @property
    def current_database(self):
        return self.dbname

    def set_database(self, name):
        raise NotImplementedError(
            'Cannot set database with MapD client. To use a different'
            ' database, use client.database({!r})'.format(name)
        )

    def exists_database(self, name):
        raise NotImplementedError()

    def list_databases(self, like=None):
        raise NotImplementedError()

    def exists_table(self, name: str, database: str=None):
        """
        Determine if the indicated table or view exists

        Parameters
        ----------
        name : string
        database : string, default None

        Returns
        -------
        if_exists : boolean
        """
        return len(self.list_tables(like=name, database=database)) > 0

    def list_tables(self, like=None, database=None):
        return self.con.get_tables()

    def get_schema(self, table_name, database=None):
        """
        Return a Schema object for the indicated table and database

        Parameters
        ----------
        table_name : string
          May be fully qualified
        database : string, default None

        Returns
        -------
        schema : ibis Schema
        """
        col_names = []
        col_types = []

        for col in self.con.get_table_details(table_name):
            col_names.append(col.name)
            col_types.append(MapDDataType.parse(col.type))

        return sch.schema(col_names, col_types)

    @property
    def version(self):
        return parse_version(pymapd.__version__)


@dt.dtype.register(MapDDataType)
def mapd_to_ibis_dtype(mapd_dtype):
    """
    Register MapD Data Types

    :param mapd_dtype:
    :return:
    """
    return mapd_dtype.to_ibis()
