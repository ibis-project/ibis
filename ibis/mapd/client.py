from ibis.compat import parse_version
from ibis.client import Database, Query, SQLClient
from ibis.mapd.compiler import MapDDialect, build_ast
from ibis.util import log
from pymapd.cursor import Cursor

try:
    from pygdf.dataframe import DataFrame as GPUDataFrame
except ImportError:
    GPUDataFrame = None

import regex as re
import pandas as pd
import pymapd

import ibis.common as com
import ibis.expr.types as ir
import ibis.expr.datatypes as dt
import ibis.expr.schema as sch

EXECUTION_TYPE_ICP = 1
EXECUTION_TYPE_ICP_GPU = 2
EXECUTION_TYPE_CURSOR = 3

fully_qualified_re = re.compile(r"(.*)\.(?:`(.*)`|(.*))")


class MapDDataType(object):

    __slots__ = 'typename', 'nullable'

    # using impala.client._HS2_TTypeId_to_dtype as reference
    # https://www.mapd.com/docs/latest/mapd-core-guide/fixed-encoding/
    dtypes = {
        'BIGINT': dt.int64,
        'BOOLEAN': dt.Boolean,
        'CHAR': dt.string,
        'DATE': dt.date,
        'DECIMAL': dt.float64,
        'DOUBLE': dt.float64,
        'INT': dt.int32,
        'INTEGER': dt.int32,
        'FLOAT': dt.float32,
        'NULL': dt.Null,
        'NUMERIC': dt.float64,
        'REAL': dt.float32,
        'SMALLINT': dt.int16,
        'STR': dt.string,
        'TEXT': dt.string,
        'TIME': dt.time,
        'TIMESTAMP': dt.timestamp,
        'VARCHAR': dt.string,
    }

    ibis_dtypes = {
        v: k for k, v in dtypes.items()
    }

    def __init__(self, typename, nullable=False):
        if typename not in self.dtypes:
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
        return self.dtypes[self.typename](nullable=self.nullable)

    @classmethod
    def from_ibis(cls, dtype, nullable=None):
        dtype_ = type(dtype)
        if dtype_ in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype_]
        elif dtype in cls.ibis_dtypes:
            typename = cls.ibis_dtypes[dtype]
        else:
            raise NotImplemented('{0} not Implemented'.format(dtype))

        if nullable is None:
            nullable = dtype.nullable
        return cls(typename, nullable=nullable)


class MapDCursor(object):
    """Cursor to allow the MapD client to reuse machinery in ibis/client.py
    """

    def __init__(self, cursor):
        self.cursor = cursor

    def to_df(self):
        if isinstance(self.cursor, Cursor):
            col_names = [c.name for c in self.cursor.description]
            result = pd.DataFrame(self.cursor.fetchall(), columns=col_names)
        elif self.cursor is None:
            result = pd.DataFrame([])
        else:
            result = self.cursor

        return result

    def __enter__(self):
        # For compatibility when constructed from Query.execute()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


class MapDQuery(Query):
    """

    """
    def _fetch(self, cursor):
        # check if cursor is a pymapd cursor.Cursor
        return self.schema().apply_to(cursor.to_df())


class MapDClient(SQLClient):
    """

    """
    database_class = Database
    sync_query = MapDQuery
    dialect = MapDDialect

    def __init__(
        self, uri: str=None, user: str=None, password: str=None,
        host: str=None, port: int=9091, dbname: str=None,
        protocol: str='binary', execution_type: int=3
    ):
        """

        Parameters
        ----------
        uri : str
        user : str
        password : str
        host : str
        port : int
        dbname : str
        protocol : {‘binary’, ‘http’, ‘https’}
        execution_type : {1, 2, 3}

        """
        self.uri = uri
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.dbname = dbname
        self.protocol = protocol

        if execution_type not in (
            EXECUTION_TYPE_ICP,
            EXECUTION_TYPE_ICP_GPU,
            EXECUTION_TYPE_CURSOR,
        ):
            raise Exception('Execution type defined not available.')

        self.execution_type = execution_type

        self.con = pymapd.connect(
            uri=uri, user=user, password=password, host=host,
            port=port, dbname=dbname, protocol=protocol
        )

    def log(self, msg):
        log(msg)

    def close(self):
        """Close MapD connection and drop any temporary objects"""
        self.con.close()

    def _build_ast(self, expr, context):
        """
        Required.

        expr:
        context:
        :return:
        """
        result = build_ast(expr, context)
        return result

    def _fully_qualified_name(self, name, database):
        if fully_qualified_re.search(name):
            return name

        database = database or self.current_database
        return '{}.{}'.format(database, name)

    def _get_table_schema(self, table_name, database=None):
        """

        Parameters
        ----------
        table_name : str
        database : str

        Returns
        -------
        schema : ibis Schema

        """
        table_name_ = table_name.split('.')
        if len(table_name_) == 2:
            database, table_name = table_name_
        return self.get_schema(table_name, database)

    def _execute(self, query, results=False):
        """

        query:
        :return:
        """
        if self.execution_type != EXECUTION_TYPE_CURSOR:
            raise NotImplemented()

        return MapDCursor(self.con.cursor().execute(query))

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
        if name == self.current_database or name is None:
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
        return bool(self.list_tables(like=name, database=database))

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

        return sch.schema([
            (col.name, MapDDataType.parse(col.type))
            for col in self.con.get_table_details(table_name)
        ])

    @property
    def version(self):
        return parse_version(pymapd.__version__)


@dt.dtype.register(MapDDataType)
def mapd_to_ibis_dtype(mapd_dtype):
    """
    Register MapD Data Types

    mapd_dtype:
    :return:
    """
    return mapd_dtype.to_ibis()
