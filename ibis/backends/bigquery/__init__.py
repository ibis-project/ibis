"""BigQuery public API."""

from __future__ import annotations

import contextlib
import warnings
from typing import TYPE_CHECKING, Any, Callable
from urllib.parse import parse_qs, urlparse

import google.auth.credentials
import google.cloud.bigquery as bq
import google.cloud.bigquery_storage_v1 as bqstorage
import pandas as pd
import pydata_google_auth
import sqlglot as sg
from pydata_google_auth import cache

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateSchema, CanListDatabases, Database
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.bigquery.client import (
    BigQueryCursor,
    bigquery_param,
    parse_project_and_dataset,
    rename_partitioned_column,
    schema_from_bigquery_table,
)
from ibis.backends.bigquery.compiler import BigQueryCompiler
from ibis.backends.bigquery.datatypes import BigQuerySchema, BigQueryType
from ibis.formats.pandas import PandasData

with contextlib.suppress(ImportError):
    from ibis.backends.bigquery.udf import udf  # noqa: F401

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import pyarrow as pa
    from google.cloud.bigquery.table import RowIterator

SCOPES = ["https://www.googleapis.com/auth/bigquery"]
EXTERNAL_DATA_SCOPES = [
    "https://www.googleapis.com/auth/bigquery",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/drive",
]
CLIENT_ID = "546535678771-gvffde27nd83kfl6qbrnletqvkdmsese.apps.googleusercontent.com"
CLIENT_SECRET = "iU5ohAF2qcqrujegE3hQ1cPt"


def _create_user_agent(application_name: str) -> str:
    user_agent = []

    if application_name:
        user_agent.append(application_name)

    user_agent_default_template = f"ibis/{ibis.__version__}"
    user_agent.append(user_agent_default_template)

    return " ".join(user_agent)


def _create_client_info(application_name):
    from google.api_core.client_info import ClientInfo

    return ClientInfo(user_agent=_create_user_agent(application_name))


def _create_client_info_gapic(application_name):
    from google.api_core.gapic_v1.client_info import ClientInfo

    return ClientInfo(user_agent=_create_user_agent(application_name))


def _anonymous_unnest_to_explode(node: sg.exp.Expression):
    """Convert `ANONYMOUS` `unnest` function calls to `EXPLODE` calls.

    This allows us to generate DuckDB-like `UNNEST` calls and let sqlglot do
    the work of transforming those into the correct BigQuery SQL.
    """
    if isinstance(node, sg.exp.Anonymous) and node.this.lower() == "unnest":
        return sg.exp.Explode(this=node.expressions[0])
    return node


class Backend(BaseSQLBackend, CanCreateSchema, CanListDatabases):
    name = "bigquery"
    compiler = BigQueryCompiler
    supports_in_memory_tables = False
    supports_python_udfs = False

    def _from_url(self, url: str, **kwargs):
        result = urlparse(url)
        params = parse_qs(result.query)
        return self.connect(
            project_id=result.netloc or params.get("project_id", [""])[0],
            dataset_id=result.path[1:] or params.get("dataset_id", [""])[0],
            **kwargs,
        )

    def do_connect(
        self,
        project_id: str | None = None,
        dataset_id: str = "",
        credentials: google.auth.credentials.Credentials | None = None,
        application_name: str | None = None,
        auth_local_webserver: bool = True,
        auth_external_data: bool = False,
        auth_cache: str = "default",
        partition_column: str | None = "PARTITIONTIME",
        client: bq.Client | None = None,
        storage_client: bqstorage.BigQueryReadClient | None = None,
    ) -> Backend:
        """Create a `Backend` for use with Ibis.

        Parameters
        ----------
        project_id
            A BigQuery project id.
        dataset_id
            A dataset id that lives inside of the project indicated by
            `project_id`.
        credentials
            Optional credentials.
        application_name
            A string identifying your application to Google API endpoints.
        auth_local_webserver
            Use a local webserver for the user authentication.  Binds a
            webserver to an open port on localhost between 8080 and 8089,
            inclusive, to receive authentication token. If not set, defaults to
            False, which requests a token via the console.
        auth_external_data
            Authenticate using additional scopes required to `query external
            data sources
            <https://cloud.google.com/bigquery/external-data-sources>`_,
            such as Google Sheets, files in Google Cloud Storage, or files in
            Google Drive. If not set, defaults to False, which requests the
            default BigQuery scopes.
        auth_cache
            Selects the behavior of the credentials cache.

            ``'default'``
                Reads credentials from disk if available, otherwise
                authenticates and caches credentials to disk.

            ``'reauth'``
                Authenticates and caches credentials to disk.

            ``'none'``
                Authenticates and does **not** cache credentials.

            Defaults to ``'default'``.
        partition_column
            Identifier to use instead of default ``_PARTITIONTIME`` partition
            column. Defaults to ``'PARTITIONTIME'``.
        client
            A ``Client`` from the ``google.cloud.bigquery`` package. If not
            set, one is created using the ``project_id`` and ``credentials``.
        storage_client
            A ``BigQueryReadClient`` from the
            ``google.cloud.bigquery_storage_v1`` package. If not set, one is
            created using the ``project_id`` and ``credentials``.

        Returns
        -------
        Backend
            An instance of the BigQuery backend.
        """
        default_project_id = client.project if client is not None else project_id

        # Only need `credentials` to create a `client` and
        # `storage_client`, so only one or the other needs to be set.
        if (client is None or storage_client is None) and credentials is None:
            scopes = SCOPES
            if auth_external_data:
                scopes = EXTERNAL_DATA_SCOPES

            if auth_cache == "default":
                credentials_cache = cache.ReadWriteCredentialsCache(
                    filename="ibis.json"
                )
            elif auth_cache == "reauth":
                credentials_cache = cache.WriteOnlyCredentialsCache(
                    filename="ibis.json"
                )
            elif auth_cache == "none":
                credentials_cache = cache.NOOP
            else:
                raise ValueError(
                    f"Got unexpected value for auth_cache = '{auth_cache}'. "
                    "Expected one of 'default', 'reauth', or 'none'."
                )

            credentials, default_project_id = pydata_google_auth.default(
                scopes,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
                credentials_cache=credentials_cache,
                use_local_webserver=auth_local_webserver,
            )

        project_id = project_id or default_project_id

        (
            self.data_project,
            self.billing_project,
            self.dataset,
        ) = parse_project_and_dataset(project_id, dataset_id)

        if client is not None:
            self.client = client
        else:
            self.client = bq.Client(
                project=self.billing_project,
                credentials=credentials,
                client_info=_create_client_info(application_name),
            )

        if storage_client is not None:
            self.storage_client = storage_client
        else:
            self.storage_client = bqstorage.BigQueryReadClient(
                credentials=credentials,
                client_info=_create_client_info_gapic(application_name),
            )

        self.partition_column = partition_column

    def _parse_project_and_dataset(self, dataset) -> tuple[str, str]:
        if not dataset and not self.dataset:
            raise ValueError("Unable to determine BigQuery dataset.")
        project, _, dataset = parse_project_and_dataset(
            self.billing_project,
            dataset or f"{self.data_project}.{self.dataset}",
        )
        return project, dataset

    @property
    def project_id(self):
        return self.data_project

    @property
    def dataset_id(self):
        return self.dataset

    def create_schema(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
        collate: str | None = None,
        **options: Any,
    ) -> None:
        create_stmt = "CREATE SCHEMA"
        if force:
            create_stmt += " IF NOT EXISTS"

        create_stmt += " "
        create_stmt += ".".join(filter(None, [database, name]))

        if collate is not None:
            create_stmt += f" DEFAULT COLLATION {collate}"

        options_str = ", ".join(f"{name}={value!r}" for name, value in options.items())
        if options_str:
            create_stmt += f" OPTIONS({options_str})"
        self.raw_sql(create_stmt)

    def drop_schema(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        drop_stmt = "DROP SCHEMA"
        if force:
            drop_stmt += " IF EXISTS"

        drop_stmt += " "
        drop_stmt += ".".join(filter(None, [database, name]))
        drop_stmt += " CASCADE" if cascade else " RESTRICT"
        self.raw_sql(drop_stmt)

    def table(self, name: str, database: str | None = None) -> ir.TableExpr:
        if database is None:
            database = f"{self.data_project}.{self.current_schema}"
        table_id = self._fully_qualified_name(name, database)
        t = super().table(table_id)
        bq_table = self.client.get_table(table_id)
        return rename_partitioned_column(t, bq_table, self.partition_column)

    def _fully_qualified_name(self, name, database):
        parts = name.split(".")
        if len(parts) == 3:
            return name

        default_project, default_dataset = self._parse_project_and_dataset(database)
        if len(parts) == 2:
            return f"{default_project}.{name}"
        elif len(parts) == 1:
            return f"{default_project}.{default_dataset}.{name}"
        raise ValueError(f"Got too many components in table name: {name}")

    def _get_schema_using_query(self, query):
        job_config = bq.QueryJobConfig(dry_run=True, use_query_cache=False)
        job = self.client.query(query, job_config=job_config)
        return BigQuerySchema.to_ibis(job.schema)

    def _get_table_schema(self, qualified_name):
        dataset, table = qualified_name.rsplit(".", 1)
        assert dataset is not None, "dataset is None"
        return self.get_schema(table, database=dataset)

    def _execute(self, stmt, results=True, query_parameters=None):
        job_config = bq.job.QueryJobConfig()
        job_config.query_parameters = query_parameters or []
        job_config.use_legacy_sql = False  # False by default in >=0.28
        query = self.client.query(
            stmt, job_config=job_config, project=self.billing_project
        )
        query.result()  # blocks until finished
        return BigQueryCursor(query)

    def compile(
        self,
        expr: ir.Expr,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        **_,
    ) -> Any:
        """Compile an Ibis expression.

        Parameters
        ----------
        expr
            Ibis expression
        limit
            For expressions yielding result sets; retrieve at most this number
            of values/rows. Overrides any limit already set on the expression.
        params
            Named unbound parameters

        Returns
        -------
        Any
            The output of compilation. The type of this value depends on the
            backend.
        """

        self._define_udf_translation_rules(expr)
        sql = self.compiler.to_ast_ensure_limit(expr, limit, params=params).compile()

        return ";\n\n".join(
            query.transform(_anonymous_unnest_to_explode).sql(
                dialect="bigquery", pretty=True
            )
            for query in sg.parse(sql, read="bigquery")
        )

    def raw_sql(self, query: str, results=False, params=None):
        query_parameters = [
            bigquery_param(
                param.type(),
                value,
                (
                    param.get_name()
                    if not isinstance(op := param.op(), ops.Alias)
                    else op.arg.name
                ),
            )
            for param, value in (params or {}).items()
        ]
        return self._execute(query, results=results, query_parameters=query_parameters)

    @property
    def current_database(self) -> str:
        warnings.warn(
            "current_database will return the current *data project* in ibis 7.0.0; "
            "use current_schema for the current BigQuery dataset",
            category=FutureWarning,
        )
        # TODO: return self.data_project in ibis 7.0.0
        return self.dataset

    @property
    def current_schema(self) -> str | None:
        return self.dataset

    def database(self, name=None):
        if name is None and not self.dataset:
            raise ValueError(
                "Unable to determine BigQuery dataset. Call "
                "client.database('my_dataset') or set_database('my_dataset') "
                "to assign your client a dataset."
            )
        return Database(name or self.dataset, self)

    def execute(self, expr, params=None, limit="default", **kwargs):
        """Compile and execute the given Ibis expression.

        Compile and execute Ibis expression using this backend client
        interface, returning results in-memory in the appropriate object type

        Parameters
        ----------
        expr
            Ibis expression to execute
        limit
            Retrieve at most this number of values/rows. Overrides any limit
            already set on the expression.
        params
            Query parameters
        kwargs
            Extra arguments specific to the backend

        Returns
        -------
        pd.DataFrame | pd.Series | scalar
            Output from execution
        """
        self._run_pre_execute_hooks(expr)

        # TODO: upstream needs to pass params to raw_sql, I think.
        kwargs.pop("timecontext", None)
        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        self._log(sql)
        cursor = self.raw_sql(sql, params=params, **kwargs)

        result = self.fetch_from_cursor(cursor, expr.as_table().schema())

        return expr.__pandas_result__(result)

    def fetch_from_cursor(self, cursor, schema):
        arrow_t = self._cursor_to_arrow(cursor)
        df = arrow_t.to_pandas(timestamp_as_object=True)
        return PandasData.convert_table(df, schema)

    def _cursor_to_arrow(
        self,
        cursor,
        *,
        method: Callable[[RowIterator], pa.Table | Iterable[pa.RecordBatch]]
        | None = None,
        chunk_size: int | None = None,
    ):
        if method is None:
            method = lambda result: result.to_arrow(
                progress_bar_type=None,
                bqstorage_client=self.storage_client,
            )
        query = cursor.query
        query_result = query.result(page_size=chunk_size)
        # workaround potentially not having the ability to create read sessions
        # in the dataset project
        orig_project = query_result._project
        query_result._project = self.billing_project
        try:
            arrow_obj = method(query_result)
        finally:
            query_result._project = orig_project
        return arrow_obj

    def to_pyarrow(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        **kwargs: Any,
    ) -> pa.Table:
        self._import_pyarrow()
        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        self._log(sql)
        cursor = self.raw_sql(sql, params=params, **kwargs)
        table = self._cursor_to_arrow(cursor)
        return expr.__pyarrow_result__(table)

    def to_pyarrow_batches(
        self,
        expr: ir.Expr,
        *,
        params: Mapping[ir.Scalar, Any] | None = None,
        limit: int | str | None = None,
        chunk_size: int = 1_000_000,
        **kwargs: Any,
    ):
        pa = self._import_pyarrow()

        schema = expr.as_table().schema()

        sql = self.compile(expr, limit=limit, params=params, **kwargs)
        self._log(sql)
        cursor = self.raw_sql(sql, params=params, **kwargs)
        batch_iter = self._cursor_to_arrow(
            cursor,
            method=lambda result: result.to_arrow_iterable(
                bqstorage_client=self.storage_client
            ),
            chunk_size=chunk_size,
        )
        return pa.RecordBatchReader.from_batches(schema.to_pyarrow(), batch_iter)

    def get_schema(self, name, database=None):
        table_id = self._fully_qualified_name(name, database)
        table_ref = bq.TableReference.from_string(table_id)
        table = self.client.get_table(table_ref)
        return schema_from_bigquery_table(table)

    def list_schemas(
        self, like: str | None = None, database: str | None = None
    ) -> list[str]:
        results = [
            dataset.dataset_id
            for dataset in self.client.list_datasets(
                project=database if database is not None else self.data_project
            )
        ]
        return self._filter_with_like(results, like)

    @ibis.util.deprecated(
        instead="use `list_schemas()`", as_of="6.1.0", removed_in="8.0.0"
    )
    def list_databases(self, like=None):
        return self.list_schemas(like=like)

    def list_tables(
        self,
        like: str | None = None,
        database: str | None = None,
        schema: str | None = None,
    ) -> list[str]:
        """List the tables in the database.

        Parameters
        ----------
        like
            A pattern to use for listing tables.
        database
            The database (project) to perform the list against.
        schema
            The schema (dataset) inside `database` to perform the list against.

            ::: {.callout-warning}
            ## `schema` refers to database hierarchy

            The `schema` parameter does **not** refer to the column names and
            types of `table`.
            :::
        """
        if database is not None and schema is None:
            util.warn_deprecated(
                "database",
                instead=(
                    f"{self.name} cannot list tables only using `database` specifier. "
                    "Include a `schema` argument."
                ),
                as_of="7.1",
                removed_in="8.0",
            )
            database = sg.parse_one(database, into=sg.exp.Table, read=self.name).sql(
                dialect=self.name
            )
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sg.exp.Table, read=self.name).sql(
                dialect=self.name
            )
        else:
            database = sg.table(schema, db=database).sql(dialect=self.name) or None

        project, dataset = self._parse_project_and_dataset(database)
        dataset_ref = bq.DatasetReference(project, dataset)
        result = [table.table_id for table in self.client.list_tables(dataset_ref)]
        return self._filter_with_like(result, like)

    def set_database(self, name):
        self.data_project, self.dataset = self._parse_project_and_dataset(name)

    @property
    def version(self):
        return bq.__version__

    def create_table(
        self,
        name: str,
        obj: pd.DataFrame | pa.Table | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
        default_collate: str | None = None,
        partition_by: str | None = None,
        cluster_by: Iterable[str] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> ir.Table:
        """Create a table in BigQuery.

        Parameters
        ----------
        name
            Name of the table to create
        obj
            The data with which to populate the table; optional, but one of `obj`
            or `schema` must be specified
        schema
            The schema of the table to create; optional, but one of `obj` or
            `schema` must be specified
        database
            The BigQuery *dataset* in which to create the table; optional
        temp
            This parameter is not yet supported in the BigQuery backend
        overwrite
            If `True`, replace the table if it already exists, otherwise fail if
            the table exists
        default_collate
            Default collation for string columns. See BigQuery's documentation
            for more details: https://cloud.google.com/bigquery/docs/reference/standard-sql/collation-concepts
        partition_by
            Partition the table by the given expression. See BigQuery's documentation
            for more details: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#partition_expression
        cluster_by
            List of columns to cluster the table by. See BigQuery's documentation
            for more details: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#clustering_column_list
        options
            BigQuery-specific table options; see the BigQuery documentation for
            details: https://cloud.google.com/bigquery/docs/reference/standard-sql/data-definition-language#table_option_list

        Returns
        -------
        Table
            The table that was just created
        """
        if obj is None and schema is None:
            raise com.IbisError("One of the `schema` or `obj` parameter is required")

        if temp:
            # TODO: these require a BQ session; figure out how to handle that
            raise NotImplementedError(
                "Temporary tables in the BigQuery backend are not yet supported"
            )

        create_stmt = "CREATE"

        if overwrite:
            create_stmt += " OR REPLACE"

        table_ref = self._fully_qualified_name(name, database)

        create_stmt += f" TABLE `{table_ref}`"

        if isinstance(obj, ir.Table) and schema is not None:
            if not schema.equals(obj.schema()):
                raise com.IbisTypeError(
                    "Provided schema and Ibis table schema are incompatible. Please "
                    "align the two schemas, or provide only one of the two arguments."
                )

        if schema is not None:
            schema_str = ", ".join(
                (
                    f"{name} {BigQueryType.from_ibis(typ)}"
                    + " NOT NULL" * (not typ.nullable)
                )
                for name, typ in schema.items()
            )
            create_stmt += f" ({schema_str})"

        if default_collate is not None:
            create_stmt += f" DEFAULT COLLATE {default_collate!r}"

        if partition_by is not None:
            create_stmt += f" PARTITION BY {partition_by}"

        if cluster_by is not None:
            create_stmt += f" CLUSTER BY {', '.join(cluster_by)}"

        if options:
            pairs = ", ".join(f"{k}={v!r}" for k, v in options.items())
            create_stmt += f" OPTIONS({pairs})"

        if obj is not None:
            import pyarrow as pa

            if isinstance(obj, (pd.DataFrame, pa.Table)):
                table = ibis.memtable(obj, schema=schema)
            else:
                table = obj

            create_stmt += f" AS ({self.compile(table)})"

        self.raw_sql(create_stmt)

        return self.table(table_ref)

    def drop_table(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        table_id = self._fully_qualified_name(name, database)
        drop_stmt = "DROP TABLE"
        if force:
            drop_stmt += " IF EXISTS"
        drop_stmt += f" `{table_id}`"
        self.raw_sql(drop_stmt)

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        or_replace = "OR REPLACE " * overwrite
        sql_select = self.compile(obj)
        table_id = self._fully_qualified_name(name, database)
        code = f"CREATE {or_replace}VIEW `{table_id}` AS {sql_select}"
        self.raw_sql(code)
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        # default_project, default_dataset = self._parse_project_and_dataset(database)
        table_id = self._fully_qualified_name(name, database)
        drop_stmt = "DROP VIEW"
        if force:
            drop_stmt += " IF EXISTS"
        drop_stmt += f" `{table_id}`"
        self.raw_sql(drop_stmt)


def compile(expr, params=None, **kwargs):
    """Compile an expression for BigQuery."""
    backend = Backend()
    return backend.compile(expr, params=params, **kwargs)


def connect(
    project_id: str | None = None,
    dataset_id: str = "",
    credentials: google.auth.credentials.Credentials | None = None,
    application_name: str | None = None,
    auth_local_webserver: bool = False,
    auth_external_data: bool = False,
    auth_cache: str = "default",
    partition_column: str | None = "PARTITIONTIME",
) -> Backend:
    """Create a :class:`Backend` for use with Ibis.

    Parameters
    ----------
    project_id
        A BigQuery project id.
    dataset_id
        A dataset id that lives inside of the project indicated by
        `project_id`.
    credentials
        Optional credentials.
    application_name
        A string identifying your application to Google API endpoints.
    auth_local_webserver
        Use a local webserver for the user authentication.  Binds a
        webserver to an open port on localhost between 8080 and 8089,
        inclusive, to receive authentication token. If not set, defaults
        to False, which requests a token via the console.
    auth_external_data
        Authenticate using additional scopes required to `query external
        data sources
        <https://cloud.google.com/bigquery/external-data-sources>`_,
        such as Google Sheets, files in Google Cloud Storage, or files in
        Google Drive. If not set, defaults to False, which requests the
        default BigQuery scopes.
    auth_cache
        Selects the behavior of the credentials cache.

        ``'default'``
            Reads credentials from disk if available, otherwise
            authenticates and caches credentials to disk.

        ``'reauth'``
            Authenticates and caches credentials to disk.

        ``'none'``
            Authenticates and does **not** cache credentials.

        Defaults to ``'default'``.
    partition_column
        Identifier to use instead of default ``_PARTITIONTIME`` partition
        column. Defaults to ``'PARTITIONTIME'``.

    Returns
    -------
    Backend
        An instance of the BigQuery backend
    """
    backend = Backend()
    return backend.connect(
        project_id=project_id,
        dataset_id=dataset_id,
        credentials=credentials,
        application_name=application_name,
        auth_local_webserver=auth_local_webserver,
        auth_external_data=auth_external_data,
        auth_cache=auth_cache,
        partition_column=partition_column,
    )


__all__ = [
    "Backend",
    "compile",
    "connect",
]
