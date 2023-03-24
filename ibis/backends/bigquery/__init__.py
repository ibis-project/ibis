"""BigQuery public API."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping
from urllib.parse import parse_qs, urlparse

import google.auth.credentials
import google.cloud.bigquery as bq
import pandas as pd
import pydata_google_auth
from google.api_core.exceptions import NotFound
from pydata_google_auth import cache

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.schema as sch
import ibis.expr.types as ir
from ibis.backends.base.sql import BaseSQLBackend
from ibis.backends.bigquery.client import (
    BigQueryCursor,
    BigQueryDatabase,
    BigQueryTable,
    bigquery_field_to_ibis_dtype,
    bigquery_param,
    ibis_schema_to_bigquery_schema,
    parse_project_and_dataset,
    rename_partitioned_column,
)
from ibis.backends.bigquery.compiler import BigQueryCompiler
from ibis.util import deprecated

with contextlib.suppress(ImportError):
    from ibis.backends.bigquery.udf import udf  # noqa: F401

if TYPE_CHECKING:
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


def _create_client_info(application_name):
    from google.api_core.client_info import ClientInfo

    user_agent = []

    if application_name:
        user_agent.append(application_name)

    user_agent_default_template = f"ibis/{ibis.__version__}"
    user_agent.append(user_agent_default_template)

    return ClientInfo(user_agent=" ".join(user_agent))


class Backend(BaseSQLBackend):
    name = "bigquery"
    compiler = BigQueryCompiler
    database_class = BigQueryDatabase
    table_class = BigQueryTable

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
    ):
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

        Returns
        -------
        Backend
            An instance of the BigQuery backend.
        """
        default_project_id = ""

        if credentials is None:
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

        self.client = bq.Client(
            project=self.billing_project,
            credentials=credentials,
            client_info=_create_client_info(application_name),
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

    def table(self, name, database=None) -> ir.TableExpr:
        t = super().table(name, database=database)
        table_id = self._fully_qualified_name(name, database)
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
        fields = self._adapt_types(job.schema)
        return sch.Schema(fields)

    def _get_table_schema(self, qualified_name):
        dataset, table = qualified_name.rsplit(".", 1)
        assert dataset is not None, "dataset is None"
        return self.get_schema(table, database=dataset)

    def _adapt_types(self, descr):
        return {col.name: bigquery_field_to_ibis_dtype(col) for col in descr}

    def _execute(self, stmt, results=True, query_parameters=None):
        job_config = bq.job.QueryJobConfig()
        job_config.query_parameters = query_parameters or []
        job_config.use_legacy_sql = False  # False by default in >=0.28
        query = self.client.query(
            stmt, job_config=job_config, project=self.billing_project
        )
        query.result()  # blocks until finished
        return BigQueryCursor(query)

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
        return self.dataset

    def database(self, name=None):
        if name is None and not self.dataset:
            raise ValueError(
                "Unable to determine BigQuery dataset. Call "
                "client.database('my_dataset') or set_database('my_dataset') "
                "to assign your client a dataset."
            )
        return self.database_class(name or self.dataset, self)

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
        # TODO: upstream needs to pass params to raw_sql, I think.
        kwargs.pop("timecontext", None)
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        self._log(sql)
        cursor = self.raw_sql(sql, params=params, **kwargs)
        schema = self.ast_schema(query_ast, **kwargs)
        result = self.fetch_from_cursor(cursor, schema)

        if hasattr(getattr(query_ast, "dml", query_ast), "result_handler"):
            result = query_ast.dml.result_handler(result)

        return result

    @deprecated(
        instead="use name in con.list_databases()", as_of="2.0", removed_in="6.0"
    )
    def exists_database(self, name):
        """Return whether a database name exists in the current connection.

        Deprecated in Ibis 2.0. Use `name in client.list_databases()`
        instead.
        """
        project, dataset = self._parse_project_and_dataset(name)
        client = self.client
        dataset_ref = client.dataset(dataset, project=project)
        try:
            client.get_dataset(dataset_ref)
        except NotFound:
            return False
        else:
            return True

    @deprecated(
        instead="use `table in con.list_tables()`", as_of="2.0", removed_in="6.0"
    )
    def exists_table(self, name: str, database: str | None = None) -> bool:
        """Return whether a table name exists in the database.

        Deprecated in Ibis 2.0. Use `name in client.list_tables()`
        instead.
        """
        table_id = self._fully_qualified_name(name, database)
        client = self.client
        try:
            client.get_table(table_id)
        except NotFound:
            return False
        else:
            return True

    def fetch_from_cursor(self, cursor, schema):
        arrow_t = self._cursor_to_arrow(cursor)
        df = arrow_t.to_pandas(timestamp_as_object=True)
        return schema.apply_to(df)

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
                bqstorage_client=None,
                create_bqstorage_client=True,
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
        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        cursor = self.raw_sql(sql, params=params, **kwargs)
        table = self._cursor_to_arrow(cursor)
        if isinstance(expr, ir.Scalar):
            assert len(table.columns) == 1, "len(table.columns) != 1"
            return table[0][0]
        elif isinstance(expr, ir.Column):
            assert len(table.columns) == 1, "len(table.columns) != 1"
            return table[0]
        else:
            return table

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

        query_ast = self.compiler.to_ast_ensure_limit(expr, limit, params=params)
        sql = query_ast.compile()
        cursor = self.raw_sql(sql, params=params, **kwargs)
        batch_iter = self._cursor_to_arrow(
            cursor,
            method=lambda result: result.to_arrow_iterable(),
            chunk_size=chunk_size,
        )
        return pa.RecordBatchReader.from_batches(schema.to_pyarrow(), batch_iter)

    def get_schema(self, name, database=None):
        table_id = self._fully_qualified_name(name, database)
        table_ref = bq.TableReference.from_string(table_id)
        bq_table = self.client.get_table(table_ref)
        return sch.infer(bq_table)

    def list_databases(self, like=None):
        results = [
            dataset.dataset_id
            for dataset in self.client.list_datasets(project=self.data_project)
        ]
        return self._filter_with_like(results, like)

    def list_tables(self, like=None, database=None):
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
        obj: pd.DataFrame | ir.Table | None = None,
        *,
        schema: ibis.Schema | None = None,
        database: str | None = None,
        temp: bool | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        if obj is None and schema is None:
            raise com.IbisError("The schema or obj parameter is required")
        if temp is True:
            raise NotImplementedError(
                "BigQuery backend does not yet support temporary tables"
            )
        if overwrite is not False:
            raise NotImplementedError(
                "BigQuery backend does not yet support overwriting tables"
            )
        if schema is not None:
            table_id = self._fully_qualified_name(name, database)
            bigquery_schema = ibis_schema_to_bigquery_schema(schema)
            table = bq.Table(table_id, schema=bigquery_schema)
            self.client.create_table(table)
        else:
            project_id, dataset = self._parse_project_and_dataset(database)
            if isinstance(obj, pd.DataFrame):
                table = ibis.memtable(obj)
            else:
                table = obj
            sql_select = self.compile(table)
            table_ref = f"`{project_id}`.`{dataset}`.`{name}`"
            self.raw_sql(f'CREATE TABLE {table_ref} AS ({sql_select})')
        return self.table(name, database=database)

    def drop_table(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        table_id = self._fully_qualified_name(name, database)
        self.client.delete_table(table_id, not_found_ok=not force)

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
        code = f"CREATE {or_replace}VIEW {table_id} AS {sql_select}"
        self.raw_sql(code)
        return self.table(name, database=database)

    def drop_view(
        self, name: str, *, database: str | None = None, force: bool = False
    ) -> None:
        self.drop_table(name=name, database=database, force=force)


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
