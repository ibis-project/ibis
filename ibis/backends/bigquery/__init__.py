"""BigQuery public API."""

from __future__ import annotations

import concurrent.futures
import contextlib
import glob
import os
import re
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import parse_qs, urlparse

import google.auth.credentials
import google.cloud.bigquery as bq
import google.cloud.bigquery_storage_v1 as bqstorage
import pandas as pd
import pydata_google_auth
import sqlglot as sg
import sqlglot.expressions as sge
from pydata_google_auth import cache

import ibis
import ibis.common.exceptions as com
import ibis.expr.operations as ops
import ibis.expr.types as ir
from ibis import util
from ibis.backends.base import CanCreateSchema, Database
from ibis.backends.base.sqlglot import SQLGlotBackend
from ibis.backends.base.sqlglot.datatypes import BigQueryType
from ibis.backends.bigquery.client import (
    BigQueryCursor,
    bigquery_param,
    parse_project_and_dataset,
    rename_partitioned_column,
    schema_from_bigquery_table,
)
from ibis.backends.bigquery.compiler import BigQueryCompiler
from ibis.backends.bigquery.datatypes import BigQuerySchema
from ibis.backends.bigquery.udf.core import PythonToJavaScriptTranslator

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path

    import pyarrow as pa
    from google.cloud.bigquery.table import RowIterator

    import ibis.expr.types as dt

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


_MEMTABLE_PATTERN = re.compile(r"^_?ibis_(?:pandas|pyarrow)_memtable_[a-z0-9]{26}$")


def _qualify_memtable(
    node: sge.Expression, *, dataset: str | None, project: str | None
) -> sge.Expression:
    """Add a BigQuery dataset and project to memtable references."""
    if isinstance(node, sge.Table) and _MEMTABLE_PATTERN.match(node.name) is not None:
        node.args["db"] = dataset
        node.args["catalog"] = project
    return node


def _remove_null_ordering_from_unsupported_window(
    node: sge.Expression,
) -> sge.Expression:
    """Remove null ordering in window frame clauses not supported by BigQuery.

    BigQuery has only partial support for NULL FIRST/LAST in RANGE windows so
    we remove it from any window frame clause that doesn't support it.

    Here's the support matrix:

    ✅ sum(x) over (order by y desc nulls last)
    🚫 sum(x) over (order by y asc nulls last)
    ✅ sum(x) over (order by y asc nulls first)
    🚫 sum(x) over (order by y desc nulls first)
    """
    if isinstance(node, sge.Window):
        order = node.args.get("order")
        if order is not None:
            for key in order.args["expressions"]:
                kargs = key.args
                if kargs.get("desc") is True and kargs.get("nulls_first", False):
                    kargs["nulls_first"] = False
                elif kargs.get("desc") is False and not kargs.setdefault(
                    "nulls_first", True
                ):
                    kargs["nulls_first"] = True
    return node


class Backend(SQLGlotBackend, CanCreateSchema):
    name = "bigquery"
    compiler = BigQueryCompiler()
    supports_in_memory_tables = True
    supports_python_udfs = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._session_dataset: bq.DatasetReference | None = None
        self._query_cache.lookup = lambda name: self.table(
            name,
            schema=self._session_dataset.dataset_id,
            database=self._session_dataset.project,
        ).op()

    def _register_in_memory_table(self, op: ops.InMemoryTable) -> None:
        self._make_session()

        raw_name = op.name

        project = self._session_dataset.project
        dataset = self._session_dataset.dataset_id

        if raw_name not in self.list_tables(schema=dataset, database=project):
            table_id = sg.table(
                raw_name, db=dataset, catalog=project, quoted=False
            ).sql(dialect=self.name)

            bq_schema = BigQuerySchema.from_ibis(op.schema)
            load_job = self.client.load_table_from_dataframe(
                op.data.to_frame(),
                table_id,
                job_config=bq.LoadJobConfig(
                    # fail if the table already exists and contains data
                    write_disposition=bq.WriteDisposition.WRITE_EMPTY,
                    schema=bq_schema,
                ),
            )
            load_job.result()

    def _read_file(
        self,
        path: str | Path,
        *,
        table_name: str | None = None,
        job_config: bq.LoadJobConfig,
    ) -> ir.Table:
        self._make_session()

        if table_name is None:
            table_name = util.gen_name(f"bq_read_{job_config.source_format}")

        table_ref = self._session_dataset.table(table_name)

        schema = self._session_dataset.dataset_id
        database = self._session_dataset.project

        # drop the table if it exists
        #
        # we could do this with write_disposition = WRITE_TRUNCATE but then the
        # concurrent append jobs aren't possible
        #
        # dropping the table first means all write_dispositions can be
        # WRITE_APPEND
        self.drop_table(table_name, schema=schema, database=database, force=True)

        if os.path.isdir(path):
            raise NotImplementedError("Reading from a directory is not supported.")
        elif str(path).startswith("gs://"):
            load_job = self.client.load_table_from_uri(
                path, table_ref, job_config=job_config
            )
            load_job.result()
        else:

            def load(file: str) -> None:
                with open(file, mode="rb") as f:
                    load_job = self.client.load_table_from_file(
                        f, table_ref, job_config=job_config
                    )
                    load_job.result()

            job_config.write_disposition = bq.WriteDisposition.WRITE_APPEND

            with concurrent.futures.ThreadPoolExecutor() as executor:
                for fut in concurrent.futures.as_completed(
                    executor.submit(load, file) for file in glob.glob(str(path))
                ):
                    fut.result()

        return self.table(table_name, schema=schema, database=database)

    def read_parquet(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ):
        """Read Parquet data into a BigQuery table.

        Parameters
        ----------
        path
            Path to a Parquet file on GCS or the local filesystem. Globs are supported.
        table_name
            Optional table name
        kwargs
            Additional keyword arguments passed to `google.cloud.bigquery.LoadJobConfig`.

        Returns
        -------
        Table
            An Ibis table expression

        """
        return self._read_file(
            path,
            table_name=table_name,
            job_config=bq.LoadJobConfig(
                source_format=bq.SourceFormat.PARQUET, **kwargs
            ),
        )

    def read_csv(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Read CSV data into a BigQuery table.

        Parameters
        ----------
        path
            Path to a CSV file on GCS or the local filesystem. Globs are supported.
        table_name
            Optional table name
        kwargs
            Additional keyword arguments passed to
            `google.cloud.bigquery.LoadJobConfig`.

        Returns
        -------
        Table
            An Ibis table expression

        """
        job_config = bq.LoadJobConfig(
            source_format=bq.SourceFormat.CSV,
            autodetect=True,
            skip_leading_rows=1,
            **kwargs,
        )
        return self._read_file(path, table_name=table_name, job_config=job_config)

    def read_json(
        self, path: str | Path, table_name: str | None = None, **kwargs: Any
    ) -> ir.Table:
        """Read newline-delimited JSON data into a BigQuery table.

        Parameters
        ----------
        path
            Path to a newline-delimited JSON file on GCS or the local
            filesystem. Globs are supported.
        table_name
            Optional table name
        kwargs
            Additional keyword arguments passed to
            `google.cloud.bigquery.LoadJobConfig`.

        Returns
        -------
        Table
            An Ibis table expression

        """
        job_config = bq.LoadJobConfig(
            source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=True,
            **kwargs,
        )
        return self._read_file(path, table_name=table_name, job_config=job_config)

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
        location: str | None = None,
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
        location
            Default location for BigQuery objects.

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
                location=location,
            )

        self.client.default_query_job_config = bq.QueryJobConfig(
            use_legacy_sql=False, allow_large_results=True
        )

        if storage_client is not None:
            self.storage_client = storage_client
        else:
            self.storage_client = bqstorage.BigQueryReadClient(
                credentials=credentials,
                client_info=_create_client_info_gapic(application_name),
            )

        self.partition_column = partition_column

    def disconnect(self) -> None:
        self.client.close()

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
        properties = [
            sge.Property(this=sg.to_identifier(name), value=sge.convert(value))
            for name, value in (options or {}).items()
        ]

        if collate is not None:
            properties.append(
                sge.CollateProperty(this=sge.convert(collate), default=True)
            )

        stmt = sge.Create(
            kind="SCHEMA",
            this=sg.table(name, db=database),
            exists=force,
            properties=sge.Properties(expressions=properties),
        )

        self.raw_sql(stmt.sql(self.name))

    def drop_schema(
        self,
        name: str,
        database: str | None = None,
        force: bool = False,
        cascade: bool = False,
    ) -> None:
        """Drop a BigQuery dataset."""
        stmt = sge.Drop(
            kind="SCHEMA",
            this=sg.table(name, db=database),
            exists=force,
            cascade=cascade,
        )

        self.raw_sql(stmt.sql(self.name))

    def table(
        self, name: str, database: str | None = None, schema: str | None = None
    ) -> ir.Table:
        if database is not None and schema is None:
            raise com.IbisInputError(
                f"The {self.name} backend cannot return a table expression using only a "
                "`database` specifier. Include a `schema` argument."
            )

        table = sg.parse_one(name, into=sge.Table, read=self.name)

        # table.catalog will be the empty string
        if table.catalog:
            if database is not None:
                raise com.IbisInputError(
                    "Cannot specify database both in the table name and as an argument"
                )
            else:
                database = table.catalog

        # table.db will be the empty string
        if table.db:
            if schema is not None:
                raise com.IbisInputError(
                    "Cannot specify schema both in the table name and as an argument"
                )
            else:
                schema = table.db

        if database is not None and schema is None:
            database = sg.parse_one(database, into=sge.Table, read=self.name)
            database.args["quoted"] = False
            database = database.sql(dialect=self.name)
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sge.Table, read=self.name)
            database.args["quoted"] = False
            database = database.sql(dialect=self.name)
        else:
            database = (
                sg.table(schema, db=database, quoted=False).sql(dialect=self.name)
                or None
            )

        project, dataset = self._parse_project_and_dataset(database)

        bq_table = self.client.get_table(
            bq.TableReference(
                bq.DatasetReference(project=project, dataset_id=dataset),
                table.name,
            )
        )

        node = ops.DatabaseTable(
            table.name,
            schema=schema_from_bigquery_table(bq_table),
            source=self,
            namespace=ops.Namespace(schema=dataset, database=project),
        )
        table_expr = node.to_expr()
        return rename_partitioned_column(table_expr, bq_table, self.partition_column)

    def _make_session(self) -> tuple[str, str]:
        if (
            self._session_dataset is None
            and (client := getattr(self, "client", None)) is not None
        ):
            job_config = bq.QueryJobConfig(use_query_cache=False)
            query = client.query(
                "SELECT 1", job_config=job_config, project=self.billing_project
            )
            query.result()

            self._session_dataset = bq.DatasetReference(
                project=query.destination.project,
                dataset_id=query.destination.dataset_id,
            )

    def _metadata(self, query: str) -> Iterator[tuple[name, dt.DataType]]:
        self._make_session()

        job = self.client.query(
            query,
            job_config=bq.QueryJobConfig(dry_run=True, use_query_cache=False),
            project=self.billing_project,
        )
        return (
            (f.name, BigQuerySchema._dtype_from_bigquery_field(f)) for f in job.schema
        )

    def _execute(self, stmt, query_parameters=None):
        self._make_session()

        job_config = bq.job.QueryJobConfig(query_parameters=query_parameters or [])
        query = self.client.query(
            stmt, job_config=job_config, project=self.billing_project
        )
        query.result()  # blocks until finished
        return BigQueryCursor(query)

    def _to_sqlglot(
        self,
        expr: ir.Expr,
        limit: str | None = None,
        params: Mapping[ir.Expr, Any] | None = None,
        **kwargs,
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
        kwargs
            Keyword arguments passed to the compiler

        Returns
        -------
        Any
            The output of compilation. The type of this value depends on the
            backend.

        """
        self._make_session()
        self._define_udf_translation_rules(expr)
        sql = super()._to_sqlglot(expr, limit=limit, params=params, **kwargs)

        query = sql.transform(
            _qualify_memtable,
            dataset=getattr(self._session_dataset, "dataset_id", None),
            project=getattr(self._session_dataset, "project", None),
        ).transform(_remove_null_ordering_from_unsupported_window)
        return query

    def raw_sql(self, query: str, params=None):
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
        with contextlib.suppress(AttributeError):
            query = query.sql(self.dialect)
        return self._execute(query, query_parameters=query_parameters)

    @property
    def current_database(self) -> str:
        return self.data_project

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

    def compile(
        self, expr: ir.Expr, limit: str | None = None, params=None, **kwargs: Any
    ):
        """Compile an Ibis expression to a SQL string."""
        query = self._to_sqlglot(expr, limit=limit, params=params, **kwargs)
        udf_sources = []
        for udf_node in expr.op().find(ops.ScalarUDF):
            compile_func = getattr(
                self, f"_compile_{udf_node.__input_type__.name.lower()}_udf"
            )
            if sql := compile_func(udf_node):
                udf_sources.append(sql.sql(self.name, pretty=True))

        sql = ";\n".join([*udf_sources, query.sql(dialect=self.name, pretty=True)])
        self._log(sql)
        return sql

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

    def insert(
        self,
        table_name: str,
        obj: pd.DataFrame | ir.Table | list | dict,
        schema: str | None = None,
        database: str | None = None,
        overwrite: bool = False,
    ):
        """Insert data into a table.

        Parameters
        ----------
        table_name
            The name of the table to which data needs will be inserted
        obj
            The source data or expression to insert
        schema
            The name of the schema that the table is located in
        database
            Name of the attached database that the table is located in.
        overwrite
            If `True` then replace existing contents of table

        """
        return super().insert(
            table_name,
            obj,
            schema=schema if schema is not None else self.current_schema,
            database=database if database is not None else self.current_database,
            overwrite=overwrite,
        )

    def fetch_from_cursor(self, cursor, schema):
        from ibis.backends.bigquery.converter import BigQueryPandasData

        arrow_t = self._cursor_to_arrow(cursor)
        df = arrow_t.to_pandas(timestamp_as_object=True)
        return BigQueryPandasData.convert_table(df, schema)

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
        self._register_in_memory_tables(expr)
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

        self._register_in_memory_tables(expr)
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

    def _gen_udf_name(self, name: str, schema: Optional[str]) -> str:
        func = ".".join(filter(None, (schema, name)))
        if "." in func:
            return ".".join(f"`{part}`" for part in func.split("."))
        return func

    def get_schema(self, name, schema: str | None = None, database: str | None = None):
        table_ref = bq.TableReference(
            bq.DatasetReference(
                project=database or self.data_project,
                dataset_id=schema or self.current_schema,
            ),
            name,
        )
        return schema_from_bigquery_table(self.client.get_table(table_ref))

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
            raise com.com.IbisInputError(
                f"{self.name} cannot list tables only using `database` specifier. "
                "Include a `schema` argument."
            )
        elif database is None and schema is not None:
            database = sg.parse_one(schema, into=sge.Table, read=self.name)
            database.args["quoted"] = False
            database = database.sql(dialect=self.name)
        else:
            database = (
                sg.table(schema, db=database, quoted=False).sql(dialect=self.name)
                or None
            )

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
        temp: bool = False,
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
            Whether the table is temporary
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

        if isinstance(obj, ir.Table) and schema is not None:
            if not schema.equals(obj.schema()):
                raise com.IbisTypeError(
                    "Provided schema and Ibis table schema are incompatible. Please "
                    "align the two schemas, or provide only one of the two arguments."
                )

        project_id, dataset = self._parse_project_and_dataset(database)

        properties = []

        if default_collate is not None:
            properties.append(
                sge.CollateProperty(this=sge.convert(default_collate), default=True)
            )

        if partition_by is not None:
            properties.append(
                sge.PartitionedByProperty(
                    this=sge.Tuple(
                        expressions=list(map(sg.to_identifier, partition_by))
                    )
                )
            )

        if cluster_by is not None:
            properties.append(
                sge.Cluster(expressions=list(map(sg.to_identifier, cluster_by)))
            )

        properties.extend(
            sge.Property(this=sg.to_identifier(name), value=sge.convert(value))
            for name, value in (options or {}).items()
        )

        if obj is not None:
            import pyarrow as pa
            import pyarrow_hotfix  # noqa: F401

            if isinstance(obj, (pd.DataFrame, pa.Table)):
                obj = ibis.memtable(obj, schema=schema)

            self._register_in_memory_tables(obj)

        if temp:
            dataset = self._session_dataset.dataset_id
            if database is not None:
                raise com.IbisInputError("Cannot specify database for temporary table")
            database = self._session_dataset.project
        else:
            dataset = database or self.current_schema

        try:
            table = sg.parse_one(name, into=sge.Table, read="bigquery")
        except sg.ParseError:
            table = sg.table(name, db=dataset, catalog=project_id)
        else:
            if table.args["db"] is None:
                table.args["db"] = dataset

            if table.args["catalog"] is None:
                table.args["catalog"] = project_id

        column_defs = [
            sge.ColumnDef(
                this=name,
                kind=BigQueryType.from_ibis(typ),
                constraints=(
                    None
                    if typ.nullable or typ.is_array()
                    else [sge.ColumnConstraint(kind=sge.NotNullColumnConstraint())]
                ),
            )
            for name, typ in (schema or {}).items()
        ]

        stmt = sge.Create(
            kind="TABLE",
            this=sge.Schema(this=table, expressions=column_defs or None),
            replace=overwrite,
            properties=sge.Properties(expressions=properties),
            expression=None if obj is None else self.compile(obj),
        )

        sql = stmt.sql(self.name)

        self.raw_sql(sql)
        return self.table(table.name, schema=table.db, database=table.catalog)

    def drop_table(
        self,
        name: str,
        *,
        schema: str | None = None,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        stmt = sge.Drop(
            kind="TABLE",
            this=sg.table(
                name,
                db=schema or self.current_schema,
                catalog=database or self.billing_project,
            ),
            exists=force,
        )
        self.raw_sql(stmt.sql(self.name))

    def create_view(
        self,
        name: str,
        obj: ir.Table,
        *,
        schema: str | None = None,
        database: str | None = None,
        overwrite: bool = False,
    ) -> ir.Table:
        stmt = sge.Create(
            kind="VIEW",
            this=sg.table(
                name,
                db=schema or self.current_schema,
                catalog=database or self.billing_project,
            ),
            expression=self.compile(obj),
            replace=overwrite,
        )
        self._register_in_memory_tables(obj)
        self.raw_sql(stmt.sql(self.name))
        return self.table(name, schema=schema, database=database)

    def drop_view(
        self,
        name: str,
        *,
        schema: str | None = None,
        database: str | None = None,
        force: bool = False,
    ) -> None:
        stmt = sge.Drop(
            kind="VIEW",
            this=sg.table(
                name,
                db=schema or self.current_schema,
                catalog=database or self.billing_project,
            ),
            exists=force,
        )
        self.raw_sql(stmt.sql(self.name))

    def _load_into_cache(self, name, expr):
        self.create_table(name, expr, schema=expr.schema(), temp=True)

    def _clean_up_cached_table(self, op):
        self.drop_table(
            op.name,
            schema=self._session_dataset.dataset_id,
            database=self._session_dataset.project,
        )

    def _get_udf_source(self, udf_node: ops.ScalarUDF):
        name = type(udf_node).__name__
        type_mapper = self.compiler.udf_type_mapper

        body = PythonToJavaScriptTranslator(udf_node.__func__).compile()
        config = udf_node.__config__
        libraries = config.get("libraries", [])

        signature = [
            sge.ColumnDef(
                this=sg.to_identifier(name),
                kind=type_mapper.from_ibis(param.annotation.pattern.dtype),
            )
            for name, param in udf_node.__signature__.parameters.items()
        ]

        lines = ['"""']

        if config.get("strict", True):
            lines.append('"use strict";')

        lines += [
            body,
            "",
            f"return {udf_node.__func_name__}({', '.join(udf_node.argnames)});",
            '"""',
        ]

        func = sge.Create(
            kind="FUNCTION",
            this=sge.UserDefinedFunction(
                this=sg.to_identifier(name), expressions=signature, wrapped=True
            ),
            # not exactly what I had in mind, but it works
            #
            # quoting is too simplistic to handle multiline strings
            expression=sge.Var(this="\n".join(lines)),
            exists=False,
            properties=sge.Properties(
                expressions=[
                    sge.TemporaryProperty(),
                    sge.ReturnsProperty(this=type_mapper.from_ibis(udf_node.dtype)),
                    sge.StabilityProperty(
                        this="IMMUTABLE" if config.get("determinism") else "VOLATILE"
                    ),
                    sge.LanguageProperty(this=sg.to_identifier("js")),
                ]
                + [
                    sge.Property(
                        this=sg.to_identifier("library"),
                        value=self.compiler.f.array(*libraries),
                    )
                ]
                * bool(libraries)
            ),
        )

        return func

    def _compile_builtin_udf(self, udf_node: ops.ScalarUDF) -> None:
        """No op."""

    def _compile_python_udf(self, udf_node: ops.ScalarUDF) -> None:
        return self._get_udf_source(udf_node)

    def _compile_pyarrow_udf(self, udf_node: ops.ScalarUDF) -> None:
        raise NotImplementedError("PyArrow UDFs are not supported in BigQuery")

    def _compile_pandas_udf(self, udf_node: ops.ScalarUDF) -> str:
        raise NotImplementedError("Pandas UDFs are not supported in BigQuery")

    def _register_udfs(self, expr: ir.Expr) -> None:
        """No op because UDFs made with CREATE TEMPORARY FUNCTION must be followed by a query."""

    @contextlib.contextmanager
    def _safe_raw_sql(self, *args, **kwargs):
        yield self.raw_sql(*args, **kwargs)


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
