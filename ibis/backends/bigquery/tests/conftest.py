from __future__ import annotations

import concurrent.futures
import contextlib
import functools
import io
import os
import pathlib
from pathlib import Path
from typing import Any, Mapping

import google.auth
from google.api_core.exceptions import NotFound
from google.cloud import bigquery as bq

import ibis
import ibis.expr.datatypes as dt
from ibis.backends.bigquery import EXTERNAL_DATA_SCOPES, Backend
from ibis.backends.bigquery.datatypes import ibis_type_to_bigquery_type
from ibis.backends.conftest import TEST_TABLES
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero, UnorderedComparator
from ibis.backends.tests.data import json_types, non_null_array_types, struct_types, win

DATASET_ID = "ibis_gbq_testing"
DEFAULT_PROJECT_ID = "ibis-gbq"
PROJECT_ID_ENV_VAR = "GOOGLE_BIGQUERY_PROJECT_ID"


@functools.singledispatch
def ibis_type_to_bq_field(typ: dt.DataType) -> Mapping[str, Any]:
    raise NotImplementedError(typ)


@ibis_type_to_bq_field.register(dt.DataType)
def _(typ: dt.DataType) -> Mapping[str, Any]:
    return {"field_type": ibis_type_to_bigquery_type(typ)}


@ibis_type_to_bq_field.register(dt.Array)
def _(typ: dt.Array) -> Mapping[str, Any]:
    return {
        "field_type": ibis_type_to_bigquery_type(typ.value_type),
        "mode": "REPEATED",
    }


@ibis_type_to_bq_field.register(dt.Struct)
def _(typ: dt.Struct) -> Mapping[str, Any]:
    return {
        "field_type": "RECORD",
        "mode": "NULLABLE" if typ.nullable else "REQUIRED",
        "fields": ibis_schema_to_bq_schema(ibis.schema(typ.fields)),
    }


def ibis_schema_to_bq_schema(schema):
    return [
        bq.SchemaField(
            name.replace(":", "").replace(" ", "_"),
            **ibis_type_to_bq_field(typ),
        )
        for name, typ in ibis.schema(schema).items()
    ]


class TestConf(UnorderedComparator, BackendTest, RoundAwayFromZero):
    """Backend-specific class with information for testing."""

    # These were moved from TestConf for use in common test suite.
    # TODO: Indicate RoundAwayFromZero and UnorderedComparator.
    # https://github.com/ibis-project/ibis-bigquery/issues/30
    supports_divide_by_zero = True
    supports_floating_modulus = False
    returned_timestamp_unit = "us"
    supports_structs = True
    supports_json = True
    check_names = False

    @staticmethod
    def format_table(name: str) -> str:
        return f"{DATASET_ID}.{name}"

    @staticmethod
    def _load_data(data_dir: Path, script_dir: Path, **_: Any) -> None:
        """Load test data into a BigQuery instance."""

        credentials, default_project_id = google.auth.default(
            scopes=EXTERNAL_DATA_SCOPES
        )

        project_id = (
            os.environ.get(PROJECT_ID_ENV_VAR, default_project_id) or DEFAULT_PROJECT_ID
        )

        client = bq.Client(project=project_id, credentials=credentials)

        testing_dataset = bq.DatasetReference(project_id, DATASET_ID)

        with contextlib.suppress(NotFound):
            client.create_dataset(testing_dataset, exists_ok=True)

        # day partitioning
        functional_alltypes_parted = bq.Table(
            bq.TableReference(testing_dataset, "functional_alltypes_parted")
        )
        functional_alltypes_parted.require_partition_filter = False
        functional_alltypes_parted.time_partitioning = bq.TimePartitioning(
            type_=bq.TimePartitioningType.DAY
        )

        # ingestion timestamp partitioning
        timestamp_table = bq.Table(
            bq.TableReference(testing_dataset, "timestamp_column_parted")
        )
        timestamp_table.schema = ibis_schema_to_bq_schema(
            dict(
                my_timestamp_parted_col="timestamp", string_col="string", int_col="int"
            )
        )
        timestamp_table.time_partitioning = bq.TimePartitioning(
            field="my_timestamp_parted_col"
        )
        client.create_table(timestamp_table, exists_ok=True)

        # ingestion date partitioning
        date_table = bq.Table(bq.TableReference(testing_dataset, "date_column_parted"))
        date_table.schema = ibis_schema_to_bq_schema(
            dict(my_date_parted_col="date", string_col="string", int_col="int")
        )
        date_table.time_partitioning = bq.TimePartitioning(field="my_date_parted_col")
        client.create_table(date_table, exists_ok=True)

        write_disposition = bq.WriteDisposition.WRITE_TRUNCATE
        make_job = lambda func, *a, **kw: func(*a, **kw).result()

        futures = []
        # 10 is because of urllib3 connection pool size
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as e:
            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_dataframe,
                    struct_types,
                    bq.TableReference(testing_dataset, "struct"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        schema=ibis_schema_to_bq_schema(
                            dict(abc="struct<a: float64, b: string, c: int64>")
                        ),
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_dataframe,
                    non_null_array_types.drop(columns=["multi_dim"]),
                    bq.TableReference(testing_dataset, "array_types"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        schema=ibis_schema_to_bq_schema(
                            dict(
                                x="array<int64>",
                                y="array<string>",
                                z="array<float64>",
                                grouper="string",
                                scalar_column="float64",
                            )
                        ),
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_file,
                    io.BytesIO(data_dir.joinpath("struct_table.avro").read_bytes()),
                    bq.TableReference(testing_dataset, "struct_table"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        source_format=bq.SourceFormat.AVRO,
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_file,
                    io.BytesIO(
                        data_dir.joinpath("functional_alltypes.csv").read_bytes()
                    ),
                    functional_alltypes_parted,
                    job_config=bq.LoadJobConfig(
                        schema=ibis_schema_to_bq_schema(
                            TEST_TABLES["functional_alltypes"]
                        ),
                        write_disposition=write_disposition,
                        source_format=bq.SourceFormat.CSV,
                        skip_leading_rows=1,
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_file,
                    io.StringIO(
                        "\n".join(
                            [
                                """{"string_col": "1st value", "numeric_col": 0.999999999}""",
                                """{"string_col": "2nd value", "numeric_col": 0.000000002}""",
                            ]
                        )
                    ),
                    bq.TableReference(testing_dataset, "numeric_table"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        schema=ibis_schema_to_bq_schema(
                            dict(string_col="string", numeric_col="decimal(38, 9)")
                        ),
                        source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_dataframe,
                    win,
                    bq.TableReference(testing_dataset, "win"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        schema=ibis_schema_to_bq_schema(
                            dict(g="string", x="int64", y="int64")
                        ),
                    ),
                )
            )

            futures.append(
                e.submit(
                    make_job,
                    client.load_table_from_file,
                    io.StringIO(
                        "\n".join(f"{{\"js\": {row}}}" for row in json_types.js)
                    ),
                    bq.TableReference(testing_dataset, "json_t"),
                    job_config=bq.LoadJobConfig(
                        write_disposition=write_disposition,
                        schema=ibis_schema_to_bq_schema(dict(js="json")),
                        source_format=bq.SourceFormat.NEWLINE_DELIMITED_JSON,
                    ),
                )
            )

            for table, schema in TEST_TABLES.items():
                futures.append(
                    e.submit(
                        make_job,
                        client.load_table_from_file,
                        io.BytesIO(data_dir.joinpath(f"{table}.csv").read_bytes()),
                        bq.TableReference(testing_dataset, table),
                        job_config=bq.LoadJobConfig(
                            schema=ibis_schema_to_bq_schema(schema),
                            write_disposition=bq.WriteDisposition.WRITE_TRUNCATE,
                            source_format=bq.SourceFormat.CSV,
                            skip_leading_rows=1,
                        ),
                    )
                )

            for fut in concurrent.futures.as_completed(futures):
                fut.result()

    @staticmethod
    def connect(data_directory: pathlib.Path) -> Backend:
        """Connect to the test project and dataset."""
        credentials, default_project_id = google.auth.default(
            scopes=EXTERNAL_DATA_SCOPES
        )

        project_id = (
            os.environ.get(PROJECT_ID_ENV_VAR, default_project_id) or DEFAULT_PROJECT_ID
        )
        return ibis.bigquery.connect(
            project_id=project_id,
            dataset_id=DATASET_ID,
            credentials=credentials,
        )
