from __future__ import annotations

import itertools
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pytest
import sqlalchemy as sa

import ibis
import ibis.expr.types as ir
from ibis.backends.conftest import TEST_TABLES, init_database
from ibis.backends.tests.base import BackendTest, RoundAwayFromZero


class TestConf(BackendTest, RoundAwayFromZero):
    supports_arrays = False
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    check_dtype = False
    returned_timestamp_unit = 's'
    supports_structs = False

    @staticmethod
    def _load_data(
        data_dir: Path, script_dir: Path, database: str | None = None, **_: Any
    ) -> None:
        """Load test data into a SQLite backend instance.

        Parameters
        ----------
        data_dir
            Location of test data
        script_dir
            Location of scripts defining schemas
        """
        if database is None:
            database = Path(
                os.environ.get(
                    "IBIS_TEST_SQLITE_DATABASE", data_dir / "ibis_testing.db"
                )
            )

        with open(script_dir / 'schema' / 'sqlite.sql') as schema:
            init_database(
                url=sa.engine.make_url("sqlite://"),
                database=str(database.absolute()),
                schema=schema,
            )

        with tempfile.TemporaryDirectory() as tempdir:
            for table in TEST_TABLES:
                basename = f"{table}.csv"
                with Path(tempdir).joinpath(basename).open("w") as f:
                    with data_dir.joinpath(basename).open("r") as lines:
                        # skip the first line
                        f.write("".join(itertools.islice(lines, 1, None)))
            subprocess.run(
                ["sqlite3", database],
                input="\n".join(
                    [
                        ".separator ,",
                        *(
                            f".import {str(path.absolute())!r} {path.stem}"
                            for path in Path(tempdir).glob("*.csv")
                        ),
                    ]
                ).encode(),
            )

    @staticmethod
    def connect(data_directory: Path):
        path = Path(
            os.environ.get(
                'IBIS_TEST_SQLITE_DATABASE', data_directory / 'ibis_testing.db'
            )
        )
        return ibis.sqlite.connect(str(path))  # type: ignore

    @property
    def functional_alltypes(self) -> ir.Table:
        t = super().functional_alltypes
        return t.mutate(timestamp_col=t.timestamp_col.cast('timestamp'))


@pytest.fixture(scope="session")
def dbpath(data_directory):
    default = str(data_directory / 'ibis_testing.db')
    return os.environ.get('IBIS_TEST_SQLITE_DATABASE', default)


@pytest.fixture(scope="session")
def con(tmp_path_factory, data_directory, script_directory, worker_id):
    return TestConf.load_data(
        data_directory,
        script_directory,
        tmp_path_factory,
        worker_id,
    ).connect(data_directory)


@pytest.fixture(scope="session")
def dialect():
    import sqlalchemy as sa

    return sa.dialects.sqlite.dialect()


@pytest.fixture(scope="session")
def translate(dialect):
    from ibis.backends.sqlite import Backend

    context = Backend.compiler.make_context()
    return lambda expr: str(
        Backend.compiler.translator_class(expr, context)
        .get_result()
        .compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def sqla_compile(dialect):
    return lambda expr: str(
        expr.compile(dialect=dialect, compile_kwargs={'literal_binds': True})
    )


@pytest.fixture(scope="session")
def alltypes(con):
    return con.table("functional_alltypes")


@pytest.fixture(scope="session")
def alltypes_sqla(alltypes):
    return alltypes.op().sqla_table


@pytest.fixture(scope="session")
def df(alltypes):
    return alltypes.execute()
