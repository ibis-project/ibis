from __future__ import annotations

import abc
import concurrent.futures
import inspect
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pytest
from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    import ibis.expr.types as ir


PYTHON_SHORT_VERSION = f"{sys.version_info.major}{sys.version_info.minor}"

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
tm = pytest.importorskip("pandas.testing")


class BackendTest(abc.ABC):
    """
    The base class for managing configuration and data loading for a backend
    that does not require Docker for testing (this includes both in-process
    backends and cloud backends like Snowflake and BigQuery).
    """

    check_dtype: bool = True
    "Check that dtypes match when comparing Pandas Series"
    check_names: bool = True
    "Check that column name matches when comparing Pandas Series"
    supports_arrays: bool = True
    "Whether backend supports Arrays / Lists"
    returned_timestamp_unit: str = "us"
    native_bool: bool = True
    "Whether backend has native boolean types"
    supports_structs: bool = True
    "Whether backend supports Structs"
    supports_json: bool = True
    "Whether backend supports operating on JSON"
    supports_map: bool = False
    "Whether backend supports mappings (currently DuckDB, Snowflake, and Trino)"
    reduction_tolerance = 1e-7
    "Used for a single test in `test_aggregation.py`. You should not need to touch this."
    stateful = True
    "Whether special handling is needed for running a multi-process pytest run."
    supports_tpch: bool = False
    "Child class defines a `load_tpch` method that loads the required TPC-H tables into a connection."
    supports_tpcds: bool = False
    "Child class defines a `load_tpcds` method that loads the required TPC-DS tables into a connection."
    force_sort = False
    "Sort results before comparing against reference computation."
    rounding_method: Literal["away_from_zero", "half_to_even"] = "away_from_zero"
    "Name of round method to use for rounding test comparisons."
    driver_supports_multiple_statements: bool = False
    "Whether the driver supports executing multiple statements in a single call."
    tpc_absolute_tolerance: float | None = None
    "Absolute tolerance for floating point comparisons with pytest.approx in TPC correctness tests."

    @property
    @abc.abstractmethod
    def deps(self) -> Iterable[str]:
        """A list of dependencies that must be present to run tests."""

    @property
    def ddl_script(self) -> Iterator[str]:
        return filter(
            None,
            map(
                str.strip,
                self.script_dir.joinpath(f"{self.name()}.sql").read_text().split(";"),
            ),
        )

    @staticmethod
    def format_table(name: str) -> str:
        return name

    def __init__(self, *, data_dir: Path, tmpdir, worker_id, **kw) -> None:
        """
        Initializes the test class -- note that none of the arguments are
        required and will be provided by `pytest` or by fixtures defined in
        `ibis/backends/conftest.py`.

        data_dir
            Directory where test data resides (will be provided by the
            `data_dir` fixture in `ibis/backends/conftest.py`)
        tmpdir
            Pytest fixture providing a temporary directory location
        worker_id
            A unique identifier for each worker used for running test
            concurrently via e.g. `pytest -n auto`
        """
        self.connection = self.connect(tmpdir=tmpdir, worker_id=worker_id, **kw)
        self.data_dir = data_dir
        self.script_dir = data_dir.parent / "schema"

    def __str__(self):
        return f"<BackendTest {self.name()}>"

    @classmethod
    def name(cls) -> str:
        backend_tests_path = inspect.getmodule(cls).__file__
        return Path(backend_tests_path).resolve().parent.parent.name

    @staticmethod
    @abc.abstractmethod
    def connect(*, tmpdir, worker_id, **kw: Any):
        """Return a connection with data loaded from `data_dir`."""

    def _transform_tpc_sql(self, parsed, *, suite, leaves):
        return parsed

    def _load_data(self, **_: Any) -> None:
        """Load test data into a backend."""
        if self.driver_supports_multiple_statements:
            with self.connection._safe_raw_sql(";".join(self.ddl_script)):
                pass
        else:
            with self.connection.begin() as con:
                for stmt in self.ddl_script:
                    con.execute(stmt)

    def stateless_load(self, **kw):
        self.preload()
        self._load_data(**kw)

        if self.supports_tpch:
            self.load_tpch()
        if self.supports_tpcds:
            self.load_tpcds()

    def stateful_load(self, fn, **kw):
        if not fn.exists():
            self.stateless_load(**kw)
            fn.touch()

    def load_tpch(self) -> None:
        """Load TPC-H data."""
        self._load_tpc(suite="h", scale_factor="0.17")

    def load_tpcds(self) -> None:
        """Load TPC-DS data."""
        self._load_tpc(suite="ds", scale_factor="0.45")

    @classmethod
    def load_data(
        cls, data_dir: Path, tmpdir: Path, worker_id: str, **kw: Any
    ) -> BackendTest:
        """Load testdata from `data_dir`."""
        # handling for multi-processes pytest

        # get the temp directory shared by all workers
        root_tmp_dir = tmpdir.getbasetemp()
        if worker_id != "master":
            root_tmp_dir = root_tmp_dir.parent

        fn = root_tmp_dir / cls.name()
        with FileLock(f"{fn}.lock"):
            cls.skip_if_missing_deps()

            inst = cls(data_dir=data_dir, tmpdir=tmpdir, worker_id=worker_id, **kw)

            if inst.stateful:
                inst.stateful_load(fn, **kw)
            else:
                inst.stateless_load(**kw)
            inst.postload(tmpdir=tmpdir, worker_id=worker_id, **kw)
            return inst

    @classmethod
    def skip_if_missing_deps(cls) -> None:
        """Add an `importorskip` for any missing dependencies."""
        for dep in cls.deps:
            pytest.importorskip(dep)

    def preload(self):  # noqa: B027
        """Code to execute before loading data."""

    def postload(self, **_):  # noqa: B027
        """Code to execute after loading data."""

    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        """Compare two Pandas Series, optionally ignoring order, dtype, and column name.

        `force_sort`, `check_dtype`, and `check_names` are set as class-level variables.
        """
        if cls.force_sort:
            left = left.sort_values().reset_index(drop=True)
            right = right.sort_values().reset_index(drop=True)
        kwargs.setdefault("check_dtype", cls.check_dtype)
        kwargs.setdefault("check_names", cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        """Compare two Pandas DataFrames optionally ignoring order, and dtype.

        `force_sort`, and `check_dtype` are set as class-level variables.
        """
        if cls.force_sort:
            columns = list(set(left.columns) & set(right.columns))
            left = left.sort_values(by=columns)
            right = right.sort_values(by=columns)
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        kwargs.setdefault("check_dtype", cls.check_dtype)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @classmethod
    def round(cls, series: pd.Series, decimals: int = 0) -> pd.Series:
        return getattr(cls, cls.rounding_method)(series, decimals)

    @staticmethod
    def away_from_zero(series: pd.Series, decimals: int = 0) -> pd.Series:
        if not decimals:
            return (-(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)).astype(
                np.int64
            )
        return series.round(decimals=decimals)

    @staticmethod
    def half_to_even(series: pd.Series, decimals: int = 0) -> pd.Series:
        result = series.round(decimals=decimals)
        return result if decimals else result.astype(np.int64)

    @staticmethod
    def default_series_rename(series: pd.Series, name: str = "tmp") -> pd.Series:
        return series.rename(name)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = self.connection.table("functional_alltypes")
        if not self.native_bool:
            return t.mutate(bool_col=t.bool_col == 1)
        return t

    @property
    def batting(self) -> ir.Table:
        return self.connection.table("batting")

    @property
    def awards_players(self) -> ir.Table:
        return self.connection.table("awards_players")

    @property
    def diamonds(self) -> ir.Table:
        return self.connection.table("diamonds")

    @property
    def astronauts(self) -> ir.Table:
        return self.connection.table("astronauts")

    @property
    def geo(self) -> ir.Table | None:
        name = "geo"
        if name in self.connection.list_tables():
            return self.connection.table(name)
        return None

    @property
    def struct(self) -> ir.Table | None:
        if self.supports_structs:
            return self.connection.table("struct")
        else:
            pytest.xfail(f"{self.name()} backend does not support struct types")

    @property
    def array_types(self) -> ir.Table | None:
        if self.supports_arrays:
            return self.connection.table("array_types")
        else:
            pytest.xfail(f"{self.name()} backend does not support array types")

    @property
    def json_t(self) -> ir.Table | None:
        from ibis import _

        if self.supports_json:
            return self.connection.table("json_t").mutate(js=_.js.cast("json"))
        else:
            pytest.xfail(f"{self.name()} backend does not support json types")

    @property
    def map(self) -> ir.Table | None:
        if self.supports_map:
            return self.connection.table("map")
        else:
            pytest.xfail(f"{self.name()} backend does not support map types")

    @property
    def win(self) -> ir.Table | None:
        return self.connection.table("win")

    @property
    def api(self):
        return self.connection

    def _tpc_table(self, name: str, benchmark: Literal["h", "ds"]):
        if not getattr(self, f"supports_tpc{benchmark}"):
            pytest.skip(
                f"{self.name()} backend does not support testing TPC-{benchmark.upper()}"
            )
        return self.connection.table(name, database=f"tpc{benchmark}")

    def h(self, name: str) -> ir.Table:
        return self._tpc_table(name, "h")

    def ds(self, name: str) -> ir.Table:
        return self._tpc_table(name, "ds")

    def list_tpc_tables(self, suite: Literal["h", "ds"]) -> frozenset[str]:
        return frozenset(
            path.with_suffix("").name
            for path in self.data_dir.joinpath(f"tpc{suite}").rglob("*.parquet")
        )


class ServiceBackendTest(BackendTest):
    """Parent class to use for backend test configuration if backend requires a
    Docker container(s) in order to run locally.

    """

    service_name: str | None = None
    "Name of service defined in compose.yaml corresponding to backend."
    data_volume = "/data"
    "Data volume defined in compose.yaml corresponding to backend."

    @property
    @abc.abstractmethod
    def test_files(self) -> Iterable[Path]:
        """Returns an iterable of test files to load into a Docker container before testing."""
        ...

    def preload(self):
        """Use `docker compose cp` to copy all files from `test_files` into a container.

        `service_name` and `data_volume` are set as class-level variables.
        """
        service = self.service_name
        data_volume = self.data_volume
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(
                    subprocess.run,
                    [
                        "docker",
                        "compose",
                        "cp",
                        str(path),
                        f"{service}:{data_volume}/{path.name}",
                    ],
                    check=True,
                )
                for path in self.test_files
            ):
                fut.result()
