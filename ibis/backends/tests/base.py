from __future__ import annotations

import abc
import concurrent.futures
import inspect
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
import toolz
from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    import ibis.expr.types as ir


# TODO: Merge into BackendTest, #2564
class RoundingConvention:
    @staticmethod
    @abc.abstractmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        """Round a series to `decimals` number of decimal values."""


# TODO: Merge into BackendTest, #2564
class RoundAwayFromZero(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        if not decimals:
            return (-(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)).astype(
                np.int64
            )
        return series.round(decimals=decimals)


# TODO: Merge into BackendTest, #2564
class RoundHalfToEven(RoundingConvention):
    @staticmethod
    def round(series: pd.Series, decimals: int = 0) -> pd.Series:
        result = series.round(decimals=decimals)
        return result if decimals else result.astype(np.int64)


# TODO: Merge into BackendTest, #2564
class UnorderedComparator:
    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        left = left.sort_values().reset_index(drop=True)
        right = right.sort_values().reset_index(drop=True)
        return super().assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        columns = list(set(left.columns) & set(right.columns))
        left = left.sort_values(by=columns)
        right = right.sort_values(by=columns)
        return super().assert_frame_equal(left, right, *args, **kwargs)


class BackendTest(abc.ABC):
    check_dtype = True
    check_names = True
    supports_arrays = True
    supports_arrays_outside_of_select = supports_arrays
    supports_window_operations = True
    supports_divide_by_zero = False
    returned_timestamp_unit = "us"
    supported_to_timestamp_units = {"s", "ms", "us"}
    supports_floating_modulus = True
    native_bool = True
    supports_structs = True
    supports_json = True
    supports_map = False  # basically nothing does except trino and snowflake
    reduction_tolerance = 1e-7
    default_identifier_case_fn = staticmethod(toolz.identity)
    stateful = True
    service_name = None
    supports_tpch = False

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

    def _transform_tpch_sql(self, parsed):
        return parsed

    def _load_data(self, **_: Any) -> None:
        """Load test data into a backend."""
        with self.connection.begin() as con:
            for stmt in self.ddl_script:
                con.exec_driver_sql(stmt)

    def stateless_load(self, **kw):
        self.preload()
        self._load_data(**kw)

        if self.supports_tpch:
            self.load_tpch()

    def stateful_load(self, fn, **kw):
        if not fn.exists():
            self.stateless_load(**kw)
            fn.touch()

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

        fn = root_tmp_dir / (getattr(cls, "service_name", None) or cls.name())
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
        kwargs.setdefault("check_dtype", cls.check_dtype)
        kwargs.setdefault("check_names", cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        kwargs.setdefault("check_dtype", cls.check_dtype)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(series: pd.Series, name: str = "tmp") -> pd.Series:
        return series.rename(name)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = self.connection.table(
            self.default_identifier_case_fn("functional_alltypes")
        )
        if not self.native_bool:
            return t.mutate(bool_col=t.bool_col == 1)
        return t

    @property
    def batting(self) -> ir.Table:
        return self.connection.table(self.default_identifier_case_fn("batting"))

    @property
    def awards_players(self) -> ir.Table:
        return self.connection.table(self.default_identifier_case_fn("awards_players"))

    @property
    def diamonds(self) -> ir.Table:
        return self.connection.table(self.default_identifier_case_fn("diamonds"))

    @property
    def astronauts(self) -> ir.Table:
        return self.connection.table(self.default_identifier_case_fn("astronauts"))

    @property
    def geo(self) -> ir.Table | None:
        name = self.default_identifier_case_fn("geo")
        if name in self.connection.list_tables():
            return self.connection.table(name)
        return None

    @property
    def struct(self) -> ir.Table | None:
        if self.supports_structs:
            return self.connection.table(self.default_identifier_case_fn("struct"))
        else:
            pytest.xfail(f"{self.name()} backend does not support struct types")

    @property
    def array_types(self) -> ir.Table | None:
        if self.supports_arrays:
            return self.connection.table(self.default_identifier_case_fn("array_types"))
        else:
            pytest.xfail(f"{self.name()} backend does not support array types")

    @property
    def json_t(self) -> ir.Table | None:
        from ibis import _

        if self.supports_json:
            return self.connection.table(
                self.default_identifier_case_fn("json_t")
            ).mutate(js=_.js.cast("json"))
        else:
            pytest.xfail(f"{self.name()} backend does not support json types")

    @property
    def map(self) -> ir.Table | None:
        if self.supports_map:
            return self.connection.table(self.default_identifier_case_fn("map"))
        else:
            pytest.xfail(f"{self.name()} backend does not support map types")

    @property
    def win(self) -> ir.Table | None:
        return self.connection.table(self.default_identifier_case_fn("win"))

    @property
    def api(self):
        return self.connection

    def make_context(self, params: Mapping[ir.Value, Any] | None = None):
        return self.api.compiler.make_context(params=params)

    @property
    def customer(self):
        return self._tpch_table("customer")

    @property
    def lineitem(self):
        return self._tpch_table("lineitem")

    @property
    def nation(self):
        return self._tpch_table("nation")

    @property
    def orders(self):
        return self._tpch_table("orders")

    @property
    def part(self):
        return self._tpch_table("part")

    @property
    def partsupp(self):
        return self._tpch_table("partsupp")

    @property
    def region(self):
        return self._tpch_table("region")

    @property
    def supplier(self):
        return self._tpch_table("supplier")

    def _tpch_table(self, name: str):
        if not self.supports_tpch:
            pytest.skip(f"{self.name()} backend does not support testing TPC-H")
        return self.connection.table(self.default_identifier_case_fn(name))


class ServiceBackendTest(BackendTest):
    data_volume = "/data"

    @property
    @abc.abstractmethod
    def test_files(self) -> Iterable[Path]:
        ...

    def preload(self):
        service = self.service_name
        data_volume = self.data_volume
        with concurrent.futures.ThreadPoolExecutor() as e:
            for fut in concurrent.futures.as_completed(
                e.submit(
                    subprocess.check_call,
                    [
                        "docker",
                        "compose",
                        "cp",
                        str(path),
                        f"{service}:{data_volume}/{path.name}",
                    ],
                )
                for path in self.test_files
            ):
                fut.result()
