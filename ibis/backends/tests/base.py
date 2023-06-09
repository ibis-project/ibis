from __future__ import annotations

import abc
import concurrent.futures
import inspect
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, NamedTuple

import numpy as np
import pandas as pd
import pandas.testing as tm
import pytest
from filelock import FileLock

if TYPE_CHECKING:
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
    additional_skipped_operations = frozenset()
    supports_divide_by_zero = False
    returned_timestamp_unit = 'us'
    supported_to_timestamp_units = {'s', 'ms', 'us'}
    supports_floating_modulus = True
    native_bool = True
    supports_structs = True
    supports_json = True
    supports_map = False  # basically nothing does except trino and snowflake
    reduction_tolerance = 1e-7

    @staticmethod
    def format_table(name: str) -> str:
        return name

    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)
        self.data_directory = data_directory

    def __str__(self):
        return f'<BackendTest {self.name()}>'

    @classmethod
    def name(cls) -> str:
        backend_tests_path = inspect.getmodule(cls).__file__
        return Path(backend_tests_path).resolve().parent.parent.name

    @staticmethod
    @abc.abstractmethod
    def connect(data_directory: Path):
        """Return a connection with data loaded from `data_directory`."""

    @staticmethod  # noqa: B027
    def _load_data(data_directory: Path, script_directory: Path, **kwargs: Any) -> None:
        """Load test data into a backend.

        Default implementation is a no-op.
        """

    @classmethod
    def load_data(
        cls, data_dir: Path, script_dir: Path, tmpdir: Path, worker_id: str, **kw: Any
    ) -> None:
        """Load testdata from `data_directory` into the backend using scripts
        in `script_directory`."""
        # handling for multi-processes pytest

        # get the temp directory shared by all workers
        root_tmp_dir = tmpdir.getbasetemp()
        if worker_id != "master":
            root_tmp_dir = root_tmp_dir.parent

        fn = root_tmp_dir / f"lockfile_{cls.name()}"
        with FileLock(f"{fn}.lock"):
            if not fn.exists():
                cls.preload(data_dir)
                cls._load_data(data_dir, script_dir, **kw)
                fn.touch()
        return cls(data_dir)

    @classmethod  # noqa: B027
    def preload(cls, data_dir: Path):
        """Code to execute before loading data."""

    @classmethod
    def assert_series_equal(
        cls, left: pd.Series, right: pd.Series, *args: Any, **kwargs: Any
    ) -> None:
        kwargs.setdefault('check_dtype', cls.check_dtype)
        kwargs.setdefault('check_names', cls.check_names)
        tm.assert_series_equal(left, right, *args, **kwargs)

    @classmethod
    def assert_frame_equal(
        cls, left: pd.DataFrame, right: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> None:
        left = left.reset_index(drop=True)
        right = right.reset_index(drop=True)
        kwargs.setdefault('check_dtype', cls.check_dtype)
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(series: pd.Series, name: str = 'tmp') -> pd.Series:
        return series.rename(name)

    @property
    def functional_alltypes(self) -> ir.Table:
        t = self.connection.table('functional_alltypes')
        if not self.native_bool:
            return t.mutate(bool_col=t.bool_col == 1)
        return t

    @property
    def batting(self) -> ir.Table:
        return self.connection.table('batting')

    @property
    def awards_players(self) -> ir.Table:
        return self.connection.table('awards_players')

    @property
    def diamonds(self) -> ir.Table:
        return self.connection.table('diamonds')

    @property
    def geo(self) -> ir.Table | None:
        if 'geo' in self.connection.list_tables():
            return self.connection.table('geo')
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

    def make_context(self, params: Mapping[ir.Value, Any] | None = None):
        return self.api.compiler.make_context(params=params)


class ServiceSpec(NamedTuple):
    name: str
    data_volume: str
    files: Iterable[Path]


class ServiceBackendTest(BackendTest):
    @classmethod
    @abc.abstractmethod
    def service_spec(data_dir: Path) -> ServiceSpec:
        ...

    @classmethod
    def preload(cls, data_dir: Path):
        spec = cls.service_spec(data_dir)
        service = spec.name
        data_volume = spec.data_volume
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
                for path in spec.files
            ):
                fut.result()
