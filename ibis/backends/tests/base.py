import abc
import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd
import pandas.testing as tm

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
            return (
                -(np.sign(series)) * np.ceil(-(series.abs()) - 0.5)
            ).astype(np.int64)
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
    bool_is_int = False

    def __init__(self, data_directory: Path) -> None:
        self.connection = self.connect(data_directory)

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
        tm.assert_frame_equal(left, right, *args, **kwargs)

    @staticmethod
    def default_series_rename(
        series: pd.Series, name: str = 'tmp'
    ) -> pd.Series:
        return series.rename(name)

    @staticmethod
    def greatest(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @staticmethod
    def least(
        f: Callable[..., ir.ValueExpr], *args: ir.ValueExpr
    ) -> ir.ValueExpr:
        return f(*args)

    @property
    def functional_alltypes(self) -> ir.TableExpr:
        t = self.connection.table('functional_alltypes')
        if self.bool_is_int:
            return t.mutate(bool_col=t.bool_col == 1)
        return t

    @property
    def batting(self) -> ir.TableExpr:
        return self.connection.table('batting')

    @property
    def awards_players(self) -> ir.TableExpr:
        return self.connection.table('awards_players')

    @property
    def geo(self) -> Optional[ir.TableExpr]:
        if 'geo' in self.connection.list_tables():
            return self.connection.table('geo')
        return None

    @property
    def api(self):
        return self.connection

    def make_context(
        self, params: Optional[Mapping[ir.ValueExpr, Any]] = None
    ):
        return self.api.compiler.make_context(params=params)
