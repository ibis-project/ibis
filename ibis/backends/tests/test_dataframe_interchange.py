from __future__ import annotations

import pyarrow as pa
import pytest
from packaging.version import parse as vparse

pytestmark = pytest.mark.skipif(
    vparse(pa.__version__) < vparse("12"), reason="pyarrow >= 12 required"
)


@pytest.mark.notimpl(["druid"])
def test_dataframe_interchange_no_execute(con, alltypes, mocker):
    t = alltypes.select("int_col", "double_col", "string_col")
    pa_df = t.to_pyarrow().__dataframe__()

    to_pyarrow = mocker.spy(con, "to_pyarrow")

    df = t.__dataframe__()

    # Schema metadata
    assert df.num_columns() == pa_df.num_columns()
    assert df.column_names() == pa_df.column_names()

    # Column access
    assert df.get_column(0).dtype == pa_df.get_column(0).dtype
    assert (
        df.get_column_by_name("int_col").dtype
        == pa_df.get_column_by_name("int_col").dtype
    )
    res = [c.dtype for c in df.get_columns()]
    sol = [c.dtype for c in pa_df.get_columns()]
    assert res == sol
    col = df.get_column(0)
    with pytest.raises(
        TypeError, match="only works on a column with categorical dtype"
    ):
        col.describe_categorical  # noqa: B018

    # Subselection
    res = df.select_columns([1, 0])
    sol = pa_df.select_columns([1, 0])
    assert res.column_names() == sol.column_names()
    res = df.select_columns_by_name(["double_col", "int_col"])
    sol = pa_df.select_columns_by_name(["double_col", "int_col"])
    assert res.column_names() == sol.column_names()

    # Nested __dataframe__ access
    df2 = df.__dataframe__()
    pa_df2 = pa_df.__dataframe__()
    assert df2.column_names() == pa_df2.column_names()

    assert not to_pyarrow.called


def test_dataframe_interchange_dataframe_methods_execute(con, alltypes, mocker):
    t = alltypes.select("int_col", "double_col", "string_col")
    pa_df = t.to_pyarrow().__dataframe__()

    to_pyarrow = mocker.spy(con, "to_pyarrow")

    df = t.__dataframe__()

    assert to_pyarrow.call_count == 0
    assert df.metadata == pa_df.metadata
    assert df.num_rows() == pa_df.num_rows()
    assert df.num_chunks() == pa_df.num_chunks()
    assert len(list(df.get_chunks())) == df.num_chunks()
    assert to_pyarrow.call_count == 1


@pytest.mark.notimpl(["druid"])
def test_dataframe_interchange_column_methods_execute(con, alltypes, mocker):
    t = alltypes.select("int_col", "double_col", "string_col")
    pa_df = t.to_pyarrow().__dataframe__()

    to_pyarrow = mocker.spy(con, "to_pyarrow")

    df = t.__dataframe__()
    col = df.get_column(0)
    pa_col = pa_df.get_column(0)

    assert to_pyarrow.call_count == 0
    assert col.size() == pa_col.size()
    assert col.offset == pa_col.offset

    assert col.describe_null == pa_col.describe_null
    assert col.null_count == pa_col.null_count
    assert col.metadata == pa_col.metadata
    assert col.num_chunks() == pa_col.num_chunks()
    assert len(list(col.get_chunks())) == pa_col.num_chunks()
    assert len(list(col.get_buffers())) == len(list(pa_col.get_buffers()))
    assert to_pyarrow.call_count == 1

    # Access another column doesn't execute
    col2 = df.get_column(1)
    pa_col2 = pa_df.get_column(1)
    assert col2.size() == pa_col2.size()


def test_dataframe_interchange_select_after_execution_no_reexecute(
    con, alltypes, mocker
):
    t = alltypes.select("int_col", "double_col", "string_col")
    pa_df = t.to_pyarrow().__dataframe__()

    to_pyarrow = mocker.spy(con, "to_pyarrow")

    df = t.__dataframe__()

    # An operation that requires loading data
    assert to_pyarrow.call_count == 0
    assert df.num_rows() == pa_df.num_rows()
    assert to_pyarrow.call_count == 1

    # Subselect columns doesn't reexecute
    df2 = df.select_columns([1, 0])
    pa_df2 = pa_df.select_columns([1, 0])
    assert df2.num_rows() == pa_df2.num_rows()
    assert df2.column_names() == pa_df2.column_names()
    assert to_pyarrow.call_count == 1
