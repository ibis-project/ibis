from __future__ import annotations

import pytest

import ibis
from ibis.backends.tests.errors import PyAthenaOperationalError
from ibis.util import gen_name


@pytest.fixture
def mock_connect(mocker):
    """Patch out ``pyathena.connect`` and return the captured mock.

    The mocked connection's cursor is wired up so the queries ``do_connect``
    issues after connecting (``current_catalog``/``current_database``) succeed,
    and ``fsspec`` is patched so no real S3 filesystem is created.
    """
    connect = mocker.patch.object(ibis.backends.athena.pyathena, "connect")
    cursor = connect.return_value.cursor.return_value.__enter__.return_value
    cursor.execute.return_value.fetchall.return_value = [("value",)]

    mocker.patch.object(ibis.backends.athena.fsspec, "filesystem")
    return connect


def test_user_agent_set_by_default(mock_connect):
    ibis.athena.connect(s3_staging_dir="s3://bucket/staging")

    _, kwargs = mock_connect.call_args
    assert kwargs["config"].user_agent_extra == f"ibis/{ibis.__version__}"


def test_user_agent_prepended_to_user_supplied(mock_connect):
    from botocore.config import Config

    ibis.athena.connect(
        s3_staging_dir="s3://bucket/staging",
        config=Config(user_agent_extra="myapp/1.0"),
    )

    _, kwargs = mock_connect.call_args
    assert kwargs["config"].user_agent_extra == f"ibis/{ibis.__version__} myapp/1.0"


def test_user_agent_preserves_other_config(mock_connect):
    from botocore.config import Config

    ibis.athena.connect(
        s3_staging_dir="s3://bucket/staging",
        config=Config(region_name="us-east-1"),
    )

    _, kwargs = mock_connect.call_args
    assert kwargs["config"].region_name == "us-east-1"
    assert kwargs["config"].user_agent_extra == f"ibis/{ibis.__version__}"


def test_create_and_drop_database(con):
    name = gen_name("db")

    # create it
    con.create_database(name)

    # create it again with force=True (i.e., IF NOT EXISTS)
    con.create_database(name, force=True)

    # create it again (should fail)
    with pytest.raises(PyAthenaOperationalError):
        con.create_database(name)

    # drop it
    con.drop_database(name)

    # drop it again with force=True (i.e., IF EXISTS)
    con.drop_database(name, force=True)

    # drop it again (should fail)
    with pytest.raises(PyAthenaOperationalError):
        con.drop_database(name)


def test_column_name_with_slash(con):
    table = ibis.memtable({"inventarnr_/_mde_dummy": [1, 2, 3]})

    renamed_table = con.execute(
        table.select("inventarnr_/_mde_dummy")
        .rename({"dummy": "inventarnr_/_mde_dummy"})
        .dummy
    )
    assert set(renamed_table.values) == {1, 2, 3}


@pytest.mark.parametrize("partitioned_by", [{"d": "date"}, [("d", "date")]])
def test_simple_partitioned_by(con, partitioned_by):
    name = gen_name("partitioned_by")
    # create a table
    t = con.create_table(
        name, schema={"x": "int", "y": "int"}, partitioned_by=partitioned_by
    )
    try:
        assert t.columns == ("x", "y", "d")
        assert t.execute().empty
        # check that it exists
        assert name in con.list_tables()
    finally:
        # drop the table
        con.drop_table(name)
        # check that it no longer exists
        assert name not in con.list_tables()
