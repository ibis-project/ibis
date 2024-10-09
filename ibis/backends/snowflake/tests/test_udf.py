from __future__ import annotations

import os
import shutil
import tempfile

import pandas.testing as tm
import pytest
from pytest import param

import ibis.expr.datatypes as dt
from ibis import udf


@udf.scalar.builtin
def soundex(s: str) -> str: ...


@udf.scalar.builtin
def jarowinkler_similarity(a: str, b: str) -> float: ...


# TODO: allow multiple signatures
@udf.scalar.builtin(name="compress")
def compress_str(data: str, method: str) -> bytes: ...


@udf.scalar.builtin(name="compress")
def compress_bytes(data: bytes, method: str) -> bytes: ...


@pytest.mark.parametrize(
    ("func", "args"),
    [
        param(soundex, ("snow",), id="soundex"),
        param(jarowinkler_similarity, ("snow", "show"), id="jarowinkler_similarity"),
    ],
)
def test_builtin_scalar_udf(con, func, args):
    expr = func(*args)

    query = f"SELECT {func.__name__}({', '.join(map(repr, args))})"
    with con._safe_raw_sql(query) as cur:
        [(expected,)] = cur.fetchall()

    assert con.execute(expr) == expected


@pytest.mark.parametrize(
    ("func", "pyargs", "snowargs"),
    [
        param(compress_str, ("snow", "zstd"), ("'snow'", "'zstd'"), id="str"),
        param(compress_bytes, (b"snow", "zstd"), ("'snow'", "'zstd'"), id="bytes"),
    ],
)
def test_compress(con, func, pyargs, snowargs):
    expr = func(*pyargs)

    query = f"SELECT compress({', '.join(snowargs)})"
    with con._safe_raw_sql(query) as cur:
        [(expected,)] = cur.fetchall()

    assert con.execute(expr) == expected


@udf.agg.builtin
def minhash(x, y) -> dt.json: ...


@udf.agg.builtin
def approximate_jaccard_index(a) -> float: ...


def test_builtin_agg_udf(con):
    ft = con.tables.FUNCTIONAL_ALLTYPES.limit(2)
    ft = ft.select(mh=minhash(100, ft.string_col).over(group_by=ft.date_string_col))
    expr = ft.agg(aji=approximate_jaccard_index(ft.mh))

    result = expr.execute()
    query = """
    SELECT approximate_jaccard_index("mh") AS "aji"
    FROM (
        SELECT minhash(100, "string_col") OVER (PARTITION BY "date_string_col") AS "mh"
        FROM (
            SELECT * FROM "FUNCTIONAL_ALLTYPES" LIMIT 2
        )
    )
    """
    with con._safe_raw_sql(query) as cur:
        expected = cur.fetch_pandas_all()

    tm.assert_frame_equal(result, expected)


def test_xgboost_model(con):
    from ibis import _

    @udf.scalar.pandas(
        packages=("joblib", "xgboost"), imports=("@MODELS/model.joblib",)
    )
    def predict_price(
        carat_scaled: float, cut_encoded: int, color_encoded: int, clarity_encoded: int
    ) -> int:
        import sys
        from pathlib import Path

        import joblib
        import pandas as pd

        import_dir = Path(sys._xoptions.get("snowflake_import_directory"))
        assert import_dir.exists(), import_dir

        model_path = import_dir / "model.joblib"
        assert model_path.exists(), model_path

        model = joblib.load(model_path)

        df = pd.concat(
            [carat_scaled, cut_encoded, color_encoded, clarity_encoded], axis=1
        )
        df.columns = ["CARAT_SCALED", "CUT_ENCODED", "COLOR_ENCODED", "CLARITY_ENCODED"]
        return model.predict(df)

    diamonds = con.tables.DIAMONDS
    expr = diamonds.mutate(
        predicted_price=predict_price(
            (_.carat - _.carat.mean()) / _.carat.std(),
            _.cut.cases(
                *(
                    (c, i)
                    for i, c in enumerate(
                        ("Fair", "Good", "Very Good", "Premium", "Ideal"), start=1
                    )
                )
            ),
            _.color.cases(*((c, i) for i, c in enumerate("DEFGHIJ", start=1))),
            _.clarity.cases(
                *(
                    (c, i)
                    for i, c in enumerate(
                        ("I1", "IF", "SI1", "SI2", "VS1", "VS2", "VVS1", "VVS2"),
                        start=1,
                    )
                )
            ),
        )
    )

    df = expr.execute()

    assert not df.empty
    assert "predicted_price" in df.columns
    assert len(df) == diamonds.count().execute()


def touch_package(pkgpath):
    # nix timestamps every file to the UNIX epoch for reproducibility
    # so we modify the utime of the _copied_ code since snowflake has
    # some annoying checks for zipping files that do not allow files
    # older than 1980 ¯\_(ツ)_/¯
    os.utime(pkgpath, None)
    for root, dirs, files in os.walk(pkgpath):
        for path in dirs + files:
            os.utime(os.path.join(root, path), None)


def add_packages(d, session):
    import parsy
    import pyarrow_hotfix
    import rich
    import sqlglot

    import ibis

    for module in (rich, parsy, sqlglot, pyarrow_hotfix):
        pkgname = module.__name__
        pkgpath = os.path.join(d, pkgname)
        shutil.copytree(os.path.dirname(module.__file__), pkgpath)
        touch_package(pkgpath)
        session.add_import(pkgname, import_path=pkgname)

    # no need to touch the package because we're using the local version
    shutil.copytree(os.path.dirname(ibis.__file__), "ibis")
    session.add_import("ibis", import_path="ibis")


@pytest.fixture
def snowpark_session():
    if not os.environ.get("SNOWFLAKE_SNOWPARK"):
        pytest.skip("SNOWFLAKE_SNOWPARK is not set")
    else:
        sp = pytest.importorskip("snowflake.snowpark")

        if connection_name := os.environ.get("SNOWFLAKE_DEFAULT_CONNECTION_NAME"):
            builder = sp.Session.builder.config("connection_name", connection_name)
        else:
            builder = sp.Session.builder.configs(
                {
                    "user": os.environ["SNOWFLAKE_USER"],
                    "account": os.environ["SNOWFLAKE_ACCOUNT"],
                    "password": os.environ["SNOWFLAKE_PASSWORD"],
                    "warehouse": os.environ["SNOWFLAKE_WAREHOUSE"],
                    "database": os.environ["SNOWFLAKE_DATABASE"],
                    "schema": os.environ["SNOWFLAKE_SCHEMA"],
                }
            )

        session = builder.create()
        session.custom_package_usage_config["enabled"] = True

        pwd = os.getcwd()

        with tempfile.TemporaryDirectory() as d:
            os.chdir(d)

            try:
                add_packages(d, session)
                yield session
            finally:
                os.chdir(pwd)
                session.clear_imports()


@pytest.mark.parametrize("execute_as", ["owner", "caller"])
def test_ibis_inside_snowpark(snowpark_session, execute_as):
    import snowflake.snowpark as sp

    def ibis_sproc(session):
        import ibis.backends.snowflake

        con = ibis.backends.snowflake.Backend.from_connection(session)

        expr = (
            con.tables.functional_alltypes.group_by("string_col")
            .agg(n=lambda t: t.count())
            .order_by("string_col")
        )

        return session.sql(ibis.to_sql(expr))

    expected = (
        snowpark_session.table('"functional_alltypes"')
        .group_by('"string_col"')
        .count()
        .rename("COUNT", '"n"')
        .order_by('"string_col"')
        .to_pandas()
    )

    local_result = ibis_sproc(snowpark_session).to_pandas()

    tm.assert_frame_equal(local_result, expected)

    name = ibis_sproc.__name__

    snowpark_session.sproc.register(
        ibis_sproc,
        name=name,
        execute_as=execute_as,
        imports=["parsy", "rich", "sqlglot", "pyarrow_hotfix", "ibis"],
        # empty struct here tells Snowflake to infer the return type from the
        # return value of the function, which is required to be a Snowpark
        # table in that case
        return_type=sp.types.StructType(),
        packages=[
            "snowflake-snowpark-python",
            "toolz",
            "atpublic",
            "pyarrow",
            "pandas",
            "numpy",
        ],
        replace=True,
    )

    remote_result = snowpark_session.call(name).to_pandas()

    tm.assert_frame_equal(remote_result, local_result)
