from __future__ import annotations

import datetime
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import sqlglot
import streamlit as st

import ibis
from ibis import _

ONE_HOUR_IN_SECONDS = datetime.timedelta(hours=1).total_seconds()

st.set_page_config(layout="wide")

# Track all queries. We display them at the bottom of the page.
ibis.options.verbose = True
sql_queries = []
ibis.options.verbose_log = lambda sql: sql_queries.append(sql)


@st.cache_data(ttl=ONE_HOUR_IN_SECONDS)
def support_matrix_df():
    resp = requests.get("https://ibis-project.org/backends/raw_support_matrix.csv")
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile() as f:
        f.write(resp.content)
        return (
            ibis.read_csv(f.name)
            .relabel({"FullOperation": "full_operation"})
            .mutate(
                short_operation=_.full_operation.split(".")[-1],
                operation_category=_.full_operation.split(".")[-2],
            )
            .to_pandas()
        )


@st.cache_data(ttl=ONE_HOUR_IN_SECONDS)
def backends_info_df():
    return pd.DataFrame(
        {
            "bigquery": ["string", "sql"],
            "clickhouse": ["string", "sql"],
            "dask": ["dataframe"],
            "datafusion": ["sql"],
            "druid": ["sqlalchemy", "sql"],
            "duckdb": ["sqlalchemy", "sql"],
            "impala": ["string", "sql"],
            "mssql": ["sqlalchemy", "sql"],
            "mysql": ["sqlalchemy", "sql"],
            "oracle": ["sqlalchemy", "sql"],
            "pandas": ["dataframe"],
            "polars": ["dataframe"],
            "postgres": ["sqlalchemy", "sql"],
            "pyspark": ["dataframe"],
            "snowflake": ["sqlalchemy", "sql"],
            "sqlite": ["sqlalchemy", "sql"],
            "trino": ["sqlalchemy", "sql"],
        }.items(),
        columns=["backend_name", "categories"],
    )


backend_info_table = ibis.memtable(backends_info_df())
support_matrix_table = ibis.memtable(support_matrix_df())


@st.cache_data(ttl=ONE_HOUR_IN_SECONDS)
def get_all_backend_categories():
    return (
        backend_info_table.select(category=_.categories.unnest())
        .distinct()
        .order_by("category")["category"]
        .to_pandas()
        .tolist()
    )


@st.cache_data(ttl=ONE_HOUR_IN_SECONDS)
def get_all_operation_categories():
    return (
        support_matrix_table.select(_.operation_category)
        .distinct()["operation_category"]
        .to_pandas()
        .tolist()
    )


@st.cache_data(ttl=ONE_HOUR_IN_SECONDS)
def get_backend_names(categories: Optional[list[str]] = None):
    backend_expr = backend_info_table.mutate(category=_.categories.unnest())
    if categories:
        backend_expr = backend_expr.filter(_.category.isin(categories))
    return (
        backend_expr.select(_.backend_name).distinct().backend_name.to_pandas().tolist()
    )


def get_selected_backend_name():
    backend_categories = get_all_backend_categories()
    selected_categories_names = st.sidebar.multiselect(
        "Backend category",
        options=backend_categories,
        default=backend_categories,
    )
    return get_backend_names(selected_categories_names)


def get_backend_subset(subset):
    return st.sidebar.multiselect("Backend name", options=subset, default=subset)


def get_selected_operation_categories():
    all_ops_categories = get_all_operation_categories()

    selected_ops_categories = st.sidebar.multiselect(
        "Operation category",
        options=sorted(all_ops_categories),
        default=None,
    )
    if not selected_ops_categories:
        selected_ops_categories = all_ops_categories
    show_geospatial = st.sidebar.checkbox("Include Geospatial ops", value=True)
    if not show_geospatial and "geospatial" in selected_ops_categories:
        selected_ops_categories.remove("geospatial")
    return selected_ops_categories


current_backend_names = get_backend_subset(get_selected_backend_name())
sort_by_coverage = st.sidebar.checkbox("Sort by API Coverage", value=False)
current_ops_categories = get_selected_operation_categories()

hide_supported_by_all_backends = st.sidebar.selectbox(
    "Operation compatibility",
    ["Show all", "Show supported by all backends", "Hide supported by all backends"],
    0,
)
show_full_ops_name = st.sidebar.checkbox("Show full operation name", False)

# Start ibis expression
table_expr = support_matrix_table

# Add index to result
if show_full_ops_name:
    table_expr = table_expr.mutate(index=_.full_operation)
else:
    table_expr = table_expr.mutate(index=_.short_operation)
table_expr = table_expr.order_by(_.index)

# Filter operations by selected categories
table_expr = table_expr.filter(_.operation_category.isin(current_ops_categories))

# Filter operation by compatibility
supported_backend_count = sum(
    getattr(table_expr, backend_name).ifelse(1, 0)
    for backend_name in current_backend_names
)
if hide_supported_by_all_backends == "Show supported by all backends":
    table_expr = table_expr.filter(
        supported_backend_count == len(current_backend_names)
    )
elif hide_supported_by_all_backends == "Hide supported by all backends":
    table_expr = table_expr.filter(
        supported_backend_count != len(current_backend_names)
    )

# Show only selected backend
table_expr = table_expr[current_backend_names + ["index"]]

# Execute query
df = table_expr.to_pandas()
df = df.set_index("index")

# Display result
all_visible_ops_count = len(df.index)
if all_visible_ops_count:
    # Compute coverage
    coverage = (
        df.sum()
        .sort_values(ascending=False)
        .map(lambda n: f"{n} ({round(100 * n / all_visible_ops_count)}%)")
        .to_frame(name="API Coverage")
        .T
    )

    table = pd.concat([coverage, df.replace({True: "✔", False: "🚫"})]).loc[
        :, slice(None) if sort_by_coverage else sorted(df.columns)
    ]
    st.dataframe(table)
else:
    st.write("No data")

with st.expander("SQL queries"):
    for sql_query in sql_queries:
        pretty_sql_query = sqlglot.transpile(
            sql_query, read="duckdb", write="duckdb", pretty=True
        )[0]
        st.code(pretty_sql_query, language="sql")

with st.expander("Source code"):
    st.code(Path(__file__).read_text())
