import datetime
import os
import tempfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
import requests
import sqlglot
import streamlit as st

import ibis
from ibis import _

ONE_HOUR_IN_SECONDS = datetime.timedelta(hours=1).total_seconds()

st.set_page_config(layout='wide')

# Track all queries. We display them at the bottom of the page.
ibis.options.verbose = True
sql_queries = []
ibis.options.verbose_log = lambda sql: sql_queries.append(sql)


def get_single_query_param(key: str, default=None):
    query_param_list = st.experimental_get_query_params().get(key)
    if query_param_list:
        return query_param_list[0]
    return default


def github_paginated_request(url):
    next_url = url
    resources = []
    headers = {
        'X-GitHub-Api-Version': '2022-11-28',
        'Accept': 'application/vnd.github+json',
    }
    # On Streamlit Cloud, token should be stored as secret.
    # See:
    # https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app/connect-to-data-sources/secrets-management
    github_token = os.environ.get('GITHUB_TOKEN')
    if github_token:
        headers['Authorization'] = f'Bearer {github_token}'
    with requests.Session() as session:
        while next_url is not None:
            resp = session.get(next_url, headers=headers)
            resp.raise_for_status()
            resources.extend(resp.json())
            next_url = resp.links.get('next', {}).get('url')
    return resources


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def ibis_versions_with_raw_data_list():
    releases = github_paginated_request(
        'https://api.github.com/repos/ibis-project/ibis/releases'
    )
    versions = (release['name'] for release in releases)
    versions = (
        version
        for version in versions
        if version.startswith("4.") and version != '4.0.0'
    )
    return list(versions) + ['dev']


ibis_versions_with_raw_data = ibis_versions_with_raw_data_list()

query_ibis_version = get_single_query_param('version', 'dev')

try:
    ibis_version_index = ibis_versions_with_raw_data.index(query_ibis_version)
except ValueError:
    ibis_version_index = ibis_versions_with_raw_data.index('dev')

current_ibis_version = st.sidebar.selectbox(
    "Ibis version", ibis_versions_with_raw_data, ibis_version_index
)


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def support_matrix_df(ibis_version: str):
    resp = requests.get(
        f"https://ibis-project.org/docs/{ibis_version}/backends/raw_support_matrix.csv"
    )
    resp.raise_for_status()

    with tempfile.NamedTemporaryFile() as f:
        f.write(resp.content)
        return (
            ibis.read_csv(f.name)
            .relabel({'FullOperation': 'full_operation'})
            .mutate(
                short_operation=_.full_operation.split(".")[-1],
                operation_category=_.full_operation.split(".")[-2],
            )
            .execute()
        )


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def backends_info_df():
    return pd.DataFrame(
        {
            "bigquery": ["string", "sql"],
            "clickhouse": ["string", "sql"],
            'dask': ["dataframe"],
            "datafusion": ["dataframe"],
            "duckdb": ["sqlalchemy", "sql"],
            "impala": ["string", "sql"],
            "mssql": ["sqlalchemy", "sql"],
            "mysql": ["sqlalchemy", "sql"],
            "pandas": ["dataframe"],
            "polars": ["dataframe"],
            "postgres": ["sqlalchemy", "sql"],
            "pyspark": ["dataframe"],
            "snowflake": ["sqlalchemy", "sql"],
            "sqlite": ["sqlalchemy", "sql"],
            "trino": ["sqlalchemy", "sql"],
        }.items(),
        columns=['backend_name', 'categories'],
    )


backend_info_table = ibis.memtable(backends_info_df())
support_matrix_table = ibis.memtable(support_matrix_df(current_ibis_version))


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def get_all_backend_categories():
    return (
        backend_info_table.select(category=_.categories.unnest())
        .distinct()
        .order_by('category')['category']
        .execute()
        .tolist()
    )


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def get_all_operation_categories():
    return (
        support_matrix_table.select(_.operation_category)
        .distinct()['operation_category']
        .execute()
        .tolist()
    )


@st.experimental_memo(ttl=ONE_HOUR_IN_SECONDS)
def get_backend_names(categories: Optional[List[str]] = None):
    backend_expr = backend_info_table.mutate(category=_.categories.unnest())
    if categories:
        backend_expr = backend_expr.filter(_.category.isin(categories))
    return (
        backend_expr.select(_.backend_name).distinct().backend_name.execute().tolist()
    )


def get_selected_backend_name():
    backend_categories = get_all_backend_categories()
    selected_categories_names = st.sidebar.multiselect(
        'Backend category',
        options=backend_categories,
        default=None,
    )
    if not selected_categories_names:
        return get_backend_names()
    return get_backend_names(selected_categories_names)


def get_selected_operation_categories():
    all_ops_categories = get_all_operation_categories()

    selected_ops_categories = st.sidebar.multiselect(
        'Operation category',
        options=sorted(all_ops_categories),
        default=None,
    )
    if not selected_ops_categories:
        selected_ops_categories = all_ops_categories
    show_geospatial = st.sidebar.checkbox('Include Geospatial ops', value=True)
    if not show_geospatial and 'geospatial' in selected_ops_categories:
        selected_ops_categories.remove("geospatial")
    return selected_ops_categories


current_backend_names = get_selected_backend_name()
current_ops_categories = get_selected_operation_categories()

hide_supported_by_all_backends = st.sidebar.selectbox(
    'Operation compatibility',
    ['Show all', 'Show supported by all backends', 'Hide supported by all backends'],
    0,
)
show_full_ops_name = st.sidebar.checkbox('Show full operation name', False)

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
if hide_supported_by_all_backends == 'Show supported by all backends':
    table_expr = table_expr.filter(
        supported_backend_count == len(current_backend_names)
    )
elif hide_supported_by_all_backends == 'Hide supported by all backends':
    table_expr = table_expr.filter(
        supported_backend_count != len(current_backend_names)
    )

# Show only selected backend
table_expr = table_expr[current_backend_names + ["index"]]

# Execute query
df = table_expr.execute()
df = df.set_index('index')

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

    table = pd.concat([coverage, df.replace({True: "✔", False: "🚫"})])
    st.dataframe(table)
else:
    st.write("No data")

with st.expander("SQL queries"):
    for sql_query in sql_queries:
        pretty_sql_query = sqlglot.transpile(
            sql_query, read='duckdb', write='duckdb', pretty=True
        )[0]
        st.code(
            pretty_sql_query,
            language='sql',
        )

with st.expander("Source code"):
    st.code(Path(__file__).read_text())
