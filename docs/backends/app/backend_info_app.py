import contextlib
import csv
from enum import Enum
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

st.set_page_config(layout='wide')


@st.cache
def load_ops_data():
    response = requests.get(
        'https://ibis-project.org/docs/dev/backends/raw_support_matrix.csv'
    )
    response.raise_for_status()
    reader = csv.DictReader(response.text.splitlines())

    data = [
        {k: v if k == "FullOperation" else v == "True" for k, v in row.items()}
        for row in reader
    ]
    for row in data:
        row['ShortOperation'] = row['FullOperation'].rsplit(".")[4]
        row['OpCategory'] = row['FullOperation'].rsplit(".")[3]
    return data


ops_data = load_ops_data()


class BackendCategory(Enum):
    string = "STRING"
    sqlalchemy = "SQLALCHEMY"
    dataframe = "DATAFRAME"
    sql = "SQL"


BACKENDS = {
    "bigquery": [BackendCategory.string, BackendCategory.sql],
    "clickhouse": [BackendCategory.string, BackendCategory.sql],
    'dask': [BackendCategory.dataframe],
    "datafusion": [BackendCategory.dataframe],
    "duckdb": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "impala": [BackendCategory.string, BackendCategory.sql],
    "mssql": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "mysql": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "pandas": [BackendCategory.dataframe],
    "polars": [BackendCategory.dataframe],
    "postgres": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "pyspark": [BackendCategory.dataframe],
    "snowflake": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "sqlite": [BackendCategory.sqlalchemy, BackendCategory.sql],
    "trino": [BackendCategory.sqlalchemy, BackendCategory.sql],
}


def get_selected_backend_name():
    all_categories_names = [cat.name for cat in iter(BackendCategory)]
    selected_categories_names = st.sidebar.multiselect(
        'Backend category',
        options=sorted(all_categories_names),
        default=None,
    )
    selected_categories = [
        getattr(BackendCategory, name) for name in selected_categories_names
    ]

    selected_backend_names = {
        backend_name
        for backend_name, backend_types in BACKENDS.items()
        if any(cat in backend_types for cat in selected_categories)
    }
    if not selected_backend_names:
        selected_backend_names = list(BACKENDS.keys())
    return selected_backend_names


def get_selected_operation_categories(all_ops_categories):
    selected_ops_categories = st.sidebar.multiselect(
        'Operation category',
        options=sorted(all_ops_categories),
        default=None,
    )
    if not selected_ops_categories:
        return all_ops_categories
    return selected_ops_categories


all_ops_categories = {row['OpCategory'] for row in ops_data}
current_backend_names = get_selected_backend_name()
current_ops_categories = get_selected_operation_categories(all_ops_categories)
show_geospatial = st.sidebar.checkbox('Include Geospatial ops', value=True)
hide_supported_by_all_backends = st.sidebar.selectbox(
    'Operation compatibility',
    ['Show all', 'Show supported by all backends', 'Hide supported by all backends'],
    0,
)
show_full_ops_name = st.sidebar.checkbox('Show full operation name', False)

df = pd.DataFrame(ops_data)
if show_full_ops_name:
    df = df.set_index("FullOperation")
else:
    df = df.set_index("ShortOperation")
df = df.sort_index()

# Show only selected operation
if not show_geospatial:
    with contextlib.suppress(ValueError):
        current_ops_categories.remove("geospatial")
df = df[df['OpCategory'].isin(current_ops_categories)]

# Show only selected backend
df = df[list(current_backend_names)]

if hide_supported_by_all_backends == 'Show supported by all backends':
    df = df[df.sum(1) == len(df.columns)]
elif hide_supported_by_all_backends == 'Hide supported by all backends':
    df = df[df.sum(1) != len(df.columns)]

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
    st.table(table)
else:
    st.write("No data")

with st.expander("Source code"):
    st.code(Path(__file__).read_text())
