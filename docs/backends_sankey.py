from __future__ import annotations

import plotly.graph_objects as go

backend_categories = {
    "SQL-generating": [
        "BigQuery",
        "ClickHouse",
        "DataFusion",
        "Druid",
        "DuckDB",
        "Exasol",
        "Flink",
        "Impala",
        "MSSQL",
        "MySQL",
        "Oracle",
        "PostgreSQL",
        "PySpark",
        "RisingWave",
        "Snowflake",
        "SQLite",
        "Trino",
    ],
    "Expression-generating": ["Dask", "Polars"],
    "Naïve execution": ["pandas"],
}

category_colors = {
    "Ibis API": "#999999",
    "Naïve execution": "#FF8C00",
    "Expression-generating": "#6A5ACD",
    "SQL-generating": "#3CB371",
}

nodes, links = [], []
node_index = {}

nodes.append({"label": "Ibis API", "color": category_colors["Ibis API"]})
node_index["Ibis API"] = 0


idx = 1
for category, backends in backend_categories.items():
    nodes.append({"label": category, "color": category_colors[category]})
    node_index[category] = idx
    links.append({"source": 0, "target": idx, "value": len(backends)})
    idx += 1

    for backend in backends:
        if backend not in node_index:
            nodes.append({"label": backend, "color": category_colors[category]})
            node_index[backend] = idx
            idx += 1
        links.append(
            {
                "source": node_index[category],
                "target": node_index[backend],
                "value": 1,
            }
        )


fig = go.Figure(
    data=[
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=[node["label"] for node in nodes],
                color=[node["color"] for node in nodes],
            ),
            link=dict(
                source=[link["source"] for link in links],
                target=[link["target"] for link in links],
                value=[link["value"] for link in links],
            ),
        )
    ]
)

fig.update_layout(
    title_text="Ibis backend types", font_size=14, margin=dict(l=30, r=30, t=80, b=30)
)
