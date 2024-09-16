from __future__ import annotations

import plotly.graph_objects as go


def to_greyish(hex_code, grey_value=128):
    hex_code = hex_code.lstrip("#")
    r, g, b = int(hex_code[0:2], 16), int(hex_code[2:4], 16), int(hex_code[4:6], 16)

    new_r = (r + grey_value) // 2
    new_g = (g + grey_value) // 2
    new_b = (b + grey_value) // 2

    new_hex_code = f"#{new_r:02x}{new_g:02x}{new_b:02x}"

    return new_hex_code


category_colors = {
    "Ibis API": "#7C65A0",
    "SQL": "#6A9BC9",
    "DataFrame": "#D58273",
}

backend_categories = {
    list(category_colors.keys())[1]: [
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
    list(category_colors.keys())[2]: ["Polars"],
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
                line=dict(color="grey", width=0.5),
                label=[node["label"] for node in nodes],
                color=[node["color"] for node in nodes],
            ),
            link=dict(
                source=[link["source"] for link in links],
                target=[link["target"] for link in links],
                value=[link["value"] for link in links],
                line=dict(color="grey", width=0.5),
                color=[to_greyish(nodes[link["target"]]["color"]) for link in links],
            ),
        )
    ],
)

fig.update_layout(
    title_text="Ibis backend types",
    font_size=20,
    # font_family="Arial",
    title_font_size=30,
    margin=dict(l=30, r=30, t=80, b=30),
    template="plotly_dark",
)
