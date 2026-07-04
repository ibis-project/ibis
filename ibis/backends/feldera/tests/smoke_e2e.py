"""End-to-end smoke test for the Ibis Feldera backend against a live Feldera."""

from __future__ import annotations

import uuid
import warnings

import pandas as pd
import pyarrow as pa

import ibis
from feldera import FelderaClient, PipelineBuilder

HOST = "http://localhost:8080"

SQL = """
CREATE TABLE penguins (
    species VARCHAR NOT NULL,
    island VARCHAR NOT NULL,
    bill_length_mm DOUBLE,
    bill_depth_mm DOUBLE,
    flipper_length_mm INTEGER,
    body_mass_g INTEGER,
    sex VARCHAR,
    year INTEGER
) WITH ('materialized' = 'true');

CREATE MATERIALIZED VIEW penguin_counts AS
SELECT species, island, COUNT(*) AS count
FROM penguins
GROUP BY species, island;
"""

ROWS = [
    {"species": "Adelie", "island": "Torgersen", "bill_length_mm": 39.1,
     "bill_depth_mm": 18.7, "flipper_length_mm": 181, "body_mass_g": 3750,
     "sex": "male", "year": 2007},
    {"species": "Adelie", "island": "Torgersen", "bill_length_mm": 39.5,
     "bill_depth_mm": 17.4, "flipper_length_mm": 186, "body_mass_g": 3800,
     "sex": "female", "year": 2007},
    {"species": "Gentoo", "island": "Biscoe", "bill_length_mm": 46.1,
     "bill_depth_mm": 13.2, "flipper_length_mm": 211, "body_mass_g": 4500,
     "sex": "female", "year": 2007},
]


def main() -> None:
    client = FelderaClient(HOST)
    name = "ibis-e2e-" + uuid.uuid4().hex[:8]
    pipe = PipelineBuilder(client, name=name, sql=SQL).create(wait=True)
    pipe.start()
    print(f"pipeline {name!r}: {pipe.status()}")
    pipe.input_pandas("penguins", pd.DataFrame(ROWS))

    con = ibis.feldera.connect(host=HOST, pipeline=name)
    print("connected:", con)
    print("version:", con.version())
    print("list_tables:", con.list_tables())

    # get_schema for a view
    sch = con.get_schema("penguin_counts")
    print("schema(penguin_counts):", sch)

    # table() should build an expression over the view
    t = con.table("penguin_counts")
    print("table expr:", t)
    print("table schema:", t.schema())

    # compile a query (no execution) — just check the SQL shape
    expr = t.order_by(ibis.desc("count"))
    sql = con.compile(expr)
    print("\ncompiled SQL:\n", sql)

    # execute end-to-end
    print("\nexecute() result:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = expr.execute()
    print(df)

    # to_pyarrow
    print("\nto_pyarrow() result:")
    tbl = con.to_pyarrow(t)
    print(tbl)

    # a more involved expression: filter + group_by + agg
    penguins = con.table("penguins")
    p = penguins.filter(penguins.body_mass_g > 3000)
    expr2 = (
        p.group_by("species")
        .agg(avg_bill=p.bill_length_mm.mean(), n=p.count())
        .order_by(ibis.desc("n"))
    )
    print("\nexpr2 SQL:\n", con.compile(expr2))
    print("\nexpr2 execute():")
    print(expr2.execute())


if __name__ == "__main__":
    main()
