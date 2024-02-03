#!/usr/bin/env python
#
# Create the geography database for the tutorial
#
# This script creates the SQLite `geography.db` database, used in the Ibis
# tutorials.
#
# The source of the `countries` table is
# [GeoNames](https://www.geonames.org/countries/).
#
# The source of the `gdp` table is
# [World Bank website](https://data.worldbank.org/indicator/NY.GDP.MKTP.CD).
#
# The source of the `independence` table has been obtained from
# [Wikipedia](https://en.wikipedia.org/wiki/List_of_national_independence_days).
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import requests

import ibis

if TYPE_CHECKING:
    from collections.abc import Mapping

SCHEMAS = {
    "countries": {
        "iso_alpha2": "string",
        "iso_alpha3": "string",
        "iso_numeric": "int",
        "fips": "string",
        "name": "string",
        "capital": "string",
        "area_km2": "float",
        "population": "int",
        "continent": "string",
    },
    "gdp": {
        "country_code": "string",
        "year": "int",
        "value": "float",
    },
    "independence": {
        "country_code": "string",
        "independence_date": "date",
        "independence_from": "string",
    },
}


def make_geography_db(
    data: Mapping[str, Any], con: ibis.backends.duckdb.Backend
) -> None:
    with tempfile.TemporaryDirectory() as d:
        for table_name, schema in SCHEMAS.items():
            ibis_schema = ibis.schema(schema)
            cols = ibis_schema.names
            path = Path(d, f"{table_name}.jsonl")
            path.write_text(
                "\n".join(json.dumps(dict(zip(cols, row))) for row in data[table_name])
            )
            con.create_table(
                table_name, obj=con.read_json(path), schema=ibis_schema, overwrite=True
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the geography SQLite database for the Ibis tutorial"
    )
    parser.add_argument(
        "-d",
        "--output-directory",
        default=tempfile.gettempdir(),
        type=str,
        help="The directory to which the database will be output",
    )
    parser.add_argument(
        "-u",
        "--input-data-url",
        default="https://storage.googleapis.com/ibis-tutorial-data/geography.json",
        type=str,
        help="The URL containing the data with which to populate the database",
    )

    args = parser.parse_args()

    response = requests.get(args.input_data_url)
    response.raise_for_status()
    input_data = response.json()
    db_path = Path(args.output_directory).joinpath("geography.duckdb")
    make_geography_db(input_data, ibis.duckdb.connect(db_path))
    print(db_path)  # noqa: T201


if __name__ == "__main__":
    main()
