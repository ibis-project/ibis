"""People whose activity doesn't count as community usage.

Maintainers, plus people employed by Voltron Data to work on Ibis. Krzysztof
Kwitt is included on the assumption that he was employed by Voltron Data; his
commits all fall between late 2022 and early 2023, so it doesn't affect recent
numbers either way.
"""

from __future__ import annotations

# Git author emails, with each person's aliases merged.
CORE_EMAILS = frozenset(
    {
        # Phillip Cloud
        "417981+cpcloud@users.noreply.github.com",
        "cpcloud@gmail.com",
        "cpcloud@fb.com",
        # Krisztián Szűcs
        "szucs.krisztian@gmail.com",
        # Jim Crist-Harif
        "jcristharif@gmail.com",
        # Gil Forsyth
        "gil@forsyth.dev",
        "gforsyth@users.noreply.github.com",
        # Nick Crews
        "nicholas.b.crews@gmail.com",
        # Deepyaman Datta
        "deepyaman.datta@utexas.edu",
        # Krzysztof Kwitt
        "krzysztofwitt93@wp.pl",
        "120908425+krzysztof-kwitt@users.noreply.github.com",
        # Cody Peterson
        "54814569+lostmygithubaccount@users.noreply.github.com",
        "cody@dkdc.dev",
        "cody.dkdc2@gmail.com",
        # Naty Clementi
        "natyclementi@gmail.com",
        # Chloe He
        "chloehe47@gmail.com",
        # Mehmet Fatih Aktas
        "mfatihaktas@gmail.com",
        # Zhenzhong Xu
        "flyfree@gmail.com",
        # Chip Huyen
        "huyenntkvn@gmail.com",
    }
)

# GitHub logins of the same people.
CORE_LOGINS = frozenset(
    {
        "cpcloud",
        "kszucs",
        "jcrist",
        "gforsyth",
        "NickCrews",
        "deepyaman",
        "krzysztof-kwitt",
        "lostmygithubaccount",
        "ncclementi",
        "chloeh13q",
        "mfatihaktas",
        "zhenzhongxu",
        "chiphuyen",
    }
)

# Backends shipped on main (or recently removed but still on conda-forge).
BACKENDS = (
    "athena",
    "bigquery",
    "clickhouse",
    "databricks",
    "datafusion",
    "druid",
    "duckdb",
    "exasol",
    "flink",
    "impala",
    "materialize",
    "mssql",
    "mysql",
    "oracle",
    "polars",
    "postgres",
    "pyspark",
    "risingwave",
    "singlestoredb",
    "snowflake",
    "sqlite",
    "trino",
)

# Extra title patterns for backends often referred to by another name.
TITLE_PATTERNS = {
    "bigquery": "bigquery|big query",
    "mssql": "mssql|sql server",
    "postgres": "postgres|postgresql",
    "pyspark": "pyspark|spark",
}
