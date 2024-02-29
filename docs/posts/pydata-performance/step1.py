from __future__ import annotations

import ibis
from ibis import _, udf


@udf.scalar.builtin
def flatten(x: list[list[str]]) -> list[str]: ...


expr = (
    ibis.read_parquet("/data/pypi-parquet/*.parquet")
    .filter(
        [
            _.path.re_search(
                r"\.(asm|c|cc|cpp|cxx|h|hpp|rs|[Ff][0-9]{0,2}(?:or)?|go)$"
            ),
            ~_.path.re_search(r"(^|/)test(|s|ing)"),
            ~_.path.contains("/site-packages/"),
        ]
    )
    .group_by(
        month=_.uploaded_on.truncate("M"),
        ext=_.path.re_extract(r"\.([a-z0-9]+)$", 1)
        .re_replace(r"cxx|cpp|cc|c|hpp|h", "C/C++")
        .re_replace("^f.*$", "Fortran")
        .replace("rs", "Rust")
        .replace("go", "Go")
        .replace("asm", "Assembly")
        .nullif(""),
    )
    .aggregate(projects=_.project_name.collect().unique())
    .order_by(_.month.desc())
    .group_by(["month", "ext"])
    .aggregate(project_count=flatten(_.projects.collect()).unique().length())
)
