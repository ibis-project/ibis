from __future__ import annotations

import ibis
from ibis import _, udf


@udf.scalar.builtin
def flatten(x: list[list[str]]) -> list[str]:  # <1>
    ...


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
        ext=_.path.re_extract(r"\.([a-z0-9]+)$", 1),
    )
    .aggregate(projects=_.project_name.collect().unique())
    .order_by(_.month.desc())
    .mutate(
        ext=_.ext.re_replace(r"cxx|cpp|cc|c|hpp|h", "C/C++")
        .re_replace("^f.*$", "Fortran")
        .replace("rs", "Rust")
        .replace("go", "Go")
        .replace("asm", "Assembly")
        .nullif(""),
    )
    .group_by(["month", "ext"])
    .aggregate(project_count=flatten(_.projects.collect()).unique().length())
    .dropna("ext")
    .order_by([_.month.desc(), _.project_count.desc()])  # <2>
)
