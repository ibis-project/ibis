SELECT
  "t3"."g",
  SUM("t3"."foo") AS "foo total"
FROM (
  SELECT
    "t2"."a",
    "t2"."b",
    "t2"."c",
    "t2"."d",
    "t2"."e",
    "t2"."f",
    "t2"."g",
    "t2"."h",
    "t2"."i",
    "t2"."j",
    "t2"."k",
    "t2"."foo"
  FROM (
    SELECT
      "t1"."a",
      "t1"."b",
      "t1"."c",
      "t1"."d",
      "t1"."e",
      "t1"."f",
      "t1"."g",
      "t1"."h",
      "t1"."i",
      "t1"."j",
      "t1"."k",
      "t1"."a" + "t1"."b" AS "foo"
    FROM (
      SELECT
        "t0"."a",
        "t0"."b",
        "t0"."c",
        "t0"."d",
        "t0"."e",
        "t0"."f",
        "t0"."g",
        "t0"."h",
        "t0"."i",
        "t0"."j",
        "t0"."k"
      FROM "alltypes" AS "t0"
      WHERE
        "t0"."f" > CAST(0 AS TINYINT)
    ) AS "t1"
  ) AS "t2"
  WHERE
    "t2"."foo" < CAST(10 AS TINYINT)
) AS "t3"
GROUP BY
  1