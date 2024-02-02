SELECT
  "t1"."g",
  SUM("t1"."foo") AS "foo total"
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
    "t0"."k",
    "t0"."a" + "t0"."b" AS "foo"
  FROM "alltypes" AS "t0"
  WHERE
    "t0"."f" > CAST(0 AS TINYINT) AND (
      "t0"."a" + "t0"."b"
    ) < CAST(10 AS TINYINT)
) AS "t1"
GROUP BY
  1