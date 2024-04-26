SELECT
  "t2"."g",
  SUM("t2"."foo") AS "foo total"
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
    "t1"."foo"
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
      "t0"."f" > CAST(0 AS TINYINT)
  ) AS "t1"
  WHERE
    "t1"."g" = 'bar'
) AS "t2"
GROUP BY
  1