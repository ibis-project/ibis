SELECT
  "t3"."g",
  SUM("t3"."foo") AS "foo total"
FROM (
  SELECT
    *
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
        *
      FROM "alltypes" AS "t0"
      WHERE
        "t0"."f" > CAST(0 AS TINYINT)
    ) AS "t1"
  ) AS "t2"
  WHERE
    "t2"."g" = 'bar'
) AS "t3"
GROUP BY
  1