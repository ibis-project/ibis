SELECT
  "t2"."b",
  "t2"."sum"
FROM (
  SELECT
    "t1"."b",
    SUM("t1"."a") AS "sum",
    MAX("t1"."a") AS "Max(a)"
  FROM (
    SELECT
      "t0"."a",
      "t0"."b"
    FROM "t" AS "t0"
    WHERE
      "t0"."b" = 'm'
  ) AS "t1"
  GROUP BY
    1
) AS "t2"
WHERE
  "t2"."Max(a)" = CAST(2 AS TINYINT)