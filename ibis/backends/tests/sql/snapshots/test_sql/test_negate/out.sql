SELECT
  NOT (
    "t0"."double_col" > CAST(0 AS TINYINT)
  ) AS "tmp"
FROM "functional_alltypes" AS "t0"