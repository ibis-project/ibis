SELECT
  CASE
    WHEN "t0"."f" > CAST(0 AS TINYINT)
    THEN "t0"."d" * CAST(2 AS TINYINT)
    WHEN "t0"."c" < CAST(0 AS TINYINT)
    THEN "t0"."a" * CAST(2 AS TINYINT)
    ELSE CAST(NULL AS BIGINT)
  END AS "tmp"
FROM "alltypes" AS "t0"