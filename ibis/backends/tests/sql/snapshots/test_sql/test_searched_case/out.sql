SELECT
  CASE
    WHEN "t0"."f" > 0
    THEN "t0"."d" * CAST(2 AS TINYINT)
    WHEN "t0"."c" < 0
    THEN "t0"."a" * CAST(2 AS TINYINT)
    ELSE NULL
  END AS "tmp"
FROM "alltypes" AS "t0"