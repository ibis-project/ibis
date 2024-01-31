SELECT
  "t2"."string_col",
  "t2"."nrows"
FROM (
  SELECT
    "t0"."string_col",
    COUNT(*) AS "nrows"
  FROM "functional_alltypes" AS "t0"
  GROUP BY
    1
  LIMIT 5
) AS "t2"
ORDER BY
  "t2"."string_col" ASC