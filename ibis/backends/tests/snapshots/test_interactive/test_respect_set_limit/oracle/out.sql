SELECT
  *
FROM (
  SELECT
    "t0"."id",
    "t0"."bool_col" = 1 AS "bool_col"
  FROM "functional_alltypes" "t0"
  FETCH FIRST 10 ROWS ONLY
) "t2"
FETCH FIRST 11 ROWS ONLY