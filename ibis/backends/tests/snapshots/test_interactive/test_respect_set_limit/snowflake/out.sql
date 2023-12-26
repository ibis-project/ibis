SELECT
  *
FROM (
  SELECT
    "t0"."id",
    "t0"."bool_col"
  FROM "FUNCTIONAL_ALLTYPES" AS "t0"
  LIMIT 10
) AS "t2"
LIMIT 11