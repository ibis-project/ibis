SELECT
  SUM("t1"."float_col") AS "Sum(float_col)"
FROM (
  SELECT
    *
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."int_col" > 0
) AS "t1"