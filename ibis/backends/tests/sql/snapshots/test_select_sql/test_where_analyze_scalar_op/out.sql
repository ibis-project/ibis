SELECT
  COUNT(*) AS "CountStar()"
FROM (
  SELECT
    *
  FROM "functional_alltypes" AS "t0"
  WHERE
    "t0"."timestamp_col" < (
      MAKE_TIMESTAMP(2010, 1, 1, 0, 0, 0.0) + INTERVAL '3' MONTH
    )
    AND "t0"."timestamp_col" < (
      CAST(CURRENT_TIMESTAMP AS TIMESTAMP) + INTERVAL '10' DAY
    )
) AS "t1"