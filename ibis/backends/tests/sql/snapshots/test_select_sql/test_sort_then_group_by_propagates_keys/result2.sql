SELECT
  "t1"."b",
  COUNT(*) AS "b_count"
FROM (
  SELECT
    "t0"."b"
  FROM "t" AS "t0"
  ORDER BY
    "t0"."b" ASC
) AS "t1"
GROUP BY
  1