SELECT
  "t2"."b",
  COUNT(*) AS "b_count"
FROM (
  SELECT
    "t1"."b"
  FROM (
    SELECT
      "t0"."a",
      "t0"."b"
    FROM "t" AS "t0"
    ORDER BY
      "t0"."b" ASC
  ) AS "t1"
) AS "t2"
GROUP BY
  1