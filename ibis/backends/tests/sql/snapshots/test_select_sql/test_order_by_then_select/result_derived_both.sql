SELECT
  "t1"."b"
FROM (
  SELECT
    *
  FROM "t" AS "t0"
  ORDER BY
    UPPER("t0"."a") || CAST("t0"."b" AS TEXT) || 'foo' ASC
) AS "t1"