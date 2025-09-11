SELECT
  "t1"."b"
FROM (
  SELECT
    *
  FROM "t" AS "t0"
  ORDER BY
    UPPER("t0"."a") ASC,
    "t0"."b" ASC
) AS "t1"