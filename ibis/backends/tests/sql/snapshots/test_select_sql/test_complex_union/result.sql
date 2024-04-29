SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      CAST("t0"."diag" + CAST(1 AS TINYINT) AS INT) AS "diag",
      "t0"."status"
    FROM "aids2_one" AS "t0"
  ) AS "t2"
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      CAST("t1"."diag" + CAST(1 AS TINYINT) AS INT) AS "diag",
      "t1"."status"
    FROM "aids2_two" AS "t1"
  ) AS "t3"
) AS "t4"