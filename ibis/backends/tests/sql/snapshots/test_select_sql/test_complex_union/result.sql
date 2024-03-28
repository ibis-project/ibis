SELECT
  "t6"."diag",
  "t6"."status"
FROM (
  SELECT
    *
  FROM (
    SELECT
      CAST("t2"."diag" AS INT) AS "diag",
      "t2"."status"
    FROM (
      SELECT
        "t0"."diag" + CAST(1 AS TINYINT) AS "diag",
        "t0"."status"
      FROM "aids2_one" AS "t0"
    ) AS "t2"
  ) AS "t4"
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      CAST("t3"."diag" AS INT) AS "diag",
      "t3"."status"
    FROM (
      SELECT
        "t1"."diag" + CAST(1 AS TINYINT) AS "diag",
        "t1"."status"
      FROM "aids2_two" AS "t1"
    ) AS "t3"
  ) AS "t5"
) AS "t6"