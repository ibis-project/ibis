SELECT
  "t8"."diag",
  "t8"."status"
FROM (
  SELECT
    *
  FROM (
    SELECT
      CAST("t4"."diag" AS INT) AS "diag",
      "t4"."status"
    FROM (
      SELECT
        "t2"."diag" + CAST(1 AS TINYINT) AS "diag",
        "t2"."status"
      FROM (
        SELECT
          "t0"."diag",
          "t0"."status"
        FROM "aids2_one" AS "t0"
      ) AS "t2"
    ) AS "t4"
  ) AS "t6"
  UNION ALL
  SELECT
    *
  FROM (
    SELECT
      CAST("t5"."diag" AS INT) AS "diag",
      "t5"."status"
    FROM (
      SELECT
        "t3"."diag" + CAST(1 AS TINYINT) AS "diag",
        "t3"."status"
      FROM (
        SELECT
          "t1"."diag",
          "t1"."status"
        FROM "aids2_two" AS "t1"
      ) AS "t3"
    ) AS "t5"
  ) AS "t7"
) AS "t8"