SELECT
  "t8"."ps_partkey",
  "t8"."value"
FROM (
  SELECT
    "t7"."ps_partkey",
    SUM("t7"."ps_supplycost" * "t7"."ps_availqty") AS "value"
  FROM (
    SELECT
      "t6"."ps_partkey",
      "t6"."ps_suppkey",
      "t6"."ps_availqty",
      "t6"."ps_supplycost",
      "t6"."ps_comment",
      "t6"."s_suppkey",
      "t6"."s_name",
      "t6"."s_address",
      "t6"."s_nationkey",
      "t6"."s_phone",
      "t6"."s_acctbal",
      "t6"."s_comment",
      "t6"."n_nationkey",
      "t6"."n_name",
      "t6"."n_regionkey",
      "t6"."n_comment"
    FROM (
      SELECT
        "t3"."ps_partkey",
        "t3"."ps_suppkey",
        "t3"."ps_availqty",
        "t3"."ps_supplycost",
        "t3"."ps_comment",
        "t4"."s_suppkey",
        "t4"."s_name",
        "t4"."s_address",
        "t4"."s_nationkey",
        "t4"."s_phone",
        "t4"."s_acctbal",
        "t4"."s_comment",
        "t5"."n_nationkey",
        "t5"."n_name",
        "t5"."n_regionkey",
        "t5"."n_comment"
      FROM "partsupp" AS "t3"
      INNER JOIN "supplier" AS "t4"
        ON "t3"."ps_suppkey" = "t4"."s_suppkey"
      INNER JOIN "nation" AS "t5"
        ON "t5"."n_nationkey" = "t4"."s_nationkey"
    ) AS "t6"
    WHERE
      "t6"."n_name" = 'GERMANY'
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
WHERE
  "t8"."value" > (
    (
      SELECT
        SUM("t7"."ps_supplycost" * "t7"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          "t6"."ps_partkey",
          "t6"."ps_suppkey",
          "t6"."ps_availqty",
          "t6"."ps_supplycost",
          "t6"."ps_comment",
          "t6"."s_suppkey",
          "t6"."s_name",
          "t6"."s_address",
          "t6"."s_nationkey",
          "t6"."s_phone",
          "t6"."s_acctbal",
          "t6"."s_comment",
          "t6"."n_nationkey",
          "t6"."n_name",
          "t6"."n_regionkey",
          "t6"."n_comment"
        FROM (
          SELECT
            "t3"."ps_partkey",
            "t3"."ps_suppkey",
            "t3"."ps_availqty",
            "t3"."ps_supplycost",
            "t3"."ps_comment",
            "t4"."s_suppkey",
            "t4"."s_name",
            "t4"."s_address",
            "t4"."s_nationkey",
            "t4"."s_phone",
            "t4"."s_acctbal",
            "t4"."s_comment",
            "t5"."n_nationkey",
            "t5"."n_name",
            "t5"."n_regionkey",
            "t5"."n_comment"
          FROM "partsupp" AS "t3"
          INNER JOIN "supplier" AS "t4"
            ON "t3"."ps_suppkey" = "t4"."s_suppkey"
          INNER JOIN "nation" AS "t5"
            ON "t5"."n_nationkey" = "t4"."s_nationkey"
        ) AS "t6"
        WHERE
          "t6"."n_name" = 'GERMANY'
      ) AS "t7"
    ) * CAST(0.0001 AS DOUBLE)
  )
ORDER BY
  "t8"."value" DESC