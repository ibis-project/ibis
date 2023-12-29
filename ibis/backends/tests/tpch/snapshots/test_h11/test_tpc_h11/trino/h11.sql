SELECT
  "t13"."ps_partkey",
  "t13"."value"
FROM (
  SELECT
    "t12"."ps_partkey",
    SUM("t12"."ps_supplycost" * "t12"."ps_availqty") AS "value"
  FROM (
    SELECT
      "t11"."ps_partkey",
      "t11"."ps_suppkey",
      "t11"."ps_availqty",
      "t11"."ps_supplycost",
      "t11"."ps_comment",
      "t11"."s_suppkey",
      "t11"."s_name",
      "t11"."s_address",
      "t11"."s_nationkey",
      "t11"."s_phone",
      "t11"."s_acctbal",
      "t11"."s_comment",
      "t11"."n_nationkey",
      "t11"."n_name",
      "t11"."n_regionkey",
      "t11"."n_comment"
    FROM (
      SELECT
        "t7"."ps_partkey",
        "t7"."ps_suppkey",
        "t7"."ps_availqty",
        "t7"."ps_supplycost",
        "t7"."ps_comment",
        "t8"."s_suppkey",
        "t8"."s_name",
        "t8"."s_address",
        "t8"."s_nationkey",
        "t8"."s_phone",
        "t8"."s_acctbal",
        "t8"."s_comment",
        "t6"."n_nationkey",
        "t6"."n_name",
        "t6"."n_regionkey",
        "t6"."n_comment"
      FROM (
        SELECT
          "t0"."ps_partkey",
          "t0"."ps_suppkey",
          "t0"."ps_availqty",
          CAST("t0"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
          "t0"."ps_comment"
        FROM "partsupp" AS "t0"
      ) AS "t7"
      INNER JOIN (
        SELECT
          "t1"."s_suppkey",
          "t1"."s_name",
          "t1"."s_address",
          "t1"."s_nationkey",
          "t1"."s_phone",
          CAST("t1"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
          "t1"."s_comment"
        FROM "supplier" AS "t1"
      ) AS "t8"
        ON "t7"."ps_suppkey" = "t8"."s_suppkey"
      INNER JOIN (
        SELECT
          "t2"."n_nationkey",
          "t2"."n_name",
          "t2"."n_regionkey",
          "t2"."n_comment"
        FROM "nation" AS "t2"
      ) AS "t6"
        ON "t6"."n_nationkey" = "t8"."s_nationkey"
    ) AS "t11"
    WHERE
      "t11"."n_name" = 'GERMANY'
  ) AS "t12"
  GROUP BY
    1
) AS "t13"
WHERE
  "t13"."value" > (
    (
      SELECT
        SUM("t12"."ps_supplycost" * "t12"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          "t11"."ps_partkey",
          "t11"."ps_suppkey",
          "t11"."ps_availqty",
          "t11"."ps_supplycost",
          "t11"."ps_comment",
          "t11"."s_suppkey",
          "t11"."s_name",
          "t11"."s_address",
          "t11"."s_nationkey",
          "t11"."s_phone",
          "t11"."s_acctbal",
          "t11"."s_comment",
          "t11"."n_nationkey",
          "t11"."n_name",
          "t11"."n_regionkey",
          "t11"."n_comment"
        FROM (
          SELECT
            "t7"."ps_partkey",
            "t7"."ps_suppkey",
            "t7"."ps_availqty",
            "t7"."ps_supplycost",
            "t7"."ps_comment",
            "t8"."s_suppkey",
            "t8"."s_name",
            "t8"."s_address",
            "t8"."s_nationkey",
            "t8"."s_phone",
            "t8"."s_acctbal",
            "t8"."s_comment",
            "t6"."n_nationkey",
            "t6"."n_name",
            "t6"."n_regionkey",
            "t6"."n_comment"
          FROM (
            SELECT
              "t0"."ps_partkey",
              "t0"."ps_suppkey",
              "t0"."ps_availqty",
              CAST("t0"."ps_supplycost" AS DECIMAL(15, 2)) AS "ps_supplycost",
              "t0"."ps_comment"
            FROM "partsupp" AS "t0"
          ) AS "t7"
          INNER JOIN (
            SELECT
              "t1"."s_suppkey",
              "t1"."s_name",
              "t1"."s_address",
              "t1"."s_nationkey",
              "t1"."s_phone",
              CAST("t1"."s_acctbal" AS DECIMAL(15, 2)) AS "s_acctbal",
              "t1"."s_comment"
            FROM "supplier" AS "t1"
          ) AS "t8"
            ON "t7"."ps_suppkey" = "t8"."s_suppkey"
          INNER JOIN (
            SELECT
              "t2"."n_nationkey",
              "t2"."n_name",
              "t2"."n_regionkey",
              "t2"."n_comment"
            FROM "nation" AS "t2"
          ) AS "t6"
            ON "t6"."n_nationkey" = "t8"."s_nationkey"
        ) AS "t11"
        WHERE
          "t11"."n_name" = 'GERMANY'
      ) AS "t12"
    ) * CAST(0.0001 AS DOUBLE)
  )
ORDER BY
  "t13"."value" DESC