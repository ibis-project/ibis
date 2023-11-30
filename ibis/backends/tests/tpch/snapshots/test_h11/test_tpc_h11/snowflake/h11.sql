SELECT
  *
FROM (
  SELECT
    *
  FROM (
    SELECT
      "t9"."ps_partkey" AS "ps_partkey",
      SUM("t9"."ps_supplycost" * "t9"."ps_availqty") AS "value"
    FROM (
      SELECT
        *
      FROM (
        SELECT
          "t3"."ps_partkey" AS "ps_partkey",
          "t3"."ps_suppkey" AS "ps_suppkey",
          "t3"."ps_availqty" AS "ps_availqty",
          "t3"."ps_supplycost" AS "ps_supplycost",
          "t3"."ps_comment" AS "ps_comment",
          "t4"."s_suppkey" AS "s_suppkey",
          "t4"."s_name" AS "s_name",
          "t4"."s_address" AS "s_address",
          "t4"."s_nationkey" AS "s_nationkey",
          "t4"."s_phone" AS "s_phone",
          "t4"."s_acctbal" AS "s_acctbal",
          "t4"."s_comment" AS "s_comment",
          "t5"."n_nationkey" AS "n_nationkey",
          "t5"."n_name" AS "n_name",
          "t5"."n_regionkey" AS "n_regionkey",
          "t5"."n_comment" AS "n_comment"
        FROM (
          SELECT
            "t0"."PS_PARTKEY" AS "ps_partkey",
            "t0"."PS_SUPPKEY" AS "ps_suppkey",
            "t0"."PS_AVAILQTY" AS "ps_availqty",
            "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
            "t0"."PS_COMMENT" AS "ps_comment"
          FROM "PARTSUPP" AS "t0"
        ) AS "t3"
        INNER JOIN (
          SELECT
            "t1"."S_SUPPKEY" AS "s_suppkey",
            "t1"."S_NAME" AS "s_name",
            "t1"."S_ADDRESS" AS "s_address",
            "t1"."S_NATIONKEY" AS "s_nationkey",
            "t1"."S_PHONE" AS "s_phone",
            "t1"."S_ACCTBAL" AS "s_acctbal",
            "t1"."S_COMMENT" AS "s_comment"
          FROM "SUPPLIER" AS "t1"
        ) AS "t4"
          ON "t3"."ps_suppkey" = "t4"."s_suppkey"
        INNER JOIN (
          SELECT
            "t2"."N_NATIONKEY" AS "n_nationkey",
            "t2"."N_NAME" AS "n_name",
            "t2"."N_REGIONKEY" AS "n_regionkey",
            "t2"."N_COMMENT" AS "n_comment"
          FROM "NATION" AS "t2"
        ) AS "t5"
          ON "t5"."n_nationkey" = "t4"."s_nationkey"
      ) AS "t8"
      WHERE
        (
          "t8"."n_name" = 'GERMANY'
        )
    ) AS "t9"
    GROUP BY
      1
  ) AS "t10"
  WHERE
    (
      "t10"."value" > (
        (
          SELECT
            SUM("t9"."ps_supplycost" * "t9"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
          FROM (
            SELECT
              *
            FROM (
              SELECT
                "t3"."ps_partkey" AS "ps_partkey",
                "t3"."ps_suppkey" AS "ps_suppkey",
                "t3"."ps_availqty" AS "ps_availqty",
                "t3"."ps_supplycost" AS "ps_supplycost",
                "t3"."ps_comment" AS "ps_comment",
                "t4"."s_suppkey" AS "s_suppkey",
                "t4"."s_name" AS "s_name",
                "t4"."s_address" AS "s_address",
                "t4"."s_nationkey" AS "s_nationkey",
                "t4"."s_phone" AS "s_phone",
                "t4"."s_acctbal" AS "s_acctbal",
                "t4"."s_comment" AS "s_comment",
                "t5"."n_nationkey" AS "n_nationkey",
                "t5"."n_name" AS "n_name",
                "t5"."n_regionkey" AS "n_regionkey",
                "t5"."n_comment" AS "n_comment"
              FROM (
                SELECT
                  "t0"."PS_PARTKEY" AS "ps_partkey",
                  "t0"."PS_SUPPKEY" AS "ps_suppkey",
                  "t0"."PS_AVAILQTY" AS "ps_availqty",
                  "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
                  "t0"."PS_COMMENT" AS "ps_comment"
                FROM "PARTSUPP" AS "t0"
              ) AS "t3"
              INNER JOIN (
                SELECT
                  "t1"."S_SUPPKEY" AS "s_suppkey",
                  "t1"."S_NAME" AS "s_name",
                  "t1"."S_ADDRESS" AS "s_address",
                  "t1"."S_NATIONKEY" AS "s_nationkey",
                  "t1"."S_PHONE" AS "s_phone",
                  "t1"."S_ACCTBAL" AS "s_acctbal",
                  "t1"."S_COMMENT" AS "s_comment"
                FROM "SUPPLIER" AS "t1"
              ) AS "t4"
                ON "t3"."ps_suppkey" = "t4"."s_suppkey"
              INNER JOIN (
                SELECT
                  "t2"."N_NATIONKEY" AS "n_nationkey",
                  "t2"."N_NAME" AS "n_name",
                  "t2"."N_REGIONKEY" AS "n_regionkey",
                  "t2"."N_COMMENT" AS "n_comment"
                FROM "NATION" AS "t2"
              ) AS "t5"
                ON "t5"."n_nationkey" = "t4"."s_nationkey"
            ) AS "t8"
            WHERE
              (
                "t8"."n_name" = 'GERMANY'
              )
          ) AS "t9"
        ) * 0.0001
      )
    )
) AS "t12"
ORDER BY
  "t12"."value" DESC NULLS LAST