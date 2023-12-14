SELECT
  "t18"."ps_partkey" AS "ps_partkey",
  "t18"."value" AS "value"
FROM (
  SELECT
    "t16"."ps_partkey" AS "ps_partkey",
    SUM("t16"."ps_supplycost" * "t16"."ps_availqty") AS "value"
  FROM (
    SELECT
      "t14"."ps_partkey" AS "ps_partkey",
      "t14"."ps_suppkey" AS "ps_suppkey",
      "t14"."ps_availqty" AS "ps_availqty",
      "t14"."ps_supplycost" AS "ps_supplycost",
      "t14"."ps_comment" AS "ps_comment",
      "t14"."s_suppkey" AS "s_suppkey",
      "t14"."s_name" AS "s_name",
      "t14"."s_address" AS "s_address",
      "t14"."s_nationkey" AS "s_nationkey",
      "t14"."s_phone" AS "s_phone",
      "t14"."s_acctbal" AS "s_acctbal",
      "t14"."s_comment" AS "s_comment",
      "t14"."n_nationkey" AS "n_nationkey",
      "t14"."n_name" AS "n_name",
      "t14"."n_regionkey" AS "n_regionkey",
      "t14"."n_comment" AS "n_comment"
    FROM (
      SELECT
        "t3"."ps_partkey" AS "ps_partkey",
        "t3"."ps_suppkey" AS "ps_suppkey",
        "t3"."ps_availqty" AS "ps_availqty",
        "t3"."ps_supplycost" AS "ps_supplycost",
        "t3"."ps_comment" AS "ps_comment",
        "t6"."s_suppkey" AS "s_suppkey",
        "t6"."s_name" AS "s_name",
        "t6"."s_address" AS "s_address",
        "t6"."s_nationkey" AS "s_nationkey",
        "t6"."s_phone" AS "s_phone",
        "t6"."s_acctbal" AS "s_acctbal",
        "t6"."s_comment" AS "s_comment",
        "t8"."n_nationkey" AS "n_nationkey",
        "t8"."n_name" AS "n_name",
        "t8"."n_regionkey" AS "n_regionkey",
        "t8"."n_comment" AS "n_comment"
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
      ) AS "t6"
        ON "t3"."ps_suppkey" = "t6"."s_suppkey"
      INNER JOIN (
        SELECT
          "t2"."N_NATIONKEY" AS "n_nationkey",
          "t2"."N_NAME" AS "n_name",
          "t2"."N_REGIONKEY" AS "n_regionkey",
          "t2"."N_COMMENT" AS "n_comment"
        FROM "NATION" AS "t2"
      ) AS "t8"
        ON "t8"."n_nationkey" = "t6"."s_nationkey"
    ) AS "t14"
    WHERE
      "t14"."n_name" = 'GERMANY'
  ) AS "t16"
  GROUP BY
    1
) AS "t18"
WHERE
  "t18"."value" > (
    (
      SELECT
        SUM("t17"."ps_supplycost" * "t17"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          "t15"."ps_partkey" AS "ps_partkey",
          "t15"."ps_suppkey" AS "ps_suppkey",
          "t15"."ps_availqty" AS "ps_availqty",
          "t15"."ps_supplycost" AS "ps_supplycost",
          "t15"."ps_comment" AS "ps_comment",
          "t15"."s_suppkey" AS "s_suppkey",
          "t15"."s_name" AS "s_name",
          "t15"."s_address" AS "s_address",
          "t15"."s_nationkey" AS "s_nationkey",
          "t15"."s_phone" AS "s_phone",
          "t15"."s_acctbal" AS "s_acctbal",
          "t15"."s_comment" AS "s_comment",
          "t15"."n_nationkey" AS "n_nationkey",
          "t15"."n_name" AS "n_name",
          "t15"."n_regionkey" AS "n_regionkey",
          "t15"."n_comment" AS "n_comment"
        FROM (
          SELECT
            "t3"."ps_partkey" AS "ps_partkey",
            "t3"."ps_suppkey" AS "ps_suppkey",
            "t3"."ps_availqty" AS "ps_availqty",
            "t3"."ps_supplycost" AS "ps_supplycost",
            "t3"."ps_comment" AS "ps_comment",
            "t7"."s_suppkey" AS "s_suppkey",
            "t7"."s_name" AS "s_name",
            "t7"."s_address" AS "s_address",
            "t7"."s_nationkey" AS "s_nationkey",
            "t7"."s_phone" AS "s_phone",
            "t7"."s_acctbal" AS "s_acctbal",
            "t7"."s_comment" AS "s_comment",
            "t9"."n_nationkey" AS "n_nationkey",
            "t9"."n_name" AS "n_name",
            "t9"."n_regionkey" AS "n_regionkey",
            "t9"."n_comment" AS "n_comment"
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
          ) AS "t7"
            ON "t3"."ps_suppkey" = "t7"."s_suppkey"
          INNER JOIN (
            SELECT
              "t2"."N_NATIONKEY" AS "n_nationkey",
              "t2"."N_NAME" AS "n_name",
              "t2"."N_REGIONKEY" AS "n_regionkey",
              "t2"."N_COMMENT" AS "n_comment"
            FROM "NATION" AS "t2"
          ) AS "t9"
            ON "t9"."n_nationkey" = "t7"."s_nationkey"
        ) AS "t15"
        WHERE
          "t15"."n_name" = 'GERMANY'
      ) AS "t17"
    ) * 0.0001
  )
ORDER BY
  "t18"."value" DESC NULLS LAST