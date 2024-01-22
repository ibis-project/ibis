SELECT
  "t11"."ps_partkey",
  "t11"."value"
FROM (
  SELECT
    "t10"."ps_partkey",
    SUM("t10"."ps_supplycost" * "t10"."ps_availqty") AS "value"
  FROM (
    SELECT
      "t9"."ps_partkey",
      "t9"."ps_suppkey",
      "t9"."ps_availqty",
      "t9"."ps_supplycost",
      "t9"."ps_comment",
      "t9"."s_suppkey",
      "t9"."s_name",
      "t9"."s_address",
      "t9"."s_nationkey",
      "t9"."s_phone",
      "t9"."s_acctbal",
      "t9"."s_comment",
      "t9"."n_nationkey",
      "t9"."n_name",
      "t9"."n_regionkey",
      "t9"."n_comment"
    FROM (
      SELECT
        "t6"."ps_partkey",
        "t6"."ps_suppkey",
        "t6"."ps_availqty",
        "t6"."ps_supplycost",
        "t6"."ps_comment",
        "t7"."s_suppkey",
        "t7"."s_name",
        "t7"."s_address",
        "t7"."s_nationkey",
        "t7"."s_phone",
        "t7"."s_acctbal",
        "t7"."s_comment",
        "t8"."n_nationkey",
        "t8"."n_name",
        "t8"."n_regionkey",
        "t8"."n_comment"
      FROM (
        SELECT
          "t0"."PS_PARTKEY" AS "ps_partkey",
          "t0"."PS_SUPPKEY" AS "ps_suppkey",
          "t0"."PS_AVAILQTY" AS "ps_availqty",
          "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
          "t0"."PS_COMMENT" AS "ps_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t0"
      ) AS "t6"
      INNER JOIN (
        SELECT
          "t1"."S_SUPPKEY" AS "s_suppkey",
          "t1"."S_NAME" AS "s_name",
          "t1"."S_ADDRESS" AS "s_address",
          "t1"."S_NATIONKEY" AS "s_nationkey",
          "t1"."S_PHONE" AS "s_phone",
          "t1"."S_ACCTBAL" AS "s_acctbal",
          "t1"."S_COMMENT" AS "s_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t1"
      ) AS "t7"
        ON "t6"."ps_suppkey" = "t7"."s_suppkey"
      INNER JOIN (
        SELECT
          "t2"."N_NATIONKEY" AS "n_nationkey",
          "t2"."N_NAME" AS "n_name",
          "t2"."N_REGIONKEY" AS "n_regionkey",
          "t2"."N_COMMENT" AS "n_comment"
        FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t2"
      ) AS "t8"
        ON "t8"."n_nationkey" = "t7"."s_nationkey"
    ) AS "t9"
    WHERE
      "t9"."n_name" = 'GERMANY'
  ) AS "t10"
  GROUP BY
    1
) AS "t11"
WHERE
  "t11"."value" > (
    (
      SELECT
        SUM("t10"."ps_supplycost" * "t10"."ps_availqty") AS "Sum(Multiply(ps_supplycost, ps_availqty))"
      FROM (
        SELECT
          "t9"."ps_partkey",
          "t9"."ps_suppkey",
          "t9"."ps_availqty",
          "t9"."ps_supplycost",
          "t9"."ps_comment",
          "t9"."s_suppkey",
          "t9"."s_name",
          "t9"."s_address",
          "t9"."s_nationkey",
          "t9"."s_phone",
          "t9"."s_acctbal",
          "t9"."s_comment",
          "t9"."n_nationkey",
          "t9"."n_name",
          "t9"."n_regionkey",
          "t9"."n_comment"
        FROM (
          SELECT
            "t6"."ps_partkey",
            "t6"."ps_suppkey",
            "t6"."ps_availqty",
            "t6"."ps_supplycost",
            "t6"."ps_comment",
            "t7"."s_suppkey",
            "t7"."s_name",
            "t7"."s_address",
            "t7"."s_nationkey",
            "t7"."s_phone",
            "t7"."s_acctbal",
            "t7"."s_comment",
            "t8"."n_nationkey",
            "t8"."n_name",
            "t8"."n_regionkey",
            "t8"."n_comment"
          FROM (
            SELECT
              "t0"."PS_PARTKEY" AS "ps_partkey",
              "t0"."PS_SUPPKEY" AS "ps_suppkey",
              "t0"."PS_AVAILQTY" AS "ps_availqty",
              "t0"."PS_SUPPLYCOST" AS "ps_supplycost",
              "t0"."PS_COMMENT" AS "ps_comment"
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."PARTSUPP" AS "t0"
          ) AS "t6"
          INNER JOIN (
            SELECT
              "t1"."S_SUPPKEY" AS "s_suppkey",
              "t1"."S_NAME" AS "s_name",
              "t1"."S_ADDRESS" AS "s_address",
              "t1"."S_NATIONKEY" AS "s_nationkey",
              "t1"."S_PHONE" AS "s_phone",
              "t1"."S_ACCTBAL" AS "s_acctbal",
              "t1"."S_COMMENT" AS "s_comment"
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."SUPPLIER" AS "t1"
          ) AS "t7"
            ON "t6"."ps_suppkey" = "t7"."s_suppkey"
          INNER JOIN (
            SELECT
              "t2"."N_NATIONKEY" AS "n_nationkey",
              "t2"."N_NAME" AS "n_name",
              "t2"."N_REGIONKEY" AS "n_regionkey",
              "t2"."N_COMMENT" AS "n_comment"
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."NATION" AS "t2"
          ) AS "t8"
            ON "t8"."n_nationkey" = "t7"."s_nationkey"
        ) AS "t9"
        WHERE
          "t9"."n_name" = 'GERMANY'
      ) AS "t10"
    ) * 0.0001
  )
ORDER BY
  "t11"."value" DESC NULLS LAST