WITH "t2" AS (
  SELECT
    "t0"."C_CUSTKEY" AS "c_custkey",
    "t0"."C_NAME" AS "c_name",
    "t0"."C_ADDRESS" AS "c_address",
    "t0"."C_NATIONKEY" AS "c_nationkey",
    "t0"."C_PHONE" AS "c_phone",
    "t0"."C_ACCTBAL" AS "c_acctbal",
    "t0"."C_MKTSEGMENT" AS "c_mktsegment",
    "t0"."C_COMMENT" AS "c_comment"
  FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."CUSTOMER" AS "t0"
)
SELECT
  "t10"."cntrycode",
  "t10"."numcust",
  "t10"."totacctbal"
FROM (
  SELECT
    "t9"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t9"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      SUBSTRING("t8"."c_phone", IFF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH("t8"."c_phone")), 2) AS "cntrycode",
      "t8"."c_acctbal"
    FROM (
      SELECT
        "t4"."c_custkey",
        "t4"."c_name",
        "t4"."c_address",
        "t4"."c_nationkey",
        "t4"."c_phone",
        "t4"."c_acctbal",
        "t4"."c_mktsegment",
        "t4"."c_comment"
      FROM "t2" AS "t4"
      WHERE
        SUBSTRING("t4"."c_phone", IFF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH("t4"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND "t4"."c_acctbal" > (
          SELECT
            AVG("t6"."c_acctbal") AS "Mean(c_acctbal)"
          FROM (
            SELECT
              "t4"."c_custkey",
              "t4"."c_name",
              "t4"."c_address",
              "t4"."c_nationkey",
              "t4"."c_phone",
              "t4"."c_acctbal",
              "t4"."c_mktsegment",
              "t4"."c_comment"
            FROM "t2" AS "t4"
            WHERE
              "t4"."c_acctbal" > 0.0
              AND SUBSTRING("t4"."c_phone", IFF((
                0 + 1
              ) >= 1, 0 + 1, 0 + 1 + LENGTH("t4"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
          ) AS "t6"
        )
        AND NOT (
          EXISTS(
            SELECT
              1
            FROM (
              SELECT
                "t1"."O_ORDERKEY" AS "o_orderkey",
                "t1"."O_CUSTKEY" AS "o_custkey",
                "t1"."O_ORDERSTATUS" AS "o_orderstatus",
                "t1"."O_TOTALPRICE" AS "o_totalprice",
                "t1"."O_ORDERDATE" AS "o_orderdate",
                "t1"."O_ORDERPRIORITY" AS "o_orderpriority",
                "t1"."O_CLERK" AS "o_clerk",
                "t1"."O_SHIPPRIORITY" AS "o_shippriority",
                "t1"."O_COMMENT" AS "o_comment"
              FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
            ) AS "t3"
            WHERE
              "t3"."o_custkey" = "t4"."c_custkey"
          )
        )
    ) AS "t8"
  ) AS "t9"
  GROUP BY
    1
) AS "t10"
ORDER BY
  "t10"."cntrycode" ASC