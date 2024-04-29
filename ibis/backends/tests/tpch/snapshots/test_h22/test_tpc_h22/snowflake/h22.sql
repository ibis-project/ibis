SELECT
  *
FROM (
  SELECT
    "t7"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t7"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      SUBSTRING("t6"."c_phone", IFF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH("t6"."c_phone")), 2) AS "cntrycode",
      "t6"."c_acctbal"
    FROM (
      SELECT
        *
      FROM (
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
      ) AS "t2"
      WHERE
        SUBSTRING("t2"."c_phone", IFF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH("t2"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND "t2"."c_acctbal" > (
          SELECT
            AVG("t3"."c_acctbal") AS "Mean(c_acctbal)"
          FROM (
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
            WHERE
              "t0"."C_ACCTBAL" > 0.0
              AND SUBSTRING("t0"."C_PHONE", IFF((
                0 + 1
              ) >= 1, 0 + 1, 0 + 1 + LENGTH("t0"."C_PHONE")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
          ) AS "t3"
        )
        AND NOT (
          EXISTS(
            SELECT
              1
            FROM "SNOWFLAKE_SAMPLE_DATA"."TPCH_SF1"."ORDERS" AS "t1"
            WHERE
              "t1"."O_CUSTKEY" = "t2"."c_custkey"
          )
        )
    ) AS "t6"
  ) AS "t7"
  GROUP BY
    1
) AS "t8"
ORDER BY
  "t8"."cntrycode" ASC