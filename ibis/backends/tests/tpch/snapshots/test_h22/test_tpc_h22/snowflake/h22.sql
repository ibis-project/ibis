SELECT
  "t7"."cntrycode",
  "t7"."numcust",
  "t7"."totacctbal"
FROM (
  SELECT
    "t6"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t6"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      IFF(
        (
          0 + 1
        ) >= 1,
        SUBSTRING("t2"."c_phone", 0 + 1, 2),
        SUBSTRING("t2"."c_phone", 0 + 1 + LENGTH("t2"."c_phone"), 2)
      ) AS "cntrycode",
      "t2"."c_acctbal"
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
      IFF(
        (
          0 + 1
        ) >= 1,
        SUBSTRING("t2"."c_phone", 0 + 1, 2),
        SUBSTRING("t2"."c_phone", 0 + 1 + LENGTH("t2"."c_phone"), 2)
      ) IN ('13', '31', '23', '29', '30', '18', '17')
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
            AND IFF(
              (
                0 + 1
              ) >= 1,
              SUBSTRING("t0"."C_PHONE", 0 + 1, 2),
              SUBSTRING("t0"."C_PHONE", 0 + 1 + LENGTH("t0"."C_PHONE"), 2)
            ) IN ('13', '31', '23', '29', '30', '18', '17')
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
  GROUP BY
    1
) AS "t7"
ORDER BY
  "t7"."cntrycode" ASC