SELECT
  *
FROM (
  SELECT
    "t9"."cntrycode" AS "cntrycode",
    COUNT(*) AS "numcust",
    SUM("t9"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      IFF(0 >= 0, SUBSTRING("t8"."c_phone", 0 + 1, 2), SUBSTRING("t8"."c_phone", 0, 2)) AS "cntrycode",
      "t8"."c_acctbal" AS "c_acctbal"
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
        FROM "CUSTOMER" AS "t0"
      ) AS "t2"
      WHERE
        IFF(0 >= 0, SUBSTRING("t2"."c_phone", 0 + 1, 2), SUBSTRING("t2"."c_phone", 0, 2)) IN ('13', '31', '23', '29', '30', '18', '17')
        AND (
          "t2"."c_acctbal" > (
            SELECT
              AVG("t5"."c_acctbal") AS "Mean(c_acctbal)"
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
                FROM "CUSTOMER" AS "t0"
              ) AS "t2"
              WHERE
                (
                  "t2"."c_acctbal" > 0.0
                )
                AND IFF(0 >= 0, SUBSTRING("t2"."c_phone", 0 + 1, 2), SUBSTRING("t2"."c_phone", 0, 2)) IN ('13', '31', '23', '29', '30', '18', '17')
            ) AS "t5"
          )
        )
        AND NOT (
          EXISTS(
            (
              SELECT
                1 AS "1"
              FROM (
                SELECT
                  *
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
                  FROM "ORDERS" AS "t1"
                ) AS "t3"
                WHERE
                  (
                    "t3"."o_custkey" = "t2"."c_custkey"
                  )
              ) AS "t4"
            )
          )
        )
    ) AS "t8"
  ) AS "t9"
  GROUP BY
    1
) AS "t10"
ORDER BY
  "t10"."cntrycode" ASC