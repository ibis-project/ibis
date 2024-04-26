WITH "t2" AS (
  SELECT
    "t0"."c_custkey",
    "t0"."c_name",
    "t0"."c_address",
    "t0"."c_nationkey",
    "t0"."c_phone",
    CAST("t0"."c_acctbal" AS DECIMAL(15, 2)) AS "c_acctbal",
    "t0"."c_mktsegment",
    "t0"."c_comment"
  FROM "hive"."ibis_sf1"."customer" AS "t0"
)
SELECT
  *
FROM (
  SELECT
    "t9"."cntrycode",
    COUNT(*) AS "numcust",
    SUM("t9"."c_acctbal") AS "totacctbal"
  FROM (
    SELECT
      SUBSTRING("t8"."c_phone", IF((
        0 + 1
      ) >= 1, 0 + 1, 0 + 1 + LENGTH("t8"."c_phone")), 2) AS "cntrycode",
      "t8"."c_acctbal"
    FROM (
      SELECT
        *
      FROM "t2" AS "t4"
      WHERE
        SUBSTRING("t4"."c_phone", IF((
          0 + 1
        ) >= 1, 0 + 1, 0 + 1 + LENGTH("t4"."c_phone")), 2) IN ('13', '31', '23', '29', '30', '18', '17')
        AND "t4"."c_acctbal" > (
          SELECT
            AVG("t6"."c_acctbal") AS "Mean(c_acctbal)"
          FROM (
            SELECT
              *
            FROM "t2" AS "t4"
            WHERE
              "t4"."c_acctbal" > CAST(0.0 AS DOUBLE)
              AND SUBSTRING("t4"."c_phone", IF((
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
                "t1"."o_orderkey",
                "t1"."o_custkey",
                "t1"."o_orderstatus",
                CAST("t1"."o_totalprice" AS DECIMAL(15, 2)) AS "o_totalprice",
                "t1"."o_orderdate",
                "t1"."o_orderpriority",
                "t1"."o_clerk",
                "t1"."o_shippriority",
                "t1"."o_comment"
              FROM "hive"."ibis_sf1"."orders" AS "t1"
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