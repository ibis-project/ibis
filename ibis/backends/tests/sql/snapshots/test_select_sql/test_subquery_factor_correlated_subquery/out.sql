SELECT
  "t8"."c_custkey",
  "t8"."c_name",
  "t8"."c_address",
  "t8"."c_nationkey",
  "t8"."c_phone",
  "t8"."c_acctbal",
  "t8"."c_mktsegment",
  "t8"."c_comment",
  "t8"."region",
  "t8"."amount",
  "t8"."odate"
FROM (
  SELECT
    "t6"."c_custkey",
    "t6"."c_name",
    "t6"."c_address",
    "t6"."c_nationkey",
    "t6"."c_phone",
    "t6"."c_acctbal",
    "t6"."c_mktsegment",
    "t6"."c_comment",
    "t4"."r_name" AS "region",
    "t7"."o_totalprice" AS "amount",
    CAST("t7"."o_orderdate" AS TIMESTAMP) AS "odate"
  FROM "tpch_region" AS "t4"
  INNER JOIN "tpch_nation" AS "t5"
    ON "t4"."r_regionkey" = "t5"."n_regionkey"
  INNER JOIN "tpch_customer" AS "t6"
    ON "t6"."c_nationkey" = "t5"."n_nationkey"
  INNER JOIN "tpch_orders" AS "t7"
    ON "t7"."o_custkey" = "t6"."c_custkey"
) AS "t8"
WHERE
  "t8"."amount" > (
    SELECT
      AVG("t10"."amount") AS "Mean(amount)"
    FROM (
      SELECT
        "t9"."c_custkey",
        "t9"."c_name",
        "t9"."c_address",
        "t9"."c_nationkey",
        "t9"."c_phone",
        "t9"."c_acctbal",
        "t9"."c_mktsegment",
        "t9"."c_comment",
        "t9"."region",
        "t9"."amount",
        "t9"."odate"
      FROM (
        SELECT
          "t6"."c_custkey",
          "t6"."c_name",
          "t6"."c_address",
          "t6"."c_nationkey",
          "t6"."c_phone",
          "t6"."c_acctbal",
          "t6"."c_mktsegment",
          "t6"."c_comment",
          "t4"."r_name" AS "region",
          "t7"."o_totalprice" AS "amount",
          CAST("t7"."o_orderdate" AS TIMESTAMP) AS "odate"
        FROM "tpch_region" AS "t4"
        INNER JOIN "tpch_nation" AS "t5"
          ON "t4"."r_regionkey" = "t5"."n_regionkey"
        INNER JOIN "tpch_customer" AS "t6"
          ON "t6"."c_nationkey" = "t5"."n_nationkey"
        INNER JOIN "tpch_orders" AS "t7"
          ON "t7"."o_custkey" = "t6"."c_custkey"
      ) AS "t9"
      WHERE
        "t9"."region" = "t8"."region"
    ) AS "t10"
  )
LIMIT 10