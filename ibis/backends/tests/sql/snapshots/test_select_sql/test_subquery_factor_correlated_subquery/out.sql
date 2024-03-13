WITH "t8" AS (
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
)
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
FROM "t8" AS "t9"
WHERE
  "t9"."amount" > (
    SELECT
      AVG("t11"."amount") AS "Mean(amount)"
    FROM (
      SELECT
        "t10"."c_custkey",
        "t10"."c_name",
        "t10"."c_address",
        "t10"."c_nationkey",
        "t10"."c_phone",
        "t10"."c_acctbal",
        "t10"."c_mktsegment",
        "t10"."c_comment",
        "t10"."region",
        "t10"."amount",
        "t10"."odate"
      FROM "t8" AS "t10"
      WHERE
        "t10"."region" = "t9"."region"
    ) AS "t11"
  )
LIMIT 10