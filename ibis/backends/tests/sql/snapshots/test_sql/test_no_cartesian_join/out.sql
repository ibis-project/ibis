SELECT
  "t13"."customer_id",
  "t13"."first_name",
  "t13"."last_name",
  "t13"."first_order",
  "t13"."most_recent_order",
  "t13"."number_of_orders",
  "t11"."total_amount" AS "customer_lifetime_value"
FROM (
  SELECT
    "t10"."customer_id",
    "t10"."first_name",
    "t10"."last_name",
    "t10"."first_order",
    "t10"."most_recent_order",
    "t10"."number_of_orders"
  FROM (
    SELECT
      "t3"."customer_id",
      "t3"."first_name",
      "t3"."last_name",
      "t7"."customer_id" AS "customer_id_right",
      "t7"."first_order",
      "t7"."most_recent_order",
      "t7"."number_of_orders"
    FROM "customers" AS "t3"
    LEFT OUTER JOIN (
      SELECT
        "t2"."customer_id",
        MIN("t2"."order_date") AS "first_order",
        MAX("t2"."order_date") AS "most_recent_order",
        COUNT("t2"."order_id") AS "number_of_orders"
      FROM "orders" AS "t2"
      GROUP BY
        1
    ) AS "t7"
      ON "t3"."customer_id" = "t7"."customer_id"
  ) AS "t10"
) AS "t13"
LEFT OUTER JOIN (
  SELECT
    "t8"."customer_id",
    SUM("t8"."amount") AS "total_amount"
  FROM (
    SELECT
      "t4"."payment_id",
      "t4"."order_id",
      "t4"."payment_method",
      "t4"."amount",
      "t5"."order_id" AS "order_id_right",
      "t5"."customer_id",
      "t5"."order_date",
      "t5"."status"
    FROM "payments" AS "t4"
    LEFT OUTER JOIN "orders" AS "t5"
      ON "t4"."order_id" = "t5"."order_id"
  ) AS "t8"
  GROUP BY
    1
) AS "t11"
  ON "t13"."customer_id" = "t11"."customer_id"