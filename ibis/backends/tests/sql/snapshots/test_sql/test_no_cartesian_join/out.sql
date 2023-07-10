SELECT
  t9.customer_id AS customer_id,
  t9.first_name AS first_name,
  t9.last_name AS last_name,
  t9.first_order AS first_order,
  t9.most_recent_order AS most_recent_order,
  t9.number_of_orders AS number_of_orders,
  t11.total_amount AS customer_lifetime_value
FROM (
  SELECT
    t0.customer_id AS customer_id,
    t0.first_name AS first_name,
    t0.last_name AS last_name,
    t5.first_order AS first_order,
    t5.most_recent_order AS most_recent_order,
    t5.number_of_orders AS number_of_orders
  FROM customers AS t0
  LEFT OUTER JOIN (
    SELECT
      t1.customer_id AS customer_id,
      MIN(t1.order_date) AS first_order,
      MAX(t1.order_date) AS most_recent_order,
      COUNT(t1.order_id) AS number_of_orders
    FROM orders AS t1
    GROUP BY
      1
  ) AS t5
    ON t0.customer_id = t5.customer_id
) AS t9
LEFT OUTER JOIN (
  SELECT
    t7.customer_id AS customer_id,
    SUM(t7.amount) AS total_amount
  FROM (
    SELECT
      t2.payment_id AS payment_id,
      t2.order_id AS order_id,
      t2.payment_method AS payment_method,
      t2.amount AS amount,
      t3.order_id AS order_id_right,
      t3.customer_id AS customer_id,
      t3.order_date AS order_date,
      t3.status AS status
    FROM payments AS t2
    LEFT OUTER JOIN orders AS t3
      ON t2.order_id = t3.order_id
  ) AS t7
  GROUP BY
    1
) AS t11
  ON t9.customer_id = t11.customer_id