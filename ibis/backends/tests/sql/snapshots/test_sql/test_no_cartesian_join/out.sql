SELECT
  t15.customer_id,
  t15.first_name,
  t15.last_name,
  t15.first_order,
  t15.most_recent_order,
  t15.number_of_orders,
  t13.total_amount AS customer_lifetime_value
FROM (
  SELECT
    t12.customer_id,
    t12.first_name,
    t12.last_name,
    t12.first_order,
    t12.most_recent_order,
    t12.number_of_orders
  FROM (
    SELECT
      t3.customer_id,
      t3.first_name,
      t3.last_name,
      t8.customer_id AS customer_id_right,
      t8.first_order,
      t8.most_recent_order,
      t8.number_of_orders
    FROM customers AS t3
    LEFT OUTER JOIN (
      SELECT
        t2.customer_id,
        MIN(t2.order_date) AS first_order,
        MAX(t2.order_date) AS most_recent_order,
        COUNT(t2.order_id) AS number_of_orders
      FROM orders AS t2
      GROUP BY
        1
    ) AS t8
      ON t3.customer_id = t8.customer_id
  ) AS t12
) AS t15
LEFT OUTER JOIN (
  SELECT
    t9.customer_id,
    SUM(t9.amount) AS total_amount
  FROM (
    SELECT
      t4.payment_id,
      t4.order_id,
      t4.payment_method,
      t4.amount,
      t5.order_id AS order_id_right,
      t5.customer_id,
      t5.order_date,
      t5.status
    FROM payments AS t4
    LEFT OUTER JOIN orders AS t5
      ON t4.order_id = t5.order_id
  ) AS t9
  GROUP BY
    1
) AS t13
  ON t15.customer_id = t13.customer_id