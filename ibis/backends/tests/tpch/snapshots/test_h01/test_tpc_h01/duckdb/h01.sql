SELECT
  t0.l_returnflag,
  t0.l_linestatus,
  t0.sum_qty,
  t0.sum_base_price,
  t0.sum_disc_price,
  t0.sum_charge,
  t0.avg_qty,
  t0.avg_price,
  t0.avg_disc,
  t0.count_order
FROM (
  SELECT
    t1.l_returnflag AS l_returnflag,
    t1.l_linestatus AS l_linestatus,
    SUM(t1.l_quantity) AS sum_qty,
    SUM(t1.l_extendedprice) AS sum_base_price,
    SUM(t1.l_extendedprice * (
      CAST(1 AS TINYINT) - t1.l_discount
    )) AS sum_disc_price,
    SUM(
      t1.l_extendedprice * (
        CAST(1 AS TINYINT) - t1.l_discount
      ) * (
        t1.l_tax + CAST(1 AS TINYINT)
      )
    ) AS sum_charge,
    AVG(t1.l_quantity) AS avg_qty,
    AVG(t1.l_extendedprice) AS avg_price,
    AVG(t1.l_discount) AS avg_disc,
    COUNT(*) AS count_order
  FROM main.lineitem AS t1
  WHERE
    t1.l_shipdate <= CAST('1998-09-02' AS DATE)
  GROUP BY
    1,
    2
) AS t0
ORDER BY
  t0.l_returnflag ASC,
  t0.l_linestatus ASC