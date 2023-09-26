SELECT
  *
FROM (
  SELECT
    t0.l_returnflag,
    t0.l_linestatus,
    SUM(t0.l_quantity) AS sum_qty,
    SUM(t0.l_extendedprice) AS sum_base_price,
    SUM((
      t0.l_extendedprice * (
        CAST(1 AS TINYINT) - t0.l_discount
      )
    )) AS sum_disc_price,
    SUM(
      (
        (
          t0.l_extendedprice * (
            CAST(1 AS TINYINT) - t0.l_discount
          )
        ) * (
          t0.l_tax + CAST(1 AS TINYINT)
        )
      )
    ) AS sum_charge,
    AVG(t0.l_quantity) AS avg_qty,
    AVG(t0.l_extendedprice) AS avg_price,
    AVG(t0.l_discount) AS avg_disc,
    COUNT(*) AS count_order
  FROM "lineitem" AS t0
  WHERE
    (
      t0.l_shipdate <= MAKE_DATE(1998, 9, 2)
    )
  GROUP BY
    1,
    2
) AS t1
ORDER BY
  t1.l_returnflag ASC,
  t1.l_linestatus ASC