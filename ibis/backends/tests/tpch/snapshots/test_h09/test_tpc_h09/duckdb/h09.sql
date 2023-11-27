SELECT
  *
FROM (
  SELECT
    t12.nation AS nation,
    t12.o_year AS o_year,
    SUM(t12.amount) AS sum_profit
  FROM (
    SELECT
      *
    FROM (
      SELECT
        (
          t0.l_extendedprice * (
            CAST(1 AS TINYINT) - t0.l_discount
          )
        ) - (
          t2.ps_supplycost * t0.l_quantity
        ) AS amount,
        EXTRACT('year' FROM t4.o_orderdate) AS o_year,
        t5.n_name AS nation,
        t3.p_name AS p_name
      FROM "lineitem" AS t0
      INNER JOIN "supplier" AS t1
        ON t1.s_suppkey = t0.l_suppkey
      INNER JOIN "partsupp" AS t2
        ON t2.ps_suppkey = t0.l_suppkey AND t2.ps_partkey = t0.l_partkey
      INNER JOIN "part" AS t3
        ON t3.p_partkey = t0.l_partkey
      INNER JOIN "orders" AS t4
        ON t4.o_orderkey = t0.l_orderkey
      INNER JOIN "nation" AS t5
        ON t1.s_nationkey = t5.n_nationkey
    ) AS t11
    WHERE
      t11.p_name LIKE '%green%'
  ) AS t12
  GROUP BY
    1,
    2
) AS t13
ORDER BY
  t13.nation ASC,
  t13.o_year DESC