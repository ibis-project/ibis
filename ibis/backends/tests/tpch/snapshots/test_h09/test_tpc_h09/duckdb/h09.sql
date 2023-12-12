SELECT
  t18.nation AS nation,
  t18.o_year AS o_year,
  t18.sum_profit AS sum_profit
FROM (
  SELECT
    t17.nation AS nation,
    t17.o_year AS o_year,
    SUM(t17.amount) AS sum_profit
  FROM (
    SELECT
      t16.amount AS amount,
      t16.o_year AS o_year,
      t16.nation AS nation,
      t16.p_name AS p_name
    FROM (
      SELECT
        (
          t0.l_extendedprice * (
            CAST(1 AS TINYINT) - t0.l_discount
          )
        ) - (
          t7.ps_supplycost * t0.l_quantity
        ) AS amount,
        EXTRACT('year' FROM t9.o_orderdate) AS o_year,
        t10.n_name AS nation,
        t8.p_name AS p_name
      FROM lineitem AS t0
      INNER JOIN supplier AS t6
        ON t6.s_suppkey = t0.l_suppkey
      INNER JOIN partsupp AS t7
        ON t7.ps_suppkey = t0.l_suppkey AND t7.ps_partkey = t0.l_partkey
      INNER JOIN part AS t8
        ON t8.p_partkey = t0.l_partkey
      INNER JOIN orders AS t9
        ON t9.o_orderkey = t0.l_orderkey
      INNER JOIN nation AS t10
        ON t6.s_nationkey = t10.n_nationkey
    ) AS t16
    WHERE
      t16.p_name LIKE '%green%'
  ) AS t17
  GROUP BY
    1,
    2
) AS t18
ORDER BY
  t18.nation ASC,
  t18.o_year DESC