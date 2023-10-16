WITH t0 AS (
  SELECT
    t2.l_extendedprice * (
      CAST(1 AS TINYINT) - t2.l_discount
    ) - t4.ps_supplycost * t2.l_quantity AS amount,
    CAST(EXTRACT(year FROM t6.o_orderdate) AS SMALLINT) AS o_year,
    t7.n_name AS nation,
    t5.p_name AS p_name
  FROM main.lineitem AS t2
  JOIN main.supplier AS t3
    ON t3.s_suppkey = t2.l_suppkey
  JOIN main.partsupp AS t4
    ON t4.ps_suppkey = t2.l_suppkey AND t4.ps_partkey = t2.l_partkey
  JOIN main.part AS t5
    ON t5.p_partkey = t2.l_partkey
  JOIN main.orders AS t6
    ON t6.o_orderkey = t2.l_orderkey
  JOIN main.nation AS t7
    ON t3.s_nationkey = t7.n_nationkey
  WHERE
    t5.p_name LIKE '%green%'
)
SELECT
  t1.nation,
  t1.o_year,
  t1.sum_profit
FROM (
  SELECT
    t0.nation AS nation,
    t0.o_year AS o_year,
    SUM(t0.amount) AS sum_profit
  FROM t0
  GROUP BY
    1,
    2
) AS t1
ORDER BY
  t1.nation ASC,
  t1.o_year DESC