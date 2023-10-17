WITH t0 AS (
  SELECT
    t2.l_extendedprice * (
      1 - t2.l_discount
    ) - t4.ps_supplycost * t2.l_quantity AS amount,
    CAST(EXTRACT(year FROM t6.o_orderdate) AS SMALLINT) AS o_year,
    t7.n_name AS nation,
    t5.p_name AS p_name
  FROM hive.ibis_sf1.lineitem AS t2
  JOIN hive.ibis_sf1.supplier AS t3
    ON t3.s_suppkey = t2.l_suppkey
  JOIN hive.ibis_sf1.partsupp AS t4
    ON t4.ps_suppkey = t2.l_suppkey AND t4.ps_partkey = t2.l_partkey
  JOIN hive.ibis_sf1.part AS t5
    ON t5.p_partkey = t2.l_partkey
  JOIN hive.ibis_sf1.orders AS t6
    ON t6.o_orderkey = t2.l_orderkey
  JOIN hive.ibis_sf1.nation AS t7
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