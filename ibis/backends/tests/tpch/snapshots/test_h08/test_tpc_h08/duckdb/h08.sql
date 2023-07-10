SELECT
  t23.o_year AS o_year,
  t23.mkt_share AS mkt_share
FROM (
  SELECT
    t22.o_year AS o_year,
    SUM(t22.nation_volume) / SUM(t22.volume) AS mkt_share
  FROM (
    SELECT
      t21.o_year AS o_year,
      t21.volume AS volume,
      t21.nation AS nation,
      t21.r_name AS r_name,
      t21.o_orderdate AS o_orderdate,
      t21.p_type AS p_type,
      CASE WHEN t21.nation = 'BRAZIL' THEN t21.volume ELSE CAST(0 AS TINYINT) END AS nation_volume
    FROM (
      SELECT
        EXTRACT('year' FROM t9.o_orderdate) AS o_year,
        t7.l_extendedprice * (
          CAST(1 AS TINYINT) - t7.l_discount
        ) AS volume,
        t12.n_name AS nation,
        t13.r_name AS r_name,
        t9.o_orderdate AS o_orderdate,
        t0.p_type AS p_type
      FROM part AS t0
      INNER JOIN lineitem AS t7
        ON t0.p_partkey = t7.l_partkey
      INNER JOIN supplier AS t8
        ON t8.s_suppkey = t7.l_suppkey
      INNER JOIN orders AS t9
        ON t7.l_orderkey = t9.o_orderkey
      INNER JOIN customer AS t10
        ON t9.o_custkey = t10.c_custkey
      INNER JOIN nation AS t11
        ON t10.c_nationkey = t11.n_nationkey
      INNER JOIN region AS t13
        ON t11.n_regionkey = t13.r_regionkey
      INNER JOIN nation AS t12
        ON t8.s_nationkey = t12.n_nationkey
    ) AS t21
    WHERE
      t21.r_name = 'AMERICA'
      AND t21.o_orderdate BETWEEN MAKE_DATE(1995, 1, 1) AND MAKE_DATE(1996, 12, 31)
      AND t21.p_type = 'ECONOMY ANODIZED STEEL'
  ) AS t22
  GROUP BY
    1
) AS t23
ORDER BY
  t23.o_year ASC