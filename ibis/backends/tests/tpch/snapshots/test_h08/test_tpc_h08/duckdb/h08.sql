SELECT
  t25.o_year,
  t25.mkt_share
FROM (
  SELECT
    t24.o_year,
    SUM(t24.nation_volume) / SUM(t24.volume) AS mkt_share
  FROM (
    SELECT
      t23.o_year,
      t23.volume,
      t23.nation,
      t23.r_name,
      t23.o_orderdate,
      t23.p_type,
      CASE WHEN t23.nation = 'BRAZIL' THEN t23.volume ELSE CAST(0 AS TINYINT) END AS nation_volume
    FROM (
      SELECT
        EXTRACT(year FROM t10.o_orderdate) AS o_year,
        t8.l_extendedprice * (
          CAST(1 AS TINYINT) - t8.l_discount
        ) AS volume,
        t15.n_name AS nation,
        t14.r_name,
        t10.o_orderdate,
        t7.p_type
      FROM part AS t7
      INNER JOIN lineitem AS t8
        ON t7.p_partkey = t8.l_partkey
      INNER JOIN supplier AS t9
        ON t9.s_suppkey = t8.l_suppkey
      INNER JOIN orders AS t10
        ON t8.l_orderkey = t10.o_orderkey
      INNER JOIN customer AS t11
        ON t10.o_custkey = t11.c_custkey
      INNER JOIN nation AS t12
        ON t11.c_nationkey = t12.n_nationkey
      INNER JOIN region AS t14
        ON t12.n_regionkey = t14.r_regionkey
      INNER JOIN nation AS t15
        ON t9.s_nationkey = t15.n_nationkey
    ) AS t23
    WHERE
      t23.r_name = 'AMERICA'
      AND t23.o_orderdate BETWEEN MAKE_DATE(1995, 1, 1) AND MAKE_DATE(1996, 12, 31)
      AND t23.p_type = 'ECONOMY ANODIZED STEEL'
  ) AS t24
  GROUP BY
    1
) AS t25
ORDER BY
  t25.o_year ASC