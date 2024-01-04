SELECT
  t19.supp_nation,
  t19.cust_nation,
  t19.l_year,
  t19.revenue
FROM (
  SELECT
    t18.supp_nation,
    t18.cust_nation,
    t18.l_year,
    SUM(t18.volume) AS revenue
  FROM (
    SELECT
      t17.supp_nation,
      t17.cust_nation,
      t17.l_shipdate,
      t17.l_extendedprice,
      t17.l_discount,
      t17.l_year,
      t17.volume
    FROM (
      SELECT
        t9.n_name AS supp_nation,
        t11.n_name AS cust_nation,
        t6.l_shipdate,
        t6.l_extendedprice,
        t6.l_discount,
        EXTRACT(year FROM t6.l_shipdate) AS l_year,
        t6.l_extendedprice * (
          CAST(1 AS TINYINT) - t6.l_discount
        ) AS volume
      FROM supplier AS t5
      INNER JOIN lineitem AS t6
        ON t5.s_suppkey = t6.l_suppkey
      INNER JOIN orders AS t7
        ON t7.o_orderkey = t6.l_orderkey
      INNER JOIN customer AS t8
        ON t8.c_custkey = t7.o_custkey
      INNER JOIN nation AS t9
        ON t5.s_nationkey = t9.n_nationkey
      INNER JOIN nation AS t11
        ON t8.c_nationkey = t11.n_nationkey
    ) AS t17
    WHERE
      (
        (
          (
            t17.cust_nation = 'FRANCE'
          ) AND (
            t17.supp_nation = 'GERMANY'
          )
        )
        OR (
          (
            t17.cust_nation = 'GERMANY'
          ) AND (
            t17.supp_nation = 'FRANCE'
          )
        )
      )
      AND t17.l_shipdate BETWEEN MAKE_DATE(1995, 1, 1) AND MAKE_DATE(1996, 12, 31)
  ) AS t18
  GROUP BY
    1,
    2,
    3
) AS t19
ORDER BY
  t19.supp_nation ASC,
  t19.cust_nation ASC,
  t19.l_year ASC