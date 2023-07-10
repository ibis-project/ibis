SELECT
  t9.s_name AS s_name,
  t9.s_address AS s_address
FROM (
  SELECT
    t0.s_suppkey AS s_suppkey,
    t0.s_name AS s_name,
    t0.s_address AS s_address,
    t0.s_nationkey AS s_nationkey,
    t0.s_phone AS s_phone,
    t0.s_acctbal AS s_acctbal,
    t0.s_comment AS s_comment,
    t5.n_nationkey AS n_nationkey,
    t5.n_name AS n_name,
    t5.n_regionkey AS n_regionkey,
    t5.n_comment AS n_comment
  FROM supplier AS t0
  INNER JOIN nation AS t5
    ON t0.s_nationkey = t5.n_nationkey
) AS t9
WHERE
  t9.n_name = 'CANADA'
  AND t9.s_suppkey IN ((
    SELECT
      t1.ps_suppkey AS ps_suppkey
    FROM partsupp AS t1
    WHERE
      t1.ps_partkey IN ((
        SELECT
          t3.p_partkey AS p_partkey
        FROM part AS t3
        WHERE
          t3.p_name LIKE 'forest%'
      ))
      AND t1.ps_availqty > (
        (
          SELECT
            SUM(t7.l_quantity) AS "Sum(l_quantity)"
          FROM (
            SELECT
              t4.l_orderkey AS l_orderkey,
              t4.l_partkey AS l_partkey,
              t4.l_suppkey AS l_suppkey,
              t4.l_linenumber AS l_linenumber,
              t4.l_quantity AS l_quantity,
              t4.l_extendedprice AS l_extendedprice,
              t4.l_discount AS l_discount,
              t4.l_tax AS l_tax,
              t4.l_returnflag AS l_returnflag,
              t4.l_linestatus AS l_linestatus,
              t4.l_shipdate AS l_shipdate,
              t4.l_commitdate AS l_commitdate,
              t4.l_receiptdate AS l_receiptdate,
              t4.l_shipinstruct AS l_shipinstruct,
              t4.l_shipmode AS l_shipmode,
              t4.l_comment AS l_comment
            FROM lineitem AS t4
            WHERE
              t4.l_partkey = t1.ps_partkey
              AND t4.l_suppkey = t1.ps_suppkey
              AND t4.l_shipdate >= MAKE_DATE(1994, 1, 1)
              AND t4.l_shipdate < MAKE_DATE(1995, 1, 1)
          ) AS t7
        ) * CAST(0.5 AS DOUBLE)
      )
  ))
ORDER BY
  t9.s_name ASC
