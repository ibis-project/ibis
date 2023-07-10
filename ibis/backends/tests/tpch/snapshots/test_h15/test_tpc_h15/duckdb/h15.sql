SELECT
  t6.s_suppkey AS s_suppkey,
  t6.s_name AS s_name,
  t6.s_address AS s_address,
  t6.s_phone AS s_phone,
  t6.total_revenue AS total_revenue
FROM (
  SELECT
    t0.s_suppkey AS s_suppkey,
    t0.s_name AS s_name,
    t0.s_address AS s_address,
    t0.s_nationkey AS s_nationkey,
    t0.s_phone AS s_phone,
    t0.s_acctbal AS s_acctbal,
    t0.s_comment AS s_comment,
    t4.l_suppkey AS l_suppkey,
    t4.total_revenue AS total_revenue
  FROM supplier AS t0
  INNER JOIN (
    SELECT
      t2.l_suppkey AS l_suppkey,
      SUM(t2.l_extendedprice * (
        CAST(1 AS TINYINT) - t2.l_discount
      )) AS total_revenue
    FROM (
      SELECT
        t1.l_orderkey AS l_orderkey,
        t1.l_partkey AS l_partkey,
        t1.l_suppkey AS l_suppkey,
        t1.l_linenumber AS l_linenumber,
        t1.l_quantity AS l_quantity,
        t1.l_extendedprice AS l_extendedprice,
        t1.l_discount AS l_discount,
        t1.l_tax AS l_tax,
        t1.l_returnflag AS l_returnflag,
        t1.l_linestatus AS l_linestatus,
        t1.l_shipdate AS l_shipdate,
        t1.l_commitdate AS l_commitdate,
        t1.l_receiptdate AS l_receiptdate,
        t1.l_shipinstruct AS l_shipinstruct,
        t1.l_shipmode AS l_shipmode,
        t1.l_comment AS l_comment
      FROM lineitem AS t1
      WHERE
        t1.l_shipdate >= MAKE_DATE(1996, 1, 1) AND t1.l_shipdate < MAKE_DATE(1996, 4, 1)
    ) AS t2
    GROUP BY
      1
  ) AS t4
    ON t0.s_suppkey = t4.l_suppkey
) AS t6
WHERE
  t6.total_revenue = (
    SELECT
      MAX(t6.total_revenue) AS "Max(total_revenue)"
    FROM (
      SELECT
        t0.s_suppkey AS s_suppkey,
        t0.s_name AS s_name,
        t0.s_address AS s_address,
        t0.s_nationkey AS s_nationkey,
        t0.s_phone AS s_phone,
        t0.s_acctbal AS s_acctbal,
        t0.s_comment AS s_comment,
        t4.l_suppkey AS l_suppkey,
        t4.total_revenue AS total_revenue
      FROM supplier AS t0
      INNER JOIN (
        SELECT
          t2.l_suppkey AS l_suppkey,
          SUM(t2.l_extendedprice * (
            CAST(1 AS TINYINT) - t2.l_discount
          )) AS total_revenue
        FROM (
          SELECT
            t1.l_orderkey AS l_orderkey,
            t1.l_partkey AS l_partkey,
            t1.l_suppkey AS l_suppkey,
            t1.l_linenumber AS l_linenumber,
            t1.l_quantity AS l_quantity,
            t1.l_extendedprice AS l_extendedprice,
            t1.l_discount AS l_discount,
            t1.l_tax AS l_tax,
            t1.l_returnflag AS l_returnflag,
            t1.l_linestatus AS l_linestatus,
            t1.l_shipdate AS l_shipdate,
            t1.l_commitdate AS l_commitdate,
            t1.l_receiptdate AS l_receiptdate,
            t1.l_shipinstruct AS l_shipinstruct,
            t1.l_shipmode AS l_shipmode,
            t1.l_comment AS l_comment
          FROM lineitem AS t1
          WHERE
            t1.l_shipdate >= MAKE_DATE(1996, 1, 1) AND t1.l_shipdate < MAKE_DATE(1996, 4, 1)
        ) AS t2
        GROUP BY
          1
      ) AS t4
        ON t0.s_suppkey = t4.l_suppkey
    ) AS t6
  )
ORDER BY
  t6.s_suppkey ASC