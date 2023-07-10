SELECT
  *
FROM (
  SELECT
    t2.c_custkey AS c_custkey,
    t2.c_name AS c_name,
    t2.c_address AS c_address,
    t2.c_nationkey AS c_nationkey,
    t2.c_phone AS c_phone,
    t2.c_acctbal AS c_acctbal,
    t2.c_mktsegment AS c_mktsegment,
    t2.c_comment AS c_comment,
    t0.r_name AS region,
    t3.o_totalprice AS amount,
    CAST(t3.o_orderdate AS TIMESTAMP) AS odate
  FROM tpch_region AS t0
  INNER JOIN tpch_nation AS t1
    ON t0.r_regionkey = t1.n_regionkey
  INNER JOIN tpch_customer AS t2
    ON t2.c_nationkey = t1.n_nationkey
  INNER JOIN tpch_orders AS t3
    ON t3.o_custkey = t2.c_custkey
) AS t7
WHERE
  (
    t7.amount > (
      SELECT
        AVG(t9.amount) AS "Mean(amount)"
      FROM (
        SELECT
          *
        FROM (
          SELECT
            t2.c_custkey AS c_custkey,
            t2.c_name AS c_name,
            t2.c_address AS c_address,
            t2.c_nationkey AS c_nationkey,
            t2.c_phone AS c_phone,
            t2.c_acctbal AS c_acctbal,
            t2.c_mktsegment AS c_mktsegment,
            t2.c_comment AS c_comment,
            t0.r_name AS region,
            t3.o_totalprice AS amount,
            CAST(t3.o_orderdate AS TIMESTAMP) AS odate
          FROM tpch_region AS t0
          INNER JOIN tpch_nation AS t1
            ON t0.r_regionkey = t1.n_regionkey
          INNER JOIN tpch_customer AS t2
            ON t2.c_nationkey = t1.n_nationkey
          INNER JOIN tpch_orders AS t3
            ON t3.o_custkey = t2.c_custkey
        ) AS t8
        WHERE
          (
            t8.region = t7.region
          )
      ) AS t9
    )
  )
LIMIT 10