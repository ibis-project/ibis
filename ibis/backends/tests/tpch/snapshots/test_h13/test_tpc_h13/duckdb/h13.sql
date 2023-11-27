SELECT
  *
FROM (
  SELECT
    t4.c_count AS c_count,
    COUNT(*) AS custdist
  FROM (
    SELECT
      t3.c_custkey AS c_custkey,
      COUNT(t3.o_orderkey) AS c_count
    FROM (
      SELECT
        t0.c_custkey AS c_custkey,
        t0.c_name AS c_name,
        t0.c_address AS c_address,
        t0.c_nationkey AS c_nationkey,
        t0.c_phone AS c_phone,
        t0.c_acctbal AS c_acctbal,
        t0.c_mktsegment AS c_mktsegment,
        t0.c_comment AS c_comment,
        t1.o_orderkey AS o_orderkey,
        t1.o_custkey AS o_custkey,
        t1.o_orderstatus AS o_orderstatus,
        t1.o_totalprice AS o_totalprice,
        t1.o_orderdate AS o_orderdate,
        t1.o_orderpriority AS o_orderpriority,
        t1.o_clerk AS o_clerk,
        t1.o_shippriority AS o_shippriority,
        t1.o_comment AS o_comment
      FROM "customer" AS t0
      LEFT OUTER JOIN "orders" AS t1
        ON t0.c_custkey = t1.o_custkey AND NOT t1.o_comment LIKE '%special%requests%'
    ) AS t3
    GROUP BY
      1
  ) AS t4
  GROUP BY
    1
) AS t5
ORDER BY
  t5.custdist DESC,
  t5.c_count DESC