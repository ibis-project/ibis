SELECT
  t0.o_orderpriority,
  COUNT(*) AS order_count
FROM "orders" AS t0
WHERE
  EXISTS(
    SELECT
      1
    FROM "lineitem"
    WHERE
      (
        l_orderkey
      ) = (
        t0.o_orderkey
      ) AND (
        l_commitdate
      ) < (
        l_receiptdate
      )
  )
  AND (
    t0.o_orderdate
  ) >= (
    MAKE_DATE(1993, 7, 1)
  )
  AND (
    t0.o_orderdate
  ) < (
    MAKE_DATE(1993, 10, 1)
  )
GROUP BY
  1
ORDER BY
  t0.o_orderpriority ASC