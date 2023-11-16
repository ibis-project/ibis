WITH t0 AS (
  SELECT t2.`a`, t2.`b`, t2.`c`, t2.`d`, t2.`g`, t2.`window_start`,
         t2.`window_end`
  FROM TABLE(TUMBLE(TABLE `table`, DESCRIPTOR(`i`), INTERVAL '10' MINUTE)) t2
)
SELECT t1.*
FROM (
  SELECT t0.*,
         (row_number() OVER (PARTITION BY t0.`window_start`, t0.`window_end` ORDER BY t0.`g` DESC) - 1) AS `rownum`
  FROM t0
) t1
WHERE t1.`rownum` <= CAST(3 AS TINYINT)