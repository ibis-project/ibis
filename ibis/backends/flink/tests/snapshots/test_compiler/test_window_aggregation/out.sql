SELECT t0.`window_start`, t0.`window_end`, t0.`g`, avg(t0.`d`) AS `mean`
FROM TABLE(TUMBLE(TABLE `table`, DESCRIPTOR(`i`), INTERVAL '15' MINUTE)) t0
GROUP BY t0.`window_start`, t0.`window_end`, t0.`g`