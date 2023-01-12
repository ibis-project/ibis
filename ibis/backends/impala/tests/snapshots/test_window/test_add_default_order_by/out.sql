SELECT t0.*, lag(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f`) AS `lag`,
       lead(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f`) - t0.`f` AS `fwd_diff`,
       first_value(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f`) AS `first`,
       last_value(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f`) AS `last`,
       lag(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`d` ASC) AS `lag2`
FROM alltypes t0