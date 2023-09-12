SELECT t0.*,
       lag(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` ASC) AS `lag`,
       lead(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` ASC) - t0.`f` AS `fwd_diff`,
       first_value(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` ASC) AS `first`,
       last_value(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`f` ASC) AS `last`,
       lag(t0.`f`) OVER (PARTITION BY t0.`g` ORDER BY t0.`d` ASC) AS `lag2`
FROM `alltypes` t0