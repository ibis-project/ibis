SELECT *, lag(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `lag`,
       lead(`f`) OVER (PARTITION BY `g` ORDER BY `f`) - `f` AS `fwd_diff`,
       first_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `first`,
       last_value(`f`) OVER (PARTITION BY `g` ORDER BY `f`) AS `last`,
       lag(`f`) OVER (PARTITION BY `g` ORDER BY `d` ASC) AS `lag2`
FROM alltypes