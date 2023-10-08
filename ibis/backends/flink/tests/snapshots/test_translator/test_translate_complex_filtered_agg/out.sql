SELECT t0.`b`, COUNT(*) AS `total`, avg(t0.`a`) AS `avg_a`,
       avg(CASE WHEN t0.`g` = 'A' THEN t0.`a` ELSE NULL END) AS `avg_a_A`,
       avg(CASE WHEN t0.`g` = 'B' THEN t0.`a` ELSE NULL END) AS `avg_a_B`
FROM table t0
GROUP BY t0.`b`