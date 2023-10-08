SELECT EXTRACT(year from t0.`i`) AS `year`,
       EXTRACT(month from t0.`i`) AS `month`, COUNT(*) AS `total`,
       count(DISTINCT t0.`b`) AS `b_unique`
FROM table t0
GROUP BY EXTRACT(year from t0.`i`), EXTRACT(month from t0.`i`)