SELECT count(DISTINCT if(t0.`value` >= 1.0, t0.`key`, NULL)) AS `CountDistinct(key, GreaterEqual(value, 1.0))`
FROM `t0` t0