SELECT count(DISTINCT if(`value` >= 1.0, `key`, NULL)) AS `CountDistinct(key, GreaterEqual(value, 1.0))`
FROM t0