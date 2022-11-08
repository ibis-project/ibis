SELECT count(DISTINCT if(`value` >= 1.0, `key`, NULL)) AS `nunique`
FROM t0