SELECT t0.`string_col`, count(1) AS `count`
FROM functional_alltypes t0
GROUP BY 1
HAVING max(t0.`double_col`) = 1