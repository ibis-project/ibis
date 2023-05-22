SELECT t0.`string_col`, count(DISTINCT t0.`int_col`) AS `nunique`
FROM functional_alltypes t0
WHERE t0.`bigint_col` > 0
GROUP BY 1