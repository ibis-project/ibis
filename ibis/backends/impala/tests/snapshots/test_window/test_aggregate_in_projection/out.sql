SELECT t0.*, t0.`f` / sum(t0.`f`) OVER () AS `normed_f`
FROM `alltypes` t0