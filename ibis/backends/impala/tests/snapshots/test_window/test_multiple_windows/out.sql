SELECT t0.`g`,
       sum(t0.`f`) OVER (PARTITION BY t0.`g`) - sum(t0.`f`) OVER () AS `result`
FROM `alltypes` t0