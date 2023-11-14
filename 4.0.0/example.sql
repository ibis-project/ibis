SELECT
  *,
  CAST(acq_ipos / num_investments AS FLOAT) AS acq_rate
FROM (
  SELECT
    COALESCE(i.investor_name, 'NO INVESTOR') AS investor_name,
    COUNT(DISTINCT c.permalink) AS num_investments,
    COUNT(
      DISTINCT
        CASE
          WHEN c.status IN ('ipo', 'acquired') THEN c.permalink
          ELSE NULL
        END
    ) AS acq_ipos
  FROM companies AS c
  LEFT JOIN investments AS i
    ON c.permalink = i.company_permalink
  GROUP BY 1
  ORDER BY 2 DESC
)
