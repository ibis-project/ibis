SELECT *,
       acq_ipos / num_investments::float AS acq_rate
FROM (
  SELECT
    CASE
      WHEN i.investor_name IS NULL THEN 'NO INVESTOR'
      ELSE i.investor_name
    END AS investor_name,
    COUNT(DISTINCT c.permalink) AS num_investments,
    COUNT(
      DISTINCT
        CASE
          WHEN c.status IN ('ipo', 'acquired') THEN
            c.permalink
          ELSE NULL
        END
    ) AS acq_ipos
  FROM crunchbase_companies AS c
  LEFT JOIN crunchbase_investments AS i
    ON c.permalink = i.company_permalink
  GROUP BY 1
  ORDER BY 2 DESC
) a
