-- =====================================================================
-- MGC CRM queries (SQLite syntax; works unchanged on PostgreSQL except
-- where noted).  Run them all with:  python run_sql.py
-- Reporting queries 1 and 3 read from leads_dedup (see schema.sql / Query 2)
-- so that the 160 duplicated enquiries are not double counted.
-- =====================================================================


-- ---------------------------------------------------------------------
-- Query 1: conversion rate by lead source, best first, sources with 200+ leads
-- ---------------------------------------------------------------------
-- (Use FROM leads instead of leads_dedup to see the raw, duplicate-inflated numbers.)
SELECT  source,
        COUNT(*)                                   AS leads,
        SUM(converted)                             AS conversions,
        ROUND(100.0 * SUM(converted) / COUNT(*), 2) AS conversion_rate_pct
FROM    leads_dedup
GROUP BY source
HAVING  COUNT(*) >= 200                 -- HAVING, not WHERE: the filter is on the aggregate
ORDER BY conversion_rate_pct DESC, leads DESC;


-- ---------------------------------------------------------------------
-- Query 2a: find the duplicate leads and show which row wins
-- ---------------------------------------------------------------------
-- Duplicate = same crm_record_hash (the CRM fingerprint). In this dump the
-- copies are identical in every column except lead_id (the copy ends in '-B')
-- and have identical created_at.
-- Winner rule (deterministic): earliest created_at; if tied, smallest lead_id.
--   => the original id beats its '-B' copy; re-running always picks the same row.
WITH ranked AS (
    SELECT  lead_id, crm_record_hash, created_at, source, city, converted,
            COUNT(*)     OVER (PARTITION BY crm_record_hash)                                   AS copies,
            ROW_NUMBER() OVER (PARTITION BY crm_record_hash ORDER BY created_at ASC, lead_id ASC) AS rn
    FROM    leads
)
SELECT  crm_record_hash,
        lead_id,
        created_at,
        source,
        city,
        converted,
        CASE WHEN rn = 1 THEN 'KEEP' ELSE 'DROP' END AS decision
FROM    ranked
WHERE   copies > 1
ORDER BY crm_record_hash, rn;


-- ---------------------------------------------------------------------
-- Query 2b: sanity check - exactly one canonical row per duplicate group
-- ---------------------------------------------------------------------
-- Expect: duplicate_groups = rows_kept (one winner each), rows_to_drop = extra copies.
WITH ranked AS (
    SELECT  crm_record_hash,
            COUNT(*)     OVER (PARTITION BY crm_record_hash) AS copies,
            ROW_NUMBER() OVER (PARTITION BY crm_record_hash ORDER BY created_at, lead_id) AS rn
    FROM    leads
)
SELECT  COUNT(DISTINCT CASE WHEN copies > 1 THEN crm_record_hash END) AS duplicate_groups,
        SUM(CASE WHEN copies > 1 AND rn = 1 THEN 1 ELSE 0 END)        AS rows_kept,
        SUM(CASE WHEN copies > 1 AND rn > 1 THEN 1 ELSE 0 END)        AS rows_to_drop
FROM    ranked;


-- ---------------------------------------------------------------------
-- Query 2c: the clean-up itself (delete the losers), then lock it in.
-- ---------------------------------------------------------------------
-- Not executed by run_sql.py (it would modify the data). Run it once, then
-- create the unique index described in schema.sql so duplicates cannot return:
--
--   DELETE FROM leads
--   WHERE lead_id IN (
--       SELECT lead_id FROM (
--           SELECT lead_id,
--                  ROW_NUMBER() OVER (PARTITION BY crm_record_hash
--                                     ORDER BY created_at ASC, lead_id ASC) AS rn
--           FROM leads) t
--       WHERE rn > 1);
--   CREATE UNIQUE INDEX uq_leads_crm_hash ON leads(crm_record_hash);


-- ---------------------------------------------------------------------
-- Query 3: conversion rate by city, normalising messy spellings in the query
-- ---------------------------------------------------------------------
-- Raw values seen: Islamabad / ISLAMABAD / ISB, Rawalpindi / RAWALPINDI / Rwp,
-- Karachi / KARACHI / khi, and plain UPPERCASE variants of the other cities.
-- Step 1 (CTE): map to ONE canonical name. Aliases are matched on UPPER(TRIM(city))
--               so case and stray spaces stop mattering.
-- Step 2: everything not aliased is Title-Cased (first letter up, rest down;
--         fine here because every city name is a single word).
WITH cleaned AS (
    SELECT  converted,
            CASE UPPER(TRIM(city))
                 WHEN 'ISB' THEN 'Islamabad'
                 WHEN 'RWP' THEN 'Rawalpindi'
                 WHEN 'KHI' THEN 'Karachi'
                 ELSE UPPER(SUBSTR(TRIM(city), 1, 1)) || LOWER(SUBSTR(TRIM(city), 2))
            END AS city_clean
    FROM    leads_dedup
)
SELECT  city_clean                                   AS city,
        COUNT(*)                                     AS leads,
        SUM(converted)                               AS conversions,
        ROUND(100.0 * SUM(converted) / COUNT(*), 2)  AS conversion_rate_pct
FROM    cleaned
GROUP BY city_clean
ORDER BY conversion_rate_pct DESC, leads DESC;
