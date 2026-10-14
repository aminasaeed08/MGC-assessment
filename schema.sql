-- =====================================================================
-- MGC CRM leads: schema (SQLite syntax; notes for PostgreSQL inline)
-- =====================================================================
-- Design choice: ONE table. Each row is one lead (one entity); every column
-- is a single-valued attribute of that lead and there are no repeating
-- groups, so splitting adds joins without removing redundancy.
-- What I would split out in production: `lead_source` and `city` as small
-- lookup tables (FK from leads) so spelling is controlled at write time.
-- I did not do it here because the dump is already messy and the brief
-- asks us to normalise inside the query.
--
-- Raw values are stored as received (city is deliberately NOT cleaned
-- here); cleaning happens in queries so the original data is never lost.

DROP VIEW  IF EXISTS leads_dedup;
DROP TABLE IF EXISTS leads;

CREATE TABLE leads (
    lead_id                     TEXT    PRIMARY KEY,             -- e.g. 'MGC-104067'; duplicates carry a '-B' suffix
    created_at                  TEXT    NOT NULL,                -- ISO-8601 'YYYY-MM-DD HH:MM:SS' (PG: TIMESTAMP)
    source                      TEXT    NOT NULL,                -- Facebook Ads, Referral, ...
    city                        TEXT    NOT NULL,                -- RAW as typed by agents: 'ISB', 'Rwp', 'KARACHI', ...
    area                        TEXT,                            -- NULL in ~5% of rows
    property_type               TEXT    NOT NULL,
    budget_pkr_lac              REAL    CHECK (budget_pkr_lac > 0),            -- NULL = not captured
    bedrooms                    INTEGER CHECK (bedrooms BETWEEN 1 AND 5),      -- NULL for plots / shops (not applicable)
    first_response_minutes      REAL    CHECK (first_response_minutes >= 0),
    calls_made                  INTEGER NOT NULL DEFAULT 0 CHECK (calls_made >= 0),
    total_call_seconds          REAL    NOT NULL DEFAULT 0 CHECK (total_call_seconds >= 0),
    whatsapp_replies            INTEGER NOT NULL DEFAULT 0 CHECK (whatsapp_replies >= 0),
    site_visits                 INTEGER NOT NULL DEFAULT 0 CHECK (site_visits >= 0),
    agent_experience_years      REAL    CHECK (agent_experience_years >= 0),
    is_overseas                 INTEGER NOT NULL CHECK (is_overseas IN (0,1)),
    referred_by_existing_client INTEGER NOT NULL CHECK (referred_by_existing_client IN (0,1)),
    has_financing_approved      INTEGER NOT NULL CHECK (has_financing_approved IN (0,1)),
    token_amount_received_pkr   REAL    NOT NULL DEFAULT 0 CHECK (token_amount_received_pkr >= 0),
    crm_record_hash             INTEGER NOT NULL,                -- PG: BIGINT (values reach ~1e10, beyond INT4)
    converted                   INTEGER NOT NULL CHECK (converted IN (0,1))
);

-- Hot paths: conversion by source / city, time-based splits, duplicate detection.
CREATE INDEX idx_leads_source     ON leads(source);
CREATE INDEX idx_leads_created_at ON leads(created_at);
CREATE INDEX idx_leads_hash       ON leads(crm_record_hash);   -- NOT unique yet: the dump contains duplicates

-- ---------------------------------------------------------------------
-- Duplicate prevention (applies AFTER the existing duplicates are removed,
-- see queries.sql Query 2c; the unique index cannot be built on the raw dump):
--
--   CREATE UNIQUE INDEX uq_leads_crm_hash ON leads(crm_record_hash);
--
-- and have the ingest path use  INSERT ... ON CONFLICT (crm_record_hash) DO NOTHING
-- (PostgreSQL / SQLite 3.24+).  The hash is the CRM's fingerprint of the lead's
-- contents, so two agents entering the same enquiry collide on it.
-- Even better: hash a normalised phone number + project, so a re-entry with a
-- typo in the name still collides.
-- ---------------------------------------------------------------------

-- Canonical (de-duplicated) leads: used by the reporting queries and by the ML code.
-- Rule: within a hash, the EARLIEST created_at wins; ties (the dump has identical
-- timestamps) are broken by the smallest lead_id, i.e. the original id beats the '-B' copy.
CREATE VIEW leads_dedup AS
SELECT *
FROM (
    SELECT l.*,
           ROW_NUMBER() OVER (
               PARTITION BY crm_record_hash
               ORDER BY created_at ASC, lead_id ASC
           ) AS rn
    FROM leads l
) t
WHERE rn = 1;
