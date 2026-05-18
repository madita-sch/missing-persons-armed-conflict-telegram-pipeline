-- Missing Persons DB Schema to set up the PostgreSQL database

BEGIN;

-- Define schema for missing persons Telegram dataset, incl. tables for cases, messages, translations, anonymized text, and extracted entities.
DROP TABLE IF EXISTS extracted_entities CASCADE;
DROP TABLE IF EXISTS message_anonymized CASCADE;
DROP TABLE IF EXISTS message_translations CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS cases CASCADE;
DROP TYPE  IF EXISTS entity_kind;

-- Create ENUM datatype for entity kinds (only allowed values of 'name', 'location', 'date', 'age')
CREATE TYPE entity_kind AS ENUM ('name', 'location', 'date', 'age');


-- Cases Table (one row per missing-person cluster)
CREATE TABLE cases (
    case_id            INTEGER PRIMARY KEY,
    name_ar            TEXT,
    name_en            TEXT,
    location_ar        TEXT,
    location_en        TEXT,
    dates_ar           TEXT,
    dates_en           TEXT,
    age                TEXT,
    message_count      INTEGER NOT NULL DEFAULT 0,
    first_seen_at      TIMESTAMPTZ,
    last_seen_at       TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    verified           BOOLEAN NOT NULL DEFAULT FALSE,
    verified_at        TIMESTAMPTZ
);

-- Indexes for efficient querying
COMMENT ON TABLE  cases       IS 'Case-level entity. One case = one missing person (cluster).';
COMMENT ON COLUMN cases.case_id IS 'Mirrors cluster_id from the NLP pipeline. -1 (unassigned) is excluded.';
COMMENT ON COLUMN cases.verified    IS 'TRUE if a practitioner has human-reviewed and confirmed the case via the dashboard.';
COMMENT ON COLUMN cases.verified_at IS 'Timestamp of when the case was verified by a practitioner.';

-- Messages Table (raw and clean Telegram posts)
CREATE TABLE messages (
    message_id    BIGINT PRIMARY KEY,                       -- equals the 'id' column from CSV
    case_id       INTEGER REFERENCES cases(case_id) ON DELETE SET NULL,
    posted_at     TIMESTAMPTZ,
    text_raw      TEXT NOT NULL,
    text_clean    TEXT,
    is_missing    BOOLEAN NOT NULL DEFAULT FALSE,
    views         INTEGER,
    forwards      INTEGER,
    reactions     INTEGER
);

-- Indexes for efficient querying
CREATE INDEX idx_messages_case_id    ON messages(case_id);
CREATE INDEX idx_messages_posted_at  ON messages(posted_at);
CREATE INDEX idx_messages_is_missing ON messages(is_missing);

COMMENT ON TABLE messages IS 'Raw Telegram messages. Linked many-to-one to cases via case_id.';


-- Message Translations Table
CREATE TABLE message_translations (
    message_id     BIGINT PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    text_clean_en  TEXT,
    names_en       TEXT,
    location_en    TEXT,
    dates_en       TEXT
);


-- Message Pseudonymized Table
CREATE TABLE message_anonymized (
    message_id        BIGINT PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    clean             TEXT,
    text_clean_anon   TEXT
);


-- Extracted entities Table, storing each extracted entity as its own row (message, kind, value_ar). 
-- Link entity_id to cases and messages for easy querying.
CREATE TABLE extracted_entities (
    entity_id   BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,  -- Link the entity to the message it was extracted from
    case_id     INTEGER REFERENCES cases(case_id) ON DELETE SET NULL,         -- Link to case for easier querying, but allow NULL if not assigned to a case yet
    kind        entity_kind NOT NULL,                   -- Add type of entity, e.g. 'name', 'location', 'date', 'age', cannot be NULL           
    value_ar    TEXT,                                   -- Original Arabic value of the entity as extracted by the NLP pipeline
    value_en    TEXT                                    -- English translation of the entity value
);

-- Indexes for efficient querying
CREATE INDEX idx_entities_message ON extracted_entities(message_id);
CREATE INDEX idx_entities_case    ON extracted_entities(case_id);
CREATE INDEX idx_entities_kind    ON extracted_entities(kind);

-- Create a view to aggregate case details with all linked messages, translations, and pseudonymized text for easy querying in the dashboard
CREATE OR REPLACE VIEW v_case_page AS
SELECT
    c.case_id,
    c.name_ar, c.name_en,
    c.location_ar, c.location_en,
    c.dates_ar,   c.dates_en,
    c.age,
    c.message_count,
    c.first_seen_at, c.last_seen_at,
    c.verified,                       
    c.verified_at, 
    -- Aggregate messages for this case into a JSON array, including translations and anonymized text, to get one row per case with all linked messages
    json_agg(
        json_build_object(
            'message_id', m.message_id,
            'posted_at',  m.posted_at,
            'text_raw',   m.text_raw,
            'text_clean', m.text_clean,
            'text_en',    t.text_clean_en,
            'text_anon',  a.text_clean_anon,
            'views',      m.views,
            'forwards',   m.forwards,
            'reactions',  m.reactions
        ) ORDER BY m.posted_at
    ) FILTER (WHERE m.message_id IS NOT NULL) AS messages
FROM cases c    -- Cases table is the main table to ensure one row per case, even if there are no linked messages
LEFT JOIN messages              m ON m.case_id    = c.case_id       -- Use LEFT JOIN to include cases with no messages
LEFT JOIN message_translations  t ON t.message_id = m.message_id    -- Left join to include translations if they exist
LEFT JOIN message_anonymized    a ON a.message_id = m.message_id    -- Left join to include anonymized text if it exists
GROUP BY c.case_id;                                                 -- Group by case_id to get one row per case with aggregated messages

COMMIT;
