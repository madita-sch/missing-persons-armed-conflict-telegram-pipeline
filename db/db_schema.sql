-- Missing Persons DB Schema


BEGIN;

DROP TABLE IF EXISTS extracted_entities CASCADE;
DROP TABLE IF EXISTS message_anonymized CASCADE;
DROP TABLE IF EXISTS message_translations CASCADE;
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS cases CASCADE;
DROP TYPE  IF EXISTS entity_kind;

CREATE TYPE entity_kind AS ENUM ('name', 'location', 'date', 'age');


-- CASES  (one row per missing-person cluster)
-- case_id == cluster_id from the dataset (excluding -1 = unassigned as non-missing cases)
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
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE  cases       IS 'Case-level entity. One case = one missing person (cluster).';
COMMENT ON COLUMN cases.case_id IS 'Mirrors cluster_id from the NLP pipeline. -1 (unassigned) is excluded.';


-- MESSAGES  (raw Telegram posts)
CREATE TABLE messages (
    message_id    BIGINT PRIMARY KEY,                       -- 'id' column from CSV
    case_id       INTEGER REFERENCES cases(case_id) ON DELETE SET NULL,
    posted_at     TIMESTAMPTZ,
    text_raw      TEXT NOT NULL,
    text_clean    TEXT,
    is_missing    BOOLEAN NOT NULL DEFAULT FALSE,
    views         INTEGER,
    forwards      INTEGER,
    reactions     INTEGER
);

CREATE INDEX idx_messages_case_id    ON messages(case_id);
CREATE INDEX idx_messages_posted_at  ON messages(posted_at);
CREATE INDEX idx_messages_is_missing ON messages(is_missing);

COMMENT ON TABLE messages IS 'Raw Telegram messages. Linked many-to-one to cases via case_id.';


-- MESSAGE TRANSLATIONS  (English versions)
CREATE TABLE message_translations (
    message_id     BIGINT PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    text_clean_en  TEXT,
    names_en       TEXT,
    location_en    TEXT,
    dates_en       TEXT
);


-- MESSAGE ANONYMIZED  (PII-stripped versions)
CREATE TABLE message_anonymized (
    message_id        BIGINT PRIMARY KEY REFERENCES messages(message_id) ON DELETE CASCADE,
    clean             TEXT,
    text_clean_anon   TEXT
);


-- EXTRACTED ENTITIES  (normalized signals per message)
-- One row per (message, kind, value_ar)
CREATE TABLE extracted_entities (
    entity_id   BIGSERIAL PRIMARY KEY,
    message_id  BIGINT NOT NULL REFERENCES messages(message_id) ON DELETE CASCADE,
    case_id     INTEGER REFERENCES cases(case_id) ON DELETE SET NULL,
    kind        entity_kind NOT NULL,
    value_ar    TEXT,
    value_en    TEXT
);

CREATE INDEX idx_entities_message ON extracted_entities(message_id);
CREATE INDEX idx_entities_case    ON extracted_entities(case_id);
CREATE INDEX idx_entities_kind    ON extracted_entities(kind);

-- Convenience view: full case page
CREATE OR REPLACE VIEW v_case_page AS
SELECT
    c.case_id,
    c.name_ar, c.name_en,
    c.location_ar, c.location_en,
    c.dates_ar,   c.dates_en,
    c.age,
    c.message_count,
    c.first_seen_at, c.last_seen_at,
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
FROM cases c
LEFT JOIN messages              m ON m.case_id    = c.case_id
LEFT JOIN message_translations  t ON t.message_id = m.message_id
LEFT JOIN message_anonymized    a ON a.message_id = m.message_id
GROUP BY c.case_id;

COMMIT;
