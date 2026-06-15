-- ============================================================================
-- Data Explorer — initial database setup
-- Runs once when the postgres container starts with an empty data volume.
-- Alembic migrations take over from here for any further schema changes.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- trigram indices for ATECO search

-- ── POD METADATA ─────────────────────────────────────────────────────────────
-- One row per POD. Address fields kept for future geocoding into `geom`.
CREATE TABLE IF NOT EXISTS pod_metadata (
    pod                 VARCHAR(30) PRIMARY KEY,
    d_dtall             DATE,                       -- activation date
    d_dtsmo             DATE,                       -- deactivation date
    d_flg1              NUMERIC,
    d_pot1              NUMERIC,                    -- declared power
    d_potma             NUMERIC,                    -- max power
    d_potc              INTEGER,                    -- contractual power class id
    d_flatt             BOOLEAN,
    d_8potim            NUMERIC,
    d_tipta             VARCHAR(20),                -- tariff type (BTA6, TD, …)
    d_49des             TEXT,                       -- tariff description
    d_viafo             VARCHAR(255),               -- address street
    d_locfo             VARCHAR(255),               -- address town
    d_frazfo            VARCHAR(255),               -- address fraction
    d_tfor              INTEGER,
    mkost               INTEGER,
    ctar1               VARCHAR(10),                -- tariff code
    fdesc               VARCHAR(255),               -- user category
    ccatete             VARCHAR(20),                -- ATECO 6-digit / sub-code
    tate3des            TEXT,                       -- ATECO description
    -- ATECO hierarchy (extracted from ccatete during ingestion)
    ateco_l1            VARCHAR(10),                -- e.g. "47", "DO", "IL"
    ateco_l2            VARCHAR(10),                -- e.g. "47.11", "DO.R"
    ateco_l3            VARCHAR(10),                -- e.g. "47.11.10"
    -- Geolocation (populated separately; nullable)
    geom                GEOGRAPHY(POINT, 4326),
    -- Bookkeeping
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_meta_ateco_l1 ON pod_metadata (ateco_l1);
CREATE INDEX IF NOT EXISTS idx_meta_ateco_l2 ON pod_metadata (ateco_l2);
CREATE INDEX IF NOT EXISTS idx_meta_ateco_l3 ON pod_metadata (ateco_l3);
CREATE INDEX IF NOT EXISTS idx_meta_fdesc    ON pod_metadata (fdesc);
CREATE INDEX IF NOT EXISTS idx_meta_geom     ON pod_metadata USING GIST (geom);

-- ── MEASUREMENTS (wide format: Q1…Q96 columns) ───────────────────────────────
-- One row per (POD, day). Partitioned by year for query speed.
CREATE TABLE IF NOT EXISTS measurements (
    pod                 VARCHAR(30) NOT NULL,
    data_misura         DATE        NOT NULL,
    tipologia           VARCHAR(10),               -- AP, AN, RCN, RLN, RLP
    matricola           VARCHAR(50),
    tipo_rilevatore     VARCHAR(20),
    codice_dispacciamento VARCHAR(50),
    trattamento         VARCHAR(10),
    tensione            INTEGER,
    potenza_contrattuale VARCHAR(20),
    k_factor            VARCHAR(10),
    effettiva           BOOLEAN,
    validata            BOOLEAN,
    totalizzatori_presenti BOOLEAN,
    intervalli_completi BOOLEAN,
    ricostruita         BOOLEAN,
    -- Fascia totalisers (kept for completeness; not used by clustering)
    totalizzatore_f1    BIGINT,
    totalizzatore_f2    BIGINT,
    totalizzatore_f3    BIGINT,
    consumo_f1          BIGINT,
    consumo_f2          BIGINT,
    consumo_f3          BIGINT,
    -- 96 quarter-hour readings (kWh × scaling factor — same as original CSV)
    q1  REAL, q2  REAL, q3  REAL, q4  REAL, q5  REAL, q6  REAL, q7  REAL, q8  REAL,
    q9  REAL, q10 REAL, q11 REAL, q12 REAL, q13 REAL, q14 REAL, q15 REAL, q16 REAL,
    q17 REAL, q18 REAL, q19 REAL, q20 REAL, q21 REAL, q22 REAL, q23 REAL, q24 REAL,
    q25 REAL, q26 REAL, q27 REAL, q28 REAL, q29 REAL, q30 REAL, q31 REAL, q32 REAL,
    q33 REAL, q34 REAL, q35 REAL, q36 REAL, q37 REAL, q38 REAL, q39 REAL, q40 REAL,
    q41 REAL, q42 REAL, q43 REAL, q44 REAL, q45 REAL, q46 REAL, q47 REAL, q48 REAL,
    q49 REAL, q50 REAL, q51 REAL, q52 REAL, q53 REAL, q54 REAL, q55 REAL, q56 REAL,
    q57 REAL, q58 REAL, q59 REAL, q60 REAL, q61 REAL, q62 REAL, q63 REAL, q64 REAL,
    q65 REAL, q66 REAL, q67 REAL, q68 REAL, q69 REAL, q70 REAL, q71 REAL, q72 REAL,
    q73 REAL, q74 REAL, q75 REAL, q76 REAL, q77 REAL, q78 REAL, q79 REAL, q80 REAL,
    q81 REAL, q82 REAL, q83 REAL, q84 REAL, q85 REAL, q86 REAL, q87 REAL, q88 REAL,
    q89 REAL, q90 REAL, q91 REAL, q92 REAL, q93 REAL, q94 REAL, q95 REAL, q96 REAL,
    -- Bookkeeping
    source_file         VARCHAR(255),
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (pod, data_misura)
) PARTITION BY RANGE (data_misura);

-- One partition per year — add new ones as new data arrives.
-- (Alembic will manage these going forward, this is just bootstrap.)
CREATE TABLE IF NOT EXISTS measurements_2023 PARTITION OF measurements
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE IF NOT EXISTS measurements_2024 PARTITION OF measurements
    FOR VALUES FROM ('2024-01-01') TO ('2025-01-01');
CREATE TABLE IF NOT EXISTS measurements_2025 PARTITION OF measurements
    FOR VALUES FROM ('2025-01-01') TO ('2026-01-01');
CREATE TABLE IF NOT EXISTS measurements_2026 PARTITION OF measurements
    FOR VALUES FROM ('2026-01-01') TO ('2027-01-01');

CREATE INDEX IF NOT EXISTS idx_meas_pod        ON measurements (pod);
CREATE INDEX IF NOT EXISTS idx_meas_date       ON measurements (data_misura);
CREATE INDEX IF NOT EXISTS idx_meas_tipologia  ON measurements (tipologia);

-- ── ATECO LOOKUP (Note esplicative ATECO 2025) ───────────────────────────────
-- Imported once from the official Excel; used to resolve codes → descriptions.
CREATE TABLE IF NOT EXISTS ateco_lookup (
    code        VARCHAR(20) PRIMARY KEY,
    description TEXT,
    level       SMALLINT          -- 1, 2, 3 (section / division / class)
);
CREATE INDEX IF NOT EXISTS idx_ateco_desc_trgm
    ON ateco_lookup USING GIN (description gin_trgm_ops);

-- ── REFERENCE PROFILES (GSE 2025 + ARERA 2024) ───────────────────────────────
-- Loaded from the supplied Excel files; small tables, no partitioning needed.
CREATE TABLE IF NOT EXISTS reference_profiles_gse (
    profile_code  VARCHAR(10)  NOT NULL,           -- PDMM, PDMF, PAUM, PAUF
    month_idx     SMALLINT     NOT NULL,           -- 1..12
    hour_idx      SMALLINT     NOT NULL,           -- 0..23
    value         REAL         NOT NULL,
    PRIMARY KEY (profile_code, month_idx, hour_idx)
);

CREATE TABLE IF NOT EXISTS reference_profiles_arera (
    power_class   VARCHAR(20)  NOT NULL,           -- "≤ 1.5 kW", "1.5–3 kW", …
    market        VARCHAR(30)  NOT NULL,           -- Maggior Tutela / Mercato Libero / Tutti
    day_type      VARCHAR(20)  NOT NULL,           -- Weekday / Saturday / Sunday
    month_idx     SMALLINT     NOT NULL,           -- 1..12  (0 = annual average)
    hour_idx      SMALLINT     NOT NULL,           -- 0..23
    value         REAL         NOT NULL,
    PRIMARY KEY (power_class, market, day_type, month_idx, hour_idx)
);
