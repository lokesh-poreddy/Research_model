-- ResearchForge-ECRM reference schema (PostgreSQL + pgvector).
--
-- The Python code in this repo runs against the SQLite equivalents created
-- by db/init_db.py (no server, no extension install required) so the whole
-- system works offline out of the box. This file is the schema as specified
-- in the design documents (exec summary Sec. 3.3, technical report Sec. 4 /
-- Sec. 10) for when you deploy against real Postgres -- e.g. to run RDE-Bench
-- at the scale the roadmap describes (GPU cluster, weeks of experiments,
-- multiple people/processes writing concurrently, which is where SQLite's
-- single-writer model stops being enough).

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;  -- pgvector, for embedding columns

CREATE TABLE IF NOT EXISTS hypotheses (
    id            SERIAL PRIMARY KEY,
    content       TEXT NOT NULL,
    gap_id        INTEGER,
    status        VARCHAR(32) DEFAULT 'pending',
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS models (
    model_id      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type    VARCHAR(64) NOT NULL,
    genome        JSONB NOT NULL,          -- architecture + hyperparameters + data_pipeline
    parent_ids    UUID[],
    generation    INTEGER DEFAULT 0,
    created_at    TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiments (
    exp_id            SERIAL PRIMARY KEY,
    hypothesis_id     INTEGER REFERENCES hypotheses(id),
    model_id          UUID REFERENCES models(model_id),
    strategy          VARCHAR(64),
    result_metric     REAL,
    train_metric      REAL,
    evidence_score    REAL,               -- RES(h,G) at time of recording
    failure_category  VARCHAR(32),        -- see diagnosis.failure_taxonomy.FailureCategory
    success           BOOLEAN,
    duration_s        REAL,
    timestamp         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS experiment_runs (
    -- Safety/audit log: one row per SafeRunner.run() call, independent of
    -- whether the underlying experiment produced a usable result. This is
    -- what the Sec. 7 "Kill Switch"/"Resource Quotas" controls need to be
    -- auditable rather than just enforced-and-forgotten.
    run_id            SERIAL PRIMARY KEY,
    exp_id            INTEGER REFERENCES experiments(exp_id),
    safety_status     VARCHAR(32) NOT NULL,   -- ok | timeout | exception | budget_exhausted | killed
    error_message     TEXT,
    duration_s        REAL,
    timestamp         TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS strategies (
    strat_id      SERIAL PRIMARY KEY,
    name          VARCHAR(64) NOT NULL UNIQUE,
    description   TEXT,
    embedding     VECTOR(256)
);

CREATE TABLE IF NOT EXISTS memory_records (
    id                        UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_summary              TEXT,
    embedding                 VECTOR(256),
    context                   JSONB,
    outcome                   JSONB,
    strategy                  VARCHAR(64),
    archived                  BOOLEAN DEFAULT FALSE,
    retrieval_count           INTEGER DEFAULT 0,
    negative_transfer_count   INTEGER DEFAULT 0,
    tier                      VARCHAR(16) DEFAULT 'short_term',  -- 'short_term' | 'long_term'
    consolidation_passes_survived INTEGER DEFAULT 0,
    created_at                TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS insights (
    insight_id    SERIAL PRIMARY KEY,
    text          TEXT,
    sources       TEXT[],
    timestamp     TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memory_strategy   ON memory_records (strategy);
CREATE INDEX IF NOT EXISTS idx_memory_archived    ON memory_records (archived);
CREATE INDEX IF NOT EXISTS idx_experiments_model  ON experiments (model_id);
CREATE INDEX IF NOT EXISTS idx_experiments_hyp    ON experiments (hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_runs_exp           ON experiment_runs (exp_id);
-- pgvector ANN index (IVFFlat); requires ANALYZE after bulk loads for good
-- recall. Swap to HNSW (`USING hnsw`) if pgvector >= 0.5.0 is available.
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_records
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
