-- ResearchForge-ECRM: PostgreSQL Schema
-- Run with: psql -U postgres -f 001_initial.sql

-- Enable pgvector extension (requires pgvector installed)
-- CREATE EXTENSION IF NOT EXISTS vector;

-- ── hypotheses ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS hypotheses (
    id          SERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    gap_id      INT,
    status      VARCHAR(32) DEFAULT 'pending',
    times_tried INT DEFAULT 0,
    best_metric FLOAT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── experiments ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS experiments (
    exp_id          SERIAL PRIMARY KEY,
    hypothesis_id   INT REFERENCES hypotheses(id),
    code            TEXT,
    result          FLOAT,
    success         BOOLEAN DEFAULT FALSE,
    train_loss      FLOAT,
    val_loss        FLOAT,
    failure_category VARCHAR(64),
    runtime_seconds  FLOAT,
    timestamp       TIMESTAMPTZ DEFAULT NOW()
);

-- ── model_genomes ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_genomes (
    id              SERIAL PRIMARY KEY,
    model_id        VARCHAR(64) UNIQUE NOT NULL,
    parent_id       VARCHAR(64),
    generation      INT DEFAULT 0,
    description     JSONB NOT NULL,
    fingerprint     VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── memory_records ────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS memory_records (
    id              SERIAL PRIMARY KEY,
    record_id       VARCHAR(64) UNIQUE NOT NULL,
    -- embedding    VECTOR(384),  -- Uncomment when pgvector is available
    summary         TEXT,
    outcome         JSONB,
    link_node       VARCHAR(64),
    failure_flags   JSONB DEFAULT '[]',
    reliability     FLOAT DEFAULT 0.0,
    domain          VARCHAR(128),
    task_id         VARCHAR(64),
    hypothesis_id   VARCHAR(64),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── strategies ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS strategies (
    id              SERIAL PRIMARY KEY,
    description     TEXT NOT NULL,
    -- embedding    VECTOR(384),
    ntr_count       INT DEFAULT 0,
    success_count   INT DEFAULT 0,
    total_uses      INT DEFAULT 0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── insights ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS insights (
    id              SERIAL PRIMARY KEY,
    content         TEXT NOT NULL,
    related_ids     JSONB DEFAULT '[]',
    relevance_score FLOAT DEFAULT 0.5,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- ── rdg_nodes (mirror from graph DB) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rdg_nodes (
    id          VARCHAR(64) PRIMARY KEY,
    node_type   VARCHAR(32) NOT NULL,
    content     TEXT,
    status      VARCHAR(32) DEFAULT 'pending',
    attributes  JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ── rdg_edges ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS rdg_edges (
    id          VARCHAR(64) PRIMARY KEY,
    from_node   VARCHAR(64) REFERENCES rdg_nodes(id),
    to_node     VARCHAR(64) REFERENCES rdg_nodes(id),
    relation    VARCHAR(64),
    confidence  FLOAT DEFAULT 1.0,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiments_hypothesis ON experiments(hypothesis_id);
CREATE INDEX IF NOT EXISTS idx_memory_records_task ON memory_records(task_id);
CREATE INDEX IF NOT EXISTS idx_memory_records_reliability ON memory_records(reliability DESC);
CREATE INDEX IF NOT EXISTS idx_rdg_nodes_type ON rdg_nodes(node_type);
