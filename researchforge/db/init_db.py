"""Creates the SQLite equivalent of db/schema.sql -- what the Python code in
this repo actually runs against. `rdg.graph.ResearchDevelopmentGraph` and
`memory.ecrm.ECRM` create their own `rdg_nodes`/`rdg_edges`/`memory_records`
tables lazily when given a `db_path`; this module exists so a whole run's
tables (including the experiments/experiment_runs audit trail that
`safety.sandbox.SafeRunner` doesn't persist on its own) can share one on-disk
database file, inspectable afterwards with any SQLite browser.
"""
from __future__ import annotations
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS hypotheses(
    id TEXT PRIMARY KEY, content TEXT, gap_id TEXT, status TEXT, created_at TEXT);

CREATE TABLE IF NOT EXISTS models(
    model_id TEXT PRIMARY KEY, model_type TEXT, genome TEXT, parent_ids TEXT,
    generation INTEGER, created_at TEXT);

CREATE TABLE IF NOT EXISTS experiments(
    exp_id TEXT PRIMARY KEY, hypothesis_id TEXT, model_id TEXT, strategy TEXT,
    result_metric REAL, train_metric REAL, evidence_score REAL,
    failure_category TEXT, success INTEGER, duration_s REAL, timestamp TEXT);

CREATE TABLE IF NOT EXISTS experiment_runs(
    run_id INTEGER PRIMARY KEY AUTOINCREMENT, exp_id TEXT, safety_status TEXT,
    error_message TEXT, duration_s REAL, timestamp TEXT);

CREATE TABLE IF NOT EXISTS strategies(
    strat_id TEXT PRIMARY KEY, name TEXT UNIQUE, description TEXT);

CREATE TABLE IF NOT EXISTS insights(
    insight_id TEXT PRIMARY KEY, text TEXT, sources TEXT, timestamp TEXT);

CREATE INDEX IF NOT EXISTS idx_experiments_model ON experiments(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_exp ON experiment_runs(exp_id);
"""


def init_sqlite(db_path: str) -> None:
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)
    con.commit()
    con.close()


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "researchforge.db"
    init_sqlite(path)
    print(f"Initialized {path}")
