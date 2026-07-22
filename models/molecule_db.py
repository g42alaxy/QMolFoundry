"""SQLite-backed for storing molecules."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "molecules.db"
_ORDER_BY_SQL = {
    "qed DESC": "qed DESC",
    "qed ASC": "qed ASC",
    "RANDOM()": "RANDOM()",
}


# ── schema ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS molecules (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    smiles       TEXT    NOT NULL,
    model_slug   TEXT    NOT NULL,
    dataset      TEXT    NOT NULL,
    backend      TEXT    NOT NULL,
    shots        INTEGER NOT NULL DEFAULT 0,
    qed          REAL,
    sa           REAL,
    logp         REAL,
    heavy_atoms  INTEGER,
    source       TEXT    NOT NULL DEFAULT 'unknown',
    generated_at TEXT    NOT NULL,
    UNIQUE (smiles, model_slug, dataset, backend)
);

CREATE INDEX IF NOT EXISTS idx_mol_lookup
    ON molecules (model_slug, dataset, backend, qed DESC);

CREATE TABLE IF NOT EXISTS run_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    model_slug     TEXT    NOT NULL,
    dataset        TEXT    NOT NULL,
    backend        TEXT    NOT NULL,
    shots          INTEGER NOT NULL DEFAULT 0,
    n_generated    INTEGER NOT NULL DEFAULT 0,
    n_valid        INTEGER NOT NULL DEFAULT 0,
    n_inserted     INTEGER NOT NULL DEFAULT 0,
    validity_rate  REAL,
    mean_qed       REAL,
    mean_sa        REAL,
    ran_at         TEXT    NOT NULL,
    source         TEXT,
    notes          TEXT
);
"""


# ── connection helpers ─────────────────────────────────────────────────────────


def _connect(path: Path = _DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    return con


@contextmanager
def get_db(path: Path = _DB_PATH):
    con = _connect(path)
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db(path: Path = _DB_PATH) -> None:
    with get_db(path) as con:
        con.executescript(_DDL)


# ── write ─────────────────────────────────────────────────────────────────────


def insert_molecules(
    rows: list[dict],
    model_slug: str,
    dataset: str,
    backend: str,
    shots: int,
    source: str,
    n_generated: int,
    notes: str = "",
    path: Path = _DB_PATH,
) -> int:
    """Insert validated molecules; skip duplicates (UNIQUE constraint).

    Each row dict must have: smiles, qed, sa, logp, heavy_atoms.
    Returns the number of newly inserted rows.
    """
    required = {"smiles", "qed", "sa", "logp", "heavy_atoms"}
    for index, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"molecule row {index} is missing fields: {sorted(missing)}")

    init_db(path)
    now = datetime.now(timezone.utc).isoformat()

    inserted = 0
    with get_db(path) as con:
        for row in rows:
            con.execute(
                """INSERT OR IGNORE INTO molecules
                   (smiles, model_slug, dataset, backend, shots,
                    qed, sa, logp, heavy_atoms, source, generated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    row["smiles"],
                    model_slug,
                    dataset,
                    backend,
                    shots,
                    row["qed"],
                    row["sa"],
                    row["logp"],
                    row["heavy_atoms"],
                    source,
                    now,
                ),
            )
            inserted += con.execute("SELECT changes()").fetchone()[0]

        n_valid = len(rows)
        validity = n_valid / n_generated if n_generated else 0.0
        qed_values = [r["qed"] for r in rows if r["qed"] is not None]
        sa_values = [r["sa"] for r in rows if r["sa"] is not None]
        mean_qed = sum(qed_values) / len(qed_values) if qed_values else None
        mean_sa = sum(sa_values) / len(sa_values) if sa_values else None

        con.execute(
            """INSERT INTO run_log
               (model_slug, dataset, backend, shots, n_generated, n_valid,
                n_inserted, validity_rate, mean_qed, mean_sa, ran_at, source, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_slug,
                dataset,
                backend,
                shots,
                n_generated,
                n_valid,
                inserted,
                validity,
                mean_qed,
                mean_sa,
                now,
                source,
                notes,
            ),
        )

    return inserted


# ── read ──────────────────────────────────────────────────────────────────────


def query_molecules(
    model_slug: str,
    dataset: str,
    backend: str,
    n: int = 16,
    min_qed: float = 0.0,
    order_by: str = "qed DESC",
    path: Path = _DB_PATH,
) -> list[dict]:
    """Return up to n molecules for a given (model, dataset, backend) combo.

    Results are ordered by QED descending by default.
    Returns [] if the DB doesn't exist or the combo has no rows.
    """
    if not path.exists():
        return []
    try:
        order_sql = _ORDER_BY_SQL[order_by]
    except KeyError as exc:
        raise ValueError(f"unsupported molecule ordering: {order_by!r}") from exc
    con = _connect(path)
    try:
        rows = con.execute(
            f"""SELECT smiles, qed, sa, logp, heavy_atoms, backend, source
                FROM molecules
                WHERE model_slug=? AND dataset=? AND backend=?
                  AND (qed IS NULL OR qed >= ?)
                ORDER BY {order_sql}
                LIMIT ?""",
            (model_slug, dataset, backend, min_qed, n),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def has_combo(
    model_slug: str, dataset: str, backend: str, min_rows: int = 1, path: Path = _DB_PATH
) -> bool:
    """Return True if the DB has at least min_rows for this combo."""
    if not path.exists():
        return False
    con = _connect(path)
    try:
        count = con.execute(
            "SELECT COUNT(*) FROM molecules WHERE model_slug=? AND dataset=? AND backend=?",
            (model_slug, dataset, backend),
        ).fetchone()[0]
        return count >= min_rows
    finally:
        con.close()


def combo_stats(path: Path = _DB_PATH) -> list[dict]:
    """Return per-(model, dataset, backend) summary statistics."""
    if not path.exists():
        return []
    con = _connect(path)
    try:
        rows = con.execute(
            """SELECT model_slug, dataset, backend,
                      COUNT(*) AS n_mols,
                      ROUND(AVG(qed),3) AS mean_qed,
                      ROUND(AVG(sa),3)  AS mean_sa,
                      MIN(generated_at) AS first_run,
                      MAX(generated_at) AS last_run
               FROM molecules
               GROUP BY model_slug, dataset, backend
               ORDER BY model_slug, dataset, backend"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def run_history(limit: int = 50, path: Path = _DB_PATH) -> list[dict]:
    """Return the most recent build-run log entries."""
    if not path.exists():
        return []
    con = _connect(path)
    try:
        rows = con.execute(
            """SELECT * FROM run_log ORDER BY ran_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()
