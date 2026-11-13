"""SQLite persistence for portfolio risk runs.

The twin database is shared with ``market_space_twin.simulator``.  Risk runs
live in their own table so the existing runs, events and idempotency tables are
untouched.  A run is keyed by a deterministic seed derived from the input
fingerprint and the referenced twin baseline, which makes replays return the
same ``risk_run_id``.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from contextlib import contextmanager
from typing import Any, Mapping

from market_space_twin.simulation.twin_store import DomainError, fingerprint, stable_json, utc_now

METHOD_VERSION = "portfolio-risk-v1"


def risk_run_seed(payload: Mapping[str, Any], twin: Mapping[str, Any] | None) -> str:
    """Deterministic id for a published comparison.

    The id depends on the input content, the method version and the referenced
    twin baseline (case id and revision), not on transient uuid run ids, so the
    same comparison replays unchanged even in a freshly created database that
    reconstructed the same twin baseline.
    """
    return hashlib.sha256(
        stable_json(
            {
                "method": METHOD_VERSION,
                "input_fingerprint": fingerprint(payload),
                "twin_baseline": None
                if twin is None
                else {
                    "case_id": twin["case_id"],
                    "revision": twin["revision"],
                    "state": twin["state"],
                },
            }
        ).encode("utf-8")
    ).hexdigest()


class RiskStore:
    def __init__(self, database: str):
        self.database = str(database)
        self._lock = threading.RLock()
        self._initialize()

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS risk_runs(
                    run_id TEXT PRIMARY KEY,
                    input_fingerprint TEXT NOT NULL,
                    twin_run_id TEXT,
                    as_of TEXT NOT NULL,
                    response TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS risk_runs_input ON risk_runs(input_fingerprint);
                """
            )

    def find_twin_baseline(self, case_id: str | None) -> dict[str, Any] | None:
        if not case_id:
            return None
        with self.connect() as connection:
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
            ).fetchone()
            if table is None:
                return None
            row = connection.execute(
                "SELECT run_id, case_id, state, revision, updated_at FROM runs WHERE case_id=? ORDER BY updated_at DESC LIMIT 1",
                (case_id,),
            ).fetchone()
        if row is None:
            return None
        return dict(row)

    def stored(self, run_id: str, input_fingerprint: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as connection:
            row = connection.execute(
                "SELECT response, input_fingerprint, twin_run_id, as_of FROM risk_runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        if row["input_fingerprint"] != input_fingerprint:
            raise DomainError("stored risk run does not match the input fingerprint")
        payload = json.loads(row["response"])
        payload["_stored_twin_run_id"] = row["twin_run_id"]
        return payload

    def remember(
        self,
        run_id: str,
        input_fingerprint: str,
        twin_run_id: str | None,
        as_of: str,
        response: Mapping[str, Any],
    ) -> None:
        with self._lock, self.connect() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO risk_runs(run_id,input_fingerprint,twin_run_id,as_of,response,created_at) VALUES(?,?,?,?,?,?)",
                (run_id, input_fingerprint, twin_run_id, as_of, stable_json(dict(response)), utc_now()),
            )

    def _stats_with(self, connection: sqlite3.Connection) -> dict[str, int]:
        names = ["risk_runs"]
        for name in ("runs", "events"):
            table = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            if table is not None:
                names.append(name)
        return {name: int(connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]) for name in names}

    def stats(self) -> dict[str, int]:
        with self.connect() as connection:
            return self._stats_with(connection)
