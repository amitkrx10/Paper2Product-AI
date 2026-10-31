"""SQLite persistence with a versioned migration runner (PostgreSQL is a documented follow-up)."""
from __future__ import annotations
import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any, Optional

MIGRATIONS: list[tuple[int, str]] = [
    (1, """
    CREATE TABLE projects(
        id TEXT PRIMARY KEY, name TEXT NOT NULL, status TEXT NOT NULL,
        created REAL NOT NULL, updated REAL NOT NULL, data TEXT NOT NULL DEFAULT '{}');
    CREATE TABLE events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, project_id TEXT NOT NULL, ts REAL NOT NULL,
        level TEXT NOT NULL, stage TEXT NOT NULL, message TEXT NOT NULL, data TEXT NOT NULL DEFAULT '{}');
    CREATE INDEX idx_events_project ON events(project_id, id);
    """),
    (2, """
    CREATE TABLE audit(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, action TEXT NOT NULL,
        actor TEXT NOT NULL, project_id TEXT, detail TEXT NOT NULL DEFAULT '{}');
    """),
]


def new_id(prefix: str = "p") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Storage:
    def __init__(self, path: str):
        self.path = path
        if path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.migrate()

    def migrate(self) -> int:
        with self._lock:
            self._conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations(version INTEGER PRIMARY KEY, applied REAL)")
            done = {r[0] for r in self._conn.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version not in done:
                    self._conn.executescript(sql)
                    self._conn.execute("INSERT INTO schema_migrations VALUES(?,?)", (version, time.time()))
            self._conn.commit()
            return max(r[0] for r in self._conn.execute("SELECT version FROM schema_migrations"))

    def ping(self) -> bool:
        with self._lock:
            return self._conn.execute("SELECT 1").fetchone()[0] == 1

    # projects -------------------------------------------------------------
    def create_project(self, name: str, data: dict) -> str:
        pid = new_id("proj")
        now = time.time()
        with self._lock:
            self._conn.execute("INSERT INTO projects VALUES(?,?,?,?,?,?)",
                               (pid, name, "created", now, now, json.dumps(data)))
            self._conn.commit()
        return pid

    def get_project(self, pid: str) -> Optional[dict]:
        with self._lock:
            r = self._conn.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        return self._proj(r) if r else None

    def list_projects(self, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM projects ORDER BY created DESC LIMIT ?", (limit,)).fetchall()
        return [self._proj(r) for r in rows]

    @staticmethod
    def _proj(r: sqlite3.Row) -> dict:
        return {"id": r["id"], "name": r["name"], "status": r["status"], "created": r["created"],
                "updated": r["updated"], "data": json.loads(r["data"])}

    def set_status(self, pid: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE projects SET status=?, updated=? WHERE id=?", (status, time.time(), pid))
            self._conn.commit()

    def update_data(self, pid: str, **fields: Any) -> None:
        with self._lock:
            r = self._conn.execute("SELECT data FROM projects WHERE id=?", (pid,)).fetchone()
            if not r:
                raise KeyError(pid)
            d = json.loads(r["data"])
            d.update(fields)
            self._conn.execute("UPDATE projects SET data=?, updated=? WHERE id=?", (json.dumps(d), time.time(), pid))
            self._conn.commit()

    # events / audit --------------------------------------------------------
    def add_event(self, pid: str, stage: str, message: str, level: str = "info", data: dict | None = None) -> int:
        with self._lock:
            cur = self._conn.execute("INSERT INTO events(project_id,ts,level,stage,message,data) VALUES(?,?,?,?,?,?)",
                                     (pid, time.time(), level, stage, message, json.dumps(data or {})))
            self._conn.commit()
            return cur.lastrowid

    def events(self, pid: str, after: int = 0) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events WHERE project_id=? AND id>? ORDER BY id", (pid, after)).fetchall()
        return [{"id": r["id"], "ts": r["ts"], "level": r["level"], "stage": r["stage"],
                 "message": r["message"], "data": json.loads(r["data"])} for r in rows]

    def audit(self, action: str, actor: str, project_id: str | None, detail: dict) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO audit(ts,action,actor,project_id,detail) VALUES(?,?,?,?,?)",
                               (time.time(), action, actor, project_id, json.dumps(detail)))
            self._conn.commit()

    def audit_log(self, project_id: str | None = None) -> list[dict]:
        with self._lock:
            if project_id:
                rows = self._conn.execute("SELECT * FROM audit WHERE project_id=? ORDER BY id", (project_id,)).fetchall()
            else:
                rows = self._conn.execute("SELECT * FROM audit ORDER BY id").fetchall()
        return [{"ts": r["ts"], "action": r["action"], "actor": r["actor"],
                 "project_id": r["project_id"], "detail": json.loads(r["detail"])} for r in rows]
