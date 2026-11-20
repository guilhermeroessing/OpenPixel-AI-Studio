"""Tarefas longas em segundo plano (ex.: animacao de corrida, ~5 min): submit() + consulta por id."""
from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable

_JOBS: dict[str, dict[str, Any]] = {}
_LOCK = threading.Lock()
_MAX_KEEP = 20


def submit(fn: Callable[[Callable[[int, int], None]], Any]) -> str:
    """Roda fn(progress) em uma thread. progress(feitos, total) atualiza a barra. Devolve o id."""
    jid = uuid.uuid4().hex[:12]
    job = {"id": jid, "status": "running", "progress": 0, "total": 1, "result": None, "error": None,
           "started": time.time()}
    with _LOCK:
        _JOBS[jid] = job
        for old in sorted(_JOBS.values(), key=lambda j: j["started"])[:-_MAX_KEEP]:
            _JOBS.pop(old["id"], None)

    def progress(done: int, total: int) -> None:
        job["progress"], job["total"] = done, total

    def run() -> None:
        try:
            job["result"] = fn(progress)
            job["status"] = "done"
        except Exception as exc:  # noqa: BLE001 - o erro vai para o cliente, nao derruba a API
            job["error"], job["status"] = str(exc), "error"

    threading.Thread(target=run, daemon=True, name=f"job-{jid}").start()
    return jid


def get(jid: str) -> dict[str, Any] | None:
    job = _JOBS.get(jid)
    if job is None:
        return None
    out = {k: job[k] for k in ("id", "status", "progress", "total", "result", "error")}
    out["elapsed"] = round(time.time() - job["started"], 1)
    return out
