# worker.py - Worker robuste (heartbeat + idle + timeout optionnel)
# - Lit la file tasks.json (config.WORKER_TASK_FILE)
# - Écrit son statut dans worker_status.json (heartbeat 1s)
# - Supporte:
#     * script absolu (fichier .py)
#     * module python (ex: "micheline.tools.patch_task") -> python -m ...
# - Timeout configurable par tâche (task["timeout_sec"]) ou global (config.WORKER_TASK_TIMEOUT_SEC)
# - CWD optionnel (task["cwd"]), ignoré si non autorisé

from __future__ import annotations
import json
import os
import subprocess
import sys
import time

import config

# Optionnel: garde-fou pour cwd
try:
    from micheline.permissions import policy as _policy
except Exception:
    _policy = None

TASK_FILE = config.WORKER_TASK_FILE
STATUS_FILE = config.WORKER_STATUS_FILE
POLL = float(getattr(config, "WORKER_POLL_INTERVAL_SEC", 2.0))
DEFAULT_TIMEOUT = float(getattr(config, "WORKER_TASK_TIMEOUT_SEC", 0))  # 0 = pas de timeout global

def _safe_json_write(path: str, obj: dict):
    try:
        base_dir = os.path.dirname(path) or "."
        os.makedirs(base_dir, exist_ok=True)
    except Exception:
        pass
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

def _safe_json_read(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def update_status(script, params, status, extra: dict | None = None):
    data = {
        "script": script,
        "params": params,
        "status": status,
        "ts": time.time()
    }
    if isinstance(extra, dict):
        data.update(extra)
    _safe_json_write(STATUS_FILE, data)

def set_idle_with_verify(retries: int = 5, delay: float = 0.2) -> bool:
    for _ in range(retries):
        update_status(None, None, "idle")
        time.sleep(delay)
        data = _safe_json_read(STATUS_FILE) or {}
        if data.get("status") == "idle":
            return True
    update_status(None, None, "idle")
    return False

def _build_command(script: str, params: list[str]) -> list[str]:
    if os.path.isfile(script):
        return [sys.executable, "-X", "utf8", script, *params]
    else:
        module_path = script.replace('.py', '').replace(os.sep, '.')
        return [sys.executable, "-m", module_path, *params]

def _allowed_cwd(cwd: str | None) -> str | None:
    if not cwd:
        return None
    try:
        cwd_abs = os.path.abspath(cwd)
        if _policy is None:
            return cwd_abs
        if _policy.is_path_allowed(cwd_abs):
            return cwd_abs
        else:
            return None
    except Exception:
        return None

def run_task(task: dict) -> bool:
    script = task.get('script')
    params = task.get('params')
    timeout_sec = task.get('timeout_sec', DEFAULT_TIMEOUT)
    cwd = _allowed_cwd(task.get('cwd'))

    if not script or not isinstance(params, list):
        print(f"--- [WORKER] Tâche mal formée ignorée : {task} ---", flush=True)
        return True

    print(f"\n--- [WORKER] Démarrage '{script}' params={params} timeout={timeout_sec or 'none'} cwd={cwd or '(default)'} ---", flush=True)
    update_status(script, params, "en_cours", {"cwd": cwd})

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["PYTHONHASHSEED"] = "42"
    env["TF_DETERMINISTIC_OPS"] = "1"
    env["TF_CUDNN_DETERMINISTIC"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"

    cmd = _build_command(script, params)
    print(f"--- [WORKER] Commande: {' '.join(cmd)} ---", flush=True)

    start = time.time()
    try:
        process = subprocess.Popen(cmd, env=env, cwd=cwd)
        # Heartbeat + timeout loop
        while True:
            ret = process.poll()
            if ret is not None:
                break
            update_status(script, params, "en_cours", {"runtime_sec": round(time.time() - start, 1)})
            # Timeout ?
            if timeout_sec and (time.time() - start) > timeout_sec:
                print(f"--- [WORKER] Timeout atteint ({timeout_sec}s). Terminaison du processus. ---", flush=True)
                try:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                except Exception:
                    pass
                update_status(script, params, "timeout", {"runtime_sec": round(time.time() - start, 1)})
                return False
            time.sleep(1.0)

        if process.returncode == 0:
            print(f"--- [WORKER] Tâche '{script}' OK. ---", flush=True)
            return True
        else:
            print(f"--- [WORKER] ERREUR: '{script}' code={process.returncode}. ---", flush=True)
            return False
    except Exception as e:
        print(f"--- [WORKER] CRASH '{script}' {params}: {e} ---", flush=True)
        return False
    finally:
        ok = set_idle_with_verify()
        if not ok:
            print("[WORKER] Avertissement: statut 'idle' non confirmé (réessayez).", flush=True)

def get_next_task() -> dict | None:
    if not os.path.exists(TASK_FILE):
        return None
    try:
        with open(TASK_FILE, 'r+', encoding='utf-8') as f:
            try:
                tasks = json.load(f)
            except json.JSONDecodeError:
                tasks = []
            if not tasks:
                return None
            next_task = tasks.pop(0)
            f.seek(0)
            f.truncate()
            json.dump(tasks, f, ensure_ascii=False, indent=2)
            return next_task
    except Exception:
        try:
            with open(TASK_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception:
            pass
        return None

def main_loop():
    print("--- [WORKER] Démarré et en attente de tâches ---", flush=True)
    set_idle_with_verify()
    while True:
        task = get_next_task()
        if task:
            run_task(task)
        else:
            set_idle_with_verify()
            time.sleep(POLL)

if __name__ == "__main__":
    if not os.path.exists(TASK_FILE):
        try:
            with open(TASK_FILE, 'w', encoding='utf-8') as f:
                json.dump([], f)
        except Exception:
            pass
    main_loop()