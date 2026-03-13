# micheline/local_vlm.py - Wrapper Vision local (Ollama) centralisé via config
# - Par défaut: valeurs de config (VLM_MODEL, VLM_HOST, VLM_TIMEOUT)
# - ENV garde la priorité si défini (OLLAMA_VLM_MODEL / OLLAMA_HOST)

from __future__ import annotations
import base64
import os
import time
from typing import Tuple, Optional, Dict, Any

import requests
import config

class LocalVLM:
    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        timeout: int = None,
    ):
        self.model = (model or os.getenv("OLLAMA_VLM_MODEL") or getattr(config, "VLM_MODEL", "llava:13b")).strip()
        self.host = (host or os.getenv("OLLAMA_HOST") or getattr(config, "VLM_HOST", "http://127.0.0.1:11434")).strip().rstrip("/")
        self.timeout = int(timeout if timeout is not None else getattr(config, "VLM_TIMEOUT", 180))

    def _b64_image(self, path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _get(self, path: str, timeout: Optional[int] = None):
        return requests.get(f"{self.host}{path}", timeout=timeout or self.timeout)

    def _post(self, path: str, json: dict, timeout: Optional[int] = None):
        return requests.post(f"{self.host}{path}", json=json, timeout=timeout or self.timeout)

    def available(self) -> bool:
        try:
            r = self._get("/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def model_present(self) -> bool:
        try:
            r = self._get("/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            data = r.json() if r.content else {}
            models = data.get("models", []) or []
            names = set()
            for m in models:
                if isinstance(m, dict):
                    n = m.get("name") or m.get("model")
                    if isinstance(n, str):
                        names.add(n)
            if self.model in names:
                return True
            base = self.model.split(":")[0]
            return any((n or "").split(":")[0] == base for n in names)
        except Exception:
            return False

    def describe(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
    ) -> Tuple[str, float, Dict[str, Any]]:
        if not os.path.isfile(image_path):
            return ("", 0.0, {"error": f"Image introuvable: {image_path}"})

        if not self.available():
            return ("", 0.0, {"error": f"Ollama non disponible sur {self.host}. Lance 'ollama serve'."})

        if not self.model_present():
            return ("", 0.0, {"error": f"Modèle '{self.model}' absent. Lance:  ollama pull {self.model}"})

        b64img = self._b64_image(image_path)
        content_prompt = prompt or "Décris précisément cette image en français. Détaille les éléments saillants et les limites."

        # 1) /api/chat
        chat_payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": content_prompt,
                    "images": [b64img],
                }
            ],
            "options": {
                "temperature": max(0.0, float(temperature)),
                "num_predict": int(max_tokens),
            },
            "stream": False
        }

        t0 = time.time()
        chat_err = None
        try:
            r = self._post("/api/chat", json=chat_payload)
            dt = time.time() - t0
            if r.status_code == 200 and r.content:
                data = r.json() or {}
                msg = ((data.get("message") or {}).get("content") or "").strip()
                if msg:
                    usage = {
                        "route": "chat",
                        "model": self.model,
                        "total_duration": data.get("total_duration"),
                        "eval_count": data.get("eval_count"),
                        "eval_duration": data.get("eval_duration"),
                        "prompt_eval_count": data.get("prompt_eval_count"),
                        "prompt_eval_duration": data.get("prompt_eval_duration"),
                    }
                    return (msg, dt, usage)
            else:
                chat_err = f"/api/chat {r.status_code}: {r.text[:400]}"
        except Exception as e:
            dt = time.time() - t0
            chat_err = f"/api/chat exception: {e}"

        # 2) Fallback /api/generate
        gen_payload = {
            "model": self.model,
            "prompt": content_prompt,
            "images": [b64img],
            "options": {
                "temperature": max(0.0, float(temperature)),
                "num_predict": int(max_tokens),
            },
            "stream": False
        }

        t1 = time.time()
        try:
            r2 = self._post("/api/generate", json=gen_payload)
            dt2 = time.time() - t1
            if r2.status_code == 200 and r2.content:
                data2 = r2.json() or {}
                msg2 = (data2.get("response") or "").strip()
                if msg2:
                    usage = {
                        "route": "generate",
                        "model": self.model,
                        "total_duration": data2.get("total_duration"),
                        "eval_count": data2.get("eval_count"),
                        "eval_duration": data2.get("eval_duration"),
                        "prompt_eval_count": data2.get("prompt_eval_count"),
                        "prompt_eval_duration": data2.get("prompt_eval_duration"),
                    }
                    return (msg2, dt2, usage)
                return ("", dt2, {"error": "Réponse vide depuis /api/generate", "route": "generate", "chat_err": chat_err})
            return ("", dt2, {"error": f"/api/generate {r2.status_code}: {r2.text[:400]}", "route": "generate", "chat_err": chat_err})
        except Exception as e:
            return ("", 0.0, {"error": f"Requête Ollama échouée (fallback generate): {e}", "route": "generate", "chat_err": chat_err})