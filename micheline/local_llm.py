# micheline/local_llm.py - Local LLM (llama.cpp) avec gestion RAM
# - Vérifie la RAM disponible AVANT de charger
# - Ajuste n_ctx dynamiquement si RAM insuffisante
# - Support déchargement/rechargement à la demande
# - Fallback sur les candidats si un modèle échoue

import os
import glob
import time
import gc
from typing import Tuple, Optional, List, Dict
import config

try:
    from llama_cpp import Llama
except Exception:
    Llama = None

try:
    import psutil
except ImportError:
    psutil = None


# ======================================================================
# Helpers RAM
# ======================================================================

def get_ram_info() -> dict:
    """Retourne les infos RAM en MB."""
    if psutil is None:
        return {"total_mb": 0, "available_mb": 0, "used_percent": 0, "available_percent": 100}
    try:
        mem = psutil.virtual_memory()
        return {
            "total_mb": round(mem.total / (1024 * 1024)),
            "available_mb": round(mem.available / (1024 * 1024)),
            "used_mb": round(mem.used / (1024 * 1024)),
            "used_percent": round(mem.percent, 1),
            "available_percent": round(100 - mem.percent, 1),
        }
    except Exception:
        return {"total_mb": 0, "available_mb": 0, "used_percent": 0, "available_percent": 100}


def estimate_model_ram_mb(model_path: str, n_ctx: int) -> float:
    """
    Estime la RAM nécessaire pour charger un modèle GGUF.
    Formule approximative:
      RAM ≈ taille_fichier * facteur_quant + kv_cache
      kv_cache ≈ n_ctx * 0.5 MB (approximation grossière)
    """
    try:
        file_mb = os.path.getsize(model_path) / (1024 * 1024)
    except Exception:
        file_mb = 0

    fname = os.path.basename(model_path).upper()

    # Facteur multiplicateur selon la quantification
    # mmap=True réduit l'empreinte réelle, mais le système peut quand même paginer
    if "BF16" in fname or "F16" in fname:
        factor = 1.1  # avec mmap, pas tout en RAM
    elif "F32" in fname:
        factor = 1.2
    elif "Q8" in fname:
        factor = 1.05
    elif "Q6" in fname:
        factor = 1.05
    elif "Q5" in fname:
        factor = 1.05
    elif "Q4" in fname:
        factor = 1.05
    elif "Q3" in fname:
        factor = 1.05
    elif "Q2" in fname:
        factor = 1.05
    else:
        factor = 1.1

    model_ram = file_mb * factor

    # KV cache: ~0.5 MB par 1024 tokens de contexte (approximation)
    kv_cache_mb = (n_ctx / 1024) * 0.5

    # Overhead llama.cpp (~200 MB)
    overhead = 200

    return model_ram + kv_cache_mb + overhead


def compute_safe_n_ctx(model_path: str, desired_n_ctx: int, ram_limit_percent: float) -> int:
    """
    Calcule un n_ctx sûr basé sur la RAM DISPONIBLE (pas la limite système).
    Ne descend jamais en dessous de LLM_N_CTX_MIN.
    """
    ram = get_ram_info()
    if ram["total_mb"] <= 0:
        return desired_n_ctx

    # Budget = RAM disponible actuellement (pas basé sur le % total)
    available_mb = ram.get("available_mb", 0)
    # On garde une marge de 2 GB pour le système
    budget_mb = max(0, available_mb - 2048)

    min_ctx = int(getattr(config, "LLM_N_CTX_MIN", 2048))

    try:
        file_mb = os.path.getsize(model_path) / (1024 * 1024)
    except Exception:
        file_mb = 0

    # RAM modèle de base (sans KV cache)
    base_ram = estimate_model_ram_mb(model_path, 0)

    # RAM restante pour le KV cache
    remaining = max(0, budget_mb - base_ram)

    # Environ 0.5 MB par 1024 tokens de KV cache
    max_ctx_from_ram = int((remaining / 0.5) * 1024) if remaining > 0 else min_ctx

    safe_ctx = min(desired_n_ctx, max(min_ctx, max_ctx_from_ram))

    if safe_ctx < desired_n_ctx:
        print(f"[RAM] n_ctx ajusté: {desired_n_ctx} → {safe_ctx} "
              f"(dispo: {available_mb:.0f} MB, budget: {budget_mb:.0f} MB, base modèle: {base_ram:.0f} MB)")

    return safe_ctx

# ======================================================================
# Helpers détection GGUF
# ======================================================================

def _find_all_gguf(directory: str) -> list:
    """Retourne les .gguf triés par priorité de quantification."""
    if not directory or not os.path.isdir(directory):
        return []

    gguf_files = []
    for root_dir, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".gguf"):
                full = os.path.join(root_dir, fname)
                try:
                    sz = os.path.getsize(full)
                except Exception:
                    sz = 0
                gguf_files.append((full, sz, fname))

    if not gguf_files:
        return []

    def _quant_priority(name: str) -> int:
        n = name.upper()
        if "Q4_K_M" in n: return 10
        if "Q4_K_S" in n: return 11
        if "Q5_K_M" in n: return 15
        if "Q5_K_S" in n: return 16
        if "Q4_K_L" in n: return 17
        if "Q5_K_L" in n: return 18
        if "Q6_K" in n:   return 20
        if "Q8_0" in n:   return 25
        if "Q4_0" in n:   return 30
        if "Q4_1" in n:   return 31
        if "Q5_0" in n:   return 32
        if "Q5_1" in n:   return 33
        if "Q3_K_M" in n: return 35
        if "Q3_K_L" in n: return 36
        if "Q3_K_S" in n: return 37
        if "Q2_K" in n:   return 40
        if "IQ" in n:     return 45
        if "BF16" in n:   return 90
        if "F16" in n:    return 91
        if "F32" in n:    return 95
        return 50

    gguf_files.sort(key=lambda x: (_quant_priority(x[2]), -x[1]))
    return gguf_files


def _diagnose_load_error(model_path: str, error: Exception) -> str:
    fname = os.path.basename(model_path)
    fupper = fname.upper()
    size_mb = 0
    try:
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
    except Exception:
        pass

    lines = [f"Échec: {fname} ({size_mb:.0f} MB)"]
    lines.append(f"  Erreur: {str(error)[:200]}")

    if "BF16" in fupper or "F16" in fupper or "F32" in fupper:
        lines.append("  ⚠ Modèle FULL PRECISION → Téléchargez Q4_K_M ou Q8_0")

    ram = get_ram_info()
    if ram["available_mb"] > 0 and size_mb > ram["available_mb"] * 0.8:
        lines.append(f"  ⚠ RAM insuffisante: {ram['available_mb']} MB dispo, modèle ~{size_mb:.0f} MB")

    return "\n".join(lines)


# ======================================================================
# Classe principale
# ======================================================================

class LocalLLM:
    def __init__(
        self,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_threads: Optional[int] = None,
        n_gpu_layers: int = 0,
        verbose: bool = False,
    ):
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python n'est pas installé.\n"
                "→ pip install llama-cpp-python"
            )

        # ============================================================
        # 1) Vérification RAM avant tout
        # ============================================================
        ram = get_ram_info()
        ram_limit = float(getattr(config, "RAM_LIMIT_PERCENT", 50))
        ram_warn = float(getattr(config, "RAM_WARN_PERCENT", 40))

        if ram["total_mb"] > 0:
            print(f"[RAM] Total: {ram['total_mb']} MB | "
                  f"Utilisé: {ram.get('used_mb', 0)} MB ({ram['used_percent']}%) | "
                  f"Disponible: {ram['available_mb']} MB")
            print(f"[RAM] Limite configurée: {ram_limit}% ({ram['total_mb'] * ram_limit / 100:.0f} MB)")

            if ram["used_percent"] >= ram_limit:
                print(f"[RAM] ⚠ RAM déjà à {ram['used_percent']}% (limite: {ram_limit}%)")
                print(f"[RAM]   Libération mémoire Python avant chargement...")
                gc.collect()
                ram = get_ram_info()

        # ============================================================
        # 2) Résolution des candidats GGUF
        # ============================================================
        candidates_to_try = []

        if model_path and os.path.isfile(model_path):
            candidates_to_try.append(model_path)
        else:
            env_path = os.getenv("MICHELINE_LLM_MODEL", "").strip()
            if env_path and os.path.isfile(env_path):
                candidates_to_try.append(env_path)

            cfg_gguf = getattr(config, "LLM_DEFAULT_GGUF", "").strip()
            if cfg_gguf and os.path.isfile(cfg_gguf) and cfg_gguf not in candidates_to_try:
                candidates_to_try.append(cfg_gguf)

            model_dir = getattr(config, "LLM_MODEL_DIR", "").strip()
            if not model_dir:
                model_dir = os.path.join("micheline", "models", "llm")

            for fpath, _sz, _fname in _find_all_gguf(model_dir):
                if fpath not in candidates_to_try:
                    candidates_to_try.append(fpath)

            fallback_dir = os.path.join("micheline", "models", "llm")
            if os.path.abspath(fallback_dir) != os.path.abspath(model_dir):
                for fpath, _sz, _fname in _find_all_gguf(fallback_dir):
                    if fpath not in candidates_to_try:
                        candidates_to_try.append(fpath)

        if not candidates_to_try:
            raise FileNotFoundError(
                "Aucun fichier .gguf trouvé.\n"
                "Placez un modèle quantifié (Q4_K_M) dans micheline/models/llm/"
            )

        # ============================================================
        # 3) Filtrer les modèles trop gros pour la RAM
        # ============================================================
        if ram["total_mb"] > 0:
            max_budget_mb = ram["total_mb"] * (ram_limit / 100.0)
            safe_candidates = []
            skipped = []

            for c in candidates_to_try:
                try:
                    fsize_mb = os.path.getsize(c) / (1024 * 1024)
                except Exception:
                    fsize_mb = 0

                estimated = estimate_model_ram_mb(c, int(getattr(config, "LLM_N_CTX", 8192)))

                if estimated <= max_budget_mb:
                    safe_candidates.append(c)
                else:
                    skipped.append((c, estimated))
                    print(f"[RAM] ⛔ Modèle trop gros: {os.path.basename(c)} "
                          f"(~{estimated:.0f} MB estimé > {max_budget_mb:.0f} MB limite)")

            if safe_candidates:
                candidates_to_try = safe_candidates
            elif skipped:
                # Tous trop gros, mais on essaie quand même le plus petit
                skipped.sort(key=lambda x: x[1])
                candidates_to_try = [skipped[0][0]]
                print(f"[RAM] ⚠ Tous les modèles dépassent la limite. "
                      f"Tentative avec le plus petit: {os.path.basename(candidates_to_try[0])}")

        # ============================================================
        # 4) Configuration
        # ============================================================
        if n_threads is None:
            n_threads = max(2, (os.cpu_count() or 4))

        desired_n_ctx = int(n_ctx if n_ctx is not None else getattr(config, "LLM_N_CTX", 8192))
        use_mmap = bool(getattr(config, "LLM_USE_MMAP", True))
        use_mlock = bool(getattr(config, "LLM_USE_MLOCK", False))

        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.model_path = None
        self.n_ctx = desired_n_ctx
        self._llm = None
        self._last_use_time = time.time()

        # ============================================================
        # 5) LoRA
        # ============================================================
        lora_path = None
        try:
            active_dir = getattr(config, "ADAPTER_ACTIVE_PATH", "") or ""
            if active_dir and os.path.isdir(active_dir):
                for name in ("adapter.gguf", "lora.gguf"):
                    p = os.path.join(active_dir, name)
                    if os.path.isfile(p):
                        lora_path = p
                        break
                if not lora_path:
                    lora_cands = glob.glob(os.path.join(active_dir, "*.gguf"))
                    if lora_cands:
                        lora_path = lora_cands[0]
                if lora_path:
                    print(f"[LLM] LoRA détecté: {lora_path}")
        except Exception as e:
            print(f"[LLM] LoRA ignoré: {e}")

        # ============================================================
        # 6) Chargement avec fallback
        # ============================================================
        errors_log = []

        for i, candidate in enumerate(candidates_to_try):
            fname = os.path.basename(candidate)
            print(f"[LLM] ── Tentative {i+1}/{len(candidates_to_try)}: {fname} ──")

            if hasattr(config, "guess_model_info"):
                info = config.guess_model_info(candidate)
                print(f"[LLM]   {info['family']} | {info['quant'] or '?'} | "
                      f"{info['params_hint'] or '?'} | {info['size_mb']} MB")

            # Ajuster n_ctx selon RAM disponible
            auto_ctx = bool(getattr(config, "LLM_MAX_N_CTX_AUTO", True))
            if auto_ctx:
                safe_ctx = compute_safe_n_ctx(candidate, desired_n_ctx, ram_limit)
            else:
                safe_ctx = desired_n_ctx

            estimated = estimate_model_ram_mb(candidate, safe_ctx)
            print(f"[LLM]   n_ctx={safe_ctx} | RAM estimée: ~{estimated:.0f} MB")

            kwargs = dict(
                model_path=candidate,
                n_ctx=safe_ctx,
                n_threads=self.n_threads,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                use_mmap=use_mmap,
                use_mlock=use_mlock,
                logits_all=False,
                embedding=False,
            )
            if lora_path:
                kwargs["lora_base"] = candidate
                kwargs["lora_path"] = lora_path

            try:
                # Libérer la mémoire avant chargement
                gc.collect()

                self._llm = Llama(**kwargs)
                self.model_path = candidate
                self.n_ctx = safe_ctx
                self._last_use_time = time.time()

                ram_after = get_ram_info()
                print(f"[LLM] ✅ Chargé: {fname} (n_ctx={safe_ctx})")
                print(f"[RAM] Après chargement: {ram_after.get('used_percent', '?')}% utilisé "
                      f"({ram_after.get('available_mb', '?')} MB dispo)")

                try:
                    config.LLM_DEFAULT_GGUF = candidate
                except Exception:
                    pass

                return  # Succès

            except Exception as e:
                diag = _diagnose_load_error(candidate, e)
                errors_log.append(diag)
                print(f"[LLM] ❌ {fname}: {type(e).__name__}: {str(e)[:150]}")

                # Nettoyage après échec
                self._llm = None
                gc.collect()
                continue

        # Tous échoués
        raise ValueError(
            f"Aucun des {len(candidates_to_try)} modèle(s) n'a pu être chargé.\n\n"
            + "\n".join(f"  • {os.path.basename(c)}" for c in candidates_to_try)
            + f"\n\n" + "\n\n".join(errors_log)
            + "\n\nSolution: utilisez un modèle Q4_K_M ou Q5_K_M (pas BF16/F16)"
        )

    # ============================================================
    # Déchargement / Rechargement
    # ============================================================

    def unload(self):
        """Décharge le modèle de la RAM."""
        if self._llm is not None:
            print(f"[LLM] Déchargement du modèle ({os.path.basename(self.model_path or '')})...")
            ram_before = get_ram_info()
            try:
                del self._llm
            except Exception:
                pass
            self._llm = None
            gc.collect()
            ram_after = get_ram_info()
            freed = ram_before.get("used_mb", 0) - ram_after.get("used_mb", 0)
            print(f"[LLM] Déchargé. RAM libérée: ~{max(0, freed)} MB")
            print(f"[RAM] {ram_after.get('used_percent', '?')}% utilisé "
                  f"({ram_after.get('available_mb', '?')} MB dispo)")

    def is_loaded(self) -> bool:
        return self._llm is not None

    def touch(self):
        """Met à jour le timestamp de dernière utilisation."""
        self._last_use_time = time.time()

    def idle_seconds(self) -> float:
        """Secondes écoulées depuis la dernière utilisation."""
        return time.time() - self._last_use_time

    # ============================================================
    # Chat
    # ============================================================

    def chat(
        self,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Tuple[str, float, dict]:
        if not messages or not isinstance(messages, list):
            return ("", 0.0, {})

        if self._llm is None:
            raise RuntimeError("Modèle non chargé. Appelez _ensure_llm_loaded().")

        self.touch()

        temperature = float(getattr(config, "LLM_CHAT_TEMPERATURE", 0.25)) if temperature is None else float(temperature)
        top_p = float(getattr(config, "LLM_CHAT_TOP_P", 0.95)) if top_p is None else float(top_p)
        max_tokens = int(getattr(config, "LLM_CHAT_MAX_TOKENS", 900)) if max_tokens is None else int(max_tokens)

        final_messages = []
        if system_prompt:
            final_messages.append({"role": "system", "content": system_prompt})
        final_messages.extend(messages)

        t0 = time.time()
        out = self._llm.create_chat_completion(
            messages=final_messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        dt = time.time() - t0

        try:
            content = out["choices"][0]["message"]["content"]
        except Exception:
            content = ""

        usage = out.get("usage", {})
        return (content.strip(), dt, usage)