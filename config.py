# config.py - Version centralisée (incluant Phase 5: garde-fou écriture, qualité, git, audit)
# - Fournit toutes les constantes et réglages (avec valeurs par défaut).
# - Charge/merge le JSON persistant (ia_config.json) pour les réglages utilisateur et l'état dynamique.
# - Expose des helpers communs (conversions d'UT, filtre de tendance, outils RAG/LLM, worker, Phase 5).

import json
import os
import numpy as np
import MetaTrader5 as mt5

# ====================== Emplacement du fichier JSON ======================
CONFIG_PATH = "ia_config.json"

# ====================== Seuils techniques généraux =======================
ADX_THRESHOLD = 0
MIN_VOLUME_RATIO = 0.0

# ====================== Map de timeframes ================================
VALID_TIMEFRAMES = ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1", "MN1"]
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
    "W1": mt5.TIMEFRAME_W1,
    "MN1": mt5.TIMEFRAME_MN1,
}

# ====================== UT d'entraînement par défaut =====================
TIMEFRAME_TO_TRAIN = mt5.TIMEFRAME_H1

# ====================== Outils de conversions UT/bars ====================
def years_to_h1_bars(years: float) -> int:
    return int(years * 260 * 24)

def months_to_h1_bars(months: float) -> int:
    return int(months * 21 * 24)

def timeframe_to_h1_bars(tf_str: str) -> int:
    s = (tf_str or "H4").upper()
    if s == "H1": return 1
    if s == "H4": return 4
    if s == "D1": return 24
    if s in ["W1", "W"]: return 24 * 5
    if s in ["MN1", "MN"]: return 24 * 21
    return 4

# ====================== Auto-détection modèle GGUF ======================
def find_gguf_in_directory(directory: str) -> str:
    """
    Recherche récursivement un fichier .gguf dans le dossier donné.
    Retourne le chemin du meilleur candidat (quantifié > full precision).
    Préfère Q4_K_M > Q5_K_M > Q8_0 > ... > BF16/F16 (dernier recours).
    """
    if not directory or not os.path.isdir(directory):
        return ""

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
        return ""

    def _quant_score(name: str) -> int:
        """Score bas = meilleur candidat."""
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

    # Tri: meilleure quantification d'abord, puis par taille décroissante à score égal
    gguf_files.sort(key=lambda x: (_quant_score(x[2]), -x[1]))
    best = gguf_files[0][0]

    if len(gguf_files) > 1:
        print(f"[LLM Auto] {len(gguf_files)} fichiers .gguf trouvés dans '{directory}'.")
        print(f"[LLM Auto] Sélection optimale: {os.path.basename(best)} (score priorité: {_quant_score(os.path.basename(best))})")
        for f_path, f_size, f_name in gguf_files:
            marker = " ← sélectionné" if f_path == best else ""
            print(f"  - {f_name} ({f_size / (1024 * 1024):.1f} MB, priorité: {_quant_score(f_name)}){marker}")
    else:
        print(f"[LLM Auto] Modèle détecté: {os.path.basename(best)} ({gguf_files[0][1] / (1024 * 1024):.1f} MB)")

    return best

def guess_model_info(gguf_path: str) -> dict:
    """
    Devine des informations sur le modèle à partir du nom du fichier GGUF.
    Retourne un dict avec: name, family, quant, size_mb, params_hint.
    """
    if not gguf_path:
        return {"name": "inconnu", "quant": "", "family": "inconnu", "size_mb": 0, "params_hint": ""}

    fname = os.path.basename(gguf_path).lower()
    name_no_ext = os.path.basename(gguf_path).replace(".gguf", "").replace(".GGUF", "")

    # Taille fichier
    try:
        size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    except Exception:
        size_mb = 0

    # Famille du modèle (heuristique par nom de fichier)
    families = [
        "llama", "qwen", "mistral", "mixtral", "phi", "gemma", "deepseek",
        "yi", "falcon", "mpt", "bloom", "codellama", "vicuna", "zephyr",
        "solar", "openchat", "starling", "neural", "hermes", "tinyllama",
        "orca", "nous-hermes", "wizardlm", "command-r", "dbrx", "internlm",
        "baichuan", "chatglm", "starcoder", "codestral", "granite"
    ]
    family = "inconnu"
    for f in families:
        if f in fname:
            family = f
            break

    # Quantification
    quants = [
        "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
        "IQ3_XXS", "IQ3_XS", "IQ3_S", "IQ3_M", "IQ4_XS", "IQ4_NL",
        "Q2_K", "Q2_K_S",
        "Q3_K_S", "Q3_K_M", "Q3_K_L",
        "Q4_0", "Q4_1", "Q4_K_S", "Q4_K_M", "Q4_K_L", "Q4_K",
        "Q5_0", "Q5_1", "Q5_K_S", "Q5_K_M", "Q5_K_L", "Q5_K",
        "Q6_K", "Q8_0", "F16", "F32", "BF16"
    ]
    quant = ""
    for q in sorted(quants, key=len, reverse=True):  # plus long d'abord pour éviter les faux positifs
        if q.lower() in fname:
            quant = q
            break

    # Estimation du nombre de paramètres (heuristique par nom)
    import re as _re
    params_hint = ""
    # Cherche des patterns comme "7b", "13b", "1_8b", "1.8b", "70b", etc.
    m = _re.search(r'(\d+[._]?\d*)\s*[bB](?:\b|[-_.])', fname)
    if m:
        raw = m.group(1).replace("_", ".")
        params_hint = f"{raw}B"

    return {
        "name": name_no_ext,
        "quant": quant,
        "family": family,
        "size_mb": round(size_mb, 1),
        "params_hint": params_hint
    }


def resolve_gguf_path(model_dir: str, raw_gguf: str) -> str:
    """
    Résout le chemin complet vers le fichier GGUF:
    1. Si raw_gguf est un chemin absolu existant -> l'utiliser
    2. Si raw_gguf est un nom de fichier, chercher dans model_dir
    3. Sinon, scanner model_dir pour trouver un .gguf automatiquement
    """
    # Cas 1: chemin absolu ou relatif existant
    if raw_gguf:
        expanded = os.path.abspath(os.path.expanduser(raw_gguf))
        if os.path.isfile(expanded):
            return expanded

    # Cas 2: nom de fichier simple -> chercher dans model_dir
    if raw_gguf and model_dir:
        candidate = os.path.join(model_dir, raw_gguf)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    # Cas 3: auto-détection dans le dossier
    if model_dir and os.path.isdir(model_dir):
        auto = find_gguf_in_directory(model_dir)
        if auto:
            return auto

    # Cas 4: raw_gguf tel quel (fallback, probablement inexistant)
    return raw_gguf or ""

# ====================== Filtre technique centralisé ======================
def apply_technical_filters(signal, entry_row):
    """
    Filtre technique centralisé appliqué avant décision finale.
    - Vérifie ADX minimal si présent
    - Vérifie momentum volume si présent
    - Refuse BUY si ma_fast < ma_slow, SELL si ma_fast > ma_slow
    - Filtre de tendance (si TREND_FILTER_ENABLED): BUY au-dessus / SELL en-dessous de trend_filter_ma
    """
    try:
        adx_val = entry_row.get('adx', np.nan)
        try: adx_val = float(adx_val)
        except Exception: adx_val = np.nan
        if np.isfinite(adx_val) and adx_val < ADX_THRESHOLD:
            return False

        vol_mom = entry_row.get('volume_momentum', np.nan)
        try: vol_mom = float(vol_mom)
        except Exception: vol_mom = np.nan
        if np.isfinite(vol_mom) and vol_mom < MIN_VOLUME_RATIO:
            return False

        ma_fast = entry_row.get('ma_fast', np.nan)
        ma_slow = entry_row.get('ma_slow', np.nan)
        try:
            ma_fast = float(ma_fast); ma_slow = float(ma_slow)
        except Exception:
            ma_fast, ma_slow = np.nan, np.nan
        if np.isfinite(ma_fast) and np.isfinite(ma_slow):
            if signal == "BUY" and ma_fast < ma_slow: return False
            if signal == "SELL" and ma_fast > ma_slow: return False

        if TREND_FILTER_ENABLED:
            close_p = entry_row.get('close', np.nan)
            ma_trend = entry_row.get('trend_filter_ma', np.nan)
            try:
                close_p = float(close_p); ma_trend = float(ma_trend)
            except Exception:
                pass
            above = None
            try:
                if 'trend_filter_above' in entry_row:
                    v = entry_row['trend_filter_above']
                    if v == v: above = bool(int(v))
                if above is None and np.isfinite(close_p) and np.isfinite(ma_trend):
                    above = close_p > ma_trend
            except Exception:
                above = None
            if above is None: return True
            if signal == "BUY" and not above: return False
            if signal == "SELL" and above: return False

        return True
    except Exception:
        return True

# ====================== Définition des features ==========================
ALWAYS_ON_FEATURES = {
    'Contexte Temporel': [
        'hour', 'day_of_week', 'london_session', 'ny_session', 'tokyo_session'
    ],
    'Contexte Multi-Temporel': [
        'trend_h4', 'trend_d1', 'trend_w1',
        'volatility_h4', 'volatility_d1'
    ],
    'Structure de Marché': [
        'distance_from_swing_high', 'distance_from_swing_low', 'swing_range_width'
    ],
    'Statistiques Avancées': [
        'log_returns', 'realized_vol', 'returns_skew', 'returns_kurt'
    ],
    'Analyse de Contexte': [
        'state_Tendance_Haussiere_Impulsion',
        'state_Tendance_Haussiere_Correction',
        'state_Tendance_Haussiere_Faible',
        'state_Tendance_Baissiere_Impulsion',
        'state_Tendance_Baissiere_Correction',
        'state_Tendance_Baissiere_Faible',
        'state_Range_Calme',
        'state_Range_Volatil',
        'state_Indetermine'
    ],
}

CATEGORIZED_FEATURES = {
    "Tendance": {
        "Moyennes Mobiles": ['ma_fast_dist', 'ma_slow_dist', 'ma_cross_age'],
        "VWAP": ['vwap_dist'],
        "SuperTrend": ['supertrend_dir', 'supertrend_dist'],
        "Choppiness": ['choppiness_index'],
        "Vortex": ['vortex_diff'],
        "Parabolic SAR": ['psar_dist']
    },
    "Momentum": {
        "RSI": ['rsi'],
        "Stochastique": ['stoch_k', 'stoch_d'],
        "CCI": ['cci'],
        "MACD": ['macd_dist', 'macd_signal_dist', 'macd_histogram_norm'],
        "Divergences": ['rsi_price_divergence', 'macd_price_divergence'],
        "Awesome Oscillator": ['awesome_oscillator']
    },
    "Volatilité & Contraction/Expansion": {
        "Bandes de Bollinger": ['bb_width', 'bb_squeeze'],
        "Canaux Keltner": ['kc_width', 'price_kc_pos'],
        "ATR": ['atr_percent'],
        "Volatilité GARCH": ['garch_volatility'],
        "Donchian Channels": ['donchian_width', 'donchian_pos']
    },
    "Volume & Order Flow": {
        "Volume en Ticks": ['tick_volume', 'volume_momentum', 'volume_price_ratio'],
        "Force Index": ['elder_force_index'],
        "Profil de Volume": ['poc_dist', 'vah_dist', 'val_dist'],
        "Flux d'Ordres Simple": ['cumulative_delta'],
        "On-Balance Volume (OBV)": ['obv_slope'],
        "Chaikin Money Flow (CMF)": ['cmf']
    },
    "Niveaux de Prix Clés": {
        "Points Pivots": ['dist_from_pivot', 'dist_from_r1', 'dist_from_s1'],
        "Retracements Fibonacci": ['dist_from_fib_382', 'dist_from_fib_500', 'dist_from_fib_618']
    },
    "Systèmes Complets": {
        "Ichimoku Kinko Hyo": ['ichi_cross', 'price_cloud_pos', 'cloud_strength', 'chikou_momentum']
    },
    "Psychologie de Marché": {
        "Patterns de Chandeliers": ['candle_pattern_score', 'candle_body_ratio']
    },
    "Analyse Inter-Marchés": {
        "Corrélation Forex": ['corr_EURUSD', 'corr_GBPUSD', 'corr_USDJPY', 'corr_AUDUSD', 'corr_USDCAD', 'corr_EURGBP', 'corr_EURJPY', 'corr_GBPJPY']
    }
}

# ====================== Paramètres d'entraînement/optimisation ===========
INITIAL_TRAINING_YEARS = 8
UPDATE_TRAINING_MONTHS = 6
INITIAL_TRAINING_BARS = years_to_h1_bars(INITIAL_TRAINING_YEARS)
UPDATE_TRAINING_BARS = months_to_h1_bars(UPDATE_TRAINING_MONTHS)

INITIAL_EPOCHS = 50
UPDATE_EPOCHS = 30
PATIENCE_INITIAL = 15
PATIENCE_UPDATE = 20

BACKTEST_BARS = 50000
OPTIMIZATION_BARS = 10000
OPTIMIZATION_EPOCHS = 10

PREDICTION_HORIZON_STR = "H4"
FUTURE_LOOKAHEAD_BARS = timeframe_to_h1_bars(PREDICTION_HORIZON_STR)

# ====================== Paramètres de trading & IA =======================
ATR_PERIOD = 14
ATR_MULTIPLIER_TP = 2.0
ATR_MULTIPLIER_SL = 1.5

SL_TIMEFRAME_STR = "D1"
TP_TIMEFRAME_STR = "H1"

SEQUENCE_LENGTH = 180
MODEL_FOLDER = "trained_models"
ENSEMBLE_MODELS = 3

# ====================== I/O MQL5 (helpers) ===============================
def get_all_feature_groups():
    return [key for category in CATEGORIZED_FEATURES.values() for key in category.keys()]

def save_config_data(config_data: dict):
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=4)
    except Exception as e:
        print(f"ERREUR CRITIQUE: Impossible de sauvegarder la configuration: {e}")

def load_config_data() -> dict:
    default_config = {
        # --- Chemin MQL5/Files (Terminal) : échanges IA<->EA
        "mql5_files_path": "",
        
        # ========= Watcher Service =========
        "watcher_service_enabled": True,  # À activer manuellement via UI ou JSON

        # --- Orchestration d’entraînement
        "training_frequency_days": 1,
        "initial_training_years": INITIAL_TRAINING_YEARS,
        "update_training_months": UPDATE_TRAINING_MONTHS,

        # --- Horizon de labellisation et UT ATR pour SL/TP
        "prediction_horizon": PREDICTION_HORIZON_STR,
        "sl_timeframe": SL_TIMEFRAME_STR,
        "tp_timeframe": TP_TIMEFRAME_STR,
        "auto_optimize_horizon_sl_tp": True,

        # --- Paires
        "selected_pairs": ["EURUSD"],
        "tradable_pairs": [],

        # --- Features (indicateurs) et états d'optimisation
        "active_feature_groups": get_all_feature_groups(),
        "optimized_feature_configs": {},
        "model_performances": {},
        "training_progress": {},
        "use_automatic_feature_selection": True,
        "optimal_sl_tp_multipliers": {},

        # --- Architecture IA
        "use_ensemble_learning": True,
        "use_multi_timeframe": True,
        "use_stacked_model": True,
        "run_on_startup": {},

        # --- Filtre de tendance configurable ---
        "trend_filter_enabled": True,
        "trend_filter_timeframe": "D1",
        "trend_filter_ma_period": 200,
        "trend_filter_bars_min": 500,
        "trend_filter_bars_margin": 50,
        "trend_filter_verbose": True,
        "trend_filter_anchor": "prev",
        "trend_min_distance_atr": 0.0,
        "countertrend_enable": False,
        "countertrend_distance_atr": 1.20,

        # Seuil décision
        "decision_conf_threshold_pct": 20.0,
        # Seuil softmax legacy
        "confidence_threshold": 0.55,

        # --- Coûts backtest (approx)
        "backtest_spread_points": 0,
        "backtest_commission_pips": 0.0,

        # --- STT/TTS ---
        "stt_enabled": True,
        "stt_backend": "vosk",
        "stt_sample_rate": 16000,
        "stt_vad_silence_ms": 800,
        "stt_device": "",
        "tts_enabled": True,
        "tts_backend": "piper",
        "tts_auto_read": False,
        "piper_voice_path": "micheline/models/tts/piper/voices/fr/fr_FR-mls_5809-medium.onnx",
        "piper_config_path": "micheline/models/tts/piper/voices/fr/fr_FR-mls_5809-medium.onnx.json",
        "piper_length_scale": 1.0,
        "piper_noise": 0.667,
        "piper_noise_w": 0.8,
        "piper_speaker_id": 0,
        "tts_rate": 175,
        "tts_volume": 1.0,
        "tts_voice_hint": "female:fr",

        # ========= RAG & Web =========
        "rag_enabled": True,
        "rag_embedding_model_dir": "micheline/models/embeddings/all-MiniLM-L6-v2",
        "rag_faiss_index_path": "micheline/rag/index/faiss/knowledge.faiss",
        "rag_chunk_size": 1000,
        "rag_chunk_overlap": 150,
        "rag_top_k": 5,
        "rag_http_timeout": 20,
        "rag_http_user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "rag_include_tables": True,
        "rag_text_exts": [".txt", ".md", ".py", ".json", ".js", ".html", ".css"],

        "always_check_web": True,
        "news_recent_days": 30,
        "news_max_articles": 8,
        "news_language": "fr",
        "news_sort_by": "publishedAt",

        # ========= LLM (texte) =========
        "llm_chat_temperature": 0.25,
        "llm_chat_top_p": 0.95,
        "llm_chat_max_tokens": 900,
        "llm_model_dir": "micheline/models/llm",           # ← NOUVEAU
        "llm_default_gguf": "",                              # ← MODIFIÉ (vide = auto-détection)
        
        # ========= RAM & Performance =========
        "ram_limit_percent": 75,              # ← MODIFIÉ: 75% au lieu de 50%
        "ram_warn_percent": 65,               # ← MODIFIÉ: 65% au lieu de 40%
        "llm_auto_unload_sec": 300,
        "llm_use_mmap": True,
        "llm_use_mlock": False,
        "llm_max_n_ctx_auto": True,
        "llm_n_ctx_min": 2048,

        # ========= VLM (vision) =========
        "vlm_model": "llava:13b",
        "vlm_host": "http://127.0.0.1:11434",
        "vlm_timeout": 180,
        "use_vlm_always": False,

        # ========= OCR =========
        "ocr_backend_pref": "auto",
        "ocr_lang_primary": "fr",
        "ocr_max_chars": 12000,
        "ocr_image_max_width": 2800,

        # ========= UI & UX =========
        "ui_show_history_on_start": 2,
        "ui_max_chat_rows": 999999999999,

        # ========= Worker / Orchestration =========
        "worker_task_file": "tasks.json",
        "worker_status_file": "worker_status.json",
        "worker_poll_interval_sec": 2.0,
        "worker_task_timeout_sec": 0,  # 0 = pas de timeout global
        "mql5_poll_interval_sec": 0.1,

        # ========= Backtests & Analyzer =========
        "analyzer_tp_window_bars": 24,
        "risk_trend_ma_min_dist_atr": 0.30,
        "risk_round_dist_pips_max": 10.0,
        "risk_extension_pvmaf_max": 1.8,
        "risk_low_margin_pct_min": 6.0,
        "risk_diverge_low_margin_pct_min": 12.0,

        # ========= Optimiseurs =========
        "optimizer_top_n_candidates": 5,
        "optimizer_horizons_to_test": ["H1", "H4", "D1"],
        "optimizer_sl_tp_timeframes_to_test": ["H1", "H4", "D1"],
        "optimizer_sl_grid": [0.5, 1.0, 1.5, 2.0],
        "optimizer_tp_grid": [1.0, 2.0, 3.0, 4.0],
        "sl_tp_pred_batch_size": 5000,

        # ========= Export / Backtest MT5 =========
        "csv_delimiter": ";",
        "preferred_tester_agent": "Agent-127.0.0.1-3000",
        
        # ========= Phase 5: Sécurité & Outils =========
        "allowed_roots": [r"C:\\"],  # vide => racine = cwd
        "allow_write_outside": True,        # garde-fou
        "allowed_write_exts": [".py", ".md", ".txt", ".json", ".yml", ".yaml", ".ini", ".toml", ".csv", ".cfg"],
        "max_patch_kb": 256,
        "audit_log_path": "micheline/logs/audit.jsonl",

        # Tests (runner)
        "test_commands": ["python -m pytest -q"],
        "sandbox_run_tests_sec": 300,

        # Qualité code
        "quality_enabled": True,
        "quality_skip_missing": True,
        "quality_require_green": True,
        "quality_apply_format": False,
        "quality_timeout_sec": 240,
        "quality_check_commands": [
            "ruff .",
            "black --check .",
            "isort --check-only .",
            "mypy .",
            "pydocstyle .",
            "bandit -q -r ."
        ],
        "quality_format_commands": [
            "isort .",
            "black .",
            "ruff --fix ."
        ],
        "patch_run_tests": False,

        # Git (optionnel)
        "git_enabled": False,
        "git_repo_root": ""
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                config_from_file = json.load(f)
            default_config.update(config_from_file)
        except (json.JSONDecodeError, IOError):
            print(f"AVERTISSEMENT: {CONFIG_PATH} corrompu ou illisible. Réinitialisation.")
            save_config_data(default_config)
    else:
        print(f"INFO: {CONFIG_PATH} non trouvé. Création par défaut.")
        save_config_data(default_config)
    return default_config

# ====================== Helpers pour récupérer les features actives ======
def get_active_groups_for_symbol(symbol: str):
    config_data = load_config_data()
    use_auto_select = config_data.get("use_automatic_feature_selection", True)
    optimized_config = config_data.get("optimized_feature_configs", {}).get(symbol)
    if use_auto_select and optimized_config and optimized_config.get("best_groups"):
        return optimized_config["best_groups"]
    else:
        return config_data.get("active_feature_groups", get_all_feature_groups())

def get_active_features_for_symbol(symbol: str):
    active_groups = get_active_groups_for_symbol(symbol)
    final_features = [feature for group in ALWAYS_ON_FEATURES.values() for feature in group]
    for category_data in CATEGORIZED_FEATURES.values():
        for group_name, features in category_data.items():
            if group_name in active_groups:
                final_features.extend(features)
    return sorted(list(set(final_features)))

def get_selected_pairs():
    return load_config_data().get("selected_pairs", [])

def get_tradable_pairs():
    return load_config_data().get("tradable_pairs", [])

# ====================== Exposition (variables lues depuis JSON) =========
CONFIG = load_config_data()

MQL5_FILES_PATH = CONFIG.get("mql5_files_path")
TRAINING_FREQUENCY_DAYS = CONFIG.get("training_frequency_days", 7)

INITIAL_TRAINING_BARS = years_to_h1_bars(CONFIG.get("initial_training_years", INITIAL_TRAINING_YEARS))
UPDATE_TRAINING_BARS = months_to_h1_bars(CONFIG.get("update_training_months", UPDATE_TRAINING_MONTHS))

PREDICTION_HORIZON_STR = CONFIG.get("prediction_horizon", PREDICTION_HORIZON_STR).upper()
FUTURE_LOOKAHEAD_BARS = timeframe_to_h1_bars(PREDICTION_HORIZON_STR)

SL_TIMEFRAME_STR = CONFIG.get("sl_timeframe", SL_TIMEFRAME_STR).upper()
TP_TIMEFRAME_STR = CONFIG.get("tp_timeframe", TP_TIMEFRAME_STR).upper()
SL_TIMEFRAME = TIMEFRAME_MAP.get(SL_TIMEFRAME_STR, mt5.TIMEFRAME_D1)
TP_TIMEFRAME = TIMEFRAME_MAP.get(TP_TIMEFRAME_STR, mt5.TIMEFRAME_H1)

REQUEST_FILE = os.path.join(MQL5_FILES_PATH, "requete_ia.txt") if MQL5_FILES_PATH else ""
RESPONSE_FILE = os.path.join(MQL5_FILES_PATH, "reponse_ia.txt") if MQL5_FILES_PATH else ""
FLAG_FILE = os.path.join(MQL5_FILES_PATH, "signal_pret.txt") if MQL5_FILES_PATH else ""

# --- Filtre de tendance (exposés) ---
TREND_FILTER_ENABLED = CONFIG.get("trend_filter_enabled", True)
TREND_FILTER_TIMEFRAME_STR = CONFIG.get("trend_filter_timeframe", "D1").upper()
TREND_FILTER_MA_PERIOD = int(CONFIG.get("trend_filter_ma_period", 200))
TREND_FILTER_BARS_MIN = int(CONFIG.get("trend_filter_bars_min", 500))
TREND_FILTER_BARS_MARGIN = int(CONFIG.get("trend_filter_bars_margin", 50))
TREND_FILTER_VERBOSE = CONFIG.get("trend_filter_verbose", True)
TREND_FILTER_ANCHOR = CONFIG.get("trend_filter_anchor", "prev").lower()

TREND_MIN_DISTANCE_ATR = float(CONFIG.get("trend_min_distance_atr", 0.0))
COUNTERTREND_ENABLE = bool(CONFIG.get("countertrend_enable", False))
COUNTERTREND_DISTANCE_ATR = float(CONFIG.get("countertrend_distance_atr", 1.2))

DECISION_CONF_THRESHOLD_PCT = float(CONFIG.get("decision_conf_threshold_pct", 20.0))
CONFIDENCE_THRESHOLD = float(CONFIG.get("confidence_threshold", 0.55))

# ====================== Filtre de tendance natif MT5 ====================
def _tf_to_timedelta(tf_str: str, bars: int):
    import pandas as pd
    s = (tf_str or "").upper()
    if s == "M1":  return pd.Timedelta(minutes=bars)
    if s == "M5":  return pd.Timedelta(minutes=5 * bars)
    if s == "M15": return pd.Timedelta(minutes=15 * bars)
    if s == "M30": return pd.Timedelta(minutes=30 * bars)
    if s == "H1":  return pd.Timedelta(hours=bars)
    if s == "H4":  return pd.Timedelta(hours=4 * bars)
    if s == "D1":  return pd.Timedelta(days=bars)
    if s in ["W1", "W"]:  return pd.Timedelta(weeks=bars)
    if s in ["MN1", "MN"]: return pd.Timedelta(days=30 * bars)
    return pd.Timedelta(hours=4 * bars)

def compute_trend_filter_columns(df, symbol: str):
    import pandas as pd
    import pandas_ta as ta

    if df is None or len(df) == 0:
        return df

    df_tmp = df.copy()
    had_time_col = 'time' in df_tmp.columns
    if had_time_col and not isinstance(df_tmp.index, pd.DatetimeIndex):
        df_tmp.set_index('time', inplace=True)

    tf_str = TREND_FILTER_TIMEFRAME_STR
    tf_const = TIMEFRAME_MAP.get(tf_str, mt5.TIMEFRAME_H4)

    start_ts = pd.Timestamp(df_tmp.index.min())
    end_ts   = pd.Timestamp(df_tmp.index.max())
    buffer_bars = TREND_FILTER_MA_PERIOD + TREND_FILTER_BARS_MARGIN
    pre_delta = _tf_to_timedelta(tf_str, int(buffer_bars))
    start_dt = (start_ts - pre_delta).to_pydatetime()
    end_dt   = (end_ts + _tf_to_timedelta(tf_str, 2)).to_pydatetime()

    df_tf = None
    try:
        rates = mt5.copy_rates_range(symbol, tf_const, start_dt, end_dt)
        if rates is not None and len(rates) > 0:
            df_tf = pd.DataFrame(rates)
    except Exception:
        df_tf = None

    if df_tf is None or df_tf.empty:
        rule_map = {"H1": "H", "H4": "4H", "D1": "D", "W1": "W", "MN1": "M"}
        rule = rule_map.get(tf_str, "4H")
        try:
            close_res = df_tmp['close'].resample(rule).last()
            ma_res = close_res.rolling(int(TREND_FILTER_MA_PERIOD), min_periods=1).mean()
            if TREND_FILTER_ANCHOR == "prev":
                ma_res = ma_res.shift(1)
            ma_up = ma_res.reindex(df_tmp.index, method='ffill')
            df_tmp['trend_filter_ma'] = ma_up.astype('float32')
            df_tmp['trend_filter_above'] = (df_tmp['close'] > df_tmp['trend_filter_ma']).astype('int8')
        except Exception:
            df_tmp['trend_filter_ma'] = pd.Series(index=df_tmp.index, dtype='float32')
            df_tmp['trend_filter_above'] = pd.Series(index=df_tmp.index, dtype='float32')
    else:
        df_tf['time'] = pd.to_datetime(df_tf['time'], unit='s')
        df_tf.set_index('time', inplace=True)
        df_tf['trend_filter_ma'] = ta.sma(df_tf['close'], int(TREND_FILTER_MA_PERIOD))
        if TREND_FILTER_ANCHOR == "prev":
            df_tf['trend_filter_ma'] = df_tf['trend_filter_ma'].shift(1)
        right = df_tf[['trend_filter_ma']].dropna().sort_index()
        df_tmp = pd.merge_asof(
            left=df_tmp.sort_index(),
            right=right,
            left_index=True, right_index=True,
            direction='backward'
        )
        df_tmp['trend_filter_ma'] = df_tmp['trend_filter_ma'].astype('float32')
        df_tmp['trend_filter_above'] = (df_tmp['close'] > df_tmp['trend_filter_ma']).astype('int8')

    if had_time_col and 'time' not in df_tmp.columns:
        df_tmp['time'] = df_tmp.index
    return df_tmp

# ====================== Décision: direction & confiance ==================
def direction_from_trend_row(entry_row):
    try:
        if 'trend_filter_above' in entry_row:
            v = entry_row['trend_filter_above']
            if v == v:
                return "BUY" if int(v) == 1 else "SELL"
        c = float(entry_row.get('close', np.nan))
        ma = float(entry_row.get('trend_filter_ma', np.nan))
        if c == c and ma == ma:
            return "BUY" if c > ma else "SELL"
        return None
    except Exception:
        return None

def compute_ensemble_confidence_pct(model_probs):
    vals = []
    for ps, pb in model_probs:
        try:
            ps = float(ps); pb = float(pb)
            vals.append(max(ps, pb) * 100.0)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0

def compute_directional_confidence_pct(model_probs, direction: str) -> float:
    dir_up = (direction or "").upper()
    vals = []
    for ps, pb in model_probs:
        try:
            ps = float(ps); pb = float(pb)
            vals.append(pb if dir_up == "BUY" else ps)
        except Exception:
            pass
    return float(np.mean(vals) * 100.0) if vals else 0.0

# ====================== STT/TTS (exposition) =============================
STT_ENABLED = CONFIG.get("stt_enabled", True)
STT_BACKEND = CONFIG.get("stt_backend", "vosk")
STT_SAMPLE_RATE = int(CONFIG.get("stt_sample_rate", 999999999999))
STT_VAD_SILENCE_MS = int(CONFIG.get("stt_vad_silence_ms", 800))
STT_DEVICE = CONFIG.get("stt_device", "")

TTS_ENABLED = CONFIG.get("tts_enabled", True)
TTS_BACKEND = CONFIG.get("tts_backend", "piper")
TTS_AUTO_READ = CONFIG.get("tts_auto_read", False)
PIPER_VOICE_PATH = CONFIG.get("piper_voice_path", "")
PIPER_CONFIG_PATH = CONFIG.get("piper_config_path", "")
PIPER_LENGTH_SCALE = float(CONFIG.get("piper_length_scale", 1.0))
PIPER_NOISE = float(CONFIG.get("piper_noise", 0.667))
PIPER_NOISE_W = float(CONFIG.get("piper_noise_w", 0.8))
PIPER_SPEAKER_ID = int(CONFIG.get("piper_speaker_id", 0))
TTS_RATE = int(CONFIG.get("tts_rate", 100))
TTS_VOLUME = float(CONFIG.get("tts_volume", 1.0))
TTS_VOICE_HINT = CONFIG.get("tts_voice_hint", "female:fr")

# ====================== RAG & Web =======================================
RAG_ENABLED = CONFIG.get("rag_enabled", True)
RAG_EMBEDDING_MODEL_DIR = CONFIG.get("rag_embedding_model_dir", "micheline/models/embeddings/all-MiniLM-L6-v2")
RAG_FAISS_INDEX_PATH = CONFIG.get("rag_faiss_index_path", "micheline/rag/index/faiss/knowledge.faiss")
RAG_CHUNK_SIZE = int(CONFIG.get("rag_chunk_size", 1000))
RAG_CHUNK_OVERLAP = int(CONFIG.get("rag_chunk_overlap", 150))
RAG_TOP_K = int(CONFIG.get("rag_top_k", 5))
RAG_HTTP_TIMEOUT = int(CONFIG.get("rag_http_timeout", 20))
RAG_HTTP_USER_AGENT = CONFIG.get("rag_http_user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
RAG_INCLUDE_TABLES = bool(CONFIG.get("rag_include_tables", True))
RAG_TEXT_EXTS = CONFIG.get("rag_text_exts", [".txt", ".md", ".py", ".json", ".js", ".html", ".css"])

ALWAYS_CHECK_WEB = bool(CONFIG.get("always_check_web", True))
NEWS_RECENT_DAYS = int(CONFIG.get("news_recent_days", 30))
NEWS_MAX_ARTICLES = int(CONFIG.get("news_max_articles", 8))
NEWS_LANGUAGE = CONFIG.get("news_language", "fr")
NEWS_SORT_BY = CONFIG.get("news_sort_by", "publishedAt")

# ====================== Phase 6: RAG Avancé & Mémoire ==================
RAG_DAILY_INGEST_ENABLED = True
RAG_DAILY_INGEST_HOUR = 12  # 3h du matin, heure locale
RAG_CORPUS_CLEAN_PATH = "micheline/rag/corpus/clean"
RAG_COMPACT_ENABLED = True
RAG_COMPACT_TRIGGER_CHUNKS = 5000  # Déclenche le compactage si > 5000 chunks
RAG_COMPACT_TARGET_CHUNKS = 3000   # Cible à atteindre après compactage

# ====================== Phase 7: Apprentissage Continu ===================
LEARNING_ENABLED = CONFIG.get("learning_enabled", True)                                               # Active/désactive toute la Phase 7 (feedback, dataset SFT, fine-tuning, adapters)
FEEDBACK_LOG_PATH = CONFIG.get("feedback_log_path", "micheline/learning/sft_feedback.jsonl")          # Fichier JSONL où sont enregistrés les 👍/👎 et corrections
SFT_DATASET_PATH = CONFIG.get("sft_dataset_path", "micheline/learning/sft_dataset.jsonl")             # Dataset SFT (prompt/completion) généré à partir des feedbacks
ADAPTERS_DIR = CONFIG.get("adapters_dir", "micheline/learning/adapters")                              # Dossier racine contenant les versions d’adapters LoRA (sous-dossiers datés)
ADAPTER_ACTIVE_NAME = CONFIG.get("adapter_active_name", "")                                           # Nom du sous-dossier d’adapter actuellement actif (ex: v2025xxxx_xxxx), vide = aucun
ADAPTER_ACTIVE_PATH = os.path.join(ADAPTERS_DIR, ADAPTER_ACTIVE_NAME) if ADAPTER_ACTIVE_NAME else ""  # Chemin complet vers l’adapter actif (utilisé par local_llm pour charger le LoRA GGUF)
FINE_TUNE_NIGHTLY = CONFIG.get("fine_tune_nightly", True)                                             # Si True, permet de planifier automatiquement un fine‑tuning (nuit/WE) via le worker
MAX_ADAPTER_VERSIONS = int(CONFIG.get("max_adapter_versions", 5))                                     # Nombre maximum de versions d’adapters à conserver (les plus anciennes pourront être purgées)
LEARNING_EVAL_QUESTIONS = CONFIG.get("learning_eval_questions", [])                                   # Liste de questions de test pour évaluer la qualité avant/après (scripts d’évaluation)
LEARNING_FT_HOUR = int(CONFIG.get("learning_ft_hour", 16))                # Heure du run auto
LEARNING_FT_DAYS = CONFIG.get("learning_ft_days", ["mon","tue","wed","thu","fri","sat","sun"])         # Jours autorisés pour l'auto-run

# ====================== LLM (texte) =====================================
LLM_CHAT_TEMPERATURE = float(CONFIG.get("llm_chat_temperature", 0.25))
LLM_CHAT_TOP_P = float(CONFIG.get("llm_chat_top_p", 0.95))
LLM_CHAT_MAX_TOKENS = int(CONFIG.get("llm_chat_max_tokens", 900))
LLM_N_CTX = int(CONFIG.get("llm_n_ctx", 8192)) # 8192 16384 plus c grand plus c long a repondre
MAX_FILE_DISPLAY_CHARS = 0              # 0 = pas de limite, sinon ex: 1_000_000
CODEBLOCK_EXPAND_DEFAULT = False        # False => hauteur limitée + scrollbar
CODEBLOCK_COLLAPSED_LINES = 10          # nb de lignes visibles sans scroll
CODEBLOCK_FORCE_MIN_LINES = 1          # nb de lignes mini pour activation de la scrollbar vertical
CHAT_ROW_VPAD = 1          # espace vertical entre les rangées (user vs assistant)
CHAT_BUBBLE_VPAD = 0       # marge verticale du canvas de la bulle
CHAT_BUBBLE_MARGIN = 1     # marge interne du fond arrondi (haut/bas)
CHAT_BUBBLE_SHADOW = 5     # ombre (ajoute de la hauteur) -> 0 pour la supprimer
CHAT_TEXT_PADY = 0         # padding vertical à l'intérieur du widget Text de la bulle
CHAT_TEXT_PADX = 10

# ====================== LLM Auto-détection ================================
LLM_MODEL_DIR = CONFIG.get("llm_model_dir", "micheline/models/llm")
_raw_gguf = CONFIG.get("llm_default_gguf", "")
LLM_DEFAULT_GGUF = resolve_gguf_path(LLM_MODEL_DIR, _raw_gguf)

if LLM_DEFAULT_GGUF and os.path.isfile(LLM_DEFAULT_GGUF):
    _info = guess_model_info(LLM_DEFAULT_GGUF)
    print(f"[LLM Config] Modèle résolu: {_info['name']}")
    print(f"[LLM Config]   Famille: {_info['family']} | Quant: {_info['quant']} | "
          f"Params: {_info['params_hint'] or '?'} | Taille: {_info['size_mb']} MB")
    print(f"[LLM Config]   Chemin: {LLM_DEFAULT_GGUF}")
elif LLM_MODEL_DIR:
    print(f"[LLM Config] Aucun .gguf trouvé dans '{LLM_MODEL_DIR}' (sera scanné au premier appel)")
    
# ====================== RAM & Performance ================================
RAM_LIMIT_PERCENT = float(CONFIG.get("ram_limit_percent", 75))
RAM_WARN_PERCENT = float(CONFIG.get("ram_warn_percent", 65))
LLM_AUTO_UNLOAD_SEC = int(CONFIG.get("llm_auto_unload_sec", 300))
LLM_USE_MMAP = bool(CONFIG.get("llm_use_mmap", True))
LLM_USE_MLOCK = bool(CONFIG.get("llm_use_mlock", False))
LLM_MAX_N_CTX_AUTO = bool(CONFIG.get("llm_max_n_ctx_auto", True))
LLM_N_CTX_MIN = int(CONFIG.get("llm_n_ctx_min", 2048))

# ====================== VLM (vision) ====================================
VLM_MODEL = CONFIG.get("vlm_model", "llava:13b")
VLM_HOST = CONFIG.get("vlm_host", "http://127.0.0.1:11434")
VLM_TIMEOUT = int(CONFIG.get("vlm_timeout", 180))
USE_VLM_ALWAYS = bool(CONFIG.get("use_vlm_always", False))

# ====================== OCR =============================================
OCR_BACKEND_PREF = CONFIG.get("ocr_backend_pref", "auto")
OCR_LANG_PRIMARY = CONFIG.get("ocr_lang_primary", "fr")
OCR_MAX_CHARS = int(CONFIG.get("ocr_max_chars", 12000))
OCR_IMAGE_MAX_WIDTH = int(CONFIG.get("ocr_image_max_width", 2800))

# ====================== UI & UX =========================================
SHOW_HISTORY_ON_START = int(CONFIG.get("ui_show_history_on_start", 2))
MAX_CHAT_ROWS = int(CONFIG.get("ui_max_chat_rows", 250))
MESSAGE_INPUT_ROWS = int(os.getenv("UI_MESSAGE_INPUT_ROWS", 8))
# — Réglages miniatures PJ —
ATTACH_THUMB = 50        # taille vignette (côté max)
ATTACH_TILE_PAD = 2       # marge externe autour de chaque vignette
ATTACH_TILE_BORDER = 0    # bordure du cadre (0 = sans bordure)
ATTACH_SHOW_FILENAME = False  # afficher le nom sous la vignette

# ====================== Worker / Orchestrateur ==========================
WORKER_TASK_FILE = CONFIG.get("worker_task_file", "tasks.json")
WORKER_STATUS_FILE = CONFIG.get("worker_status_file", "worker_status.json")
WORKER_POLL_INTERVAL_SEC = float(CONFIG.get("worker_poll_interval_sec", 2.0))
WORKER_TASK_TIMEOUT_SEC = float(CONFIG.get("worker_task_timeout_sec", 0))
MQL5_POLL_INTERVAL_SEC = float(CONFIG.get("mql5_poll_interval_sec", 0.1))

# ====================== Backtests & Analyzer =============================
ANALYZER_TP_WINDOW_BARS = int(CONFIG.get("analyzer_tp_window_bars", 24))
RISK_TREND_MA_MIN_DIST_ATR = float(CONFIG.get("risk_trend_ma_min_dist_atr", 0.30))
RISK_ROUND_DIST_PIPS_MAX = float(CONFIG.get("risk_round_dist_pips_max", 10.0))
RISK_EXTENSION_PVMAF_MAX = float(CONFIG.get("risk_extension_pvmaf_max", 1.8))
RISK_LOW_MARGIN_PCT_MIN = float(CONFIG.get("risk_low_margin_pct_min", 6.0))
RISK_DIVERGE_LOW_MARGIN_PCT_MIN = float(CONFIG.get("risk_diverge_low_margin_pct_min", 12.0))

# ====================== Optimiseurs =====================================
OPTIMIZER_TOP_N_CANDIDATES = int(CONFIG.get("optimizer_top_n_candidates", 5))
OPTIMIZER_HORIZONS_TO_TEST = CONFIG.get("optimizer_horizons_to_test", ["H1", "H4", "D1"])
OPTIMIZER_SL_TP_TIMEFRAMES_TO_TEST = CONFIG.get("optimizer_sl_tp_timeframes_to_test", ["H1", "H4", "D1"])
OPTIMIZER_SL_GRID = CONFIG.get("optimizer_sl_grid", [0.5, 1.0, 1.5, 2.0])
OPTIMIZER_TP_GRID = CONFIG.get("optimizer_tp_grid", [1.0, 2.0, 3.0, 4.0])
SL_TP_PRED_BATCH_SIZE = int(CONFIG.get("sl_tp_pred_batch_size", 5000))

# ====================== Export / Backtest MT5 ============================
CSV_DELIMITER = CONFIG.get("csv_delimiter", ";")
PREFERRED_TESTER_AGENT = CONFIG.get("preferred_tester_agent", "Agent-127.0.0.1-3000")

# ====================== Phase 5: Sécurité & Outils =======================
ALLOWED_ROOTS = CONFIG.get("allowed_roots", ["C:\\"])
ALLOW_WRITE_OUTSIDE = bool(CONFIG.get("allow_write_outside", True))
ALLOWED_WRITE_EXTS = CONFIG.get("allowed_write_exts", [".py", ".md", ".txt", ".json", ".yml", ".yaml", ".ini", ".toml", ".csv", ".cfg"])
MAX_PATCH_KB = int(CONFIG.get("max_patch_kb", 256))
AUDIT_LOG_PATH = CONFIG.get("audit_log_path", "micheline/logs/audit.jsonl")

TEST_COMMANDS = CONFIG.get("test_commands", ["python -m pytest -q"])
SANDBOX_RUN_TESTS_SEC = int(CONFIG.get("sandbox_run_tests_sec", 300))

QUALITY_ENABLED = bool(CONFIG.get("quality_enabled", True))
QUALITY_SKIP_MISSING = bool(CONFIG.get("quality_skip_missing", True))
QUALITY_REQUIRE_GREEN = bool(CONFIG.get("quality_require_green", True))
QUALITY_APPLY_FORMAT = bool(CONFIG.get("quality_apply_format", False))
QUALITY_TIMEOUT_SEC = int(CONFIG.get("quality_timeout_sec", 240))
QUALITY_CHECK_COMMANDS = CONFIG.get("quality_check_commands", [
    "ruff .",
    "black --check .",
    "isort --check-only .",
    "mypy .",
    "pydocstyle .",
    "bandit -q -r ."
])
QUALITY_FORMAT_COMMANDS = CONFIG.get("quality_format_commands", [
    "isort .",
    "black .",
    "ruff --fix ."
])
PATCH_RUN_TESTS = bool(CONFIG.get("patch_run_tests", False))

GIT_ENABLED = bool(CONFIG.get("git_enabled", False))
GIT_REPO_ROOT = CONFIG.get("git_repo_root", "")

# ====================== Phase 7: Apprentissage Continu (code-only) ===================
LEARNING_ENABLED = True  # phase 7 on/off
FEEDBACK_LOG_PATH = "micheline/learning/sft_feedback.jsonl"
SFT_DATASET_PATH = "micheline/learning/sft_dataset.jsonl"
ADAPTERS_DIR = "micheline/learning/adapters"
ADAPTER_ACTIVE_NAME = ""  # ex: "v2025xxxx_xxxx"
ADAPTER_ACTIVE_PATH = os.path.join(ADAPTERS_DIR, ADAPTER_ACTIVE_NAME) if ADAPTER_ACTIVE_NAME else ""

# Auto fine-tuning (horaire et jours) -> 100% code-only
FINE_TUNE_NIGHTLY = True                      # pilote auto activé
LEARNING_FT_HOUR = 16                         # déclenchement 14:00 (heure locale)
LEARNING_FT_DAYS = ["mon","tue","wed","thu","fri","sat","sun"]  # jours autorisés

MAX_ADAPTER_VERSIONS = 5
LEARNING_EVAL_QUESTIONS = [
    "Explique la logique direction=tendance avec filtre D1/MA200.",
    "Donne un exemple de réglage ATR SL/TP pour EURUSD H1.",
    "Propose un plan de backtest propre pour EURUSD.",
    "Pourquoi utiliser LoRA plutôt que full fine‑tuning ?",
    "Explique un pipeline RAG minimal et ses limites.",
    "Corrige ce code Python d’agrégation pandas (erreurs fréquentes).",
    "Donne 3 bonnes pratiques pour éviter la fuite de données en ML.",
    "Que faire en cas de divergence H4 vs vote modèle ?",
    "Comment évaluer un adapter LoRA (avant/après) ?",
    "Propose un format de dataset SFT minimal (prompt/completion)."
]

# ====================== Phase Intel: Entity Registry & Watchers ======================
# (Bloc 1 — Registry)

ENTITY_REGISTRY_DB_PATH = "micheline/intel/db/entity_registry.sqlite"

# Auto-discovery : seuils de validation
ENTITY_AUTO_APPROVE_MIN_SOURCES = 2  # Nombre minimum de sources pour auto-approval
ENTITY_AUTO_APPROVE_TRUSTED_DOMAINS = [  # Domaines considérés comme fiables
    ".gov", ".mil", "reuters.com", "bloomberg.com", "ft.com", "fr.investing.com", "boursorama.com",
    "federalreserve.gov", "ecb.europa.eu", "imf.org", "worldbank.org", "ecb.europa.eu", "boursier.com"
]

# Importance score : seuils
ENTITY_CRITICAL_THRESHOLD = 0.7  # Entités avec score >= 0.7 = critiques
ENTITY_HIGH_IMPORTANCE = 0.8
ENTITY_MEDIUM_IMPORTANCE = 0.5

# Trust score : seuils pour sources
SOURCE_HIGH_TRUST = 0.8  # Sources officielles vérifiées
SOURCE_MEDIUM_TRUST = 0.6
SOURCE_LOW_TRUST = 0.3

# Topics surveillés (à étendre selon besoins)
INTEL_TOPICS = [
    "geo",           # Géopolitique
    "oil",           # Pétrole/Énergie
    "rates",         # Taux d'intérêt
    "fx",            # Foreign Exchange (devises)
    "sanctions",     # Sanctions économiques
    "military",      # Mouvements militaires
    "central_bank",  # Banques centrales
    "tariffs",       # Tarifs/Commerce
    "elections",     # Élections
    "covid",         # Santé publique/Pandémies
]

# ====================== Phase Intel: Watchers (Bloc 2) ======================

# Chemins
WATCHER_EVENTS_DB_PATH = "micheline/intel/db/raw_events.sqlite"

# User-Agent pour le watcher (à personnaliser selon ton projet)
WATCHER_USER_AGENT = "MichelineBot/1.0 (Market Intelligence Bot; Contact: killdrago@hotmail.com)"

# Rate limiting (secondes entre deux requêtes au même domaine)
WATCHER_RATE_LIMIT_SEC = 5.0

# Intervalles de polling par type de source (minutes)
WATCHER_RSS_INTERVAL_MIN = 5       # RSS: toutes les 5 minutes
WATCHER_WEB_INTERVAL_MIN = 15      # Pages web classiques: toutes les 15 minutes
WATCHER_OFFICIAL_INTERVAL_MIN = 30 # Documents officiels: toutes les 30 minutes
WATCHER_SOCIAL_INTERVAL_MIN = 3    # Réseaux sociaux: toutes les 3 minutes

# Respect robots.txt (RFC 9309)
WATCHER_RESPECT_ROBOTS_TXT = True

# Limites d'extraction
WATCHER_MAX_CONTENT_LENGTH = 50000  # Caractères max par article
WATCHER_MIN_CONTENT_LENGTH = 50     # Minimum pour considérer valide

# Timeout HTTP (secondes)
WATCHER_HTTP_TIMEOUT = 15

# Batch size pour le traitement des événements non traités
WATCHER_PROCESS_BATCH_SIZE = 100

WATCHER_RETENTION_DAYS = 7
WATCHER_PURGE_EVERY_SEC = 3600

# ====================== (Bloc 3) ======================

ENABLE_EVENT_CARDS = True
EVENT_CARDS_RETENTION_DAYS = 7
