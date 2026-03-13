# ai_bot.py - Bot de trading robuste aux changements de features
# - Direction = tendance + confiance directionnelle (moyenne p_dir)
# - Scaler enrichi: {"scaler": RobustScaler, "features": [liste_features_du_training]}
# - Alignement automatique des colonnes au runtime si la liste des features du training est disponible

import os
import time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from joblib import load as joblib_load

import config
import mql5_communicator as com
import news_analyzer as news
from trainer import create_features
from model_manager import EnsembleAIBrain
from ia_utils import classify_detailed_market_state
from config import apply_technical_filters


def parse_mql5_data(data_string: str) -> pd.DataFrame:
    lines = (data_string or "").strip().split(";")
    records = []
    for line in lines:
        parts = line.split(",")
        if len(parts) == 6:
            try:
                records.append({
                    "time": int(parts[0]),
                    "open": float(parts[1]),
                    "high": float(parts[2]),
                    "low": float(parts[3]),
                    "close": float(parts[4]),
                    "tick_volume": int(parts[5]),
                })
            except Exception:
                pass
    df = pd.DataFrame(records)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


# Mémoire des cerveaux par symbole
ai_brains: dict[str, EnsembleAIBrain] = {}


def _load_scaler_info(scaler_path: str):
    """
    Charge un scaler au format:
      - ancien: RobustScaler
      - nouveau: {"scaler": RobustScaler, "features": [noms_features_du_training]}
    Retourne (scaler, feature_list | None)
    """
    loaded = joblib_load(scaler_path)
    if isinstance(loaded, dict) and "scaler" in loaded:
        return loaded["scaler"], list(loaded.get("features") or [])
    return loaded, None


def load_ai_brains() -> bool:
    """
    Charge EnsembleAIBrain + scaler/scaler_features pour chaque paire tradable.
    IMPORTANT:
      - On instancie les modèles avec le nombre de features du training (si connu),
        pour éviter une reconstruction non souhaitée des modèles.
    """
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

    tradable_symbols = config.get_tradable_pairs()
    if not tradable_symbols:
        print("Aucune paire tradable définie (config.tradable_pairs).")
        return False

    for symbol in tradable_symbols:
        try:
            scaler_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_scaler.joblib")
            if not os.path.exists(scaler_path):
                print(f"[LOAD] Scaler manquant pour {symbol} -> ignoré.")
                continue

            # Charge scaler (+ features du training si dispo)
            scaler, scaler_features = _load_scaler_info(scaler_path)
            if scaler_features and len(scaler_features) > 0:
                n_features = len(scaler_features)
            else:
                # fallback: nombre de features "actives" depuis la config courante
                n_features = len(config.get_active_features_for_symbol(symbol))

            # Instancie l'ensemble avec le bon nb de features (critique pour charger les modèles existants)
            brain = EnsembleAIBrain(symbol=symbol, num_features=n_features)
            brain.scaler = scaler
            brain.scaler_features = scaler_features or None  # liste des features du training (si connue)

            ai_brains[symbol] = brain
            print(f"[LOAD] {symbol}: modèles chargés (n_features={n_features}, scaler_features={len(scaler_features or [])}).")
        except Exception as e:
            print(f"Impossible de charger {symbol}: {e}")

    return bool(ai_brains)


def _build_scaled_matrix_for_inference(brain: EnsembleAIBrain, features_df: pd.DataFrame, active_features: list[str]) -> tuple[np.ndarray, list[str]]:
    """
    Construit la matrice X (toutes lignes) à partir du DataFrame de features.
    - Si brain.scaler_features est présent: aligne sur le set du training (order strict + colonnes manquantes à 0.0)
    - Sinon: utilise active_features disponibles dans le DF (vérifie la compatibilité n_features_in_)
    Retourne (X_scaled_all, used_feature_names)
    Lève ValueError si incompatible et aucun alignement possible.
    """
    # 1) Avec liste de features du training -> alignement strict
    if getattr(brain, "scaler_features", None):
        used_names = list(brain.scaler_features)
        for col in used_names:
            if col not in features_df.columns:
                features_df[col] = 0.0
        X = features_df[used_names].values
        try:
            X_scaled = brain.scaler.transform(X)
        except Exception as e:
            raise ValueError(f"[SCALE] Transform échoue (training-feature set): {e}")
        return X_scaled, used_names

    # 2) Ancien scaler: on essaie avec les features actives disponibles
    used_names = [f for f in active_features if f in features_df.columns]
    X = features_df[used_names].values

    # Vérifie la compat compatibilité n_features_in_ si possible
    nfi = getattr(brain.scaler, "n_features_in_", None)
    if nfi is not None and X.shape[1] != int(nfi):
        raise ValueError(f"[SCALE] Mismatch features: X={X.shape[1]} vs scaler expects {int(nfi)}")

    try:
        X_scaled = brain.scaler.transform(X)
    except Exception as e:
        raise ValueError(f"[SCALE] Transform échoue (legacy scaler): {e}")

    return X_scaled, used_names


def run_ai_bot():
    """
    Boucle principale du bot IA (temps réel) avec garde-fou journalier:
      - Si une perte est signalée pour une paire (via loss_lock.jsonl), on ignore
        tous les signaux de cette paire jusqu'à minuit (local).
    Fichier lu: <MQL5_FILES_PATH>/loss_lock.jsonl (JSONL ou 'SYMBOL;YYYY-MM-DD;1')
    """
    if not load_ai_brains():
        print("Aucun cerveau chargé. Arrêt du bot.")
        return

    # ------------- Helpers (auto-contenus) -------------
    import json
    from datetime import datetime

    # Dictionnaire runtime: { "EURUSD": "YYYY-MM-DD" }
    daily_loss_lock = {}
    last_lock_mtime = None

    # Chemin du fichier de lock (dans Common/MQL5/Files si configuré)
    lock_file = os.path.join(
        (config.MQL5_FILES_PATH or os.getcwd()),
        "loss_lock.jsonl"
    )

    def _today_str() -> str:
        return datetime.now().strftime("%Y-%m-%d")

    def _is_symbol_locked_today(sym: str) -> bool:
        d = daily_loss_lock.get((sym or "").upper())
        return isinstance(d, str) and d == _today_str()

    def _refresh_daily_loss_lock_from_file():
        """
        Lit loss_lock.jsonl si modifié; format accepté:
          - JSONL: {"symbol":"EURUSD","date":"2025-10-13","lock":true}
          - CSV:   EURUSD;2025-10-13;1
        Conserve uniquement les verrous du jour courant.
        """
        nonlocal last_lock_mtime, daily_loss_lock
        # Purge des verrous d'hier
        today = _today_str()
        if daily_loss_lock:
            daily_loss_lock = {s: d for s, d in daily_loss_lock.items() if d == today}

        try:
            if not os.path.isfile(lock_file):
                return
            mtime = os.path.getmtime(lock_file)
            if last_lock_mtime is not None and mtime <= last_lock_mtime:
                return  # pas de changement
            last_lock_mtime = mtime

            with open(lock_file, "r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    s = (ln or "").strip()
                    if not s:
                        continue
                    sym, date_s, lock = None, None, True
                    # Tentative JSON
                    try:
                        obj = json.loads(s)
                        sym = str(obj.get("symbol") or obj.get("pair") or "").upper()
                        date_s = str(obj.get("date") or obj.get("day") or "").strip()
                        if "lock" in obj:
                            lock = bool(obj.get("lock"))
                    except Exception:
                        # Fallback CSV: SYMBOL;YYYY-MM-DD;1
                        parts = s.split(";")
                        if len(parts) >= 2:
                            sym = (parts[0] or "").strip().upper()
                            date_s = (parts[1] or "").strip()
                            if len(parts) >= 3:
                                v = (parts[2] or "").strip().lower()
                                lock = v in ("1", "true", "yes", "y", "lock")

                    if not sym or not date_s:
                        continue
                    # Verrouiller ou déverrouiller pour la date spécifiée
                    if lock:
                        daily_loss_lock[sym] = date_s
                    else:
                        if daily_loss_lock.get(sym) == date_s:
                            daily_loss_lock.pop(sym, None)
        except Exception as e:
            print(f"[LOCK] Lecture/parse de {lock_file} échouée: {e}")

    # ------------- Boucle principale -------------
    while True:
        try:
            if not com.wait_for_request():
                continue

            # Rafraîchit le fichier de locks (si modifié)
            _refresh_daily_loss_lock_from_file()

            raw_request = com.read_request()
            if not raw_request:
                continue

            parts = raw_request.strip().split(";", 1)
            if len(parts) != 2:
                continue

            symbol, raw_data_h1 = parts[0].upper(), parts[1]

            # Si symbole non chargé -> nettoie le flag et ignore
            if symbol not in ai_brains:
                if os.path.exists(config.FLAG_FILE):
                    try:
                        os.remove(config.FLAG_FILE)
                    except Exception:
                        pass
                continue

            # Garde-fou: si perte aujourd'hui => HOLD
            if _is_symbol_locked_today(symbol):
                print(f"[{symbol}] Garde-fou actif (perte aujourd’hui). Signal forcé = HOLD.")
                com.write_response({"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0})
                continue

            df_h1 = parse_mql5_data(raw_data_h1)
            if df_h1.empty or len(df_h1) < config.SEQUENCE_LENGTH:
                com.write_response({"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0})
                continue

            # Features
            active_groups = config.get_active_groups_for_symbol(symbol)
            features_df_h1 = create_features(df_h1.copy(), symbol, active_groups)

            # Log contexte détaillé (optionnel)
            try:
                last_state = classify_detailed_market_state(features_df_h1).iloc[-1]
                print(f"[Filtre marché] Contexte actuel = {last_state}")
            except Exception:
                pass

            brain = ai_brains[symbol]
            active_features = config.get_active_features_for_symbol(symbol)

            # Scaling robuste (alignment auto si possible)
            try:
                X_scaled_all, used_feature_names = _build_scaled_matrix_for_inference(
                    brain, features_df_h1.copy(), active_features
                )
            except ValueError as e:
                print(f"[AI_BOT] Erreur scaling {symbol}: {e}")
                com.write_response({"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0})
                continue

            # Séquence la plus récente
            if X_scaled_all.shape[0] < config.SEQUENCE_LENGTH:
                com.write_response({"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0})
                continue

            last_sequence_scaled = X_scaled_all[-config.SEQUENCE_LENGTH:]
            input_data = np.expand_dims(last_sequence_scaled, axis=0)  # (1, SEQ_LEN, n_features)

            # Probas par modèle
            per_model_probs = []
            for model in brain.models:
                try:
                    p = model.model.predict(input_data, verbose=0)[0]
                    ps, pb = float(p[0]), float(p[1])
                except Exception as e:
                    print(f"[PRED] Erreur prédiction modèle: {e}")
                    ps, pb = 0.0, 0.0
                per_model_probs.append((ps, pb))

            # Direction = filtre de tendance (D1/MA200 selon config)
            direction_trend = config.direction_from_trend_row(features_df_h1.iloc[-1])

            # Confiance directionnelle = moyenne des p_dir (en %)
            dir_conf_pct = (
                config.compute_directional_confidence_pct(per_model_probs, direction_trend)
                if direction_trend is not None else 0.0
            )

            final_signal = "HOLD"
            try:
                news_sentiment = news.get_news_sentiment(symbol)
            except Exception:
                news_sentiment = 0.0

            if direction_trend is not None and dir_conf_pct >= config.DECISION_CONF_THRESHOLD_PCT:
                # Gating news (léger, ajustable)
                news_ok = (
                    (direction_trend == "BUY" and news_sentiment >= 0.0) or
                    (direction_trend == "SELL" and news_sentiment <= 0.0)
                )
                if news_ok and apply_technical_filters(direction_trend, features_df_h1.iloc[-1]):
                    final_signal = direction_trend

            # SL/TP
            if final_signal != "HOLD":
                current_price = float(features_df_h1.iloc[-1]["close"])
                current_atr = float(features_df_h1.iloc[-1].get("atr", 0.0))
                if current_atr <= 0:
                    com.write_response({"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0})
                    continue

                cfg = config.load_config_data()
                sl_tp_config = cfg.get("optimal_sl_tp_multipliers", {}).get(symbol, {})
                sl_multiplier = float(sl_tp_config.get("sl", config.ATR_MULTIPLIER_SL))
                tp_multiplier = float(sl_tp_config.get("tp", config.ATR_MULTIPLIER_TP))

                if final_signal == "BUY":
                    stop_loss = current_price - (current_atr * sl_multiplier)
                    take_profit = current_price + (current_atr * tp_multiplier)
                else:
                    stop_loss = current_price + (current_atr * sl_multiplier)
                    take_profit = current_price - (current_atr * tp_multiplier)

                # Lot size centralisé (optionnel)
                lot_size = float(getattr(config, "BOT_DEFAULT_LOT_SIZE", 0.01))

                # Digits: 5 par défaut (ou 3 si JPY)
                digits = 3 if "JPY" in symbol.upper() else 5
                response = {
                    "signal": final_signal,
                    "lot_size": lot_size,
                    "take_profit": round(take_profit, digits),
                    "stop_loss": round(stop_loss, digits),
                }
            else:
                response = {"signal": "HOLD", "lot_size": 0.0, "take_profit": 0.0, "stop_loss": 0.0}

            com.write_response(response)

        except KeyboardInterrupt:
            print("Arrêt demandé (Ctrl+C).")
            break
        except Exception as e:
            print(f"[AI_BOT] ERREUR INATTENDUE: {e}")
            # Sécurise le flag file pour ne pas bloquer l'EA
            if os.path.exists(config.FLAG_FILE):
                try:
                    os.remove(config.FLAG_FILE)
                except Exception:
                    pass
            time.sleep(3)


if __name__ == "__main__":
    load_dotenv()
    if not os.getenv("NEWS_API_KEY"):
        print("AVERTISSEMENT: NEWS_API_KEY non trouvée dans .env (news gating=neutre)")
    run_ai_bot()