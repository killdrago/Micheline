# generate_backtest_signals.py - Génération IA de signaux pour MT5 (CSV)
# Robustesse:
# - Écrit TOUJOURS un CSV (au moins l'en-tête), même si 0 signal ou erreur (stub)
# - Duplique systématiquement en miroir vers Common\Files\signals
# - Logs: rows_count, listing de dossier après export

import os
import sys
import csv
import argparse
from datetime import datetime, timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from joblib import load as joblib_load

import config
from ia_utils import connect_to_mt5
from trainer import create_features
from model_manager import EnsembleAIBrain
from config import apply_technical_filters


def parse_dt(s: str) -> datetime:
    s = s.strip()
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y.%m.%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y.%m.%d %H:%M",
        "%Y-%m-%d",
        "%Y.%m.%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    return datetime.strptime(s.replace(".", "-"), "%Y-%m-%d %H:%M:%S")


def tf_to_minutes(tf: str) -> int:
    tf = tf.upper()
    tf_map = {"M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440, "W1": 10080, "MN1": 43200}
    return tf_map.get(tf, 60)


def get_common_files_path() -> str:
    return os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Terminal", "Common", "Files")


def export_signals(symbol: str, timeframe: str, rows: List[dict], out_base: str, pretty: bool = True, subdir: str = "signals") -> str:
    base = out_base or os.getcwd()
    out_dir = os.path.join(base, subdir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{symbol}_{timeframe}_signals.csv")
    tmp_file = out_file + ".tmp"

    with open(tmp_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter=config.CSV_DELIMITER)
        if pretty:
            w.writerow(["time", "heure", "direction", "sl_points", "tp_points", "comment"])
            for r in rows:
                d = r["time"]
                w.writerow([d.strftime("%Y-%m-%d"), d.strftime("%H:%M:%S"), r["dir"], r["sl"], r["tp"], r.get("comment", "")])
        else:
            w.writerow(["time", "direction", "sl_points", "tp_points", "comment"])
            for r in rows:
                d = r["time"]
                w.writerow([d.strftime("%Y-%m-%d %H:%M:%S"), r["dir"], r["sl"], r["tp"], r.get("comment", "")])

    os.replace(tmp_file, out_file)
    print(f"Exported: {out_file}")
    return out_file


def list_dir(path: str):
    try:
        print(f"[DEBUG] Listing: {path}")
        names = os.listdir(path)
        if not names:
            print(" - (vide)")
        for n in names:
            print(" -", n)
    except Exception as e:
        print(f"[DEBUG] Impossible de lister le dossier ({path}): {e}")


def extract_terminal_hash(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    p = os.path.normpath(path_str)
    parts = p.split(os.sep)
    try:
        idx = parts.index("Terminal")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    except ValueError:
        pass
    return None


def find_tester_agent_files_path(term_hash: str, prefer_agent: Optional[str] = None) -> Optional[str]:
    if not term_hash:
        return None
    base = os.path.join(os.environ.get("APPDATA", ""), "MetaQuotes", "Tester", term_hash)
    if not os.path.isdir(base):
        return None
    agents = []
    try:
        for d in os.listdir(base):
            full = os.path.join(base, d)
            if d.startswith("Agent-") and os.path.isdir(os.path.join(full, "MQL5", "Files")):
                agents.append(d)
    except Exception:
        pass
    if not agents:
        return None
    if prefer_agent and prefer_agent in agents:
        selected = prefer_agent
    elif config.PREFERRED_TESTER_AGENT in agents:
        selected = config.PREFERRED_TESTER_AGENT
    else:
        local_sorted = sorted(agents, key=lambda s: (not s.startswith("Agent-127.0.0.1-"), s))
        selected = local_sorted[0]
    return os.path.join(base, selected, "MQL5", "Files")


def adapt_out_path_for_tester(user_out_common: Optional[str], prefer_agent: Optional[str]) -> str:
    if user_out_common:
        term_hash = extract_terminal_hash(user_out_common)
        if term_hash:
            path_tester = find_tester_agent_files_path(term_hash, prefer_agent)
            if path_tester:
                return path_tester
        return user_out_common
    term_path_cfg = getattr(config, "MQL5_FILES_PATH", None) or config.load_config_data().get("mql5_files_path", "")
    term_hash = extract_terminal_hash(term_path_cfg)
    if term_hash:
        path_tester = find_tester_agent_files_path(term_hash, prefer_agent)
        if path_tester:
            return path_tester
    return get_common_files_path()


def get_historical_data_range(symbol: str, timeframe_const, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    rates = mt5.copy_rates_range(symbol, timeframe_const, start_dt, end_dt)
    if rates is None or len(rates) == 0:
        return pd.DataFrame()
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df


def _load_scaler_info(path: str):
    obj = joblib_load(path)
    if isinstance(obj, dict) and "scaler" in obj:
        return obj["scaler"], list(obj.get("features") or [])
    return obj, None


def _build_scaled_matrix(df_features: pd.DataFrame, scaler, expected_feature_names: Optional[List[str]], fallback_feature_names: List[str]):
    if expected_feature_names:
        used = list(expected_feature_names)
        for col in used:
            if col not in df_features.columns:
                df_features[col] = 0.0
        X = df_features[used].values
        Xs = scaler.transform(X)
        return Xs, used
    used = [f for f in fallback_feature_names if f in df_features.columns]
    X = df_features[used].values
    nfi = getattr(scaler, "n_features_in_", None)
    if nfi is not None and X.shape[1] != int(nfi):
        raise ValueError("Mismatch features: X={} vs scaler expects {}.".format(X.shape[1], int(nfi)))
    Xs = scaler.transform(X)
    return Xs, used


def safe_export_stub(symbol: str, tf_str: str, out_base: str, reason: str, pretty: bool = True):
    print(f"[STUB] Génération d’un CSV vide (en‑tête) à cause de: {reason}")
    rows: List[dict] = []  # en-tête seul
    try:
        path_main = export_signals(symbol, tf_str, rows, out_base=out_base, pretty=pretty, subdir="signals")
        list_dir(os.path.dirname(path_main))
    except Exception as e:
        print(f"[STUB] Export principal échoué: {e}")
    try:
        common = get_common_files_path()
        path_mirror = export_signals(symbol, tf_str, rows, out_base=common, pretty=pretty, subdir="signals")
        print("[STUB] Copie miroir Common OK")
        list_dir(os.path.dirname(path_mirror))
    except Exception as e:
        print(f"[STUB] Export miroir Common échoué: {e}")


def _simulate_trade_outcome(df: pd.DataFrame, entry_idx: int, direction: str, sl_points: int, tp_points: int, point_size: float) -> str:
    """
    Simule le résultat d'un trade à partir de l'index 'entry_idx' dans df.
    Retourne: 'WIN' si TP atteint en premier, 'LOSS' si SL atteint en premier, 'OPEN' sinon.
    """
    if entry_idx < 0 or entry_idx >= len(df):
        return 'OPEN'

    try:
        entry_price = float(df['close'].iloc[entry_idx])
    except Exception:
        return 'OPEN'

    if not np.isfinite(entry_price) or point_size <= 0:
        return 'OPEN'

    if direction == "BUY":
        stop_loss_price = entry_price - sl_points * point_size
        take_profit_price = entry_price + tp_points * point_size
    else:  # SELL
        stop_loss_price = entry_price + sl_points * point_size
        take_profit_price = entry_price - tp_points * point_size

    # Fenêtre de simulation bornée (pour éviter de parcourir tout le df)
    sim_window = df.iloc[entry_idx + 1: entry_idx + 1 + 200]
    for _, row in sim_window.iterrows():
        hi = float(row.get('high', np.nan))
        lo = float(row.get('low', np.nan))
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue

        if direction == "BUY":
            if lo <= stop_loss_price:
                return 'LOSS'
            if hi >= take_profit_price:
                return 'WIN'
        else:
            if hi >= stop_loss_price:
                return 'LOSS'
            if lo <= take_profit_price:
                return 'WIN'

    return 'OPEN'


def generate_signals(symbol: str, tf_str: str, start_dt: datetime, end_dt: datetime,
                     sl_points: int, tp_points: int, out_common: Optional[str] = None,
                     to_common: bool = False, pretty: bool = True, prefer_agent: Optional[str] = None):
    # Choix dossier cible
    out_base = get_common_files_path() if to_common else adapt_out_path_for_tester(out_common, prefer_agent)
    print("[OUT] Dossier cible:", out_base)

    # Connexion MT5
    if not connect_to_mt5():
        print("ERREUR: connexion MT5 impossible.", flush=True)
        # On écrit quand même un stub
        safe_export_stub(symbol, tf_str.upper(), out_base, "CONNECT_MT5_FAIL", pretty=pretty)
        sys.exit(1)

    tf_const = config.TIMEFRAME_MAP.get(tf_str.upper(), mt5.TIMEFRAME_H1)

    # Buffer pour séquences
    buf_minutes = tf_to_minutes(tf_str) * (config.SEQUENCE_LENGTH + 2)
    start_fetch = start_dt - timedelta(minutes=buf_minutes)

    # Historique
    df_hist = get_historical_data_range(symbol, tf_const, start_fetch, end_dt)
    if df_hist.empty:
        print("ERREUR: Aucune donnée MT5 sur la plage demandée.", flush=True)
        safe_export_stub(symbol, tf_str.upper(), out_base, "NO_HISTORY", pretty=pretty)
        mt5.shutdown()
        return

    # Features
    cfg = config.load_config_data()
    active_groups = config.get_active_groups_for_symbol(symbol)
    active_features = config.get_active_features_for_symbol(symbol)
    df_features = create_features(df_hist.copy(), symbol, active_groups)

    try:
        df_features = config.compute_trend_filter_columns(df_features.copy(), symbol)
    except Exception:
        pass

    # Scaler
    scaler_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_scaler.joblib")
    if not os.path.exists(scaler_path):
        print(f"ERREUR: Scaler introuvable pour {symbol} ({scaler_path}).", flush=True)
        safe_export_stub(symbol, tf_str.upper(), out_base, "NO_SCALER", pretty=pretty)
        mt5.shutdown()
        return

    try:
        scaler, scaler_features = _load_scaler_info(scaler_path)
    except Exception as e:
        print(f"ERREUR: Impossible de charger scaler: {e}", flush=True)
    # Matrice scaled (toutes lignes)
    try:
        X_scaled_all, used_feature_names = _build_scaled_matrix(df_features.copy(), scaler, scaler_features, active_features)
    except Exception as e:
        print("ERREUR: Echec scaling:", e)
        safe_export_stub(symbol, tf_str.upper(), out_base, "SCALING_FAIL", pretty=pretty)
        mt5.shutdown()
        return

    if X_scaled_all.shape[0] <= config.SEQUENCE_LENGTH:
        print("ERREUR: Données insuffisantes pour séquences.", flush=True)
        safe_export_stub(symbol, tf_str.upper(), out_base, "NOT_ENOUGH_SEQUENCES", pretty=pretty)
        mt5.shutdown()
        return

    # Séquences
    seq_mat = X_scaled_all
    all_sequences = np.array([seq_mat[i:i + config.SEQUENCE_LENGTH] for i in range(len(seq_mat) - config.SEQUENCE_LENGTH)])
    if all_sequences.size == 0:
        print("ERREUR: Aucune séquence créée.", flush=True)
        safe_export_stub(symbol, tf_str.upper(), out_base, "NO_SEQUENCE_CREATED", pretty=pretty)
        mt5.shutdown()
        return

    # Ensemble
    brain = EnsembleAIBrain(symbol=symbol, num_features=len(used_feature_names))

    # Récupération de la taille du point (pour la micro-simulation)
    try:
        si = mt5.symbol_info(symbol)
        point_size = si.point if si else (0.0001 if "JPY" not in symbol.upper() else 0.001)
    except Exception:
        point_size = 0.0001 if "JPY" not in symbol.upper() else 0.001

    rows: List[dict] = []

    # GARDE-FOU: mémorise la date de perte pour bloquer le reste de la journée
    last_loss_date = None

    try:
        pred_bs = int(getattr(config, "PRED_BATCH_SIZE", 512))
        individual_preds = [m.model.predict(all_sequences, batch_size=pred_bs, verbose=0) for m in brain.models]

        for i in range(len(all_sequences)):
            idx = i + config.SEQUENCE_LENGTH
            if idx >= len(df_features):
                continue

            entry_time = df_features.iloc[idx]["time"].to_pydatetime()
            if entry_time < start_dt or entry_time > end_dt:
                continue

            # Bloque tous les signaux après une perte pour la journée
            if last_loss_date == entry_time.date():
                continue

            entry_row = df_features.iloc[idx]
            per_model_probs = []
            for preds in individual_preds:
                try:
                    ps = float(preds[i][0]) if preds.shape[1] > 0 else 0.0
                    pb = float(preds[i][1]) if preds.shape[1] > 1 else 0.0
                except Exception:
                    ps, pb = 0.0, 0.0
                per_model_probs.append((ps, pb))

            direction_trend = config.direction_from_trend_row(entry_row)
            dir_conf_pct = config.compute_directional_confidence_pct(per_model_probs, direction_trend) if direction_trend is not None else 0.0

            final_signal = "HOLD"
            if direction_trend is not None and dir_conf_pct >= config.DECISION_CONF_THRESHOLD_PCT:
                if apply_technical_filters(direction_trend, entry_row):
                    final_signal = direction_trend

            if final_signal != "HOLD":
                # Micro-simulation: si perte -> verrouille la journée
                outcome = _simulate_trade_outcome(df_hist, idx, final_signal, sl_points, tp_points, point_size)
                if outcome == 'LOSS':
                    print(f"[{entry_time.strftime('%Y-%m-%d %H:%M')}] {symbol} -> Micro-simulation: LOSS, verrouillage de la journée.")
                    last_loss_date = entry_time.date()
                    continue

                rows.append({
                    "time": entry_time,
                    "dir": final_signal,
                    "sl": int(sl_points),
                    "tp": int(tp_points),
                    "comment": "Micheline IA | conf={:.1f}%".format(dir_conf_pct),
                })
    except Exception as e:
        print("[PRED] Erreur prédiction:", e)

    print(f"[INFO] rows_count={len(rows)}")

    # Export principal
    try:
        p_main = export_signals(symbol, tf_str.upper(), rows, out_base=out_base, pretty=pretty, subdir="signals")
        list_dir(os.path.dirname(p_main))
    except Exception as e:
        print("[EXPORT] principal échoué:", e)

    # Export miroir systématique vers Common
    try:
        p_mirror = export_signals(symbol, tf_str.upper(), rows, out_base=get_common_files_path(), pretty=pretty, subdir="signals")
        print("[INFO] Copie miroir ->", p_mirror)
        list_dir(os.path.dirname(p_mirror))
    except Exception as e:
        print("[EXPORT] miroir Common échoué:", e)

    mt5.shutdown()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", required=True)
    p.add_argument("--tf", required=True, help="M1/M5/M15/M30/H1/H4/D1/W1/MN1")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    cfg = config.load_config_data()
    p.add_argument("--sl", type=int, default=int(cfg.get("backtest_default_sl_points", 200)))
    p.add_argument("--tp", type=int, default=int(cfg.get("backtest_default_tp_points", 300)))
    p.add_argument("--to-common", action="store_true")
    p.add_argument("--out-common", type=str, default=None)
    p.add_argument("--prefer-agent", type=str, default=config.PREFERRED_TESTER_AGENT)
    p.add_argument("--pretty", dest="pretty", action="store_true")
    p.add_argument("--raw", dest="pretty", action="store_false")
    p.set_defaults(pretty=True)

    args = p.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf.upper()
    start_dt = parse_dt(args.start)
    end_dt = parse_dt(args.end)
    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    generate_signals(
        symbol=symbol,
        tf_str=tf,
        start_dt=start_dt,
        end_dt=end_dt,
        sl_points=args.sl,
        tp_points=args.tp,
        out_common=args.out_common,
        to_common=args.to_common,
        pretty=args.pretty,
        prefer_agent=args.prefer_agent
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        try:
            base = get_common_files_path()
            os.makedirs(os.path.join(base, "signals"), exist_ok=True)
            with open(os.path.join(base, "signals", "gen_log.txt"), "a", encoding="utf-8") as f:
                import traceback
                f.write("[ERROR] {}: {}\n{}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), e, traceback.format_exc()))
        except Exception:
            pass
        raise