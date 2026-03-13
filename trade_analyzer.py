# trade_analyzer.py - Backtest détaillé avec colonnes TrendFilter + Qualité/R multiple + ATR% + coûts + round numbers
# + Diagnostics de risque (flags par trade) et RISK_AUDIT en fin de CSV
# + Gating "tout vert": un trade n'est pris que si aucun flag de risque n'est déclenché

import numpy as np
import pandas as pd
import sys
import os
import MetaTrader5 as mt5
import config
from model_manager import AIBrain, EnsembleAIBrain
from trainer import create_features
from ia_utils import connect_to_mt5, get_historical_data, classify_detailed_market_state
from config import apply_technical_filters

def _ensure_trend_filter_columns(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Assure la présence des colonnes trend_filter_ma + trend_filter_above via
    le calcul natif MT5 centralisé dans config.compute_trend_filter_columns.
    """
    if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('time', drop=False)
    if all(c in df.columns for c in ['trend_filter_ma','trend_filter_above']):
        return df
    try:
        return config.compute_trend_filter_columns(df.copy(), symbol)
    except Exception:
        return df

def analyze_trades_for_symbol(symbol):
    report_path = f"backtest_report_{symbol}.csv"
    if os.path.exists(report_path):
        try:
            os.remove(report_path)
        except OSError as e:
            print(f"\nERREUR: Fichier rapport bloqué: {e}\n")
            print(f"ANALYSIS_FILE_LOCKED:{symbol}", flush=True)
            return

    active_features = config.get_active_features_for_symbol(symbol)
    try:
        cfg = config.load_config_data()
        brain = EnsembleAIBrain(symbol=symbol, num_features=len(active_features))
    except Exception as e:
        print(f"ERREUR: Impossible de charger le modèle: {e}")
        return

    df_history = get_historical_data(symbol, config.TIMEFRAME_TO_TRAIN, config.BACKTEST_BARS)
    if df_history is None or df_history.empty:
        return

    active_groups = cfg.get("optimized_feature_configs", {}).get(symbol, {}).get("best_groups", cfg.get("active_feature_groups"))
    df_features = create_features(df_history.copy(), symbol, active_groups)

    # Assure la présence des colonnes trend filter pour debug, même si le filtre n'est pas appliqué
    df_features = _ensure_trend_filter_columns(df_features.copy(), symbol)

    if getattr(config, "TREND_FILTER_VERBOSE", False):
        try:
            vc = df_features['trend_filter_above'].value_counts(dropna=False)
            print("TrendFilter_Above counts:", vc.to_dict(), flush=True)
        except Exception:
            pass

    # Contexte d'état de marché
    df_features['contexte_detaille'] = classify_detailed_market_state(df_features)

    # Séquences
    sequences = df_features[active_features].values

    # Traiter par lots pour limiter la RAM
    num_possible_sequences = len(sequences) - config.SEQUENCE_LENGTH
    if num_possible_sequences <= 0:
        print("Pas assez de données pour créer des séquences.")
        return

    PROCESSING_BATCH_SIZE = 2048  # Ajustez selon la RAM
    batched_preds = [[] for _ in brain.models]

    print(f"Début des prédictions sur {num_possible_sequences} séquences (par lots de {PROCESSING_BATCH_SIZE})...")

    for start_idx in range(0, num_possible_sequences, PROCESSING_BATCH_SIZE):
        end_idx = min(start_idx + PROCESSING_BATCH_SIZE, num_possible_sequences)

        batch_sequences = np.array([
            sequences[i: i + config.SEQUENCE_LENGTH] for i in range(start_idx, end_idx)
        ])

        if batch_sequences.size == 0:
            continue

        for model_idx, model_brain in enumerate(brain.models):
            preds_for_batch = model_brain.model.predict(batch_sequences, batch_size=512, verbose=0)
            batched_preds[model_idx].append(preds_for_batch)

    individual_preds = [np.vstack(preds_list) for preds_list in batched_preds]

    print("Prédictions terminées.")

    trade_log = []
    cfg = config.load_config_data()
    sl_tp_config = cfg.get("optimal_sl_tp_multipliers", {}).get(symbol, {})
    sl_multiplier = sl_tp_config.get("sl", config.ATR_MULTIPLIER_SL)
    tp_multiplier = sl_tp_config.get("tp", config.ATR_MULTIPLIER_TP)
    total_pips, gains, pertes = 0.0, 0, 0

    digits = 3 if "JPY" in symbol.upper() else 5
    TP_WINDOW_BARS = int(getattr(config, "ANALYZER_TP_WINDOW_BARS", 24))  # Centralisé

    # Hypothèses coût (spread/commission) depuis la config (valeurs par défaut: 0)
    assumed_spread_points = cfg.get("backtest_spread_points", 0)
    assumed_commission_pips = cfg.get("backtest_commission_pips", 0.0)

    # Conversion points -> pips (approx) via symbol_info
    try:
        si = mt5.symbol_info(symbol)
        point_size = si.point if si else (0.0001 if "JPY" not in symbol.upper() else 0.001)
    except Exception:
        point_size = 0.0001 if "JPY" not in symbol.upper() else 0.001
    pip_size = 0.0001 if "JPY" not in symbol.upper() else 0.01
    est_spread_pips = (assumed_spread_points * point_size) / pip_size if assumed_spread_points else 0.0
    est_commission_pips = float(assumed_commission_pips or 0.0)

    # ---------- Helpers locaux déjà présents ----------
    def get_session_label(row):
        try:
            if int(row.get('london_session', 0)) == 1:
                return "London"
            if int(row.get('ny_session', 0)) == 1:
                return "NY"
            if int(row.get('tokyo_session', 0)) == 1:
                return "Tokyo"
        except Exception:
            pass
        return "Off"

    def round_number_dist_pips(price: float) -> float:
        """
        Distance en pips au plus proche round number (00 ou 50).
        - 00 = multiples de 100 pips
        - 50 = multiples de 50 pips
        """
        if price is None or np.isnan(price):
            return np.nan
        price_in_pips = price / pip_size
        near100 = round(price_in_pips / 100.0) * 100.0
        dist100 = abs(price_in_pips - near100)
        near50 = round(price_in_pips / 50.0) * 50.0
        dist50 = abs(price_in_pips - near50)
        return float(min(dist100, dist50))

    def compute_risk_info(entry_row, direction, proba_buy_ens, proba_sell_ens, marge_proba, session_label):
        flags = {
            "Risk_DistTrendLow": 0,
            "Risk_NearRound": 0,
            "Risk_Extended": 0,
            "Risk_LowMargin": 0,
            "Risk_OffSession": 0,  # non bloquant seul (utilisé pour info)
            "Risk_DivergeH4": 0
        }
        reasons = []

        try:
            atr = float(entry_row.get('atr', np.nan))
            price = float(entry_row.get('close', np.nan))
            trend_ma = float(entry_row.get('trend_filter_ma', np.nan))
            pvmaf = float(entry_row.get('Price_vs_MA_Fast', np.nan))
            rnd = float(entry_row.get('Round_Number_Dist_Pips', np.nan))
            trend_h4 = float(entry_row.get('trend_h4', entry_row.get('Trend_H4', np.nan)))
        except Exception:
            atr, price, trend_ma, pvmaf, rnd, trend_h4 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Dist à la MA de tendance (en ATR)
        dist_trend_ma_atr = abs((price - trend_ma) / atr) if (np.isfinite(price) and np.isfinite(trend_ma) and np.isfinite(atr) and atr > 0) else np.inf
        if np.isfinite(dist_trend_ma_atr) and dist_trend_ma_atr < float(getattr(config, "RISK_TREND_MA_MIN_DIST_ATR", 0.30)):
            flags["Risk_DistTrendLow"] = 1
            reasons.append("Proche_MA_Tendance")

        if np.isfinite(rnd) and rnd < float(getattr(config, "RISK_ROUND_DIST_PIPS_MAX", 10.0)):
            flags["Risk_NearRound"] = 1
            reasons.append("Proche_Round")

        if np.isfinite(pvmaf) and abs(pvmaf) > float(getattr(config, "RISK_EXTENSION_PVMAF_MAX", 1.8)):
            flags["Risk_Extended"] = 1
            reasons.append("Sur_Extension")

        if np.isfinite(marge_proba) and marge_proba < float(getattr(config, "RISK_LOW_MARGIN_PCT_MIN", 6.0)):
            flags["Risk_LowMargin"] = 1
            reasons.append("Marge_Faible")

        if session_label == "Off":
            # pour info dans le CSV; pas bloquant seul
            flags["Risk_OffSession"] = 1
            reasons.append("Session_Off")

        dir_sign = 1 if direction == "BUY" else -1
        trend_sign = 1 if (np.isfinite(trend_h4) and trend_h4 > 0) else (-1 if (np.isfinite(trend_h4) and trend_h4 < 0) else 0)
        if trend_sign != 0 and dir_sign * trend_sign < 0:
            # divergence H4 vs direction
            diverge_margin_min = float(getattr(config, "RISK_DIVERGE_LOW_MARGIN_PCT_MIN", 12.0))
            if np.isfinite(marge_proba) and marge_proba < diverge_margin_min:
                flags["Risk_DivergeH4"] = 1
                reasons.append("Divergence_H4")

        score = int(flags["Risk_DistTrendLow"] + flags["Risk_NearRound"] + flags["Risk_Extended"] + flags["Risk_LowMargin"] + flags["Risk_DivergeH4"])
        is_risky = 1 if score > 0 else 0
        reason_str = ",".join(reasons)
        return is_risky, score, flags, reason_str, dist_trend_ma_atr
    # ------------------------------------------------------

    if not individual_preds:
        print("ERREUR: Aucun modèle n'a produit de prédictions.")
        print(f"ANALYSIS_FINISHED:{symbol}", flush=True)
        return

    # ====== GARDE-FOU: mémorise la date d'une perte pour bloquer le reste de la journée ======
    last_loss_date = None
    # ========================================================================================

    for i in range(len(individual_preds[0])):
        idx = i + config.SEQUENCE_LENGTH
        if idx >= len(df_features):
            continue
        entry_row = df_features.iloc[idx]

        # Ne pas traiter de nouveaux signaux si une perte est déjà survenue aujourd'hui
        try:
            entry_date = entry_row['time'].date()
        except Exception:
            entry_date = pd.to_datetime(entry_row['time']).date()
        if last_loss_date == entry_date:
            continue

        # Probas par modèle -> (SELL, BUY) pour chaque modèle
        per_model_probs = []
        votes_list = []
        votes_details = {}
        for model_idx, pred in enumerate(individual_preds, start=1):
            p = np.array(pred[i]).reshape(-1)
            ps = float(p[0]) if p.size > 0 else 0.0
            pb = float(p[1]) if p.size > 1 else 0.0
            per_model_probs.append((ps, pb))
            vote_dir = "BUY" if pb >= ps else "SELL"
            conf_pct = max(ps, pb) * 100.0
            votes_list.append(vote_dir)
            votes_details[f'Vote{model_idx}'] = vote_dir
            votes_details[f'%{model_idx}'] = f"({conf_pct:.1f}%)"

        # Confiance directionnelle (moyenne des p_dir)
        direction_trend = config.direction_from_trend_row(entry_row)
        dir_conf_pct = config.compute_directional_confidence_pct(per_model_probs, direction_trend) if direction_trend is not None else 0.0

        # Statistiques ensemble (pour logs)
        try:
            proba_buy_ens = float(np.mean([pb for (ps, pb) in per_model_probs]))
            proba_sell_ens = float(np.mean([ps for (ps, pb) in per_model_probs]))
        except Exception:
            proba_buy_ens, proba_sell_ens = 0.0, 0.0
        marge_proba = abs(proba_buy_ens - proba_sell_ens) * 100.0  # en points %

        # Consensus (pour info)
        votes_identiques = "Oui" if (len(votes_list) > 0 and all(v == votes_list[0] for v in votes_list)) else "Non"
        if votes_list:
            c_buy = votes_list.count("BUY")
            c_sell = votes_list.count("SELL")
            if c_buy > c_sell:
                consensus_str = f"{c_buy}/{len(votes_list)} BUY"
            elif c_sell > c_buy:
                consensus_str = f"{c_sell}/{len(votes_list)} SELL"
            else:
                consensus_str = f"{c_buy}/{len(votes_list)} EGALITE"
        else:
            consensus_str = "HOLD"

        # Règle de décision + GATING "tout vert"
        final_signal = "HOLD"
        if direction_trend is not None and dir_conf_pct >= config.DECISION_CONF_THRESHOLD_PCT:
            if apply_technical_filters(direction_trend, entry_row):
                # Calcule round dist à l'instant de l'entrée (en pips)
                try:
                    entry_close_for_gate = float(entry_row.get('close'))
                except Exception:
                    entry_close_for_gate = np.nan
                round_dist_for_gate = round_number_dist_pips(entry_close_for_gate)
                session_label_gate = get_session_label(entry_row)

                tmp_row_gate = entry_row.copy()
                tmp_row_gate['Round_Number_Dist_Pips'] = round_dist_for_gate

                risky_gate, risk_score_gate, risk_flags_gate, risk_reasons_gate, _ = compute_risk_info(
                    tmp_row_gate, direction_trend, proba_buy_ens, proba_sell_ens, marge_proba, session_label_gate
                )

                # Autoriser le trade seulement si le score de risque structurel == 0
                if risk_score_gate == 0:
                    final_signal = direction_trend

        if final_signal != "HOLD":
            entry_price = float(entry_row['close'])
            entry_time = entry_row['time']
            atr = float(entry_row.get('atr', 0.0))
            if atr <= 0:
                continue

            # Métriques utiles (existantes)
            ma_fast = float(entry_row.get('ma_fast', np.nan))
            ma_slow = float(entry_row.get('ma_slow', np.nan))
            ma_spread = (ma_fast - ma_slow) / atr if atr > 0 and np.isfinite(ma_fast) and np.isfinite(ma_slow) else np.nan
            price_vs_ma_fast = (entry_price - ma_fast) / atr if atr > 0 and np.isfinite(ma_fast) else np.nan
            price_vs_ma_slow = (entry_price - ma_slow) / atr if atr > 0 and np.isfinite(ma_slow) else np.nan

            vol_mom = float(entry_row.get('volume_momentum', np.nan))
            adx_val = float(entry_row.get('adx', entry_row.get('ADX_14', np.nan)))
            di_plus = float(entry_row.get('DMP_14', np.nan))
            di_minus = float(entry_row.get('DMN_14', np.nan))
            trend_h4 = float(entry_row.get('trend_h4', np.nan))
            trend_d1 = float(entry_row.get('trend_d1', np.nan))
            session_label = get_session_label(entry_row)

            # Trend filter logs
            trend_ma = float(entry_row.get('trend_filter_ma', np.nan))
            trend_above = entry_row.get('trend_filter_above', np.nan)
            try:
                trend_above = int(trend_above) if not pd.isna(trend_above) else ""
            except Exception:
                pass
            dist_trend_ma_atr = (entry_price - trend_ma) / atr if atr > 0 and np.isfinite(trend_ma) else np.nan

            # Volatilité relative et coûts supposés
            atr_percent = atr / entry_price if entry_price > 0 else np.nan
            round_dist_pips = round_number_dist_pips(entry_price)

            # Fenêtre pour l'info TP atteignable
            end_window = min(idx + TP_WINDOW_BARS, len(df_features) - 1)

            pip_size_local = 0.0001 if "JPY" not in symbol.upper() else 0.01
            resultat = "N/A"
            report_entry = {}
            exit_price = None
            issue_order = None

            # ---------- RISK FLAGS (pour CSV) ----------
            tmp_row = entry_row.copy()
            tmp_row['Round_Number_Dist_Pips'] = round_dist_pips
            risky, risk_score, risk_flags, risk_reasons, dist_trend_abs_atr = compute_risk_info(
                tmp_row, final_signal, proba_buy_ens, proba_sell_ens, marge_proba, session_label
            )
            # ----------------------------------------------------------

            # BUY
            if final_signal == "BUY":
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
                tp_hit_24h = bool((df_features['high'].iloc[idx + 1:end_window + 1] >= take_profit).any()) if idx + 1 <= end_window else False

                for k in range(idx + 1, len(df_features)):
                    if df_features['low'].iloc[k] <= stop_loss:
                        pertes += 1
                        resultat = "Perte"
                        resultat_pips = (stop_loss - entry_price) / pip_size_local
                        exit_price = stop_loss
                        issue_order = "SL_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        mfe_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        mae_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        r_multiple = (exit_price - entry_price) / (atr * sl_multiplier)  # < 0 en perte

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Raison_Perte": "SL_Trop_Serré" if tp_hit_24h else "Prediction_Incorrecte",
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui" if tp_hit_24h else "Non",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mfe_up_atr, 2),
                            "Excursion_Down_ATR": round(mae_dn_atr, 2)
                        }

                        # GARDE-FOU: verrouille le reste de la journée après une perte
                        last_loss_date = entry_date
                        break

                    if df_features['high'].iloc[k] >= take_profit:
                        gains += 1
                        resultat = "Gain"
                        resultat_pips = (take_profit - entry_price) / pip_size_local
                        exit_price = take_profit
                        issue_order = "TP_First"

                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        drawdown_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 and atr > 0 else 0.0
                        mfe_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        mae_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        r_multiple = (exit_price - entry_price) / (atr * sl_multiplier)  # > 0 en gain

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Drawdown_Max_ATR": round(drawdown_atr, 2),
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mfe_up_atr, 2),
                            "Excursion_Down_ATR": round(mae_dn_atr, 2)
                        }
                        break

            # SELL
            elif final_signal == "SELL":
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
                tp_hit_24h = bool((df_features['low'].iloc[idx + 1:end_window + 1] <= take_profit).any()) if idx + 1 <= end_window else False

                for k in range(idx + 1, len(df_features)):
                    if df_features['high'].iloc[k] >= stop_loss:
                        pertes += 1
                        resultat = "Perte"
                        resultat_pips = (entry_price - stop_loss) / pip_size_local  # < 0 pour SELL
                        exit_price = stop_loss
                        issue_order = "SL_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        mfe_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        mae_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        r_multiple = (entry_price - exit_price) / (atr * sl_multiplier)  # < 0 en perte

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Raison_Perte": "SL_Trop_Serré" if tp_hit_24h else "Prediction_Incorrecte",
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui" if tp_hit_24h else "Non",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mae_up_atr, 2),   # défavorable pour SELL
                            "Excursion_Down_ATR": round(mfe_dn_atr, 2)  # favorable pour SELL
                        }

                        # GARDE-FOU: verrouille le reste de la journée après une perte
                        last_loss_date = entry_date
                        break

                    if df_features['low'].iloc[k] <= take_profit:
                        gains += 1
                        resultat = "Gain"
                        resultat_pips = (entry_price - take_profit) / pip_size_local
                        exit_price = take_profit
                        issue_order = "TP_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        drawup_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 and atr > 0 else 0.0
                        mfe_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        mae_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        r_multiple = (entry_price - exit_price) / (atr * sl_multiplier)  # > 0 en gain

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Drawdown_Max_ATR": round(drawup_atr, 2),
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mae_up_atr, 2),   # défavorable pour SELL
                            "Excursion_Down_ATR": round(mfe_dn_atr, 2)  # favorable pour SELL
                        }
                        break

            if resultat != "N/A":
                total_pips += resultat_pips

                base_info = {
                    "Date": entry_time.strftime('%Y-%m-%d'),
                    "Heure": entry_time.strftime('%H:%M'),
                    "Direction": final_signal,
                    "Entrée": round(entry_price, digits) if np.isfinite(entry_price) else "",
                    "Sortie": round(exit_price, digits) if (exit_price is not None and np.isfinite(exit_price)) else "",
                    "Resultat": resultat
                }
                base_info.update(report_entry)
                base_info.update(votes_details)

                diag_info = {
                    # Stats ensemble
                    "Proba_Buy": round(proba_buy_ens * 100.0, 2),
                    "Proba_Sell": round(proba_sell_ens * 100.0, 2),
                    "Marge_Proba": round(marge_proba, 2),
                    "Ensemble_Conf_Pct": round(dir_conf_pct, 2),

                    # Consensus affichage
                    "Consensus": consensus_str,
                    "Votes_Identiques": votes_identiques,

                    # Indicateurs
                    "ATR": round(atr, 6),
                    "SLxATR": float(sl_multiplier),
                    "TPxATR": float(tp_multiplier),
                    "MA_Fast": round(ma_fast, digits) if np.isfinite(ma_fast) else "",
                    "MA_Slow": round(ma_slow, digits) if np.isfinite(ma_slow) else "",
                    "MA_Spread": round(ma_spread, 3) if np.isfinite(ma_spread) else "",
                    "Price_vs_MA_Fast": round(price_vs_ma_fast, 3) if np.isfinite(price_vs_ma_fast) else "",
                    "Price_vs_MA_Slow": round(price_vs_ma_slow, 3) if np.isfinite(price_vs_ma_slow) else "",
                    "Trend_H4": round(trend_h4, 5) if np.isfinite(trend_h4) else "",
                    "Trend_D1": round(trend_d1, 5) if np.isfinite(trend_d1) else "",
                    # Trend filter debug
                    "TrendFilter_TF": getattr(config, "TREND_FILTER_TIMEFRAME_STR", "D1"),
                    "TrendFilter_Period": int(getattr(config, "TREND_FILTER_MA_PERIOD", 200)),
                    "TrendFilter_MA": round(trend_ma, digits) if np.isfinite(trend_ma) else "",
                    "TrendFilter_Above": trend_above,
                    "TrendFilter_Dist_ATR": round(dist_trend_ma_atr, 3) if np.isfinite(dist_trend_ma_atr) else "",
                    # Coûts et round numbers
                    "Est_Spread_Pips": round(est_spread_pips, 2),
                    "Est_Commission_Pips": round(est_commission_pips, 2),
                    "Round_Number_Dist_Pips": round(round_dist_pips, 1) if np.isfinite(round_dist_pips) else "",
                    "Session": session_label,
                    # Risk flags
                    "Risk_IsRisky": risky,
                    "Risk_Score": int(risk_score),
                    "Risk_Reasons": risk_reasons,
                    "Risk_DistTrendLow": risk_flags["Risk_DistTrendLow"],
                    "Risk_NearRound": risk_flags["Risk_NearRound"],
                    "Risk_Extended": risk_flags["Risk_Extended"],
                    "Risk_LowMargin": risk_flags["Risk_LowMargin"],
                    "Risk_OffSession": risk_flags["Risk_OffSession"],
                    "Risk_DivergeH4": risk_flags["Risk_DivergeH4"],
                    "Risk_DistTrendAbs_ATR": round(dist_trend_abs_atr, 3) if np.isfinite(dist_trend_abs_atr) else ""
                }

                base_info.update(diag_info)
                trade_log.append(base_info)

    if not trade_log:
        print("Aucun trade n'a été fermé pendant la simulation.")
        print(f"ANALYSIS_FINISHED:{symbol}", flush=True)
        return

    # ===================== Construction du DataFrame et ordre des colonnes =====================
    report_df = pd.DataFrame(trade_log)

    # 1) Colonnes de base
    base_cols = [
        "Date", "Heure", "Direction",
        "Entrée", "Sortie",
        "Resultat", "Resultat_Pips", "R_Multiple", "Raison_Perte",
        "Drawdown_Max_ATR",
        "TPatteint",
        "Contexte_Entree",
    ]

    # 2) Votes présents (pas d'ajout de colonnes vides)
    vote_nums = sorted(
        {int(c[4:]) for c in report_df.columns
         if c.startswith("Vote") and c[4:].isdigit()}
    )
    vote_cols = []
    for n in vote_nums:
        vote_cols.append(f"Vote{n}")
        if f"%{n}" in report_df.columns:
            vote_cols.append(f"%{n}")

    # 3) Colonnes explications (anciennes + nouvelles + risk)
    extra_columns = [
        "Proba_Buy", "Proba_Sell", "Marge_Proba", "Ensemble_Conf_Pct",
        "Consensus", "Votes_Identiques",
        "ATR", "SLxATR", "TPxATR",
        "MA_Fast", "MA_Slow", "MA_Spread", "Price_vs_MA_Fast", "Price_vs_MA_Slow",
        "Trend_H4", "Trend_D1",
        # Trend filter debug
        "TrendFilter_TF", "TrendFilter_Period", "TrendFilter_MA", "TrendFilter_Above", "TrendFilter_Dist_ATR",
        # Coûts et round numbers
        "Est_Spread_Pips", "Est_Commission_Pips", "Round_Number_Dist_Pips",
        "Excursion_Up_ATR", "Excursion_Down_ATR",
        "Issue_Order",
        "Session",
        # Risk flags
        "Risk_IsRisky", "Risk_Score", "Risk_Reasons",
        "Risk_DistTrendLow", "Risk_NearRound", "Risk_Extended",
        "Risk_LowMargin", "Risk_OffSession", "Risk_DivergeH4",
        "Risk_DistTrendAbs_ATR"
    ]

    # 4) Ordre final
    final_order = (
        [c for c in base_cols if c in report_df.columns] +
        vote_cols +
        [c for c in extra_columns if c in report_df.columns]
    )
    report_df = report_df.loc[:, final_order]

    # 5) Conversion des colonnes numériques
    for col in [
        'Resultat_Pips', 'R_Multiple', 'Drawdown_Max_ATR', 'Entrée', 'Sortie',
        'Proba_Buy', 'Proba_Sell', 'Marge_Proba', 'Ensemble_Conf_Pct',
        'MA_Fast', 'MA_Slow', 'MA_Spread', 'Price_vs_MA_Fast', 'Price_vs_MA_Slow',
        'Trend_H4', 'Trend_D1',
        'TrendFilter_Period', 'TrendFilter_MA', 'TrendFilter_Above', 'TrendFilter_Dist_ATR',
        'Est_Spread_Pips', 'Est_Commission_Pips', 'Round_Number_Dist_Pips',
        'Excursion_Up_ATR', 'Excursion_Down_ATR',
        'Risk_IsRisky','Risk_Score',
        'Risk_DistTrendLow','Risk_NearRound','Risk_Extended','Risk_LowMargin','Risk_OffSession','Risk_DivergeH4',
        'Risk_DistTrendAbs_ATR'
    ]:
        if col in report_df.columns:
            report_df[col] = pd.to_numeric(report_df[col], errors='coerce')

    # ===================== Totaux en tête de fichier =====================
    somme_pips = float(pd.to_numeric(report_df['Resultat_Pips'], errors='coerce').sum())
    nb_gain = int((report_df['Resultat'] == 'Gain').sum())
    nb_perte = int((report_df['Resultat'] == 'Perte').sum())

    if 'Raison_Perte' in report_df.columns:
        nb_pred_incorrecte = int((report_df['Raison_Perte'].fillna('') == 'Prediction_Incorrecte').sum())
        nb_sl_trop_serre = int((report_df['Raison_Perte'].fillna('') == 'SL_Trop_Serré').sum())
    else:
        nb_pred_incorrecte = 0
        nb_sl_trop_serre = 0

    def fmt_num(x):
        try:
            return f"{x:.2f}".replace('.', ',')  # virgule comme séparateur décimal
        except Exception:
            return str(x)

    # ===================== Export CSV =====================
    with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
        # 5 lignes de résumé en tête
        f.write(f"Somme_Resultat_Pips;{fmt_num(somme_pips)}\n")
        f.write(f"Nombre_Gain;{nb_gain}\n")
        f.write(f"Nombre_Perte;{nb_perte}\n")
        f.write(f"Prediction_Incorrecte;{nb_pred_incorrecte}\n")
        f.write(f"SL_Trop_Serré;{nb_sl_trop_serre}\n\n")

        # Puis le tableau détaillé
        report_df.to_csv(f, index=False, sep=';', decimal=',')

        # ------------- RISK_AUDIT (tables à la fin) -------------
        def write_table(title, headers, rows):
            f.write("\n")
            f.write(f"RISK_AUDIT:{title}\n")
            f.write(";".join(headers) + "\n")
            for r in rows:
                vals = []
                for v in r:
                    if isinstance(v, (int, np.integer)):
                        vals.append(str(int(v)))
                    elif isinstance(v, (float, np.floating)):
                        vals.append(fmt_num(v))
                    else:
                        vals.append(str(v))
                f.write(";".join(vals) + "\n")

        def risk_bins(df, col, bins, label=None):
            s = pd.to_numeric(df[col], errors='coerce')
            cat = pd.cut(s, bins=bins, include_lowest=True)
            total = df.groupby(cat, observed=False).size()
            err = df.groupby(cat, observed=False).apply(lambda g: (g['Raison_Perte'].fillna('') == 'Prediction_Incorrecte').sum())
            gain = df.groupby(cat, observed=False).apply(lambda g: (g['Resultat'] == 'Gain').sum())
            tab = pd.DataFrame({"N_Total": total, "N_Gain": gain, "N_PredIncorrecte": err}).fillna(0)
            tab["Err_Rate_%"] = (tab["N_PredIncorrecte"] / tab["N_Total"].replace(0, np.nan)) * 100.0
            tab = tab.reset_index()
            first_col = tab.columns[0]
            tab.rename(columns={first_col: label or col}, inplace=True)
            return tab

        try:
            if "TrendFilter_Dist_ATR" in report_df.columns:
                tdf = report_df.copy()
                tdf["Abs_TF_Dist_ATR"] = tdf["TrendFilter_Dist_ATR"].abs()
                t = risk_bins(
                    tdf, "Abs_TF_Dist_ATR",
                    [-np.inf, 0.15, 0.30, 0.45, 0.60, 0.90, np.inf],
                    label="Abs_TF_Dist_ATR"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Abs(TrendFilter_Dist_ATR)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            if "Round_Number_Dist_Pips" in report_df.columns:
                t = risk_bins(
                    report_df, "Round_Number_Dist_Pips",
                    [-np.inf, 5, 8, 10, 15, 20, 50, np.inf],
                    label="Round_Dist_Pips"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Round_Number_Dist_Pips", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            if "Price_vs_MA_Fast" in report_df.columns:
                tdf = report_df.copy()
                tdf["Abs_PvMAF"] = tdf["Price_vs_MA_Fast"].abs()
                t = risk_bins(
                    tdf, "Abs_PvMAF",
                    [-np.inf, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, np.inf],
                    label="Abs_Price_vs_MA_Fast"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Abs(Price_vs_MA_Fast)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            if "Marge_Proba" in report_df.columns:
                t = risk_bins(
                    report_df, "Marge_Proba",
                    [-np.inf, 3, 5, 7, 10, 15, 20, 30, np.inf],
                    label="Marge_Proba_pts"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Marge_Proba (points %)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            if "Session" in report_df.columns:
                grp = (report_df.assign(is_err = report_df['Raison_Perte'].fillna('') == 'Prediction_Incorrecte')
                       .groupby("Session", observed=False)
                       .agg(N_Total=('Session','size'),
                            N_Gain=('Resultat',lambda s: (s=='Gain').sum()),
                            N_PredIncorrecte=('is_err','sum')))
                grp["Err_Rate_%"] = (grp["N_PredIncorrecte"] / grp["N_Total"].replace(0, np.nan)) * 100.0
                grp = grp.reset_index()
                rows = [[row.Session, int(row.N_Total), int(row.N_Gain), int(row["N_PredIncorrecte"]), float(row["Err_Rate_%"])]
                        for _, row in grp.iterrows()]
                write_table("Session", ["Session","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            if "Trend_H4" in report_df.columns and "Direction" in report_df.columns:
                def sign(v):
                    try:
                        vv = float(v)
                        return 1 if vv>0 else (-1 if vv<0 else 0)
                    except Exception:
                        return 0
                tdf = report_df.copy()
                tdf["dir_sign"] = tdf["Direction"].map({'BUY':1,'SELL':-1}).fillna(0)
                tdf["trend_h4_sign"] = tdf["Trend_H4"].apply(sign)
                tdf["diverge_h4"] = (tdf["dir_sign"] * tdf["trend_h4_sign"] < 0).astype(int)
                grp = (tdf.assign(is_err = tdf['Raison_Perte'].fillna('') == 'Prediction_Incorrecte')
                       .groupby("diverge_h4", observed=False)
                       .agg(N_Total=('diverge_h4','size'),
                            N_Gain=('Resultat',lambda s: (s=='Gain').sum()),
                            N_PredIncorrecte=('is_err','sum')))
                grp["Err_Rate_%"] = (grp["N_PredIncorrecte"] / grp["N_Total"].replace(0, np.nan)) * 100.0
                grp = grp.reset_index()
                rows = [[int(row.diverge_h4), int(row.N_Total), int(row.N_Gain), int(row["N_PredIncorrecte"]), float(row["Err_Rate_%"])]
                        for _, row in grp.iterrows()]
                write_table("Divergence H4 vs Direction (0/1)", ["DivergeH4","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

    print(f"Rapport de backtest de {len(report_df)} trades sauvegardé dans : {report_path}")
    print(f"ANALYSIS_FINISHED:{symbol}", flush=True)
    
    def round_number_dist_pips(price: float) -> float:
        """
        Distance en pips au plus proche round number (00 ou 50).
        - 00 = multiples de 100 pips
        - 50 = multiples de 50 pips
        """
        if price is None or np.isnan(price):
            return np.nan
        price_in_pips = price / pip_size
        near100 = round(price_in_pips / 100.0) * 100.0
        dist100 = abs(price_in_pips - near100)
        near50 = round(price_in_pips / 50.0) * 50.0
        dist50 = abs(price_in_pips - near50)
        return float(min(dist100, dist50))

    # ---------- RISK: fonction de flags par trade ----------
    def compute_risk_info(entry_row, direction, proba_buy_ens, proba_sell_ens, marge_proba, session_label):
        flags = {
            "Risk_DistTrendLow": 0,
            "Risk_NearRound": 0,
            "Risk_Extended": 0,
            "Risk_LowMargin": 0,
            "Risk_OffSession": 0,  # non bloquant seul (utilisé pour info)
            "Risk_DivergeH4": 0
        }
        reasons = []

        try:
            atr = float(entry_row.get('atr', np.nan))
            price = float(entry_row.get('close', np.nan))
            trend_ma = float(entry_row.get('trend_filter_ma', np.nan))
            pvmaf = float(entry_row.get('Price_vs_MA_Fast', np.nan))
            rnd = float(entry_row.get('Round_Number_Dist_Pips', np.nan))
            trend_h4 = float(entry_row.get('trend_h4', entry_row.get('Trend_H4', np.nan)))
        except Exception:
            atr, price, trend_ma, pvmaf, rnd, trend_h4 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Dist à la MA de tendance (en ATR)
        dist_trend_ma_atr = abs((price - trend_ma) / atr) if (np.isfinite(price) and np.isfinite(trend_ma) and np.isfinite(atr) and atr > 0) else np.inf
        if np.isfinite(dist_trend_ma_atr) and dist_trend_ma_atr < float(getattr(config, "RISK_TREND_MA_MIN_DIST_ATR", 0.30)):
            flags["Risk_DistTrendLow"] = 1
            reasons.append("Proche_MA_Tendance")

        if np.isfinite(rnd) and rnd < float(getattr(config, "RISK_ROUND_DIST_PIPS_MAX", 10.0)):
            flags["Risk_NearRound"] = 1
            reasons.append("Proche_Round")

        if np.isfinite(pvmaf) and abs(pvmaf) > float(getattr(config, "RISK_EXTENSION_PVMAF_MAX", 1.8)):
            flags["Risk_Extended"] = 1
            reasons.append("Sur_Extension")

        if np.isfinite(marge_proba) and marge_proba < float(getattr(config, "RISK_LOW_MARGIN_PCT_MIN", 6.0)):
            flags["Risk_LowMargin"] = 1
            reasons.append("Marge_Faible")

        if session_label == "Off":
            # pour info dans le CSV; pas bloquant seul
            flags["Risk_OffSession"] = 1
            reasons.append("Session_Off")

        dir_sign = 1 if direction == "BUY" else -1
        trend_sign = 1 if (np.isfinite(trend_h4) and trend_h4 > 0) else (-1 if (np.isfinite(trend_h4) and trend_h4 < 0) else 0)
        if trend_sign != 0 and dir_sign * trend_sign < 0:
            # divergence H4 vs direction
            diverge_margin_min = float(getattr(config, "RISK_DIVERGE_LOW_MARGIN_PCT_MIN", 12.0))
            if np.isfinite(marge_proba) and marge_proba < diverge_margin_min:
                flags["Risk_DivergeH4"] = 1
                reasons.append("Divergence_H4")

        score = int(flags["Risk_DistTrendLow"] + flags["Risk_NearRound"] + flags["Risk_Extended"] + flags["Risk_LowMargin"] + flags["Risk_DivergeH4"])
        is_risky = 1 if score > 0 else 0
        reason_str = ",".join(reasons)
        return is_risky, score, flags, reason_str, dist_trend_ma_atr

    # ------------------------------------------------------

    if not individual_preds:
        print("ERREUR: Aucun modèle n'a produit de prédictions.")
        print(f"ANALYSIS_FINISHED:{symbol}", flush=True)
        return

    for i in range(len(individual_preds[0])):
        idx = i + config.SEQUENCE_LENGTH
        if idx >= len(df_features):
            continue
        entry_row = df_features.iloc[idx]

        # Probas par modèle -> (SELL, BUY) pour chaque modèle
        per_model_probs = []
        votes_list = []
        votes_details = {}
        for model_idx, pred in enumerate(individual_preds, start=1):
            p = np.array(pred[i]).reshape(-1)
            ps = float(p[0]) if p.size > 0 else 0.0
            pb = float(p[1]) if p.size > 1 else 0.0
            per_model_probs.append((ps, pb))
            vote_dir = "BUY" if pb >= ps else "SELL"
            conf_pct = max(ps, pb) * 100.0
            votes_list.append(vote_dir)
            votes_details[f'Vote{model_idx}'] = vote_dir
            votes_details[f'%{model_idx}'] = f"({conf_pct:.1f}%)"

        # Confiance directionnelle (moyenne des p_dir)
        direction_trend = config.direction_from_trend_row(entry_row)
        dir_conf_pct = config.compute_directional_confidence_pct(per_model_probs, direction_trend) if direction_trend is not None else 0.0

        # Statistiques ensemble (pour logs)
        try:
            proba_buy_ens = float(np.mean([pb for (ps, pb) in per_model_probs]))
            proba_sell_ens = float(np.mean([ps for (ps, pb) in per_model_probs]))
        except Exception:
            proba_buy_ens, proba_sell_ens = 0.0, 0.0
        marge_proba = abs(proba_buy_ens - proba_sell_ens) * 100.0  # en points %

        # Consensus (pour info)
        votes_identiques = "Oui" if (len(votes_list) > 0 and all(v == votes_list[0] for v in votes_list)) else "Non"
        if votes_list:
            c_buy = votes_list.count("BUY")
            c_sell = votes_list.count("SELL")
            if c_buy > c_sell:
                consensus_str = f"{c_buy}/{len(votes_list)} BUY"
            elif c_sell > c_buy:
                consensus_str = f"{c_sell}/{len(votes_list)} SELL"
            else:
                consensus_str = f"{c_buy}/{len(votes_list)} EGALITE"
        else:
            consensus_str = "HOLD"

        # Règle de décision + GATING "tout vert"
        final_signal = "HOLD"
        if direction_trend is not None and dir_conf_pct >= config.DECISION_CONF_THRESHOLD_PCT:
            if apply_technical_filters(direction_trend, entry_row):
                # Calcule round dist à l'instant de l'entrée (en pips)
                try:
                    entry_close_for_gate = float(entry_row.get('close'))
                except Exception:
                    entry_close_for_gate = np.nan
                round_dist_for_gate = round_number_dist_pips(entry_close_for_gate)
                session_label_gate = get_session_label(entry_row)

                tmp_row_gate = entry_row.copy()
                tmp_row_gate['Round_Number_Dist_Pips'] = round_dist_for_gate

                risky_gate, risk_score_gate, risk_flags_gate, risk_reasons_gate, _ = compute_risk_info(
                    tmp_row_gate, direction_trend, proba_buy_ens, proba_sell_ens, marge_proba, session_label_gate
                )

                # Autoriser le trade seulement si le score de risque structurel == 0
                if risk_score_gate == 0:
                    final_signal = direction_trend
                else:
                    pass

        if final_signal != "HOLD":
            entry_price = float(entry_row['close'])
            entry_time = entry_row['time']
            atr = float(entry_row.get('atr', 0.0))
            if atr <= 0:
                continue

            # Métriques utiles (existantes)
            ma_fast = float(entry_row.get('ma_fast', np.nan))
            ma_slow = float(entry_row.get('ma_slow', np.nan))
            ma_spread = (ma_fast - ma_slow) / atr if atr > 0 and np.isfinite(ma_fast) and np.isfinite(ma_slow) else np.nan
            price_vs_ma_fast = (entry_price - ma_fast) / atr if atr > 0 and np.isfinite(ma_fast) else np.nan
            price_vs_ma_slow = (entry_price - ma_slow) / atr if atr > 0 and np.isfinite(ma_slow) else np.nan

            vol_mom = float(entry_row.get('volume_momentum', np.nan))
            adx_val = float(entry_row.get('adx', entry_row.get('ADX_14', np.nan)))
            di_plus = float(entry_row.get('DMP_14', np.nan))
            di_minus = float(entry_row.get('DMN_14', np.nan))
            trend_h4 = float(entry_row.get('trend_h4', np.nan))
            trend_d1 = float(entry_row.get('trend_d1', np.nan))
            session_label = get_session_label(entry_row)

            # Trend filter logs
            trend_ma = float(entry_row.get('trend_filter_ma', np.nan))
            trend_above = entry_row.get('trend_filter_above', np.nan)
            try:
                trend_above = int(trend_above) if not pd.isna(trend_above) else ""
            except Exception:
                pass
            dist_trend_ma_atr = (entry_price - trend_ma) / atr if atr > 0 and np.isfinite(trend_ma) else np.nan

            # Volatilité relative et coûts supposés
            atr_percent = atr / entry_price if entry_price > 0 else np.nan
            round_dist_pips = round_number_dist_pips(entry_price)

            # Fenêtre pour l'info TP atteignable
            end_window = min(idx + TP_WINDOW_BARS, len(df_features) - 1)

            pip_size_local = 0.0001 if "JPY" not in symbol.upper() else 0.01
            resultat = "N/A"
            report_entry = {}
            exit_price = None
            issue_order = None

            # ---------- RISK FLAGS (pour CSV) ----------
            tmp_row = entry_row.copy()
            tmp_row['Round_Number_Dist_Pips'] = round_dist_pips
            risky, risk_score, risk_flags, risk_reasons, dist_trend_abs_atr = compute_risk_info(
                tmp_row, final_signal, proba_buy_ens, proba_sell_ens, marge_proba, session_label
            )
            # ----------------------------------------------------------

            # BUY
            if final_signal == "BUY":
                stop_loss = entry_price - (atr * sl_multiplier)
                take_profit = entry_price + (atr * tp_multiplier)
                tp_hit_24h = bool((df_features['high'].iloc[idx + 1:end_window + 1] >= take_profit).any()) if idx + 1 <= end_window else False

                for k in range(idx + 1, len(df_features)):
                    if df_features['low'].iloc[k] <= stop_loss:
                        pertes += 1
                        resultat = "Perte"
                        resultat_pips = (stop_loss - entry_price) / pip_size_local
                        exit_price = stop_loss
                        issue_order = "SL_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        mfe_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        mae_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        r_multiple = (exit_price - entry_price) / (atr * sl_multiplier)  # < 0 en perte

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Raison_Perte": "SL_Trop_Serré" if tp_hit_24h else "Prediction_Incorrecte",
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui" if tp_hit_24h else "Non",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mfe_up_atr, 2),
                            "Excursion_Down_ATR": round(mae_dn_atr, 2)
                        }
                        break

                    if df_features['high'].iloc[k] >= take_profit:
                        gains += 1
                        resultat = "Gain"
                        resultat_pips = (take_profit - entry_price) / pip_size_local
                        exit_price = take_profit
                        issue_order = "TP_First"

                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        drawdown_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 and atr > 0 else 0.0
                        mfe_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        mae_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        r_multiple = (exit_price - entry_price) / (atr * sl_multiplier)  # > 0 en gain

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Drawdown_Max_ATR": round(drawdown_atr, 2),
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mfe_up_atr, 2),
                            "Excursion_Down_ATR": round(mae_dn_atr, 2)
                        }
                        break

            # SELL
            elif final_signal == "SELL":
                stop_loss = entry_price + (atr * sl_multiplier)
                take_profit = entry_price - (atr * tp_multiplier)
                tp_hit_24h = bool((df_features['low'].iloc[idx + 1:end_window + 1] <= take_profit).any()) if idx + 1 <= end_window else False

                for k in range(idx + 1, len(df_features)):
                    if df_features['high'].iloc[k] >= stop_loss:
                        pertes += 1
                        resultat = "Perte"
                        resultat_pips = (entry_price - stop_loss) / pip_size_local  # < 0 pour SELL
                        exit_price = stop_loss
                        issue_order = "SL_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        mfe_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        mae_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        r_multiple = (entry_price - exit_price) / (atr * sl_multiplier)  # < 0 en perte

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Raison_Perte": "SL_Trop_Serré" if tp_hit_24h else "Prediction_Incorrecte",
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui" if tp_hit_24h else "Non",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mae_up_atr, 2),   # défavorable pour SELL
                            "Excursion_Down_ATR": round(mfe_dn_atr, 2)  # favorable pour SELL
                        }
                        break

                    if df_features['low'].iloc[k] <= take_profit:
                        gains += 1
                        resultat = "Gain"
                        resultat_pips = (entry_price - take_profit) / pip_size_local
                        exit_price = take_profit
                        issue_order = "TP_First"

                        high_slice = df_features['high'].iloc[idx + 1:k + 1]
                        low_slice = df_features['low'].iloc[idx + 1:k + 1]
                        drawup_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 and atr > 0 else 0.0
                        mfe_dn_atr = ((entry_price - low_slice.min()) / atr) if len(low_slice) > 0 else 0.0
                        mae_up_atr = ((high_slice.max() - entry_price) / atr) if len(high_slice) > 0 else 0.0
                        r_multiple = (entry_price - exit_price) / (atr * sl_multiplier)  # > 0 en gain

                        report_entry = {
                            "Resultat": resultat,
                            "Resultat_Pips": round(resultat_pips, 2),
                            "R_Multiple": round(r_multiple, 3),
                            "Drawdown_Max_ATR": round(drawup_atr, 2),
                            "Entrée": round(entry_price, digits),
                            "Sortie": round(exit_price, digits),
                            "TPatteint": "Oui",
                            "Issue_Order": issue_order,
                            "Excursion_Up_ATR": round(mae_up_atr, 2),   # défavorable pour SELL
                            "Excursion_Down_ATR": round(mfe_dn_atr, 2)  # favorable pour SELL
                        }
                        break

            if resultat != "N/A":
                total_pips += resultat_pips

                base_info = {
                    "Date": entry_time.strftime('%Y-%m-%d'),
                    "Heure": entry_time.strftime('%H:%M'),
                    "Direction": final_signal,
                    "Contexte_Entree": entry_row['contexte_detaille']
                }
                base_info.update(report_entry)
                base_info.update(votes_details)

                diag_info = {
                    # Stats ensemble
                    "Proba_Buy": round(proba_buy_ens * 100.0, 2),
                    "Proba_Sell": round(proba_sell_ens * 100.0, 2),
                    "Marge_Proba": round(marge_proba, 2),
                    "Ensemble_Conf_Pct": round(dir_conf_pct, 2),

                    # Consensus affichage
                    "Consensus": consensus_str,
                    "Votes_Identiques": votes_identiques,

                    # Indicateurs
                    "ATR": round(atr, 6),
                    "SLxATR": float(sl_multiplier),
                    "TPxATR": float(tp_multiplier),
                    "MA_Fast": round(ma_fast, digits) if np.isfinite(ma_fast) else "",
                    "MA_Slow": round(ma_slow, digits) if np.isfinite(ma_slow) else "",
                    "MA_Spread": round(ma_spread, 3) if np.isfinite(ma_spread) else "",
                    "Price_vs_MA_Fast": round(price_vs_ma_fast, 3) if np.isfinite(price_vs_ma_fast) else "",
                    "Price_vs_MA_Slow": round(price_vs_ma_slow, 3) if np.isfinite(price_vs_ma_slow) else "",
                    "Trend_H4": round(trend_h4, 5) if np.isfinite(trend_h4) else "",
                    "Trend_D1": round(trend_d1, 5) if np.isfinite(trend_d1) else "",
                    # Trend filter debug
                    "TrendFilter_TF": getattr(config, "TREND_FILTER_TIMEFRAME_STR", "D1"),
                    "TrendFilter_Period": int(getattr(config, "TREND_FILTER_MA_PERIOD", 200)),
                    "TrendFilter_MA": round(trend_ma, digits) if np.isfinite(trend_ma) else "",
                    "TrendFilter_Above": trend_above,
                    "TrendFilter_Dist_ATR": round(dist_trend_ma_atr, 3) if np.isfinite(dist_trend_ma_atr) else "",
                    # Coûts et round numbers
                    "Est_Spread_Pips": round(est_spread_pips, 2),
                    "Est_Commission_Pips": round(est_commission_pips, 2),
                    "Round_Number_Dist_Pips": round(round_dist_pips, 1) if np.isfinite(round_dist_pips) else "",
                    "Session": session_label,
                    # Risk flags
                    "Risk_IsRisky": risky,
                    "Risk_Score": int(risk_score),
                    "Risk_Reasons": risk_reasons,
                    "Risk_DistTrendLow": risk_flags["Risk_DistTrendLow"],
                    "Risk_NearRound": risk_flags["Risk_NearRound"],
                    "Risk_Extended": risk_flags["Risk_Extended"],
                    "Risk_LowMargin": risk_flags["Risk_LowMargin"],
                    "Risk_OffSession": risk_flags["Risk_OffSession"],
                    "Risk_DivergeH4": risk_flags["Risk_DivergeH4"],
                    "Risk_DistTrendAbs_ATR": round(dist_trend_abs_atr, 3) if np.isfinite(dist_trend_abs_atr) else ""
                }

                base_info.update(diag_info)
                trade_log.append(base_info)

    if not trade_log:
        print("Aucun trade n'a été fermé pendant la simulation.")
        print(f"ANALYSIS_FINISHED:{symbol}", flush=True)
        return

    # ===================== Construction du DataFrame et ordre des colonnes =====================
    report_df = pd.DataFrame(trade_log)

    # 1) Colonnes de base
    base_cols = [
        "Date", "Heure", "Direction",
        "Entrée", "Sortie",
        "Resultat", "Resultat_Pips", "R_Multiple", "Raison_Perte",
        "Drawdown_Max_ATR",
        "TPatteint",
        "Contexte_Entree",
    ]

    # 2) Votes présents (pas d'ajout de colonnes vides)
    vote_nums = sorted(
        {int(c[4:]) for c in report_df.columns
         if c.startswith("Vote") and c[4:].isdigit()}
    )
    vote_cols = []
    for n in vote_nums:
        vote_cols.append(f"Vote{n}")
        if f"%{n}" in report_df.columns:
            vote_cols.append(f"%{n}")

    # 3) Colonnes explications (anciennes + nouvelles + risk)
    extra_columns = [
        "Proba_Buy", "Proba_Sell", "Marge_Proba", "Ensemble_Conf_Pct",
        "Consensus", "Votes_Identiques",
        "ATR", "SLxATR", "TPxATR",
        "MA_Fast", "MA_Slow", "MA_Spread", "Price_vs_MA_Fast", "Price_vs_MA_Slow",
        "Trend_H4", "Trend_D1",
        # Trend filter debug
        "TrendFilter_TF", "TrendFilter_Period", "TrendFilter_MA", "TrendFilter_Above", "TrendFilter_Dist_ATR",
        # Coûts et round numbers
        "Est_Spread_Pips", "Est_Commission_Pips", "Round_Number_Dist_Pips",
        "Excursion_Up_ATR", "Excursion_Down_ATR",
        "Issue_Order",
        "Session",
        # Risk flags
        "Risk_IsRisky", "Risk_Score", "Risk_Reasons",
        "Risk_DistTrendLow", "Risk_NearRound", "Risk_Extended",
        "Risk_LowMargin", "Risk_OffSession", "Risk_DivergeH4",
        "Risk_DistTrendAbs_ATR"
    ]

    # 4) Ordre final
    final_order = (
        [c for c in base_cols if c in report_df.columns] +
        vote_cols +
        [c for c in extra_columns if c in report_df.columns]
    )
    report_df = report_df.loc[:, final_order]

    # 5) Conversion des colonnes numériques
    for col in [
        'Resultat_Pips', 'R_Multiple', 'Drawdown_Max_ATR', 'Entrée', 'Sortie',
        'Proba_Buy', 'Proba_Sell', 'Marge_Proba', 'Ensemble_Conf_Pct',
        'MA_Fast', 'MA_Slow', 'MA_Spread', 'Price_vs_MA_Fast', 'Price_vs_MA_Slow',
        'Trend_H4', 'Trend_D1',
        'TrendFilter_Period', 'TrendFilter_MA', 'TrendFilter_Above', 'TrendFilter_Dist_ATR',
        'Est_Spread_Pips', 'Est_Commission_Pips', 'Round_Number_Dist_Pips',
        'Excursion_Up_ATR', 'Excursion_Down_ATR',
        'Risk_IsRisky','Risk_Score',
        'Risk_DistTrendLow','Risk_NearRound','Risk_Extended','Risk_LowMargin','Risk_OffSession','Risk_DivergeH4',
        'Risk_DistTrendAbs_ATR'
    ]:
        if col in report_df.columns:
            report_df[col] = pd.to_numeric(report_df[col], errors='coerce')

    # ===================== Totaux en tête de fichier =====================
    somme_pips = float(pd.to_numeric(report_df['Resultat_Pips'], errors='coerce').sum())
    nb_gain = int((report_df['Resultat'] == 'Gain').sum())
    nb_perte = int((report_df['Resultat'] == 'Perte').sum())

    if 'Raison_Perte' in report_df.columns:
        nb_pred_incorrecte = int((report_df['Raison_Perte'].fillna('') == 'Prediction_Incorrecte').sum())
        nb_sl_trop_serre = int((report_df['Raison_Perte'].fillna('') == 'SL_Trop_Serré').sum())
    else:
        nb_pred_incorrecte = 0
        nb_sl_trop_serre = 0

    def fmt_num(x):
        try:
            return f"{x:.2f}".replace('.', ',')  # virgule comme séparateur décimal
        except Exception:
            return str(x)

    # ===================== Export CSV =====================
    with open(report_path, 'w', encoding='utf-8-sig', newline='') as f:
        # 5 lignes de résumé en tête
        f.write(f"Somme_Resultat_Pips;{fmt_num(somme_pips)}\n")
        f.write(f"Nombre_Gain;{nb_gain}\n")
        f.write(f"Nombre_Perte;{nb_perte}\n")
        f.write(f"Prediction_Incorrecte;{nb_pred_incorrecte}\n")
        f.write(f"SL_Trop_Serré;{nb_sl_trop_serre}\n\n")

        # Puis le tableau détaillé
        report_df.to_csv(f, index=False, sep=';', decimal=',')

        # ------------- RISK_AUDIT (tables à la fin) -------------
        def write_table(title, headers, rows):
            f.write("\n")
            f.write(f"RISK_AUDIT:{title}\n")
            f.write(";".join(headers) + "\n")
            for r in rows:
                vals = []
                for v in r:
                    if isinstance(v, (int, np.integer)):
                        vals.append(str(int(v)))
                    elif isinstance(v, (float, np.floating)):
                        vals.append(fmt_num(v))
                    else:
                        vals.append(str(v))
                f.write(";".join(vals) + "\n")

        # Helper pour tableaux par bins (observed=False pour éviter les FutureWarning)
        def risk_bins(df, col, bins, label=None):
            s = pd.to_numeric(df[col], errors='coerce')
            cat = pd.cut(s, bins=bins, include_lowest=True)
            total = df.groupby(cat, observed=False).size()
            err = df.groupby(cat, observed=False).apply(lambda g: (g['Raison_Perte'].fillna('') == 'Prediction_Incorrecte').sum())
            gain = df.groupby(cat, observed=False).apply(lambda g: (g['Resultat'] == 'Gain').sum())
            tab = pd.DataFrame({"N_Total": total, "N_Gain": gain, "N_PredIncorrecte": err}).fillna(0)
            tab["Err_Rate_%"] = (tab["N_PredIncorrecte"] / tab["N_Total"].replace(0, np.nan)) * 100.0
            tab = tab.reset_index()
            first_col = tab.columns[0]
            tab.rename(columns={first_col: label or col}, inplace=True)
            return tab

        try:
            # 1) Abs(TrendFilter_Dist_ATR)
            if "TrendFilter_Dist_ATR" in report_df.columns:
                tdf = report_df.copy()
                tdf["Abs_TF_Dist_ATR"] = tdf["TrendFilter_Dist_ATR"].abs()
                t = risk_bins(
                    tdf, "Abs_TF_Dist_ATR",
                    [-np.inf, 0.15, 0.30, 0.45, 0.60, 0.90, np.inf],
                    label="Abs_TF_Dist_ATR"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Abs(TrendFilter_Dist_ATR)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            # 2) Round_Number_Dist_Pips
            if "Round_Number_Dist_Pips" in report_df.columns:
                t = risk_bins(
                    report_df, "Round_Number_Dist_Pips",
                    [-np.inf, 5, 8, 10, 15, 20, 50, np.inf],
                    label="Round_Dist_Pips"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Round_Number_Dist_Pips", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            # 3) Abs(Price_vs_MA_Fast)
            if "Price_vs_MA_Fast" in report_df.columns:
                tdf = report_df.copy()
                tdf["Abs_PvMAF"] = tdf["Price_vs_MA_Fast"].abs()
                t = risk_bins(
                    tdf, "Abs_PvMAF",
                    [-np.inf, 0.5, 1.0, 1.5, 1.8, 2.0, 2.5, np.inf],
                    label="Abs_Price_vs_MA_Fast"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Abs(Price_vs_MA_Fast)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            # 4) Marge_Proba (% points)
            if "Marge_Proba" in report_df.columns:
                t = risk_bins(
                    report_df, "Marge_Proba",
                    [-np.inf, 3, 5, 7, 10, 15, 20, 30, np.inf],
                    label="Marge_Proba_pts"
                )
                rows = [[str(ix), int(r.N_Total), int(r.N_Gain), int(r["N_PredIncorrecte"]), float(r["Err_Rate_%"])]
                        for ix, r in t.iterrows()]
                write_table("Marge_Proba (points %)", ["Bin","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            # 5) Session
            if "Session" in report_df.columns:
                grp = (report_df.assign(is_err = report_df['Raison_Perte'].fillna('') == 'Prediction_Incorrecte')
                       .groupby("Session", observed=False)
                       .agg(N_Total=('Session','size'),
                            N_Gain=('Resultat',lambda s: (s=='Gain').sum()),
                            N_PredIncorrecte=('is_err','sum')))
                grp["Err_Rate_%"] = (grp["N_PredIncorrecte"] / grp["N_Total"].replace(0, np.nan)) * 100.0
                grp = grp.reset_index()
                rows = [[row.Session, int(row.N_Total), int(row.N_Gain), int(row["N_PredIncorrecte"]), float(row["Err_Rate_%"])]
                        for _, row in grp.iterrows()]
                write_table("Session", ["Session","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass

        try:
            # 6) Divergence H4 vs Direction
            if "Trend_H4" in report_df.columns and "Direction" in report_df.columns:
                def sign(v):
                    try:
                        vv = float(v); 
                        return 1 if vv>0 else (-1 if vv<0 else 0)
                    except Exception:
                        return 0
                tdf = report_df.copy()
                tdf["dir_sign"] = tdf["Direction"].map({'BUY':1,'SELL':-1}).fillna(0)
                tdf["trend_h4_sign"] = tdf["Trend_H4"].apply(sign)
                tdf["diverge_h4"] = (tdf["dir_sign"] * tdf["trend_h4_sign"] < 0).astype(int)
                grp = (tdf.assign(is_err = tdf['Raison_Perte'].fillna('') == 'Prediction_Incorrecte')
                       .groupby("diverge_h4", observed=False)
                       .agg(N_Total=('diverge_h4','size'),
                            N_Gain=('Resultat',lambda s: (s=='Gain').sum()),
                            N_PredIncorrecte=('is_err','sum')))
                grp["Err_Rate_%"] = (grp["N_PredIncorrecte"] / grp["N_Total"].replace(0, np.nan)) * 100.0
                grp = grp.reset_index()
                rows = [[int(row.diverge_h4), int(row.N_Total), int(row.N_Gain), int(row["N_PredIncorrecte"]), float(row["Err_Rate_%"])]
                        for _, row in grp.iterrows()]
                write_table("Divergence H4 vs Direction (0/1)", ["DivergeH4","N_Total","N_Gain","N_PredIncorrecte","Err_Rate_%"], rows)
        except Exception:
            pass
        # ---------------------------------------------------------

    print(f"Rapport de backtest de {len(report_df)} trades sauvegardé dans : {report_path}")
    print(f"ANALYSIS_FINISHED:{symbol}", flush=True)

def main():
    if len(sys.argv) < 2:
        print("ERREUR: Spécifiez un symbole.")
        return
    symbol_to_analyze = sys.argv[1].upper()
    if not connect_to_mt5():
        sys.exit(1)
    analyze_trades_for_symbol(symbol_to_analyze)
    mt5.shutdown()

if __name__ == "__main__":
    main()