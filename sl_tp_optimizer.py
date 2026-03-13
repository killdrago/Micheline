# sl_tp_optimizer.py - Optimisation SL/TP centralisée (logs TF réduits + seeds)
# - Grilles, batchs et seuils via config
# - Conserve la logique existante

import os
# Réduction de logs TF
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import numpy as np
import pandas as pd
import sys
import random
import MetaTrader5 as mt5
import config
import pandas_ta as ta
from numba import njit
from model_manager import AIBrain, EnsembleAIBrain
from ia_utils import get_historical_data, connect_to_mt5
from trainer import create_features
from sklearn.preprocessing import RobustScaler
from joblib import load as joblib_load

# Option: silence logger TF Python si TensorFlow est importé via model_manager
try:
    import tensorflow as tf
    tf.random.set_seed(42)
    try:
        tf.get_logger().setLevel('ERROR')
    except Exception:
        pass
except Exception:
    pass

# Seeds numpy/python
random.seed(42)
np.random.seed(42)

@njit
def fast_trade_simulator_asymmetric(prices_high, prices_low, entry_indices, entry_prices, atrs_tp, atrs_sl, signals, sl_multiplier, tp_multiplier):
    total_profit = 0.0
    n_trades = len(entry_indices)
    for i in range(n_trades):
        entry_idx, entry_price = entry_indices[i], entry_prices[i]
        atr_tp, atr_sl, signal = atrs_tp[i], atrs_sl[i], signals[i]
        if signal == 1:
            stop_loss = entry_price - (atr_sl * sl_multiplier)
            take_profit = entry_price + (atr_tp * tp_multiplier)
            for k in range(entry_idx + 1, len(prices_high)):
                if prices_low[k] <= stop_loss: total_profit -= abs(entry_price - stop_loss); break
                if prices_high[k] >= take_profit: total_profit += abs(take_profit - entry_price); break
        else:
            stop_loss = entry_price + (atr_sl * sl_multiplier)
            take_profit = entry_price - (atr_tp * tp_multiplier)
            for k in range(entry_idx + 1, len(prices_high)):
                if prices_high[k] >= stop_loss: total_profit -= abs(entry_price - stop_loss); break
                if prices_low[k] <= take_profit: total_profit += abs(entry_price - take_profit); break
    return total_profit

@njit
def fast_sequence_creator(data, start_index, end_index, sequence_length):
    num_sequences = end_index - start_index
    batch_sequences = np.empty((num_sequences, sequence_length, data.shape[1]), dtype=np.float64)
    for i in range(num_sequences):
        original_index = start_index + i
        batch_sequences[i] = data[original_index : original_index + sequence_length]
    return batch_sequences

def optimize_sl_tp_for_symbol(symbol):
    print(f"\n--- DÉBUT DE L'OPTIMISATION SL/TP POUR {symbol} ---", flush=True)
    sl_tf_name, tp_tf_name = config.SL_TIMEFRAME_STR, config.TP_TIMEFRAME_STR
    sl_tf, tp_tf = config.SL_TIMEFRAME, config.TP_TIMEFRAME
    
    cfg = config.load_config_data()
    active_features = config.get_active_features_for_symbol(symbol)
    
    try:
        brain = EnsembleAIBrain(symbol=symbol, num_features=len(active_features))
    except Exception as e:
        print(f"   ERREUR: Impossible de charger le modèle. {e}", flush=True); return

    df_history = get_historical_data(symbol, config.TIMEFRAME_TO_TRAIN, config.BACKTEST_BARS)
    if df_history is None or df_history.empty: return
    df_history.set_index('time', inplace=True)
    
    df_features = pd.DataFrame(index=df_history.index)
    df_features[['open', 'high', 'low', 'close', 'tick_volume']] = df_history[['open', 'high', 'low', 'close', 'tick_volume']]

    # ATRs selon UT SL/TP
    for tf_name, tf_const, col_name in [(sl_tf_name, sl_tf, 'atr_sl'), (tp_tf_name, tp_tf, 'atr_tp')]:
        if tf_const == config.TIMEFRAME_TO_TRAIN:
            df_features[col_name] = ta.atr(df_features['high'], df_features['low'], df_features['close'], length=config.ATR_PERIOD)
        else:
            df_other_tf = get_historical_data(symbol, tf_const, len(df_features) // 24 + 200)
            if df_other_tf is not None and not df_other_tf.empty:
                df_other_tf[col_name] = ta.atr(df_other_tf['high'], df_other_tf['low'], df_other_tf['close'], length=config.ATR_PERIOD)
                df_other_tf.set_index('time', inplace=True)
                df_features = pd.merge_asof(df_features.sort_index(), df_other_tf[[col_name]].sort_index(), left_index=True, right_index=True, direction='backward')
    
    df_features.ffill(inplace=True)
    df_features.reset_index(inplace=True)

    active_groups = cfg.get("optimized_feature_configs", {}).get(symbol, {}).get("best_groups", cfg.get("active_feature_groups"))
    df_features_full = create_features(df_features.copy(), symbol, active_groups)

    all_predictions = []
    num_sequences = len(df_features_full) - config.SEQUENCE_LENGTH
    if num_sequences <= 0: 
        print("   ERREUR: Pas assez de données.", flush=True)
        return
    
    # Prépare features et scaler
    features_to_scale = [f for f in active_features if f in df_features_full.columns]
    feature_values = df_features_full[active_features].values
    scaler_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_scaler.joblib")
    if not os.path.exists(scaler_path):
        print(f"   ERREUR: Scaler non trouvé pour {symbol}", flush=True)
        return
    scaler = joblib_load(scaler_path)

    batch_size_pred = int(getattr(config, "SL_TP_PRED_BATCH_SIZE", 5000))
    keras_pred_batch = int(getattr(config, "PRED_BATCH_SIZE", 512))

    for i in range(0, num_sequences, batch_size_pred):
        end_index = min(i + batch_size_pred, num_sequences)
        batch_sequences = fast_sequence_creator(feature_values, i, end_index, config.SEQUENCE_LENGTH)
        if batch_sequences.shape[0] > 0:
            shape = batch_sequences.shape
            batch_sequences_reshaped = batch_sequences.reshape(-1, shape[2])
            batch_sequences_scaled_reshaped = scaler.transform(batch_sequences_reshaped)
            batch_sequences_scaled = batch_sequences_scaled_reshaped.reshape(shape)
            preds = brain.predict(batch_sequences_scaled, batch_size=keras_pred_batch)
            all_predictions.extend(preds)

    potential_indices, potential_signals = [], []
    for i, pred in enumerate(all_predictions):
        if pred[1] > config.CONFIDENCE_THRESHOLD: 
            potential_indices.append(i + config.SEQUENCE_LENGTH); potential_signals.append(1)
        elif pred[0] > config.CONFIDENCE_THRESHOLD: 
            potential_indices.append(i + config.SEQUENCE_LENGTH); potential_signals.append(-1)

    entry_indices = np.array(potential_indices, dtype=np.int64)
    signals = np.array(potential_signals, dtype=np.int64)
    if len(entry_indices) == 0:
        print("   -> Aucun point d'entrée trouvé. Impossible d'optimiser.", flush=True)
        print(f"OPTIMIZER_FINISHED:NoSignal:{symbol}", flush=True)
        return

    entry_prices = df_features_full['close'].iloc[entry_indices].values
    atrs_tp = df_features_full['atr_tp'].iloc[entry_indices].values
    atrs_sl = df_features_full['atr_sl'].iloc[entry_indices].values
    prices_high, prices_low = df_features_full['high'].values, df_features_full['low'].values

    # Grilles centralisées (ou valeurs par défaut)
    sl_range = np.array(getattr(config, "OPTIMIZER_SL_GRID", [0.5, 1.0, 1.5, 2.0]), dtype=np.float64)
    tp_range = np.array(getattr(config, "OPTIMIZER_TP_GRID", [1.0, 2.0, 3.0, 4.0]), dtype=np.float64)

    best_profit, best_params = -np.inf, (config.ATR_MULTIPLIER_SL, config.ATR_MULTIPLIER_TP)
    for sl in sl_range:
        for tp in tp_range:
            profit = fast_trade_simulator_asymmetric(prices_high, prices_low, entry_indices, entry_prices, atrs_tp, atrs_sl, signals, sl, tp)
            if profit > best_profit: best_profit, best_params = profit, (sl, tp)

    best_sl, best_tp = best_params
    print(f"\n--- OPTIMISATION TERMINÉE POUR {symbol} ---", flush=True)
    print(f"Meilleurs paramètres: SL({sl_tf_name})={best_sl:.2f}, TP({tp_tf_name})={best_tp:.1f}", flush=True)
    
    config_data = config.load_config_data()
    config_data.setdefault("optimal_sl_tp_multipliers", {})[symbol] = {"sl": float(best_sl), "tp": float(best_tp), "sl_timeframe": sl_tf_name, "tp_timeframe": tp_tf_name}
    config.save_config_data(config_data)
    print("Paramètres optimaux sauvegardés.", flush=True)
    print(f"OPTIMIZER_FINISHED:Terminé:{symbol}", flush=True)

def main():
    if len(sys.argv) < 2: 
        print("ERREUR: Spécifiez un symbole.", flush=True)
        return
    symbol = sys.argv[1].upper()
    if not connect_to_mt5(): 
        sys.exit(1)
    optimize_sl_tp_for_symbol(symbol)
    mt5.shutdown()

if __name__ == "__main__":
    main()