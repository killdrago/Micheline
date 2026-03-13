# full_optimizer.py - Version avec seeds + shuffle=False + réduction de logs TF

import os
# Réduction de logs TF avant import tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import sys
import itertools
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from numba import njit
import config
import MetaTrader5 as mt5
import pandas_ta as ta
from sklearn.preprocessing import RobustScaler
from datetime import datetime
from ia_utils import get_historical_data, connect_to_mt5
from trainer import create_features
from sl_tp_optimizer import fast_trade_simulator_asymmetric

# Seeds (reproductibilité)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

HORIZONS_TO_TEST = ['H1', 'H4', 'D1']
SL_TP_TIMEFRAMES_TO_TEST = ['H1', 'H4', 'D1']

def build_temp_model(num_features):
    model = Sequential([
        Input(shape=(config.SEQUENCE_LENGTH, num_features)),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_labels_for_horizon(df_prices, horizon_str):
    lookahead = config.timeframe_to_h1_bars(horizon_str)
    price_diff = df_prices['close'].shift(-lookahead) - df_prices['close']
    labels = (price_diff > 0).astype(int)
    return tf.keras.utils.to_categorical(labels, num_classes=2)

def train_and_predict_for_horizon(df_features, active_features, horizon_str):
    print(f"  -> Entraînement du modèle de test pour l'horizon {horizon_str}...", flush=True)
    lookahead = config.timeframe_to_h1_bars(horizon_str)
    sequences = df_features[active_features].values
    num_sequences_possible = len(sequences) - config.SEQUENCE_LENGTH - lookahead
    if num_sequences_possible <= 0:
        return None
    all_X = np.array([sequences[i:i+config.SEQUENCE_LENGTH] for i in range(num_sequences_possible)])
    all_y = create_labels_for_horizon(df_features, horizon_str)[:num_sequences_possible]
    if all_X.size == 0:
        return None
    model = build_temp_model(all_X.shape[2])
    model.fit(
        all_X, all_y,
        epochs=config.OPTIMIZATION_EPOCHS,
        batch_size=128,
        validation_split=0.15,
        verbose=0,
        shuffle=False  # IMPORTANT: pas de mélange temporel
    )
    all_possible_sequences = np.array([sequences[i:i+config.SEQUENCE_LENGTH] for i in range(len(sequences) - config.SEQUENCE_LENGTH)])
    predictions = model.predict(all_possible_sequences, batch_size=512, verbose=0)
    tf.keras.backend.clear_session()
    return predictions

@njit
def find_best_multipliers_numba(prices_high, prices_low, entry_indices, entry_prices, atrs_tp, atrs_sl, signals):
    sl_range = np.arange(0.5, 2.1, 0.5)
    tp_range = np.arange(1.0, 4.1, 1.0)
    best_profit = -1e9
    best_sl, best_tp = 1.5, 2.0
    for sl in sl_range:
        for tp in tp_range:
            profit = fast_trade_simulator_asymmetric(prices_high, prices_low, entry_indices, entry_prices, atrs_tp, atrs_sl, signals, sl, tp)
            if profit > best_profit:
                best_profit, best_sl, best_tp = profit, sl, tp
    return best_profit, best_sl, best_tp

def run_full_optimization(symbol):
    print(f"\n--- DÉBUT DE L'OPTIMISATION COMPLÈTE POUR {symbol} ---", flush=True)
    
    cfg = config.load_config_data()
    df_history = get_historical_data(symbol, config.TIMEFRAME_TO_TRAIN, config.BACKTEST_BARS)
    if df_history is None:
        return
        
    active_groups = cfg.get("optimized_feature_configs", {}).get(symbol, {}).get("best_groups", cfg.get("active_feature_groups"))
    df_features = create_features(df_history.copy(), symbol, active_groups)
    active_features = config.get_active_features_for_symbol(symbol)

    scaler = RobustScaler()
    features_to_scale = [f for f in active_features if f in df_features.columns]
    if features_to_scale:
        scaler.fit(df_features[features_to_scale].values)
        df_features[features_to_scale] = scaler.transform(df_features[features_to_scale].values)
    
    atr_series_cache = {}
    df_features_indexed = df_features.set_index('time')
    for tf_str in set(SL_TP_TIMEFRAMES_TO_TEST):
        tf_const = config.TIMEFRAME_MAP[tf_str]
        if tf_const == config.TIMEFRAME_TO_TRAIN:
            atr_series = ta.atr(df_history['high'], df_history['low'], df_history['close'], length=config.ATR_PERIOD)
            temp_df = pd.DataFrame({'atr_' + tf_str: atr_series.values}, index=df_history['time'])
            atr_series_cache[tf_str] = temp_df
        else:
            df_temp_atr = get_historical_data(symbol, tf_const, len(df_features) // 24 + 200)
            if df_temp_atr is not None:
                df_temp_atr['atr_' + tf_str] = ta.atr(df_temp_atr['high'], df_temp_atr['low'], df_temp_atr['close'], length=config.ATR_PERIOD)
                df_temp_atr.set_index('time', inplace=True)
                atr_series_cache[tf_str] = df_temp_atr[['atr_' + tf_str]]

    overall_best_profit = -np.inf
    overall_best_params = {}
    best_params_per_horizon = {}

    for horizon in HORIZONS_TO_TEST:
        print(f"\n--- Analyse pour l'Horizon: {horizon} ---", flush=True)
        predictions = train_and_predict_for_horizon(df_features, active_features, horizon)
        if predictions is None:
            continue

        potential_indices, potential_signals = [], []
        for i, pred in enumerate(predictions):
            idx = i + config.SEQUENCE_LENGTH
            if idx < len(df_history) and pred[1] > config.CONFIDENCE_THRESHOLD:
                potential_indices.append(idx); potential_signals.append(1)
            elif idx < len(df_history) and pred[0] > config.CONFIDENCE_THRESHOLD:
                potential_indices.append(idx); potential_signals.append(-1)
        
        if not potential_indices:
            continue
        
        entry_indices = np.array(potential_indices, dtype=np.int64)
        signals = np.array(potential_signals, dtype=np.int64)
        entry_prices = df_history['close'].values[entry_indices]
        prices_high, prices_low = df_history['high'].values, df_history['low'].values

        horizon_best_profit, horizon_best_params = -np.inf, {}
        for sl_tf_str, tp_tf_str in itertools.product(SL_TP_TIMEFRAMES_TO_TEST, repeat=2):
            temp_df = pd.DataFrame(index=df_features_indexed.index)
            if sl_tf_str in atr_series_cache:
                temp_df = pd.merge_asof(temp_df, atr_series_cache[sl_tf_str], left_index=True, right_index=True, direction='backward')
            if tp_tf_str in atr_series_cache and sl_tf_str != tp_tf_str:
                temp_df = pd.merge_asof(temp_df, atr_series_cache[tp_tf_str], left_index=True, right_index=True, direction='backward')
            temp_df.ffill(inplace=True)
            
            if f'atr_{sl_tf_str}' not in temp_df.columns or f'atr_{tp_tf_str}' not in temp_df.columns:
                continue
            atrs_sl = temp_df[f'atr_{sl_tf_str}'].values[entry_indices]
            atrs_tp = temp_df[f'atr_{tp_tf_str}'].values[entry_indices]
            current_profit, best_sl_mult, best_tp_mult = find_best_multipliers_numba(
                prices_high, prices_low, entry_indices, entry_prices, atrs_tp, atrs_sl, signals
            )
            if current_profit > horizon_best_profit:
                horizon_best_profit = current_profit
                horizon_best_params = {
                    "prediction_horizon": horizon,
                    "sl_timeframe": sl_tf_str,
                    "tp_timeframe": tp_tf_str,
                    "sl_multiplier": best_sl_mult,
                    "tp_multiplier": best_tp_mult,
                    "profit": current_profit
                }
            if current_profit > overall_best_profit:
                overall_best_profit = current_profit
                overall_best_params = horizon_best_params.copy()

        if horizon_best_params:
            best_params_per_horizon[horizon] = horizon_best_params

    if not overall_best_params:
        print("Aucune combinaison profitable trouvée.", flush=True)
        print(f"FULL_OPTIMIZER_FINISHED:NoProfit:{symbol}", flush=True)
        return

    config_data = config.load_config_data()
    config_data["prediction_horizon"] = overall_best_params['prediction_horizon']
    config_data["sl_timeframe"] = overall_best_params['sl_timeframe']
    config_data["tp_timeframe"] = overall_best_params['tp_timeframe']
    config_data.setdefault("optimal_sl_tp_multipliers", {})[symbol] = {
        "sl": overall_best_params['sl_multiplier'],
        "tp": overall_best_params['tp_multiplier']
    }
    config_data.setdefault("full_optimization_status", {})[symbol] = {
        "last_run": datetime.now().strftime("%Y-%m-%d"),
        "best_overall": overall_best_params,
        "best_per_horizon": best_params_per_horizon
    }
    config.save_config_data(config_data)
    print("\nConfiguration optimale sauvegardée.", flush=True)
    print(f"FULL_OPTIMIZER_FINISHED:Terminé:{symbol}", flush=True)

def main():
    if len(sys.argv) < 2:
        print("ERREUR: Spécifiez un symbole.", flush=True)
        return
    symbol_to_optimize = sys.argv[1].upper()
    if not connect_to_mt5():
        sys.exit(1)
    run_full_optimization(symbol_to_optimize)
    mt5.shutdown()

if __name__ == "__main__":
    main()