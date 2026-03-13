# trainer.py - Version corrigée avec gestion sûre du scaler (chunk_num None en --update)
# + Filtre de Tendance configurable (UT & période SMA) via config.compute_trend_filter_columns
# + Réduction de logs TensorFlow (TF_CPP_MIN_LOG_LEVEL, oneDNN off)
# - Apprentissage continu par "chunks" ou mise à jour (--update)
# - Cohérence des features (ALWAYS_ON + groupes actifs/optimisés)
# - Multi-timeframe optionnel
# - Scaler: chargement SÛR, refit si mismatch, SAUVEGARDE dict {"scaler":..., "features":[...]} pour garder l'ordre
# - Callbacks d'entraînement avec logs de progression

import os
# Réduction de logs TF avant import tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # masque INFO/WARNING/ERROR C++ TF
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # désactive les opts oneDNN (moins de logs)

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import config
import json
from datetime import datetime
import sys
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau
from model_manager import AIBrain, EnsembleAIBrain
from scipy.signal import find_peaks
from scipy.stats import linregress
from sklearn.preprocessing import RobustScaler
from arch import arch_model
import joblib
from ia_utils import classify_detailed_market_state, connect_to_mt5
import random

# Seeds (reproductibilité)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# Optionnel: réduit le logger python TF
try:
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

# ------------------------------
# Récupération de données MT5
# ------------------------------

def get_historical_data_chunk(symbol, timeframe, total_years_config, chunk_num_to_get):
    """Récupère les données pour une année spécifique (chunk) de l'historique."""
    bars_per_year = config.years_to_h1_bars(1)
    offset = (total_years_config - chunk_num_to_get) * bars_per_year
    
    print(f"Récupération de l'année {chunk_num_to_get}/{total_years_config} pour {symbol} (environ {bars_per_year} barres)...", flush=True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, offset, bars_per_year)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def get_historical_data_update(symbol, timeframe, num_bars):
    """Récupère les données les plus récentes pour une mise à jour."""
    print(f"Récupération des {num_bars} barres les plus récentes pour {symbol}...", flush=True)
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

# ------------------------------
# Features & indicateurs
# ------------------------------

def calculate_pivot_points(df, atr_series):
    df_temp = df.copy()
    if not isinstance(df_temp.index, pd.DatetimeIndex):
        if 'time' in df_temp.columns:
            df_temp.set_index('time', inplace=True)
        elif df_temp.index.name == 'time':
            pass
        else:
            return df
    df_daily = df_temp['close'].resample('D').ohlc().rename(columns=lambda x: f"D_{x}")
    df_daily['pivot'] = (df_daily['D_high'] + df_daily['D_low'] + df_daily['D_close']) / 3
    df_daily['r1'] = 2 * df_daily['pivot'] - df_daily['D_low']
    df_daily['s1'] = 2 * df_daily['pivot'] - df_daily['D_high']
    pivots = df_daily[['pivot', 'r1', 's1']].shift(1)
    df_temp.reset_index(inplace=True)
    merged = pd.merge(df_temp, pivots, how='left', left_on=df_temp['time'].dt.date, right_on=pivots.index.date)
    merged.set_index('time', inplace=True)
    merged.drop(columns=['key_0'], inplace=True, errors='ignore')
    merged[['pivot', 'r1', 's1']] = merged[['pivot', 'r1', 's1']].ffill()
    df['dist_from_pivot'] = (df['close'] - merged['pivot']) / atr_series
    df['dist_from_r1'] = (df['close'] - merged['r1']) / atr_series
    df['dist_from_s1'] = (df['close'] - merged['s1']) / atr_series
    return df

def calculate_fibonacci_levels(df, atr_series):
    lookback = 120
    high_roll = df['high'].rolling(window=lookback, min_periods=lookback).max()
    low_roll = df['low'].rolling(window=lookback, min_periods=lookback).min()
    df['fib_382'] = high_roll - (high_roll - low_roll) * 0.382
    df['fib_500'] = high_roll - (high_roll - low_roll) * 0.5
    df['fib_618'] = high_roll - (high_roll - low_roll) * 0.618
    df['dist_from_fib_382'] = (df['close'] - df['fib_382']) / atr_series
    df['dist_from_fib_500'] = (df['close'] - df['fib_500']) / atr_series
    df['dist_from_fib_618'] = (df['close'] - df['fib_618']) / atr_series
    df.drop(columns=['fib_382', 'fib_500', 'fib_618'], inplace=True, errors='ignore')
    return df

def calculate_volume_profile(df, atr_series):
    lookback = 240
    def get_poc(series):
        if series.isnull().all():
            return np.nan
        volume_at_price = df.loc[series.index, 'tick_volume'].groupby(pd.cut(series, bins=20), observed=False).sum()
        if not volume_at_price.empty:
            return volume_at_price.idxmax().mid
        return np.nan
    poc_series = df['close'].rolling(window=lookback, min_periods=lookback).apply(get_poc, raw=False)
    df['poc_dist'] = (df['close'] - poc_series) / atr_series
    df['vah_dist'] = (df['high'].rolling(lookback).max() - df['close']) / atr_series
    df['val_dist'] = (df['close'] - df['low'].rolling(lookback).min()) / atr_series
    return df

def calculate_garch_volatility(log_returns):
    garch_series = pd.Series(index=log_returns.index, dtype=float)
    valid_returns = log_returns.dropna()
    if len(valid_returns) < 252:
        return garch_series
    model = arch_model(valid_returns, vol='Garch', p=1, q=1, dist='Normal', rescale=True)
    try:
        res = model.fit(disp='off', show_warning=False)
        garch_series.loc[res.resid.index] = np.sqrt(res.conditional_volatility)
    except Exception:
        pass
    return garch_series.ffill()

def create_correlation_features(df, symbol):
    REFERENCE_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY']
    if 'time' not in df.columns:
        df.reset_index(inplace=True)
    df.set_index('time', inplace=True)
    df['returns_main'] = df['close'].pct_change()
    
    for other_pair in REFERENCE_PAIRS:
        if other_pair == symbol:
            continue
        df_other = get_historical_data_update(other_pair, config.TIMEFRAME_TO_TRAIN, len(df) + 50)
        if df_other is None or df_other.empty:
            continue
        df_other.set_index('time', inplace=True)
        df_other[f'returns_{other_pair}'] = df_other['close'].pct_change()
        
        original_index = df.index
        df_merged = pd.merge_asof(df.sort_index(), df_other[[f'returns_{other_pair}']], left_index=True, right_index=True)
        df_merged.index = original_index
        
        correlation_col = f'corr_{other_pair}'
        df_merged[correlation_col] = df_merged['returns_main'].rolling(window=50, min_periods=50).corr(df_merged[f'returns_{other_pair}'])
        df = df_merged.drop(columns=[f'returns_{other_pair}'])

    df.drop(columns=['returns_main'], inplace=True)
    df.reset_index(inplace=True)
    return df

def _calculate_swing_features(df, atr_series):
    high_peaks, _ = find_peaks(df['high'], distance=5, prominence=0.001)
    low_peaks, _ = find_peaks(-df['low'], distance=5, prominence=0.001)
    df['swing_high'] = np.nan
    df.loc[df.index[high_peaks], 'swing_high'] = df['high'].iloc[high_peaks]
    df['swing_low'] = np.nan
    df.loc[df.index[low_peaks], 'swing_low'] = df['low'].iloc[low_peaks]
    df[['swing_high', 'swing_low']] = df[['swing_high', 'swing_low']].ffill().bfill()
    df['distance_from_swing_high'] = (df['close'] - df['swing_high']) / atr_series
    df['distance_from_swing_low'] = (df['close'] - df['swing_low']) / atr_series
    df['swing_range_width'] = (df['swing_high'] - df['swing_low']) / atr_series
    return df.drop(columns=['swing_high', 'swing_low'], errors='ignore')

def create_multi_timeframe_features(symbol, df_h1):
    if 'time' in df_h1.columns and not isinstance(df_h1.index, pd.DatetimeIndex):
        df_h1.set_index('time', inplace=True)
    for tf, period_factor, name in [
        (mt5.TIMEFRAME_H4, 4, 'h4'),
        (mt5.TIMEFRAME_D1, 24, 'd1'),
        (mt5.TIMEFRAME_W1, 24*5, 'w1')
    ]:
        df_tf = get_historical_data_update(symbol, tf, len(df_h1) // period_factor + 200)
        if df_tf is not None and not df_tf.empty:
            df_tf.set_index('time', inplace=True)
            df_tf[f'trend_{name}'] = ta.sma(df_tf['close'], 5) / ta.sma(df_tf['close'], 20) - 1
            df_tf[f'volatility_{name}'] = ta.atr(df_tf['high'], df_tf['low'], df_tf['close'], 14) / df_tf['close']
            df_h1 = pd.merge_asof(
                left=df_h1.sort_index(), right=df_tf[[f'trend_{name}', f'volatility_{name}']].sort_index(),
                left_index=True, right_index=True, direction='backward'
            )
    df_h1.ffill(inplace=True)
    return df_h1

def create_features(df, symbol, active_feature_groups):
    if 'Corrélation Forex' in active_feature_groups:
        df = create_correlation_features(df.copy(), symbol)

    active_options = set(active_feature_groups)
    if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df.set_index('time', inplace=True)
    
    df.ta.adx(length=14, append=True)
    atr_series = ta.atr(df['high'], df['low'], df['close'], length=config.ATR_PERIOD).replace(0, 1e-9)
    ma_fast = ta.sma(df['close'], 20)
    ma_slow = ta.sma(df['close'], 50)

    # === Filtre de tendance configurable (natif MT5 via config) ===
    try:
        df = config.compute_trend_filter_columns(df, symbol)
    except Exception as e:
        print(f"[TrendFilter] Impossible de calculer le filtre de tendance: {e}", flush=True)
    
    # Always-On context
    df['hour'] = df.index.hour / 23.0
    df['day_of_week'] = df.index.dayofweek / 6.0
    df['london_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
    df['ny_session'] = ((df.index.hour >= 13) & (df.index.hour < 21)).astype(int)
    df['tokyo_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['realized_vol'] = df['log_returns'].rolling(20).std() * np.sqrt(252)
    df['returns_skew'] = df['log_returns'].rolling(60).skew()
    df['returns_kurt'] = df['log_returns'].rolling(60).kurt()
    df = _calculate_swing_features(df, atr_series)
    
    # Groupes optionnels
    if 'Moyennes Mobiles' in active_options:
        df['ma_fast_dist'] = (df['close'] - ma_fast) / atr_series
        df['ma_slow_dist'] = (df['close'] - ma_slow) / atr_series
        ma_cross = (ma_fast > ma_slow).astype(int).diff().fillna(0)
        df['ma_cross_age'] = ma_cross.groupby((ma_cross != 0).cumsum()).cumcount()
    if 'VWAP' in active_options:
        vwap = ta.vwap(df['high'], df['low'], df['close'], df['tick_volume'])
        if vwap is not None:
            df['vwap_dist'] = (df['close'] - vwap) / atr_series
    if 'SuperTrend' in active_options:
        st = ta.supertrend(df['high'], df['low'], df['close'], length=10, multiplier=3)
        if st is not None and not st.empty:
            df['supertrend_dir'] = st.iloc[:, 1]
            df['supertrend_dist'] = (df['close'] - st.iloc[:, 0]) / atr_series
    if 'RSI' in active_options:
        df['rsi'] = ta.rsi(df['close'], 14)
    if 'Stochastique' in active_options:
        stoch = ta.stoch(df['high'], df['low'], df['close'])
        if stoch is not None and not stoch.empty:
            df['stoch_k'] = stoch.iloc[:, 0]
            df['stoch_d'] = stoch.iloc[:, 1]
    if 'CCI' in active_options:
        df['cci'] = ta.cci(df['high'], df['low'], df['close'], 20)
    if 'MACD' in active_options or 'Divergences' in active_options:
        macd_df = ta.macd(df['close'])
        if macd_df is not None and not macd_df.empty:
            if 'MACD' in active_options:
                df['macd_dist'] = (macd_df.iloc[:, 0] - macd_df.iloc[:, 1]) / atr_series
                df['macd_signal_dist'] = macd_df.iloc[:, 1] / atr_series
                df['macd_histogram_norm'] = macd_df.iloc[:, 2] / atr_series
            if 'Divergences' in active_options:
                df['macd_price_divergence'] = df['close'].pct_change(14) - macd_df.iloc[:, 0].pct_change(14)
    if 'Divergences' in active_options:
        if 'rsi' not in df.columns:
            df['rsi'] = ta.rsi(df['close'], 14)
        df['rsi_price_divergence'] = df['close'].pct_change(14) - df['rsi'].pct_change(14)
    if 'Bandes de Bollinger' in active_options or 'Canaux Keltner' in active_options:
        bb = ta.bbands(df['close'], 20)
        kc = ta.kc(df['high'], df['low'], df['close'], 20, atr_length=config.ATR_PERIOD)
        if bb is not None and not bb.empty:
            bb_lower = bb.iloc[:, 0]
            bb_mid   = bb.iloc[:, 1].replace(0, 1e-9)
            bb_upper = bb.iloc[:, 2]
            if 'Bandes de Bollinger' in active_options:
                df['bb_width'] = (bb_upper - bb_lower) / bb_mid
                if kc is not None and not kc.empty:
                    kc_lower = kc.iloc[:, 2]
                    kc_mid   = kc.iloc[:, 1].replace(0, 1e-9)
                    kc_upper = kc.iloc[:, 0]
                    kc_width = (kc_upper - kc_lower) / kc_mid
                    df['bb_squeeze'] = (df['bb_width'] < kc_width).astype(int)
        if 'Canaux Keltner' in active_options and kc is not None and not kc.empty:
            kc_lower = kc.iloc[:, 2]
            kc_mid   = kc.iloc[:, 1].replace(0, 1e-9)
            kc_upper = kc.iloc[:, 0]
            df['kc_width'] = (kc_upper - kc_lower) / kc_mid
            df['price_kc_pos'] = (df['close'] - kc_lower) / (kc_upper - kc_lower).replace(0, 1e-9)
    if 'ATR' in active_options:
        df['atr_percent'] = atr_series / df['close']
    if 'Volume en Ticks' in active_options:
        df['volume_momentum'] = df['tick_volume'] / df['tick_volume'].rolling(50).mean().replace(0,1)
        df['volume_price_ratio'] = df['tick_volume'] / (df['high'] - df['low']).replace(0,1e-9)
    if 'Force Index' in active_options:
        df['elder_force_index'] = ta.efi(df['close'], df['tick_volume'], 13)
    if 'Profil de Volume' in active_options:
        df = calculate_volume_profile(df, atr_series)
    if 'Flux d\'Ordres Simple' in active_options:
        price_change = df['close'].diff()
        df['cumulative_delta'] = (np.sign(price_change) * df['tick_volume']).rolling(50).sum()
    if 'Points Pivots' in active_options:
        df = calculate_pivot_points(df, atr_series)
    if 'Retracements Fibonacci' in active_options:
        df = calculate_fibonacci_levels(df, atr_series)
    if 'Ichimoku Kinko Hyo' in active_options:
        ichimoku_data = ta.ichimoku(df['high'], df['low'], df['close'])
        ichi = None
        if ichimoku_data is not None:
            if isinstance(ichimoku_data, (list, tuple)) and len(ichimoku_data) > 0:
                ichi = ichimoku_data[0]
            elif isinstance(ichimoku_data, pd.DataFrame):
                ichi = ichimoku_data
        if ichi is not None and not ichi.empty and all(c in ichi.columns for c in ['ITS_9', 'IKS_26', 'ISA_9', 'ISB_26']):
            df['ichi_cross'] = (ichi['ITS_9'] - ichi['IKS_26']) / atr_series
            df['price_cloud_pos'] = (df['close'] - (ichi['ISA_9'] + ichi['ISB_26']) / 2) / atr_series
            df['cloud_strength'] = (ichi['ISA_9'] - ichi['ISB_26']) / atr_series
            df['chikou_momentum'] = (df['close'] - df['close'].shift(26)) / atr_series
    if 'Choppiness' in active_options:
        df['choppiness_index'] = ta.chop(df['high'], df['low'], df['close'], 14)
    if 'Awesome Oscillator' in active_options:
        df['awesome_oscillator'] = ta.ao(df['high'], df['low'])
    if 'Volatilité GARCH' in active_options:
        if 'log_returns' not in df.columns:
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['garch_volatility'] = calculate_garch_volatility(df['log_returns'])
    if 'Patterns de Chandeliers' in active_options:
        df.ta.cdl_pattern("all", append=True)
        cdl_cols = [c for c in df.columns if c.startswith('CDL_')]
        if cdl_cols:
            df['candle_pattern_score'] = df[cdl_cols].sum(axis=1) / 100.0
            df.drop(columns=cdl_cols, inplace=True, errors='ignore')
        df['candle_body_ratio'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'])).replace([np.inf, -np.inf], 0).fillna(0)
    if 'Vortex' in active_options:
        vortex_df = ta.vortex(df['high'], df['low'], df['close'], append=False)
        if vortex_df is not None and not vortex_df.empty:
            df['vortex_diff'] = vortex_df.iloc[:, 0] - vortex_df.iloc[:, 1]
    if 'Parabolic SAR' in active_options:
        psar_df = ta.psar(df['high'], df['low'], df['close'], append=False)
        if psar_df is not None and not psar_df.empty:
            psar_values = psar_df.iloc[:, 0].fillna(psar_df.iloc[:, 1])
            df['psar_dist'] = (df['close'] - psar_values) / atr_series
    if 'Donchian Channels' in active_options:
        lookback = 20
        upper = df['high'].rolling(window=lookback).max()
        lower = df['low'].rolling(window=lookback).min()
        df['donchian_width'] = (upper - lower) / atr_series
        df['donchian_pos'] = (df['close'] - lower) / (upper - lower).replace(0, 1e-9)
    if 'On-Balance Volume (OBV)' in active_options:
        obv = ta.obv(df['close'], df['tick_volume'])
        if obv is not None and not obv.empty:
            obv_slope = obv.rolling(14).apply(lambda x: linregress(np.arange(len(x)), x)[0] if len(x) > 1 else 0.0, raw=False)
            df['obv_slope'] = obv_slope
    if 'Chaikin Money Flow (CMF)' in active_options:
        df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['tick_volume'])
    
    if isinstance(df.index, pd.DatetimeIndex):
        df.reset_index(inplace=True)
    df.rename(columns={'ADX_14': 'adx'}, inplace=True, errors='ignore')
    df['atr'] = atr_series.values
    df['ma_fast'] = ma_fast.values
    df['ma_slow'] = ma_slow.values
    
    # Etats de marché avancés
    df_copy = df.copy()
    if 'time' in df_copy.columns:
        df_copy.set_index('time', inplace=True)
    market_states = classify_detailed_market_state(df_copy)
    state_dummies = pd.get_dummies(market_states, prefix='state').astype('float32')
    
    df.set_index('time', inplace=True, drop=False)
    df = pd.concat([df, state_dummies], axis=1)
    df.reset_index(drop=True, inplace=True)
    
    # Normaliser la liste finale de features en fonction de la config active
    all_possible_features = config.get_active_features_for_symbol(symbol)
    for feature in set(all_possible_features):
        if feature not in df.columns:
            df[feature] = 0.0
    
    feature_cols_in_df = [col for col in all_possible_features if col in df.columns]
    df[feature_cols_in_df] = df[feature_cols_in_df].apply(pd.to_numeric, errors='coerce')

    df.fillna(0, inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df[feature_cols_in_df] = df[feature_cols_in_df].astype('float32')

    return df

# ------------------------------
# Callbacks & Séquences
# ------------------------------

class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"PROGRESS_EPOCH:{epoch + 1}/{self.total_epochs} (Train Acc: {logs.get('accuracy',0)*100:.2f}%, Val Acc: {logs.get('val_accuracy',0)*100:.2f}%, Val Loss: {logs.get('val_loss',0):.4f})", flush=True)

def create_sequences(df_scaled, features_list, df_original_prices):
    X, y = [], []
    if not features_list or len(df_scaled) < config.SEQUENCE_LENGTH + config.FUTURE_LOOKAHEAD_BARS:
        return np.array(X), np.array(y)
    df_features_values = df_scaled[features_list].values
    for i in range(len(df_features_values) - config.SEQUENCE_LENGTH - config.FUTURE_LOOKAHEAD_BARS):
        X.append(df_features_values[i:i+config.SEQUENCE_LENGTH])
        current_price = df_original_prices['close'].iloc[i + config.SEQUENCE_LENGTH]
        future_price = df_original_prices['close'].iloc[i + config.SEQUENCE_LENGTH + config.FUTURE_LOOKAHEAD_BARS]
        y.append([0, 1] if future_price > current_price else [1, 0])
    return np.array(X), np.array(y)

# ------------------------------
# Entraînement principal
# ------------------------------
def train_or_update_model(symbol, chunk_num=None, is_update=False):
    cfg = config.load_config_data()
    active_features = config.get_active_features_for_symbol(symbol)
    num_active_features = len(active_features)

    # Détermine le mode + hyperparamètres depuis config
    if chunk_num:
        total_years = cfg.get('initial_training_years', config.INITIAL_TRAINING_YEARS)
        mode = f"Apprentissage Continu - Année {chunk_num}/{total_years}"
        historical_df = get_historical_data_chunk(symbol, config.TIMEFRAME_TO_TRAIN, total_years, chunk_num)
        epochs = config.INITIAL_EPOCHS
        patience = getattr(config, "PATIENCE_INITIAL", 10)
    elif is_update:
        mode = "Mise à Jour"
        historical_df = get_historical_data_update(symbol, config.TIMEFRAME_TO_TRAIN, config.UPDATE_TRAINING_BARS)
        epochs = config.UPDATE_EPOCHS
        patience = getattr(config, "PATIENCE_UPDATE", 10)
    else:
        print("ERREUR: Mode d'entraînement non spécifié.", flush=True)
        return

    print(f"Mode: {mode}", flush=True)
    print(f"Hyperparamètres -> epochs={epochs}, patience={patience}, batch_size=256", flush=True)

    # Sanité des données
    min_len = config.SEQUENCE_LENGTH + config.FUTURE_LOOKAHEAD_BARS + 50
    if historical_df is None or len(historical_df) < min_len:
        print(f"ERREUR CRITIQUE: Données insuffisantes pour ce bloc (min {min_len}).", flush=True)
        return
    
    # Multi-timeframe (optionnel)
    if cfg.get("use_multi_timeframe", True):
        df_with_mtf = create_multi_timeframe_features(symbol, historical_df.copy())
    else:
        df_with_mtf = historical_df.copy()

    # Features
    active_groups = config.get_active_groups_for_symbol(symbol)
    features_df = create_features(df_with_mtf, symbol, active_groups)
    features_to_scale = [f for f in active_features if f in features_df.columns]
    if not features_to_scale:
        print("ERREUR: Aucune feature active disponible après création des features.", flush=True)
        return
    
    # ----- Scaler: chargement/fit SÛR + sauvegarde dict {"scaler":..., "features":[...]} -----
    scaler_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_scaler.joblib")
    use_prev_scaler = os.path.exists(scaler_path) and ((chunk_num is not None and chunk_num > 1) or is_update)

    if use_prev_scaler:
        try:
            prev = joblib.load(scaler_path)
            if isinstance(prev, dict):
                scaler = prev.get("scaler")
                prev_features = prev.get("features", [])
            else:
                scaler = prev
                prev_features = []
            # Si mismatch de dimension OU mismatch de set de features → refit
            need_refit = False
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ != len(features_to_scale):
                need_refit = True
            if prev_features and prev_features != features_to_scale:
                need_refit = True
            if need_refit:
                print(f"Scaler incompatible avec les features actuelles. Refit du scaler...", flush=True)
                scaler = RobustScaler()
                scaler.fit(features_df[features_to_scale].values)
                joblib.dump({"scaler": scaler, "features": features_to_scale}, scaler_path)
            else:
                # Met à jour le format (si ancien) pour inclure la liste des features
                if not isinstance(prev, dict):
                    joblib.dump({"scaler": scaler, "features": features_to_scale}, scaler_path)
        except Exception as e:
            print(f"Impossible de charger le scaler existant ({e}). Refit en cours...", flush=True)
            scaler = RobustScaler()
            scaler.fit(features_df[features_to_scale].values)
            joblib.dump({"scaler": scaler, "features": features_to_scale}, scaler_path)
    else:
        scaler = RobustScaler()
        scaler.fit(features_df[features_to_scale].values)
        joblib.dump({"scaler": scaler, "features": features_to_scale}, scaler_path)
    # --------------------------------------

    # Mise à l'échelle + séquences
    df_scaled = features_df.copy()
    df_scaled[features_to_scale] = scaler.transform(features_df[features_to_scale].values)
    
    X_train, y_train = create_sequences(df_scaled, features_to_scale, features_df)
    if X_train.size == 0:
        print("Aucune séquence d'entraînement créée pour ce bloc.", flush=True)
        return

    brain = EnsembleAIBrain(symbol=symbol, num_features=num_active_features)
    reduce_patience = max(1, patience // 2)

    for model_brain in brain.models:
        print(f"\n--- Apprentissage Continu de l'Expert V{model_brain.model_variant} pour {symbol} ({mode}) ---", flush=True)

        # Patience déjà définie plus haut (PATIENCE_INITIAL ou PATIENCE_UPDATE)
        reduce_patience = max(1, patience // 3)
        min_delta_acc = 1e-4  # amélioration minimale utile sur val_accuracy

        callbacks = [
            ProgressCallback(epochs),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                mode='max',
                patience=reduce_patience,
                factor=0.5,
                min_lr=1e-6,
                min_delta=min_delta_acc,
                cooldown=2,
                verbose=0
            ),
            EarlyStopping(
                monitor='val_accuracy',
                mode='max',
                patience=patience,
                min_delta=min_delta_acc,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model_brain.model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=256,
            validation_split=0.15, verbose=0, callbacks=callbacks,
            shuffle=False  # pas de mélange temporel
        )

        # Info: meilleure époque sur val_acc
        try:
            import numpy as np
            best_epoch = int(np.argmax(history.history['val_accuracy']) + 1)
            best_val_acc = float(history.history['val_accuracy'][best_epoch - 1]) * 100.0
            print(f"V{model_brain.model_variant} meilleur val_acc à l’époque {best_epoch} "
                  f"({best_val_acc:.2f}%), entraînement arrêté à {len(history.history['loss'])} époques.", flush=True)
        except Exception:
            pass

        model_brain.model.save(model_brain.model_path)

    # Met à jour la config
    config_data = config.load_config_data()
    if chunk_num:
        config_data.setdefault("training_progress", {}).setdefault(symbol, {})
        config_data["training_progress"][symbol]["last_chunk_completed"] = chunk_num
        
        if chunk_num >= 1:
            config_data.setdefault("model_performances", {}).setdefault(symbol, {})
            config_data["model_performances"][symbol]["last_update"] = datetime.now().strftime("%Y-%m-%d")
            config_data["model_performances"][symbol]["num_features"] = num_active_features
        
        if chunk_num == cfg.get('initial_training_years', config.INITIAL_TRAINING_YEARS):
            config_data["model_performances"][symbol]["last_training"] = datetime.now().strftime("%Y-%m-%d")

    if is_update:
        config_data.setdefault("model_performances", {}).setdefault(symbol, {})
        config_data["model_performances"][symbol]["last_training"] = datetime.now().strftime("%Y-%m-%d")
        config_data["model_performances"][symbol]["num_features"] = num_active_features

    config.save_config_data(config_data)
    print(f"Bloc d'apprentissage terminé et sauvegardé pour {symbol}.", flush=True)
    
# ------------------------------
# Entrée principale
# ------------------------------

def main():
    if len(sys.argv) < 2:
        print("ERREUR: Spécifiez les paramètres.", flush=True)
        return
    
    symbol_to_train = sys.argv[1]
    chunk_num = None
    is_update = False

    if "--chunk" in sys.argv:
        try:
            index = sys.argv.index("--chunk") + 1
            chunk_num = int(sys.argv[index])
        except (ValueError, IndexError):
            print("ERREUR: Argument --chunk invalide.", flush=True)
            return
    elif "--update" in sys.argv:
        is_update = True
    
    if not connect_to_mt5():
        return
    if not os.path.exists(config.MODEL_FOLDER):
        os.makedirs(config.MODEL_FOLDER)
    
    train_or_update_model(symbol_to_train, chunk_num=chunk_num, is_update=is_update)
    mt5.shutdown()

if __name__ == "__main__":
    main()