# ia_utils.py
# Ce fichier centralise les fonctions utilitaires, notamment la classification de l'état du marché.

import os
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import pandas_ta as ta
import config

def connect_to_mt5():
    """Initialise la connexion à MetaTrader 5."""
    mql5_path_str = config.MQL5_FILES_PATH
    terminal_path = None
    if mql5_path_str and "MQL5/Files" in mql5_path_str:
        terminal_path = mql5_path_str.split("MQL5/Files")[0] + "terminal64.exe"
    
    if terminal_path and os.path.exists(terminal_path):
        if mt5.initialize(path=terminal_path):
            print("Connecté à MetaTrader 5 via le chemin spécifique.")
            return True
            
    if mt5.initialize():
        print("Connecté à MetaTrader 5 (connexion par défaut).")
        return True
        
    print("ERREUR: Impossible de se connecter à MetaTrader5.")
    return False

def get_historical_data(symbol, timeframe, num_bars):
    """Récupère les données historiques depuis MT5."""
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"Aucune donnée historique reçue pour {symbol} sur {timeframe}.")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

def classify_detailed_market_state(df_features):
    """
    Analyse un DataFrame de features pour déterminer l'état détaillé du marché
    pour chaque bougie. Cette fonction est conçue pour être appliquée sur un DataFrame entier.

    Retourne: Une série Pandas avec l'état du marché pour chaque ligne.
    """
    # Indicateurs nécessaires pour la classification
    close = df_features['close']
    high = df_features['high']
    low = df_features['low']

    ma_slow = ta.sma(close, length=50)
    ma_fast = ta.sma(close, length=20)
    ema_short = ta.ema(close, length=9)

    adx_series = ta.adx(high, low, close, length=14)
    if adx_series is not None and isinstance(adx_series, pd.DataFrame) and 'ADX_14' in adx_series.columns:
        adx = adx_series['ADX_14']
    else:
        adx = pd.Series(0.0, index=df_features.index)

    bbands = ta.bbands(close, length=20)
    if bbands is not None and not bbands.empty:
        # bbands colonnes (par défaut pandas_ta): [BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0]
        # On prend: lower, middle, upper
        bb_lower = bbands.iloc[:, 0]
        bb_mid = bbands.iloc[:, 1].replace(0, 1e-9)
        bb_upper = bbands.iloc[:, 2]
        bb_width = (bb_upper - bb_lower) / bb_mid
    else:
        bb_width = pd.Series(0.0, index=df_features.index)

    # Logique de classification
    conditions = [
        # TENDANCES HAUSSIÈRES
        (adx > 25) & (ma_fast > ma_slow) & (close > ema_short),
        (adx > 25) & (ma_fast > ma_slow) & (close <= ema_short),
        (adx <= 25) & (ma_fast > ma_slow),

        # TENDANCES BAISSIÈRES
        (adx > 25) & (ma_fast < ma_slow) & (close < ema_short),
        (adx > 25) & (ma_fast < ma_slow) & (close >= ema_short),
        (adx <= 25) & (ma_fast < ma_slow),

        # RANGES
        (adx < 20) & (bb_width < bb_width.rolling(50).mean()),
        (adx < 20) & (bb_width >= bb_width.rolling(50).mean())
    ]
    
    choices = [
        "Tendance_Haussiere_Impulsion",
        "Tendance_Haussiere_Correction",
        "Tendance_Haussiere_Faible",
        "Tendance_Baissiere_Impulsion",
        "Tendance_Baissiere_Correction",
        "Tendance_Baissiere_Faible",
        "Range_Calme",
        "Range_Volatil"
    ]
    
    return pd.Series(np.select(conditions, choices, default='Indetermine'), index=df_features.index)