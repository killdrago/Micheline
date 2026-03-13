# meta_trainer.py - Superviseur (stacked) robuste au scaler dict + alignement des features
# - Supporte scaler sauvegardé comme {"scaler": RobustScaler, "features": [noms_features_training]}
# - Aligne automatiquement les colonnes avant transform
# - Seeds, logs TF réduits, no shuffle

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("PYTHONHASHSEED", "42")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")

import random
random.seed(42)

import numpy as np
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)
try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import sys
from datetime import datetime
import config
from trainer import ProgressCallback, create_sequences, create_features
from ia_utils import connect_to_mt5, get_historical_data
from model_manager import EnsembleAIBrain
import MetaTrader5 as mt5
from joblib import load as joblib_load


def build_meta_model(num_base_models: int) -> Sequential:
    input_shape = (num_base_models * 2,)
    model = Sequential([
        Input(shape=input_shape),
        Dense(16, activation="relu"),
        Dropout(0.3),
        Dense(8, activation="relu"),
        Dense(2, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _load_scaler_info(path: str):
    """
    Charge le scaler:
      - nouveau format: {"scaler": RobustScaler, "features": [noms_features_training]}
      - ancien format: RobustScaler seul
    Retourne (scaler, features_list_or_None)
    """
    obj = joblib_load(path)
    if isinstance(obj, dict) and "scaler" in obj:
        return obj["scaler"], list(obj.get("features") or [])
    return obj, None


def train_meta_model_fast(symbol: str):
    print(f"\n--- DÉBUT DE L'ENTRAÎNEMENT DU SUPERVISEUR POUR {symbol} ---")

    active_features = config.get_active_features_for_symbol(symbol)
    if not active_features:
        print("ERREUR: Impossible de déterminer les features actives.")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "FEATURES_UNAVAILABLE"
        config.save_config_data(cfg)
        sys.exit(2)

    # Charger les modèles de base (experts)
    try:
        brain = EnsembleAIBrain(symbol, len(active_features))
    except Exception as e:
        print(f"ERREUR: Impossible de charger les modèles d’experts: {e}")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "MISSING_BASE_MODELS"
        config.save_config_data(cfg)
        sys.exit(2)

    # Charger scaler avec support dict
    scaler_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_scaler.joblib")
    try:
        scaler, expected_features = _load_scaler_info(scaler_path)
        print(f"[META] Scaler chargé. format={'dict' if expected_features else 'raw'}, "
              f"n_expected_features={len(expected_features) if expected_features else 'N/A'}")
    except Exception as e:
        print(f"ERREUR: Scaler introuvable ou illisible: {e}")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "MISSING_SCALER"
        config.save_config_data(cfg)
        sys.exit(2)

    # Données historiques
    num_bars = config.INITIAL_TRAINING_BARS
    df_history = get_historical_data(symbol, config.TIMEFRAME_TO_TRAIN, num_bars)
    if df_history is None or df_history.empty:
        print("ERREUR: Données historiques introuvables.")
        sys.exit(2)

    # Features
    active_groups = config.get_active_groups_for_symbol(symbol)
    features_df = create_features(df_history.copy(), symbol, active_groups)

    # Préparer la liste de colonnes à scaler
    if expected_features:
        # Aligne strictement sur la liste du training
        for col in expected_features:
            if col not in features_df.columns:
                features_df[col] = 0.0
        features_to_scale = expected_features
    else:
        # Fallback: Features actives présentes
        features_to_scale = [f for f in active_features if f in features_df.columns]
        nfi = getattr(scaler, "n_features_in_", None)
        if nfi is not None and len(features_to_scale) != int(nfi):
            print(f"ERREUR FATALE: Le scaler attend {int(nfi)} features, mais on en a {len(features_to_scale)}.")
            cfg = config.load_config_data()
            cfg.setdefault("model_performances", {}).setdefault(symbol, {})
            cfg["model_performances"][symbol]["meta_blocked"] = "FEATURE_MISMATCH"
            config.save_config_data(cfg)
            sys.exit(2)

    # Transform
    try:
        df_scaled = features_df.copy()
        X = features_df[features_to_scale].values
        df_scaled[features_to_scale] = scaler.transform(X)
    except Exception as e:
        print(f"ERREUR: Transformation avec le scaler impossible: {e}")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "SCALER_TRANSFORM_ERROR"
        config.save_config_data(cfg)
        sys.exit(2)

    # Séquences
    X_sequences, y_target = create_sequences(df_scaled, features_to_scale, features_df)
    if X_sequences.size == 0:
        print("ERREUR: Impossible de créer des séquences.")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "NO_SEQUENCES"
        config.save_config_data(cfg)
        sys.exit(2)

    # Prédictions des modèles de base -> features méta
    try:
        base_preds = [m.model.predict(X_sequences, batch_size=512, verbose=0) for m in brain.models]
        X_meta_features = np.concatenate(base_preds, axis=1)
    except Exception as e:
        print(f"ERREUR: Prédiction des modèles de base impossible: {e}")
        cfg = config.load_config_data()
        cfg.setdefault("model_performances", {}).setdefault(symbol, {})
        cfg["model_performances"][symbol]["meta_blocked"] = "BASE_MODELS_PREDICT_ERROR"
        config.save_config_data(cfg)
        sys.exit(2)

    # Entraînement du superviseur
    meta_model = build_meta_model(len(brain.models))
    callbacks_meta = [
        ProgressCallback(30),
        EarlyStopping(monitor="val_accuracy", patience=7, mode="max", restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", patience=4),
    ]
    history = meta_model.fit(
        X_meta_features, y_target,
        epochs=30, batch_size=64,
        validation_split=0.2, callbacks=callbacks_meta,
        verbose=0, shuffle=False
    )

    # Sauvegarde
    meta_model_path = os.path.join(config.MODEL_FOLDER, f"{symbol}_meta.keras")
    meta_model.save(meta_model_path)

    final_accuracy = max(history.history["val_accuracy"]) * 100
    print("\n--- ENTRAÎNEMENT DU SUPERVISEUR TERMINÉ ---")
    print(f"Meilleure performance du Superviseur: {final_accuracy:.2f}%")

    config_data = config.load_config_data()
    config_data.setdefault("model_performances", {}).setdefault(symbol, {})
    # Nettoie tout blocage précédent
    try:
        config_data["model_performances"][symbol].pop("meta_blocked", None)
    except Exception:
        pass
    config_data["model_performances"][symbol].update({
        "meta_accuracy": final_accuracy,
        "last_meta_training": datetime.now().strftime("%Y-%m-%d"),
        "stacked": True
    })
    config.save_config_data(config_data)
    print(f"FINAL_ACCURACY:{final_accuracy}", flush=True)


def main():
    if len(sys.argv) < 2:
        print("ERREUR: Spécifiez un symbole.")
        sys.exit(2)
    symbol_to_train = sys.argv[1].upper()
    if not connect_to_mt5():
        sys.exit(1)
    train_meta_model_fast(symbol_to_train)
    mt5.shutdown()


if __name__ == "__main__":
    main()