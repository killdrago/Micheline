# feature_optimizer.py - Version centralisée (seeds + shuffle=False + TF logs réduits)
# - TOP_N_CANDIDATES lu via config.OPTIMIZER_TOP_N_CANDIDATES

import os
# Réduction de logs TF avant import tensorflow
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import sys
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from sklearn.preprocessing import RobustScaler
import config
from ia_utils import get_historical_data, connect_to_mt5
from trainer import create_features, create_sequences

# Seeds (reproductibilité)
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.get_logger().setLevel('ERROR')
except Exception:
    pass

TOP_N_CANDIDATES = int(getattr(config, "OPTIMIZER_TOP_N_CANDIDATES", 5))

def build_temp_model(num_features):
    model = Sequential([
        Input(shape=(config.SEQUENCE_LENGTH, num_features)),
        LSTM(units=32, return_sequences=False),
        Dropout(0.2),
        Dense(units=2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def evaluate_feature_combination(symbol, feature_groups, df_base):
    print(f"    -> Test de la combinaison: {feature_groups}", flush=True)
    
    df_features = create_features(df_base.copy(), symbol, active_feature_groups=feature_groups)
    
    temp_config = config.load_config_data()
    temp_config.setdefault("optimized_feature_configs", {})[symbol] = {"best_groups": feature_groups}
    
    active_features = config.get_active_features_for_symbol(symbol)
    features_to_scale = [f for f in active_features if f in df_features.columns]
    if not features_to_scale: 
        return 0.0

    scaler = RobustScaler()
    train_split_index = int(len(df_features) * 0.85)
    scaler.fit(df_features[features_to_scale][:train_split_index].values)
    
    df_scaled = df_features.copy()
    df_scaled[features_to_scale] = scaler.transform(df_features[features_to_scale].values)
    
    X, y = create_sequences(df_scaled, features_to_scale, df_features)
    if X.size == 0: 
        return 0.0
        
    model = build_temp_model(X.shape[2])
    print(f"       -> Entraînement du modèle de test ({config.OPTIMIZATION_EPOCHS} époques)...", flush=True)
    history = model.fit(
        X, y,
        epochs=config.OPTIMIZATION_EPOCHS,
        batch_size=64,
        validation_split=0.15,
        verbose=0,
        shuffle=False   # IMPORTANT: pas de mélange temporel
    )
    
    accuracy = np.mean(history.history['val_accuracy'][-2:]) * 100
    del model
    tf.keras.backend.clear_session()
    
    print(f"      => Précision obtenue: {accuracy:.2f}%", flush=True)
    return accuracy

def run_optimization(symbol_to_optimize):
    print(f"\n--- LANCEMENT DE L'OPTIMISATION DES INDICATEURS POUR {symbol_to_optimize} ---", flush=True)
    df_base = get_historical_data(symbol_to_optimize, config.TIMEFRAME_TO_TRAIN, config.OPTIMIZATION_BARS)
    if df_base is None: 
        return

    config_data = config.load_config_data()
    all_possible_groups = config.get_all_feature_groups()
    
    checkpoint = config_data.get("optimizer_checkpoint", {}).get(symbol_to_optimize, {})
    if checkpoint:
        selected_groups = checkpoint.get("selected_groups", [])
        best_overall_accuracy = checkpoint.get("best_accuracy", 0.0)
        iteration_1_scores = checkpoint.get("iteration_1_scores", None)
        print(f"*** Reprise de l'optimisation à partir d'un checkpoint. Base: {selected_groups} ({best_overall_accuracy:.2f}%) ***", flush=True)
    else:
        selected_groups = []
        best_overall_accuracy = 0.0
        iteration_1_scores = None
        print("--- Démarrage d'une nouvelle session d'optimisation. ---", flush=True)
    
    iteration_count = len(selected_groups) + 1
    while True:
        iteration_results = {}
        
        must_run_full_iteration = False
        if (iteration_count == 1 and iteration_1_scores is None) or (iteration_count > 1 and iteration_1_scores is None):
            must_run_full_iteration = True
            print(f"\n--- Itération {iteration_count} (Classement en cours) ---", flush=True)
            print(f"OPTIMIZER_PROGRESS:Classement {iteration_count}...", flush=True)
            groups_to_test = [g for g in all_possible_groups if g not in selected_groups]
        else:
            print(f"\n--- Itération {iteration_count} (Base: {selected_groups}, Précision: {best_overall_accuracy:.2f}%) ---", flush=True)
            candidate_pool = [g for g in iteration_1_scores if g not in selected_groups]
            groups_to_test = candidate_pool[:TOP_N_CANDIDATES]
            print(f"--- Test des {len(groups_to_test)} meilleurs candidats restants ---", flush=True)
        
        if not groups_to_test:
            print("\n--- Plus aucun candidat à tester. Fin de l'optimisation. ---", flush=True)
            break

        for i, group in enumerate(groups_to_test):
            progress_msg_detail = f"It. {iteration_count}: Test {i+1}/{len(groups_to_test)}"
            print(f"OPTIMIZER_PROGRESS:{progress_msg_detail}", flush=True)
            accuracy = evaluate_feature_combination(symbol_to_optimize, selected_groups + [group], df_base)
            iteration_results[group] = accuracy
        
        if must_run_full_iteration:
            iteration_1_scores = {k: v for k, v in sorted(iteration_results.items(), key=lambda item: item[1], reverse=True)}

        if not iteration_results:
            print("\n--- AUCUNE AMÉLIORATION TROUVÉE. FIN DE L'OPTIMISATION. ---", flush=True)
            break

        best_addition = max(iteration_results, key=iteration_results.get)
        if iteration_results[best_addition] > best_overall_accuracy:
            best_overall_accuracy = iteration_results[best_addition]
            selected_groups.append(best_addition)
            print(f"*** Meilleur ajout: '{best_addition}' | Nouvelle précision: {best_overall_accuracy:.2f}% ***", flush=True)
            
            config_data.setdefault("optimizer_checkpoint", {}).setdefault(symbol_to_optimize, {})
            config_data["optimizer_checkpoint"][symbol_to_optimize] = {
                "selected_groups": selected_groups,
                "best_accuracy": best_overall_accuracy,
                "iteration_1_scores": iteration_1_scores
            }
            config.save_config_data(config_data)
            print("--- Checkpoint de progression sauvegardé. ---", flush=True)
        else:
            print("\n--- AUCUNE AMÉLIORATION TROUVÉE. FIN DE L'OPTIMISATION. ---", flush=True)
            break
        
        iteration_count += 1
            
    config_data.setdefault("optimized_feature_configs", {})[symbol_to_optimize] = {
        "best_groups": selected_groups, "accuracy": best_overall_accuracy
    }
    if "optimizer_checkpoint" in config_data and symbol_to_optimize in config_data["optimizer_checkpoint"]:
        del config_data["optimizer_checkpoint"][symbol_to_optimize]
        if not config_data["optimizer_checkpoint"]:
            del config_data["optimizer_checkpoint"]
    config.save_config_data(config_data)
    
    print(f"\n--- OPTIMISATION DES INDICATEURS TERMINÉE POUR {symbol_to_optimize} ---", flush=True)
    print(f"Meilleure combinaison sauvegardée: {selected_groups} ({best_overall_accuracy:.2f}%)", flush=True)
    print(f"OPTIMIZER_FINISHED:Terminé", flush=True)

def main():
    if len(sys.argv) < 2:
        print("ERREUR: Spécifiez un symbole.", flush=True)
        return
    symbol_to_optimize = sys.argv[1].upper()
    if not connect_to_mt5():
        sys.exit(1)
    run_optimization(symbol_to_optimize)
    import MetaTrader5 as mt5
    mt5.shutdown()

if __name__ == "__main__":
    main()