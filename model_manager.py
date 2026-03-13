# model_manager.py - Version simplifiée sans scheduler personnalisé

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, 
    Bidirectional, Conv1D, MaxPooling1D, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import config
import os

class AIBrain:
    """
    Version améliorée avec Batch Normalization (sans scheduler personnalisé).
    """
    def __init__(self, symbol: str, num_features: int, model_variant: int = 0):
        if not symbol:
            raise ValueError("Un nom de symbole est requis pour initialiser AIBrain.")
        if num_features <= 0:
            raise ValueError("Le nombre de features doit être un entier positif.")
            
        self.symbol = symbol
        self.num_features = num_features
        self.model_variant = model_variant
        self.model_path = os.path.join(config.MODEL_FOLDER, f"{self.symbol}_v{model_variant}.keras")
        self.model = self._load_or_build_model()

    def _load_or_build_model(self):
        """Charge un modèle existant ou en construit un nouveau."""
        if os.path.exists(self.model_path):
            try:
                print(f"Chargement du modèle existant pour {self.symbol} (variante {self.model_variant})...")
                model = load_model(self.model_path)
                if model.input_shape[2] != self.num_features or model.output_shape[1] != 2:
                    print(f"AVERTISSEMENT: Architecture incompatible. Reconstruction du modèle.")
                    return self._build_model()
                print("Modèle chargé avec succès.")
                return model
            except Exception as e:
                print(f"Erreur lors du chargement: {e}. Construction d'un nouveau modèle.")
                return self._build_model()
        else:
            print(f"Construction d'un nouveau modèle pour {self.symbol} (variante {self.model_variant}).")
            return self._build_model()

    def _build_model(self):
        """Construit l'architecture de modèle appropriée en fonction de la variante."""
        if self.model_variant == 0:
            return self._build_attention_model()
        elif self.model_variant == 1:
            return self._build_cnn_lstm_model()
        else:
            return self._build_bidirectional_model()

    def _build_attention_model(self):
        """Modèle avec mécanisme d'attention et Batch Norm."""
        inputs = Input(shape=(config.SEQUENCE_LENGTH, self.num_features))
        
        lstm1 = LSTM(128, return_sequences=True, kernel_regularizer=l2(0.01))(inputs)
        lstm1 = BatchNormalization()(lstm1)
        lstm1 = Dropout(0.2)(lstm1)
        
        attention = MultiHeadAttention(num_heads=8, key_dim=64)(lstm1, lstm1)
        attention = LayerNormalization()(attention + lstm1)
        
        lstm2 = LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01))(attention)
        lstm2 = BatchNormalization()(lstm2)
        lstm2 = Dropout(0.3)(lstm2)
        
        lstm3 = LSTM(32, return_sequences=False)(lstm2)
        lstm3 = BatchNormalization()(lstm3)
        lstm3 = Dropout(0.3)(lstm3)
        
        dense1 = Dense(64, kernel_regularizer=l2(0.01))(lstm3)
        dense1 = BatchNormalization()(dense1)
        dense1 = Activation('relu')(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32)(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Activation('relu')(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(2, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Modèle Attention avec Batch Norm construit pour {self.symbol}")
        return model

    def _build_cnn_lstm_model(self):
        """Modèle hybride CNN-LSTM avec Batch Norm."""
        inputs = Input(shape=(config.SEQUENCE_LENGTH, self.num_features))
        
        cnn = Conv1D(filters=64, kernel_size=3)(inputs)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        cnn = MaxPooling1D(pool_size=2)(cnn)
        
        cnn = Conv1D(filters=32, kernel_size=3)(cnn)
        cnn = BatchNormalization()(cnn)
        cnn = Activation('relu')(cnn)
        
        lstm = LSTM(64, return_sequences=True)(cnn)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.3)(lstm)
        
        lstm = LSTM(32)(lstm)
        lstm = BatchNormalization()(lstm)
        lstm = Dropout(0.3)(lstm)
        
        dense = Dense(32)(lstm)
        dense = BatchNormalization()(dense)
        dense = Activation('relu')(dense)
        dense = Dropout(0.2)(dense)
        
        outputs = Dense(2, activation='softmax')(dense)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Modèle CNN-LSTM avec Batch Norm construit pour {self.symbol}")
        return model

    def _build_bidirectional_model(self):
        """Modèle LSTM bidirectionnel avec Batch Norm."""
        inputs = Input(shape=(config.SEQUENCE_LENGTH, self.num_features))
        
        bilstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
        bilstm1 = BatchNormalization()(bilstm1)
        bilstm1 = Dropout(0.3)(bilstm1)
        
        bilstm2 = Bidirectional(LSTM(32))(bilstm1)
        bilstm2 = BatchNormalization()(bilstm2)
        bilstm2 = Dropout(0.3)(bilstm2)
        
        dense1 = Dense(64)(bilstm2)
        dense1 = BatchNormalization()(dense1)
        dense1 = Activation('relu')(dense1)
        dense1 = Dropout(0.3)(dense1)
        
        dense2 = Dense(32)(dense1)
        dense2 = BatchNormalization()(dense2)
        dense2 = Activation('relu')(dense2)
        dense2 = Dropout(0.2)(dense2)
        
        outputs = Dense(2, activation='softmax')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        print(f"Modèle Bidirectionnel avec Batch Norm construit pour {self.symbol}")
        return model

class EnsembleAIBrain:
    """Ensemble de plusieurs modèles pour des prédictions plus robustes"""
    def __init__(self, symbol: str, num_features: int):
        self.symbol = symbol
        self.num_features = num_features
        self.models = []
        for i in range(config.ENSEMBLE_MODELS):
            brain = AIBrain(symbol, num_features, model_variant=i)
            self.models.append(brain)
    
    def predict(self, X, use_weighted_average=True, **kwargs):
        predictions = []
        weights = []
        for i, brain in enumerate(self.models):
            # Passe les kwargs (ex: batch_size) au predict Keras
            pred = brain.model.predict(X, verbose=0, **kwargs)
            predictions.append(pred)
            if use_weighted_average:
                confidence = np.max(pred, axis=1).mean()
                weights.append(confidence)
            else:
                weights.append(1.0)
        weights = np.array(weights)
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(self.models)) / len(self.models)
            
        weighted_predictions = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, weights):
            weighted_predictions += pred * weight
        return weighted_predictions
    
    def fit(self, X, y, **kwargs):
        histories = []
        for brain in self.models:
            print(f"Entraînement du modèle {brain.model_variant} pour {self.symbol}...")
            history = brain.model.fit(X, y, **kwargs)
            histories.append(history)
            brain.model.save(brain.model_path)
        return histories
    
    def save_all(self):
        for brain in self.models:
            brain.model.save(brain.model_path)