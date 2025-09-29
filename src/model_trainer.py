#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para definir y entrenar modelos de predicción.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import joblib
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

# Configurar el estilo de las gráficas
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = (15, 8)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class StockPredictor:
    def __init__(self, ticker, model_type='lstm', sequence_length=60, prediction_horizon=1):
        """
        Inicializa el predictor de acciones.
        
        Args:
            ticker (str): Símbolo del ticker.
            model_type (str): Tipo de modelo ('lstm', 'gru', 'bilstm').
            sequence_length (int): Longitud de la secuencia de entrada.
            prediction_horizon (int): Horizonte de predicción (días en el futuro).
        """
        self.ticker = ticker
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model = None
        self.history = None
        self.scaler = None
        self.feature_cols = None
        
        # Cargar datos si existen
        self.load_data()
    
    def load_data(self):
        """Carga los datos de entrenamiento y prueba."""
        try:
            self.X_train = np.load(f"data/{self.ticker}_X_train.npy")
            self.y_train = np.load(f"data/{self.ticker}_y_train.npy")
            self.X_test = np.load(f"data/{self.ticker}_X_test.npy")
            self.y_test = np.load(f"data/{self.ticker}_y_test.npy")
            
            # Cargar el scaler
            self.scaler = joblib.load(f"models/scaler_{self.ticker}.pkl")
            
            # Cargar las características
            with open(f"models/features_{self.ticker}.txt", "r") as f:
                self.feature_cols = f.read().splitlines()
            
            # Obtener el índice de 'Close'
            self.close_idx = self.feature_cols.index('Close')
            
            logger.info(f"Datos cargados para {self.ticker}")
        except Exception as e:
            logger.error(f"Error al cargar los datos para {self.ticker}: {str(e)}")
            raise
    
    def build_model(self):
        """Construye la arquitectura del modelo."""
        input_shape = (self.X_train.shape[1], self.X_train.shape[2])
        
        model = Sequential()
        
        if self.model_type == 'lstm':
            model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
        
        elif self.model_type == 'gru':
            model.add(GRU(units=50, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(GRU(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(GRU(units=50))
            model.add(Dropout(0.2))
        
        elif self.model_type == 'bilstm':
            model.add(Bidirectional(LSTM(units=50, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50, return_sequences=True)))
            model.add(Dropout(0.2))
            model.add(Bidirectional(LSTM(units=50)))
            model.add(Dropout(0.2))
        
        # Capa de salida
        model.add(Dense(units=1))
        
        # Compilar el modelo
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        
        self.model = model
        logger.info(f"Modelo {self.model_type} construido para {self.ticker}")
        
        return model
    
    def train(self, epochs=100, batch_size=32, validation_split=0.2):
        """
        Entrena el modelo.
        
        Args:
            epochs (int): Número de épocas de entrenamiento.
            batch_size (int): Tamaño del lote.
            validation_split (float): Proporción de datos para validación.
            
        Returns:
            history: Historial de entrenamiento.
        """
        if self.model is None:
            self.build_model()
        
        # Definir callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True
        )
        
        model_checkpoint = ModelCheckpoint(
            f"models/{self.ticker}_{self.model_type}_model.h5",
            save_best_only=True,
            monitor='val_loss'
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Entrenar el modelo
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping, model_checkpoint, reduce_lr],
            verbose=1
        )
        
        self.history = history
        
        # Guardar el historial de entrenamiento
        history_dict = {key: [float(val) for val in history.history[key]] for key in history.history.keys()}
        with open(f"models/{self.ticker}_{self.model_type}_history.json", "w") as f:
            json.dump(history_dict, f)
        
        logger.info(f"Modelo {self.model_type} entrenado para {self.ticker}")
        
        return history
    
    def load_trained_model(self):
        """Carga un modelo entrenado."""
        try:
            self.model = load_model(f"models/{self.ticker}_{self.model_type}_model.h5")
            logger.info(f"Modelo cargado desde models/{self.ticker}_{self.model_type}_model.h5")
        except Exception as e:
            logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
    
    def predict(self, X=None):
        """
        Hace predicciones con el modelo.
        
        Args:
            X (numpy.array, optional): Datos de entrada. Si es None, usa X_test.
            
        Returns:
            numpy.array: Predicciones.
        """
        if self.model is None:
            try:
                self.load_trained_model()
            except:
                logger.error("No se pudo cargar el modelo. Asegúrate de entrenar el modelo primero.")
                return None
        
        if X is None:
            X = self.X_test
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def evaluate(self):
        """
        Evalúa el rendimiento del modelo.
        
        Returns:
            dict: Métricas de evaluación.
        """
        if self.model is None:
            try:
                self.load_trained_model()
            except:
                logger.error("No se pudo cargar el modelo. Asegúrate de entrenar el modelo primero.")
                return None
        
        # Hacer predicciones
        y_pred = self.predict()
        
        # Calcular métricas
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2)
        }
        
        # Guardar las métricas
        with open(f"models/{self.ticker}_{self.model_type}_metrics.json", "w") as f:
            json.dump(metrics, f)
        
        logger.info(f"Evaluación del modelo {self.model_type} para {self.ticker}:")
        logger.info(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}, R²: {r2}")
        
        return metrics
    
    def plot_training_history(self):
        """Grafica el historial de entrenamiento."""
        if self.history is None:
            try:
                with open(f"models/{self.ticker}_{self.model_type}_history.json", "r") as f:
                    self.history = json.load(f)
            except:
                logger.error("No se encontró el historial de entrenamiento.")
                return
        
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'])
        plt.plot(self.history['val_loss'])
        plt.title('Pérdida del modelo')
        plt.ylabel('Pérdida')
        plt.xlabel('Época')
        plt.legend(['Entrenamiento', 'Validación'], loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f"plots/{self.ticker}_{self.model_type}_history.png")
        plt.close()
        
        logger.info(f"Gráfico de historial guardado en plots/{self.ticker}_{self.model_type}_history.png")
    
    def plot_predictions(self):
        """Grafica las predicciones vs valores reales."""
        if self.model is None:
            try:
                self.load_trained_model()
            except:
                logger.error("No se pudo cargar el modelo. Asegúrate de entrenar el modelo primero.")
                return
        
        # Hacer predicciones
        y_pred = self.predict()
        
        # Invertir la normalización para obtener los precios reales
        # Crear un array con ceros en todas las columnas excepto la de cierre
        dummy_test = np.zeros((len(self.y_test), len(self.feature_cols)))
        dummy_test[:, self.close_idx] = self.y_test.flatten()
        # Invertir la normalización
        y_test_inv = self.scaler.inverse_transform(dummy_test)[:, self.close_idx]
        
        # Hacer lo mismo para las predicciones
        dummy_pred = np.zeros((len(y_pred), len(self.feature_cols)))
        dummy_pred[:, self.close_idx] = y_pred.flatten()
        y_pred_inv = self.scaler.inverse_transform(dummy_pred)[:, self.close_idx]
        
        # Crear un DataFrame con las fechas, valores reales y predicciones
        try:
            dates_test = np.load(f"data/{self.ticker}_dates_test.npy", allow_pickle=True)
            df_pred = pd.DataFrame({
                'Date': dates_test[:len(y_test_inv)],
                'Real': y_test_inv,
                'Predicción': y_pred_inv
            })
        except:
            # Si no hay fechas guardadas, usar índices
            df_pred = pd.DataFrame({
                'Index': range(len(y_test_inv)),
                'Real': y_test_inv,
                'Predicción': y_pred_inv
            })
        
        # Graficar
        plt.figure(figsize=(15, 8))
        plt.plot(df_pred['Real'], label='Valores reales')
        plt.plot(df_pred['Predicción'], label='Predicciones')
        plt.title(f'Predicción de precios para {self.ticker} usando {self.model_type.upper()}')
        plt.xlabel('Tiempo')
        plt.ylabel('Precio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{self.ticker}_{self.model_type}_predictions.png")
        plt.close()
        
        # Guardar las predicciones
        df_pred.to_csv(f"results/{self.ticker}_{self.model_type}_predictions.csv", index=False)
        
        logger.info(f"Gráfico de predicciones guardado en plots/{self.ticker}_{self.model_type}_predictions.png")
        logger.info(f"Predicciones guardadas en results/{self.ticker}_{self.model_type}_predictions.csv")
    
    def predict_future(self, days=30):
        """
        Predice el precio futuro para los próximos días.
        
        Args:
            days (int): Número de días a predecir.
            
        Returns:
            pandas.DataFrame: DataFrame con las predicciones.
        """
        if self.model is None:
            try:
                self.load_trained_model()
            except:
                logger.error("No se pudo cargar el modelo. Asegúrate de entrenar el modelo primero.")
                return None
        
        # Obtener la última secuencia
        last_sequence = self.X_test[-1].copy()
        
        # Lista para almacenar predicciones
        predictions = []
        
        # Predecir para cada día
        for _ in range(days):
            # Hacer una predicción
            pred = self.model.predict(np.array([last_sequence]))
            
            # Agregar la predicción a la lista
            predictions.append(pred[0, 0])
            
            # Actualizar la secuencia para la próxima predicción
            # Crear un nuevo punto con la predicción en la columna de cierre
            new_point = last_sequence[-1].copy()
            new_point[self.close_idx] = pred[0, 0]
            
            # Desplazar la secuencia y agregar el nuevo punto
            last_sequence = np.vstack([last_sequence[1:], new_point])
        
        # Convertir las predicciones a precios reales
        dummy = np.zeros((len(predictions), len(self.feature_cols)))
        dummy[:, self.close_idx] = predictions
        future_prices = self.scaler.inverse_transform(dummy)[:, self.close_idx]
        
        # Crear fechas futuras
        try:
            dates_test = np.load(f"data/{self.ticker}_dates_test.npy", allow_pickle=True)
            last_date = pd.to_datetime(dates_test[-1])
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        except:
            last_date = datetime.now()
            future_dates = [last_date + pd.Timedelta(days=i+1) for i in range(days)]
        
        # Crear DataFrame con las predicciones
        df_future = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices
        })
        
        # Guardar las predicciones futuras
        df_future.to_csv(f"results/{self.ticker}_{self.model_type}_future_predictions.csv", index=False)
        
        # Graficar
        plt.figure(figsize=(15, 8))
        plt.plot(df_future['Date'], df_future['Predicted_Price'], label='Predicciones futuras')
        plt.title(f'Predicción futura de precios para {self.ticker} usando {self.model_type.upper()}')
        plt.xlabel('Fecha')
        plt.ylabel('Precio')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"plots/{self.ticker}_{self.model_type}_future_predictions.png")
        plt.close()
        
        logger.info(f"Predicciones futuras guardadas en results/{self.ticker}_{self.model_type}_future_predictions.csv")
        logger.info(f"Gráfico de predicciones futuras guardado en plots/{self.ticker}_{self.model_type}_future_predictions.png")
        
        return df_future

def train_models_for_tickers(tickers, model_types=['lstm', 'gru', 'bilstm']):
    """
    Entrena modelos para múltiples tickers.
    
    Args:
        tickers (list): Lista de tickers.
        model_types (list): Lista de tipos de modelos.
    """
    for ticker in tickers:
        for model_type in model_types:
            try:
                logger.info(f"Entrenando modelo {model_type} para {ticker}")
                predictor = StockPredictor(ticker, model_type)
                predictor.train()
                predictor.evaluate()
                predictor.plot_training_history()
                predictor.plot_predictions()
                predictor.predict_future()
            except Exception as e:
                logger.error(f"Error al entrenar el modelo {model_type} para {ticker}: {str(e)}")

def main():
    # Definir los tickers para los que se quiere entrenar modelos
    tickers = [
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        "AMZN",   # Amazon
        "META",   # Meta (Facebook)
        "TSLA",   # Tesla
    ]
    
    # Definir los tipos de modelos a entrenar
    model_types = ['lstm', 'gru', 'bilstm']
    
    # Entrenar modelos para cada ticker
    train_models_for_tickers(tickers, model_types)

if __name__ == "__main__":
    main()
