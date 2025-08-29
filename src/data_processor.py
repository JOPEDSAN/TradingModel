#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para procesar y preparar los datos para el entrenamiento.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import glob
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_processing.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_data(data_dir='data'):
    """
    Carga todos los archivos CSV de la carpeta de datos.
    
    Args:
        data_dir (str): Directorio donde se encuentran los datos.
        
    Returns:
        dict: Diccionario con los DataFrames de cada ticker.
    """
    all_data = {}
    
    csv_files = glob.glob(os.path.join(data_dir, "*_data.csv"))
    
    for file in csv_files:
        ticker = os.path.basename(file).split('_')[0].replace('-', '.')
        try:
            data = pd.read_csv(file)
            all_data[ticker] = data
            logger.info(f"Datos cargados para {ticker}: {len(data)} filas")
        except Exception as e:
            logger.error(f"Error al cargar {file}: {str(e)}")
    
    return all_data

def calculate_features(df):
    """
    Calcula características técnicas adicionales para los datos.
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos históricos.
        
    Returns:
        pandas.DataFrame: DataFrame con las características adicionales.
    """
    # Asegurarse de que 'date' está en formato datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Hacer una copia para evitar warnings
    data = df.copy()
    
    # Precios
    data['return'] = data['Close'].pct_change() # Retorno diario
    data['log_return'] = np.log(data['Close'] / data['Close'].shift(1)) # Log-retorno
    
    # Medias móviles
    data['sma_5'] = data['Close'].rolling(window=5).mean()
    data['sma_10'] = data['Close'].rolling(window=10).mean()
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['sma_50'] = data['Close'].rolling(window=50).mean()
    data['sma_200'] = data['Close'].rolling(window=200).mean()
    
    # Cruce de medias móviles
    data['sma_5_10'] = data['sma_5'] - data['sma_10']
    data['sma_10_20'] = data['sma_10'] - data['sma_20']
    data['sma_20_50'] = data['sma_20'] - data['sma_50']
    data['sma_50_200'] = data['sma_50'] - data['sma_200']
    
    # Volatilidad
    data['volatility_5'] = data['log_return'].rolling(window=5).std()
    data['volatility_10'] = data['log_return'].rolling(window=10).std()
    data['volatility_20'] = data['log_return'].rolling(window=20).std()
    
    # Momentum
    data['momentum_5'] = data['Close'] / data['Close'].shift(5) - 1
    data['momentum_10'] = data['Close'] / data['Close'].shift(10) - 1
    data['momentum_20'] = data['Close'] / data['Close'].shift(20) - 1
    
    # RSI (Relative Strength Index)
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['rsi_14'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    data['ema_12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['ema_26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['macd'] = data['ema_12'] - data['ema_26']
    data['macd_signal'] = data['macd'].ewm(span=9, adjust=False).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # Bollinger Bands
    data['bb_middle'] = data['Close'].rolling(window=20).mean()
    data['bb_std'] = data['Close'].rolling(window=20).std()
    data['bb_upper'] = data['bb_middle'] + 2 * data['bb_std']
    data['bb_lower'] = data['bb_middle'] - 2 * data['bb_std']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    
    # Características de volumen
    data['volume_change'] = data['Volume'].pct_change()
    data['volume_sma_5'] = data['Volume'].rolling(window=5).mean()
    data['volume_sma_10'] = data['Volume'].rolling(window=10).mean()
    data['volume_ratio'] = data['Volume'] / data['volume_sma_5']
    
    # Limpiar NaN
    data = data.dropna()
    
    return data

def create_features_for_all(data_dict):
    """
    Calcula características para todos los tickers.
    
    Args:
        data_dict (dict): Diccionario con los DataFrames de cada ticker.
        
    Returns:
        dict: Diccionario con los DataFrames procesados.
    """
    processed_data = {}
    
    for ticker, data in data_dict.items():
        try:
            processed = calculate_features(data)
            processed_data[ticker] = processed
            logger.info(f"Características calculadas para {ticker}: {len(processed)} filas")
        except Exception as e:
            logger.error(f"Error al procesar {ticker}: {str(e)}")
    
    return processed_data

def create_sequences(data, sequence_length=60, prediction_horizon=1, target_col='Close'):
    """
    Crea secuencias para entrenamiento de modelos de series temporales.
    
    Args:
        data (pandas.DataFrame): DataFrame con los datos procesados.
        sequence_length (int): Longitud de la secuencia de entrada.
        prediction_horizon (int): Horizonte de predicción (días en el futuro).
        target_col (str): Columna objetivo para la predicción.
        
    Returns:
        tuple: X (secuencias de entrada), y (valores objetivo)
    """
    X, y = [], []
    
    # Seleccionar todas las columnas numéricas para las características
    feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Eliminar 'date' y 'ticker' si existen en feature_cols
    feature_cols = [col for col in feature_cols if col not in ['date', 'ticker']]
    
    # Convertir DataFrame a numpy array
    data_array = data[feature_cols].values
    
    # Normalizar los datos
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_array)
    
    # Obtener el índice de la columna objetivo
    target_idx = feature_cols.index(target_col)
    
    # Crear secuencias
    for i in range(len(data_scaled) - sequence_length - prediction_horizon + 1):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(data_scaled[i + sequence_length + prediction_horizon - 1, target_idx])
    
    return np.array(X), np.array(y), scaler, feature_cols

def prepare_dataset(processed_data, sequence_length=60, prediction_horizon=1, target_col='Close'):
    """
    Prepara los conjuntos de datos para entrenamiento y prueba.
    
    Args:
        processed_data (dict): Diccionario con los DataFrames procesados.
        sequence_length (int): Longitud de la secuencia de entrada.
        prediction_horizon (int): Horizonte de predicción (días en el futuro).
        target_col (str): Columna objetivo para la predicción.
        
    Returns:
        dict: Diccionario con los datos preparados para cada ticker.
    """
    prepared_data = {}
    
    for ticker, data in processed_data.items():
        try:
            X, y, scaler, feature_cols = create_sequences(
                data, sequence_length, prediction_horizon, target_col
            )
            
            # Guardar el scaler para uso posterior
            joblib.dump(scaler, f"models/scaler_{ticker}.pkl")
            
            # Guardar la lista de características
            with open(f"models/features_{ticker}.txt", "w") as f:
                f.write("\n".join(feature_cols))
            
            # Dividir en entrenamiento y prueba (80% - 20%)
            train_size = int(len(X) * 0.8)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            prepared_data[ticker] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_test': X_test,
                'y_test': y_test,
                'scaler': scaler,
                'feature_cols': feature_cols,
                'dates_train': data['date'].iloc[sequence_length:train_size + sequence_length].values,
                'dates_test': data['date'].iloc[train_size + sequence_length:train_size + sequence_length + len(X_test)].values
            }
            
            logger.info(f"Datos preparados para {ticker}: {len(X_train)} ejemplos de entrenamiento, {len(X_test)} ejemplos de prueba")
            
            # Guardar los datos preparados
            np.save(f"data/{ticker}_X_train.npy", X_train)
            np.save(f"data/{ticker}_y_train.npy", y_train)
            np.save(f"data/{ticker}_X_test.npy", X_test)
            np.save(f"data/{ticker}_y_test.npy", y_test)
            
        except Exception as e:
            logger.error(f"Error al preparar los datos para {ticker}: {str(e)}")
    
    return prepared_data

def main():
    # Cargar los datos
    data_dict = load_data()
    
    # Calcular características
    processed_data = create_features_for_all(data_dict)
    
    # Guardar los datos procesados
    for ticker, data in processed_data.items():
        output_file = f"data/{ticker.replace('.', '-')}_processed.csv"
        data.to_csv(output_file, index=False)
        logger.info(f"Datos procesados guardados en {output_file}")
    
    # Preparar los datos para el entrenamiento
    prepare_dataset(processed_data, sequence_length=60, prediction_horizon=1, target_col='Close')

if __name__ == "__main__":
    main()
