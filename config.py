#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Archivo de configuración global para el proyecto de predicción de inversiones.
"""

import os
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class ModelConfig:
    """Configuración de modelos."""
    sequence_length: int = 60
    prediction_horizon: int = 1
    batch_size: int = 32
    epochs: int = 100
    validation_split: float = 0.2
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    lstm_units: int = 50
    patience: int = 20
    reduce_lr_patience: int = 5
    min_lr: float = 0.0001

@dataclass
class DataConfig:
    """Configuración de datos."""
    train_test_split: float = 0.8
    default_years: int = 5
    target_column: str = 'Close'
    feature_range: tuple = (0, 1)
    interpolation_method: str = 'linear'

@dataclass
class APIConfig:
    """Configuración de APIs."""
    fred_api_key: str = os.getenv("FRED_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    finnhub_api_key: str = os.getenv("FINNHUB_API_KEY", "")
    alpha_vantage_api_key: str = os.getenv("ALPHAVANTAGE_API_KEY", "")

@dataclass
class FileConfig:
    """Configuración de archivos y directorios."""
    data_dir: str = "data"
    models_dir: str = "models"
    plots_dir: str = "plots"
    results_dir: str = "results"
    logs_dir: str = "logs"

# Tickers por defecto
DEFAULT_TICKERS = [
    # Índices principales
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones
    "^IXIC",  # NASDAQ
    
    # Grandes tecnológicas
    "AAPL",   # Apple
    "MSFT",   # Microsoft
    "GOOGL",  # Alphabet (Google)
    "AMZN",   # Amazon
    "META",   # Meta (Facebook)
    "TSLA",   # Tesla
]

# Tipos de modelos disponibles
AVAILABLE_MODELS = ['lstm', 'gru', 'bilstm']

# Configuración de logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'pipeline.log',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Instancias de configuración
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
API_CONFIG = APIConfig()
FILE_CONFIG = FileConfig()

def validate_config():
    """
    Valida la configuración del proyecto.
    
    Returns:
        List[str]: Lista de mensajes de validación (warnings/errors).
    """
    messages = []
    
    # Validar APIs
    if not API_CONFIG.fred_api_key:
        messages.append("WARNING: FRED_API_KEY no configurada. Funcionalidades de datos económicos limitadas.")
    
    if not API_CONFIG.news_api_key:
        messages.append("WARNING: NEWS_API_KEY no configurada. Funcionalidades de análisis de noticias limitadas.")
    
    if not API_CONFIG.finnhub_api_key:
        messages.append("WARNING: FINNHUB_API_KEY no configurada. Funcionalidades de noticias limitadas.")
    
    # Validar directorios
    for directory in [FILE_CONFIG.data_dir, FILE_CONFIG.models_dir, 
                     FILE_CONFIG.plots_dir, FILE_CONFIG.results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            messages.append(f"INFO: Directorio creado: {directory}")
    
    # Validar configuración de modelo
    if MODEL_CONFIG.sequence_length <= 0:
        messages.append("ERROR: sequence_length debe ser mayor a 0")
    
    if MODEL_CONFIG.batch_size <= 0:
        messages.append("ERROR: batch_size debe ser mayor a 0")
    
    if not 0 < DATA_CONFIG.train_test_split < 1:
        messages.append("ERROR: train_test_split debe estar entre 0 y 1")
    
    return messages

def print_config():
    """Imprime la configuración actual."""
    print("=== Configuración del Proyecto ===")
    print(f"Modelo - Sequence Length: {MODEL_CONFIG.sequence_length}")
    print(f"Modelo - Batch Size: {MODEL_CONFIG.batch_size}")
    print(f"Modelo - Epochs: {MODEL_CONFIG.epochs}")
    print(f"Datos - Train/Test Split: {DATA_CONFIG.train_test_split}")
    print(f"Datos - Años por defecto: {DATA_CONFIG.default_years}")
    print(f"APIs configuradas: {len([k for k, v in vars(API_CONFIG).items() if v])}")
    print(f"Tickers por defecto: {len(DEFAULT_TICKERS)}")
    print("================================")