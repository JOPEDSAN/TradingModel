#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para ejecutar el pipeline completo de predicción de inversiones.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import traceback

# Importar configuración
try:
    from config import (
        MODEL_CONFIG, DATA_CONFIG, API_CONFIG, FILE_CONFIG, 
        DEFAULT_TICKERS, AVAILABLE_MODELS, LOGGING_CONFIG,
        validate_config, print_config
    )
except ImportError:
    print("ERROR: No se pudo importar el archivo de configuración. Usando configuración por defecto.")
    DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
    AVAILABLE_MODELS = ['lstm', 'gru', 'bilstm']

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Pipeline de predicción de inversiones')
    
    parser.add_argument('--download', action='store_true', help='Descargar datos históricos')
    parser.add_argument('--process', action='store_true', help='Procesar datos y calcular características')
    parser.add_argument('--train', action='store_true', help='Entrenar modelos')
    parser.add_argument('--predict', action='store_true', help='Hacer predicciones')
    parser.add_argument('--all', action='store_true', help='Ejecutar todo el pipeline')
    
    parser.add_argument('--tickers', nargs='+', default=None, help='Lista de tickers a procesar')
    parser.add_argument('--models', nargs='+', default=['lstm', 'gru', 'bilstm'], 
                        help='Lista de modelos a entrenar')
    parser.add_argument('--years', type=int, default=5, help='Años de datos históricos a descargar')
    parser.add_argument('--seq-length', type=int, default=60, 
                        help='Longitud de la secuencia para modelos')
    parser.add_argument('--horizon', type=int, default=1, 
                        help='Horizonte de predicción (días en el futuro)')
    parser.add_argument('--future-days', type=int, default=30, 
                        help='Días futuros a predecir')
    
    return parser.parse_args()

def main():
    try:
        # Validar configuración
        config_messages = validate_config()
        for message in config_messages:
            if message.startswith("ERROR"):
                logger.error(message)
                return 1
            elif message.startswith("WARNING"):
                logger.warning(message)
            else:
                logger.info(message)
        
        # Mostrar configuración
        print_config()
        
        args = parse_args()
        
        # Tickers por defecto si no se especifican
        if args.tickers is None:
            args.tickers = DEFAULT_TICKERS
        
        # Validar modelos
        invalid_models = [m for m in args.models if m not in AVAILABLE_MODELS]
        if invalid_models:
            logger.error(f"Modelos no válidos: {invalid_models}. Modelos disponibles: {AVAILABLE_MODELS}")
            return 1
        
        # Crear directorios si no existen
        directories = [FILE_CONFIG.data_dir, FILE_CONFIG.models_dir, 
                      FILE_CONFIG.results_dir, FILE_CONFIG.plots_dir]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Directorio creado: {directory}")
        
    except Exception as e:
        logger.error(f"Error en la inicialización: {str(e)}")
        logger.error(traceback.format_exc())
        return 1
    
    # Ejecutar pipeline completo si se indica --all
    if args.all:
        args.download = args.process = args.train = args.predict = True
    
    # 1. Descarga de datos
    if args.download:
        logger.info("Iniciando descarga de datos históricos...")
        from src.data_downloader import download_multiple_stocks, save_data
        
        # Calcular fechas
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365 * args.years)).strftime('%Y-%m-%d')
        
        # Descargar datos
        data_dict = download_multiple_stocks(args.tickers, start_date, end_date)
        save_data(data_dict)
        
        # Descargar datos económicos
        logger.info("Iniciando descarga de datos macroeconómicos...")
        try:
            from src.economic_data import get_economic_indicators, save_economic_data
            economic_data = get_economic_indicators(start_date, end_date)
            if economic_data is not None:
                save_economic_data(economic_data)
        except Exception as e:
            logger.error(f"Error al descargar datos económicos: {str(e)}")
        
        # Descargar y analizar noticias (solo último mes debido a limitaciones de API)
        logger.info("Iniciando descarga y análisis de noticias...")
        try:
            from src.news_analyzer import FinancialNewsAnalyzer
            news_start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            analyzer = FinancialNewsAnalyzer()
            analyzer.process_multiple_tickers(args.tickers, news_start, end_date)
        except Exception as e:
            logger.error(f"Error al procesar noticias: {str(e)}")
        
        logger.info("Descarga de datos completada.")
    
    # 2. Procesamiento de datos
    if args.process:
        logger.info("Iniciando procesamiento de datos...")
        from src.data_processor import load_data, create_features_for_all, prepare_dataset
        
        # Cargar datos
        data_dict = load_data()
        
        # Cargar datos económicos si existen
        economic_data = None
        try:
            if os.path.exists('data/economic_indicators.csv'):
                import pandas as pd
                logger.info("Cargando datos económicos...")
                economic_data = pd.read_csv('data/economic_indicators.csv', index_col=0)
        except Exception as e:
            logger.error(f"Error al cargar datos económicos: {str(e)}")
        
        # Cargar datos de sentimiento de noticias si existen
        sentiment_data = None
        try:
            if os.path.exists('data/news_sentiment_daily.csv'):
                import pandas as pd
                logger.info("Cargando datos de sentimiento de noticias...")
                sentiment_data = pd.read_csv('data/news_sentiment_daily.csv')
        except Exception as e:
            logger.error(f"Error al cargar datos de sentimiento: {str(e)}")
        
        # Calcular características incluyendo datos económicos y de noticias
        processed_data = create_features_for_all(data_dict, economic_data, sentiment_data)
        
        # Guardar datos procesados
        for ticker, data in processed_data.items():
            output_file = f"data/{ticker.replace('.', '-')}_processed.csv"
            data.to_csv(output_file, index=False)
        
        # Preparar datos para entrenamiento
        prepare_dataset(processed_data, 
                       sequence_length=args.seq_length, 
                       prediction_horizon=args.horizon, 
                       target_col='Close')
        
        logger.info("Procesamiento de datos completado.")
    
    # 3. Entrenamiento de modelos
    if args.train:
        logger.info("Iniciando entrenamiento de modelos...")
        from src.model_trainer import train_models_for_tickers
        
        # Entrenar modelos
        train_models_for_tickers(args.tickers, args.models)
        
        logger.info("Entrenamiento de modelos completado.")
    
    # 4. Predicciones
    if args.predict:
        logger.info("Realizando predicciones...")
        from src.model_trainer import StockPredictor
        
        # Realizar predicciones
        for ticker in args.tickers:
            for model_type in args.models:
                try:
                    predictor = StockPredictor(ticker, model_type)
                    predictor.evaluate()
                    predictor.plot_predictions()
                    predictor.predict_future(days=args.future_days)
                except Exception as e:
                    logger.error(f"Error al predecir con {model_type} para {ticker}: {str(e)}")
        
        logger.info("Predicciones completadas.")
    
    logger.info("Pipeline de predicción de inversiones completado.")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
