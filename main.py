#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para ejecutar el pipeline completo de predicción de inversiones.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta

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
    args = parse_args()
    
    # Tickers por defecto si no se especifican
    if args.tickers is None:
        args.tickers = [
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
    
    # Crear directorios si no existen
    for directory in ['data', 'models', 'results', 'plots']:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
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
        
        logger.info("Descarga de datos completada.")
    
    # 2. Procesamiento de datos
    if args.process:
        logger.info("Iniciando procesamiento de datos...")
        from src.data_processor import load_data, create_features_for_all, prepare_dataset
        
        # Cargar datos
        data_dict = load_data()
        
        # Calcular características
        processed_data = create_features_for_all(data_dict)
        
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

if __name__ == "__main__":
    main()
