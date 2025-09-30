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

def get_asset_groups():
    """
    Define los grupos de activos disponibles para análisis.
    
    Returns:
        dict: Diccionario con diferentes categorías de activos
    """
    return {
        'indices': {
            'description': 'Índices bursátiles principales',
            'tickers': [
                "^GSPC",   # S&P 500
                "^DJI",    # Dow Jones Industrial Average
                "^IXIC",   # NASDAQ Composite
                "^RUT",    # Russell 2000
                "^VIX",    # CBOE Volatility Index
                "^FTSE",   # FTSE 100 (Reino Unido)
                "^GDAXI",  # DAX (Alemania)
                "^FCHI",   # CAC 40 (Francia)
                "^N225",   # Nikkei 225 (Japón)
                "^HSI",    # Hang Seng (Hong Kong)
            ]
        },
        'stocks': {
            'description': 'Acciones individuales principales',
            'tickers': [
                # FAANG + Microsoft
                "AAPL",    # Apple
                "AMZN",    # Amazon
                "META",    # Meta (Facebook)
                "NFLX",    # Netflix
                "GOOGL",   # Alphabet (Google)
                "MSFT",    # Microsoft
                
                # Otras tecnológicas importantes
                "TSLA",    # Tesla
                "NVDA",    # NVIDIA
                "AMD",     # Advanced Micro Devices
                "INTC",    # Intel
                
                # Financieras
                "JPM",     # JPMorgan Chase
                "BAC",     # Bank of America
                "WFC",     # Wells Fargo
                
                # Otros sectores
                "JNJ",     # Johnson & Johnson
                "PG",      # Procter & Gamble
                "KO",      # Coca-Cola
            ]
        },
        'etfs': {
            'description': 'ETFs principales',
            'tickers': [
                "SPY",     # SPDR S&P 500 ETF
                "QQQ",     # Invesco QQQ (NASDAQ-100)
                "VTI",     # Vanguard Total Stock Market ETF
                "IWM",     # iShares Russell 2000 ETF
                "EFA",     # iShares MSCI EAFE ETF (Internacional)
                "EEM",     # iShares MSCI Emerging Markets ETF
                "GLD",     # SPDR Gold Shares
                "TLT",     # iShares 20+ Year Treasury Bond ETF
                "HYG",     # iShares iBoxx High Yield Corporate Bond ETF
                "XLF",     # Financial Select Sector SPDR Fund
                "XLK",     # Technology Select Sector SPDR Fund
                "XLE",     # Energy Select Sector SPDR Fund
            ]
        },
        'crypto': {
            'description': 'Criptomonedas principales (a través de Yahoo Finance)',
            'tickers': [
                "BTC-USD", # Bitcoin
                "ETH-USD", # Ethereum
                "BNB-USD", # Binance Coin
                "ADA-USD", # Cardano
                "XRP-USD", # XRP
                "SOL-USD", # Solana
                "DOGE-USD",# Dogecoin
                "DOT-USD", # Polkadot
                "AVAX-USD",# Avalanche
                "MATIC-USD",# Polygon
            ]
        },
        'commodities': {
            'description': 'Materias primas y futuros',
            'tickers': [
                "GC=F",    # Gold Futures
                "SI=F",    # Silver Futures
                "CL=F",    # Crude Oil Futures
                "NG=F",    # Natural Gas Futures
                "HG=F",    # Copper Futures
                "ZC=F",    # Corn Futures
                "ZW=F",    # Wheat Futures
                "ZS=F",    # Soybean Futures
                "KC=F",    # Coffee Futures
                "CT=F",    # Cotton Futures
            ]
        }
    }

def select_tickers(args):
    """
    Selecciona los tickers basándose en los argumentos proporcionados.
    
    Args:
        args: Argumentos de línea de comandos
        
    Returns:
        list: Lista de tickers seleccionados
    """
    asset_groups = get_asset_groups()
    selected_tickers = []
    
    # Si se especificaron tickers específicos, usarlos
    if args.tickers:
        return args.tickers
    
    # Recopilar tickers según las opciones seleccionadas
    if args.indices:
        selected_tickers.extend(asset_groups['indices']['tickers'])
        logger.info(f"Añadidos {len(asset_groups['indices']['tickers'])} índices bursátiles")
    
    if args.stocks:
        selected_tickers.extend(asset_groups['stocks']['tickers'])
        logger.info(f"Añadidas {len(asset_groups['stocks']['tickers'])} acciones individuales")
    
    if args.etfs:
        selected_tickers.extend(asset_groups['etfs']['tickers'])
        logger.info(f"Añadidos {len(asset_groups['etfs']['tickers'])} ETFs")
    
    if args.crypto:
        selected_tickers.extend(asset_groups['crypto']['tickers'])
        logger.info(f"Añadidas {len(asset_groups['crypto']['tickers'])} criptomonedas")
    
    if args.commodities:
        selected_tickers.extend(asset_groups['commodities']['tickers'])
        logger.info(f"Añadidas {len(asset_groups['commodities']['tickers'])} materias primas")
    
    # Si no se seleccionó nada, usar índices principales por defecto
    if not selected_tickers:
        selected_tickers = asset_groups['indices']['tickers'][:5]  # Top 5 índices
        logger.info("No se especificaron activos. Usando índices principales por defecto.")
    
    # Eliminar duplicados manteniendo el orden
    seen = set()
    unique_tickers = []
    for ticker in selected_tickers:
        if ticker not in seen:
            seen.add(ticker)
            unique_tickers.append(ticker)
    
    # Aplicar límite máximo para evitar sobrecarga
    try:
        from config import ASSET_SELECTION_CONFIG
        max_tickers = ASSET_SELECTION_CONFIG.get('max_tickers_per_run', 20)
        if len(unique_tickers) > max_tickers:
            logger.warning(f"Se han seleccionado {len(unique_tickers)} activos, limitando a {max_tickers} para evitar sobrecarga")
            unique_tickers = unique_tickers[:max_tickers]
    except ImportError:
        # Si no se puede importar config, usar límite por defecto
        if len(unique_tickers) > 20:
            logger.warning(f"Limitando a 20 activos para evitar sobrecarga")
            unique_tickers = unique_tickers[:20]
    
    return unique_tickers

def print_selected_assets(tickers):
    """
    Muestra información sobre los activos seleccionados.
    
    Args:
        tickers (list): Lista de tickers seleccionados
    """
    asset_groups = get_asset_groups()
    
    print(f"\n📊 ACTIVOS SELECCIONADOS ({len(tickers)} en total):")
    print("=" * 50)
    
    # Categorizar los tickers seleccionados
    categorized = {category: [] for category in asset_groups.keys()}
    uncategorized = []
    
    for ticker in tickers:
        found = False
        for category, group in asset_groups.items():
            if ticker in group['tickers']:
                categorized[category].append(ticker)
                found = True
                break
        if not found:
            uncategorized.append(ticker)
    
    # Mostrar por categorías
    for category, group_tickers in categorized.items():
        if group_tickers:
            category_name = asset_groups[category]['description']
            print(f"\n🏷️  {category_name.upper()}:")
            for ticker in group_tickers:
                print(f"   • {ticker}")
    
    if uncategorized:
        print(f"\n🔍 OTROS:")
        for ticker in uncategorized:
            print(f"   • {ticker}")
    
    print("=" * 50)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Trading Model - Pipeline de predicción de inversiones')
    
    # Acciones del pipeline
    parser.add_argument('--download', action='store_true', help='Descargar datos históricos')
    parser.add_argument('--process', action='store_true', help='Procesar datos y calcular características')
    parser.add_argument('--train', action='store_true', help='Entrenar modelos')
    parser.add_argument('--predict', action='store_true', help='Hacer predicciones')
    parser.add_argument('--all', action='store_true', help='Ejecutar todo el pipeline')
    
    # Selección de activos
    parser.add_argument('--tickers', nargs='+', default=None, 
                        help='Lista específica de tickers a procesar (ej: AAPL MSFT ^GSPC)')
    parser.add_argument('--indices', action='store_true', 
                        help='Usar índices bursátiles principales (S&P 500, Dow Jones, NASDAQ, etc.)')
    parser.add_argument('--stocks', action='store_true', 
                        help='Usar acciones individuales principales (FAANG + tecnológicas)')
    parser.add_argument('--etfs', action='store_true', 
                        help='Usar ETFs principales (SPY, QQQ, VTI, etc.)')
    parser.add_argument('--crypto', action='store_true', 
                        help='Usar criptomonedas principales (BTC, ETH, etc.)')
    parser.add_argument('--commodities', action='store_true', 
                        help='Usar materias primas (oro, petróleo, etc.)')
    
    # Configuración de modelos
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
        
        # Seleccionar tickers basándose en los argumentos
        selected_tickers = select_tickers(args)
        args.tickers = selected_tickers
        
        # Mostrar activos seleccionados
        print_selected_assets(args.tickers)
        
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
