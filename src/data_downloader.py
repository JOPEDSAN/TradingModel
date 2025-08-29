#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para obtener datos históricos de Yahoo Finance.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_download.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def download_stock_data(ticker, start_date, end_date, interval='1d'):
    """
    Descarga datos históricos de un ticker específico.
    
    Args:
        ticker (str): Símbolo del ticker.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        interval (str): Intervalo de los datos ('1d', '1h', etc.).
        
    Returns:
        pandas.DataFrame: DataFrame con los datos históricos.
    """
    try:
        logger.info(f"Descargando datos para {ticker} desde {start_date} hasta {end_date}")
        data = yf.download(ticker, start=start_date, end=end_date, interval=interval)
        
        if data.empty:
            logger.warning(f"No se encontraron datos para {ticker}")
            return None
            
        # Resetear el índice para tener la fecha como columna
        data.reset_index(inplace=True)
        
        # Renombrar columnas para consistencia
        data.columns = [col if col != 'Date' else 'date' for col in data.columns]
        data.columns = [col if col != 'Datetime' else 'date' for col in data.columns]
        
        # Agregar columna de ticker
        data['ticker'] = ticker
        
        return data
    
    except Exception as e:
        logger.error(f"Error al descargar datos para {ticker}: {str(e)}")
        return None

def download_multiple_stocks(tickers, start_date, end_date, interval='1d'):
    """
    Descarga datos históricos para múltiples tickers.
    
    Args:
        tickers (list): Lista de símbolos de tickers.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        interval (str): Intervalo de los datos ('1d', '1h', etc.).
        
    Returns:
        dict: Diccionario con los DataFrames de cada ticker.
    """
    all_data = {}
    
    for ticker in tqdm(tickers, desc="Descargando datos de acciones"):
        data = download_stock_data(ticker, start_date, end_date, interval)
        if data is not None:
            all_data[ticker] = data
    
    return all_data

def save_data(data_dict, output_dir='data'):
    """
    Guarda los datos en archivos CSV.
    
    Args:
        data_dict (dict): Diccionario con los DataFrames de cada ticker.
        output_dir (str): Directorio de salida.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for ticker, data in data_dict.items():
        output_file = os.path.join(output_dir, f"{ticker.replace('.', '-')}_data.csv")
        data.to_csv(output_file, index=False)
        logger.info(f"Datos guardados en {output_file}")

def main():
    # Ejemplo de uso
    # Define los tickers a descargar
    tickers = [
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
        
        # Sectores financieros
        "JPM",    # JPMorgan Chase
        "BAC",    # Bank of America
        "GS",     # Goldman Sachs
        
        # ETFs importantes
        "SPY",    # SPDR S&P 500 ETF
        "QQQ",    # Invesco QQQ (Nasdaq-100)
        "VTI",    # Vanguard Total Stock Market ETF
    ]
    
    # Define el rango de fechas (5 años de datos)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 5)).strftime('%Y-%m-%d')
    
    # Descarga los datos
    data_dict = download_multiple_stocks(tickers, start_date, end_date)
    
    # Guarda los datos
    save_data(data_dict)

if __name__ == "__main__":
    main()
