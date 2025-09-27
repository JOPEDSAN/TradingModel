#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para obtener datos macroeconómicos (inflación, tipos de interés, etc.) 
para enriquecer el modelo de predicción.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import json
from pandas_datareader import data as pdr
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la clave API de FRED
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
if not FRED_API_KEY:
    print("ADVERTENCIA: No se ha configurado FRED_API_KEY en el archivo .env")
    print("Algunas funcionalidades pueden no estar disponibles")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("economic_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def get_fred_data(series_ids, start_date, end_date):
    """
    Obtiene datos económicos de FRED (Federal Reserve Economic Data).
    
    Args:
        series_ids (dict): Diccionario con nombres de series y sus IDs.
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        
    Returns:
        pandas.DataFrame: DataFrame con los datos económicos.
    """
    if not FRED_API_KEY:
        logger.error("No se ha configurado FRED_API_KEY. No se pueden obtener datos de FRED.")
        return None
    
    fred = Fred(api_key=FRED_API_KEY)
    data = {}
    
    for name, series_id in series_ids.items():
        try:
            logger.info(f"Obteniendo datos para {name} (ID: {series_id})")
            series = fred.get_series(series_id, start_date, end_date)
            data[name] = series
        except Exception as e:
            logger.error(f"Error al obtener datos para {name}: {str(e)}")
    
    # Crear DataFrame y manejar frecuencias diferentes
    df = pd.DataFrame(data)
    
    # Rellenar valores faltantes (interpolación lineal)
    df = df.resample('D').asfreq()
    df = df.interpolate(method='linear')
    
    return df

def get_inflation_data(start_date, end_date, country='US'):
    """
    Obtiene datos de inflación.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        country (str): Código del país (por defecto 'US').
        
    Returns:
        pandas.Series: Serie con los datos de inflación.
    """
    series_ids = {
        'US': 'CPIAUCSL',  # Índice de Precios al Consumidor para EE.UU.
        'EU': 'CP0000EZ19M086NEST',  # IPC para la Eurozona
        'UK': 'GBRCPIALLMINMEI',  # IPC para Reino Unido
        'JP': 'JPNCPIALLMINMEI',  # IPC para Japón
    }
    
    if country not in series_ids:
        logger.error(f"País {country} no soportado para datos de inflación.")
        return None
    
    if not FRED_API_KEY:
        logger.error("No se ha configurado FRED_API_KEY. No se pueden obtener datos de inflación.")
        return None
    
    fred = Fred(api_key=FRED_API_KEY)
    
    try:
        # Obtener IPC
        cpi = fred.get_series(series_ids[country], start_date, end_date)
        
        # Calcular tasa de inflación anual (cambio porcentual respecto al año anterior)
        inflation = cpi.pct_change(periods=12) * 100
        
        # Rellenar valores faltantes
        inflation = inflation.resample('D').asfreq()
        inflation = inflation.interpolate(method='linear')
        
        return inflation
    
    except Exception as e:
        logger.error(f"Error al obtener datos de inflación para {country}: {str(e)}")
        return None

def get_interest_rates(start_date, end_date, country='US'):
    """
    Obtiene datos de tipos de interés.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        country (str): Código del país (por defecto 'US').
        
    Returns:
        pandas.DataFrame: DataFrame con los datos de tipos de interés.
    """
    series_ids = {
        'US': {
            'FED_FUNDS_RATE': 'FEDFUNDS',  # Tasa de fondos federales
            'TREASURY_3M': 'DTB3',  # Tasa del Tesoro a 3 meses
            'TREASURY_10Y': 'DGS10',  # Tasa del Tesoro a 10 años
        },
        'EU': {
            'ECB_RATE': 'ECBASSETS',  # Tasa del BCE
        },
    }
    
    if country not in series_ids:
        logger.error(f"País {country} no soportado para datos de tipos de interés.")
        return None
    
    return get_fred_data(series_ids[country], start_date, end_date)

def get_gdp_data(start_date, end_date, country='US'):
    """
    Obtiene datos de PIB.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        country (str): Código del país (por defecto 'US').
        
    Returns:
        pandas.Series: Serie con los datos de PIB.
    """
    series_ids = {
        'US': 'GDP',  # PIB nominal de EE.UU.
        'EU': 'EUNGDP',  # PIB nominal de la Eurozona
        'UK': 'UKNGDP',  # PIB nominal de Reino Unido
        'JP': 'JPNGDPNQDSMEI',  # PIB nominal de Japón
    }
    
    if country not in series_ids:
        logger.error(f"País {country} no soportado para datos de PIB.")
        return None
    
    if not FRED_API_KEY:
        logger.error("No se ha configurado FRED_API_KEY. No se pueden obtener datos de PIB.")
        return None
    
    fred = Fred(api_key=FRED_API_KEY)
    
    try:
        # Obtener PIB
        gdp = fred.get_series(series_ids[country], start_date, end_date)
        
        # Calcular tasa de crecimiento (cambio porcentual)
        gdp_growth = gdp.pct_change() * 100
        
        # Rellenar valores faltantes (interpolación para datos trimestrales)
        gdp_daily = gdp.resample('D').asfreq()
        gdp_daily = gdp_daily.interpolate(method='linear')
        
        gdp_growth_daily = gdp_growth.resample('D').asfreq()
        gdp_growth_daily = gdp_growth_daily.interpolate(method='linear')
        
        return pd.DataFrame({
            'GDP': gdp_daily,
            'GDP_Growth': gdp_growth_daily
        })
    
    except Exception as e:
        logger.error(f"Error al obtener datos de PIB para {country}: {str(e)}")
        return None

def get_currency_strength(start_date, end_date, base_currency='USD'):
    """
    Obtiene datos de fortaleza de divisas frente a una moneda base.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        base_currency (str): Moneda base (por defecto 'USD').
        
    Returns:
        pandas.DataFrame: DataFrame con los datos de tipos de cambio.
    """
    # Definir pares de divisas a descargar
    if base_currency == 'USD':
        pairs = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCAD=X', 'USDCHF=X', 'AUDUSD=X']
    elif base_currency == 'EUR':
        pairs = ['EUR=X', 'EURGBP=X', 'EURJPY=X', 'EURCAD=X', 'EURCHF=X', 'EURAUD=X']
    else:
        logger.error(f"Moneda base {base_currency} no soportada.")
        return None
    
    # Descargar datos
    try:
        yf.pdr_override()
        data = pdr.get_data_yahoo(pairs, start=start_date, end=end_date)['Adj Close']
        
        # Renombrar columnas
        data.columns = [col.replace('=X', '') for col in data.columns]
        
        # Calcular variación diaria
        data_pct = data.pct_change() * 100
        data_pct.columns = [col + '_change' for col in data.columns]
        
        # Combinar datos originales y variaciones
        result = pd.concat([data, data_pct], axis=1)
        
        return result
    
    except Exception as e:
        logger.error(f"Error al obtener datos de fortaleza de divisas: {str(e)}")
        return None

def get_commodity_prices(start_date, end_date):
    """
    Obtiene precios de materias primas.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        
    Returns:
        pandas.DataFrame: DataFrame con los precios de materias primas.
    """
    # Tickers de materias primas
    commodities = {
        'Gold': 'GC=F',
        'Silver': 'SI=F',
        'Crude_Oil': 'CL=F',
        'Natural_Gas': 'NG=F',
        'Copper': 'HG=F',
        'Corn': 'ZC=F',
        'Wheat': 'ZW=F'
    }
    
    try:
        yf.pdr_override()
        data = pdr.get_data_yahoo(list(commodities.values()), start=start_date, end=end_date)['Adj Close']
        
        # Renombrar columnas
        data.columns = list(commodities.keys())
        
        # Calcular variación diaria
        data_pct = data.pct_change() * 100
        data_pct.columns = [col + '_change' for col in data.columns]
        
        # Combinar datos originales y variaciones
        result = pd.concat([data, data_pct], axis=1)
        
        return result
    
    except Exception as e:
        logger.error(f"Error al obtener precios de materias primas: {str(e)}")
        return None

def get_economic_indicators(start_date, end_date, country='US'):
    """
    Obtiene un conjunto completo de indicadores económicos.
    
    Args:
        start_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
        country (str): Código del país (por defecto 'US').
        
    Returns:
        pandas.DataFrame: DataFrame con todos los indicadores económicos.
    """
    # Obtener datos de diferentes fuentes
    inflation = get_inflation_data(start_date, end_date, country)
    interest_rates = get_interest_rates(start_date, end_date, country)
    gdp = get_gdp_data(start_date, end_date, country)
    currencies = get_currency_strength(start_date, end_date, 'USD' if country == 'US' else 'EUR')
    commodities = get_commodity_prices(start_date, end_date)
    
    # Crear una lista de DataFrames disponibles
    dfs = []
    
    if inflation is not None:
        inflation = inflation.to_frame('Inflation')
        dfs.append(inflation)
    
    if interest_rates is not None:
        dfs.append(interest_rates)
    
    if gdp is not None:
        dfs.append(gdp)
    
    if currencies is not None:
        dfs.append(currencies)
    
    if commodities is not None:
        dfs.append(commodities)
    
    # Combinar todos los DataFrames
    if dfs:
        result = pd.concat(dfs, axis=1)
        
        # Manejar valores faltantes
        result = result.interpolate(method='linear')
        
        return result
    else:
        logger.error("No se pudieron obtener datos económicos.")
        return None

def save_economic_data(data, output_file='data/economic_indicators.csv'):
    """
    Guarda los datos económicos en un archivo CSV.
    
    Args:
        data (pandas.DataFrame): DataFrame con los datos económicos.
        output_file (str): Ruta del archivo de salida.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Guardar datos
    data.to_csv(output_file)
    logger.info(f"Datos económicos guardados en {output_file}")

def main():
    """Función principal para la descarga de datos económicos."""
    # Definir el rango de fechas (20 años)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365 * 20)).strftime('%Y-%m-%d')
    
    # Obtener indicadores económicos
    logger.info("Obteniendo indicadores económicos...")
    economic_data = get_economic_indicators(start_date, end_date, 'US')
    
    if economic_data is not None:
        # Guardar datos
        save_economic_data(economic_data)
    else:
        logger.error("No se pudieron obtener datos económicos.")

if __name__ == "__main__":
    main()
