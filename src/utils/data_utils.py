#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidades para el procesamiento de datos en el proyecto de predicción de inversiones.
"""

import pandas as pd
import numpy as np
import logging

# Configurar logging
logger = logging.getLogger(__name__)

def merge_economic_data(df, economic_data, date_column='date'):
    """
    Fusiona datos económicos con los datos de precios.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de precios.
        economic_data (pandas.DataFrame): DataFrame con datos económicos.
        date_column (str): Nombre de la columna de fecha.
        
    Returns:
        pandas.DataFrame: DataFrame combinado.
    """
    # Verificar si hay datos económicos
    if economic_data is None or economic_data.empty:
        logger.warning("No hay datos económicos para fusionar.")
        return df
    
    # Convertir fechas a datetime si no lo están
    df[date_column] = pd.to_datetime(df[date_column])
    economic_data.index = pd.to_datetime(economic_data.index)
    
    # Reindexar datos económicos para que coincidan con las fechas de precios
    # Usamos 'ffill' para llenar hacia adelante los días sin datos
    economic_daily = economic_data.reindex(
        pd.date_range(start=economic_data.index.min(), end=economic_data.index.max())
    ).ffill()
    
    # Crear un DataFrame con las fechas del DataFrame original
    dates_df = pd.DataFrame({'date': df[date_column]})
    dates_df.set_index('date', inplace=True)
    
    # Fusionar con datos económicos
    merged = dates_df.join(economic_daily, how='left')
    
    # Rellenar valores faltantes
    merged = merged.ffill()  # Reemplaza fillna(method='ffill')
    merged = merged.bfill()  # Reemplaza fillna(method='bfill')
    
    # Resetear el índice
    merged.reset_index(inplace=True)
    
    # Fusionar con el DataFrame original
    result = pd.merge(df, merged, on=date_column, how='left')
    
    return result

def merge_news_sentiment(df, sentiment_data, ticker, date_column='date'):
    """
    Fusiona datos de sentimiento de noticias con los datos de precios.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de precios.
        sentiment_data (pandas.DataFrame): DataFrame con datos de sentimiento.
        ticker (str): Símbolo del ticker.
        date_column (str): Nombre de la columna de fecha.
        
    Returns:
        pandas.DataFrame: DataFrame combinado.
    """
    # Verificar si hay datos de sentimiento
    if sentiment_data is None or sentiment_data.empty:
        logger.warning("No hay datos de sentimiento para fusionar.")
        return df
    
    # Filtrar datos de sentimiento para el ticker específico y para el mercado general
    ticker_sentiment = sentiment_data[sentiment_data['ticker'].isin([ticker, 'MARKET'])]
    
    if ticker_sentiment.empty:
        logger.warning(f"No hay datos de sentimiento para {ticker} o MARKET.")
        return df
    
    # Convertir fechas a datetime si no lo están
    df[date_column] = pd.to_datetime(df[date_column])
    ticker_sentiment['date'] = pd.to_datetime(ticker_sentiment['date'])
    
    # Pivotar para tener una columna para cada ticker
    sentiment_pivot = ticker_sentiment.pivot_table(
        index='date',
        columns='ticker',
        values=['combined_sentiment', 'vader_sentiment', 'textblob_sentiment', 'vader_pos', 'vader_neg']
    )
    
    # Aplanar las columnas multinivel
    sentiment_pivot.columns = [f"{col[1]}_{col[0]}" for col in sentiment_pivot.columns]
    
    # Crear un DataFrame con las fechas del DataFrame original
    dates_df = pd.DataFrame({'date': df[date_column]})
    dates_df.set_index('date', inplace=True)
    
    # Fusionar con datos de sentimiento
    merged = dates_df.join(sentiment_pivot, how='left')
    
    # Rellenar valores faltantes (usar la media de los datos disponibles)
    for column in merged.columns:
        if column != 'date':
            # Usar la media de los últimos 7 días si está disponible
            merged[column] = merged[column].fillna(merged[column].rolling(window=7, min_periods=1).mean())
    
    # Si aún hay valores faltantes, rellenar con 0 (neutro)
    merged.fillna(0, inplace=True)
    
    # Resetear el índice
    merged.reset_index(inplace=True)
    
    # Fusionar con el DataFrame original
    result = pd.merge(df, merged, on=date_column, how='left')
    
    return result
