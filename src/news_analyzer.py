#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para obtener noticias financieras y realizar análisis de sentimiento.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
import json
import re
from newspaper import Article
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from dotenv import load_dotenv
from tqdm import tqdm
import time
import nltk
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

# Obtener claves API
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("news_data.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Verificar si se han descargado los recursos necesarios para NLTK
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class FinancialNewsAnalyzer:
    def __init__(self):
        """Inicializa el analizador de noticias financieras."""
        self.news_api_key = NEWS_API_KEY
        self.finnhub_api_key = FINNHUB_API_KEY
        self.alpha_vantage_api_key = ALPHA_VANTAGE_API_KEY
        self.vader = SentimentIntensityAnalyzer()
        self.news_data = None
    
    def get_news_api_articles(self, query, from_date, to_date, language='en', sort_by='publishedAt'):
        """
        Obtiene artículos de noticias desde NewsAPI.
        
        Args:
            query (str): Consulta de búsqueda.
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            language (str): Código de idioma (por defecto 'en').
            sort_by (str): Criterio de ordenación (por defecto 'publishedAt').
            
        Returns:
            list: Lista de artículos de noticias.
        """
        if not self.news_api_key:
            logger.error("No se ha configurado NEWS_API_KEY. No se pueden obtener noticias de NewsAPI.")
            return []
        
        url = 'https://newsapi.org/v2/everything'
        
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'language': language,
            'sortBy': sort_by,
            'apiKey': self.news_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'ok':
                return data['articles']
            else:
                logger.error(f"Error en la respuesta de NewsAPI: {data.get('message', 'Desconocido')}")
                return []
                
        except Exception as e:
            logger.error(f"Error al obtener noticias de NewsAPI: {str(e)}")
            return []
    
    def get_finnhub_company_news(self, ticker, from_date, to_date):
        """
        Obtiene noticias de empresas desde Finnhub.
        
        Args:
            ticker (str): Símbolo del ticker.
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            
        Returns:
            list: Lista de noticias de la empresa.
        """
        if not self.finnhub_api_key:
            logger.error("No se ha configurado FINNHUB_API_KEY. No se pueden obtener noticias de Finnhub.")
            return []
        
        url = 'https://finnhub.io/api/v1/company-news'
        
        params = {
            'symbol': ticker,
            'from': from_date,
            'to': to_date,
            'token': self.finnhub_api_key
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data
                
        except Exception as e:
            logger.error(f"Error al obtener noticias de Finnhub para {ticker}: {str(e)}")
            return []
    
    def get_alpha_vantage_news(self, ticker=None, topics=None):
        """
        Obtiene noticias desde Alpha Vantage.
        
        Args:
            ticker (str, optional): Símbolo del ticker.
            topics (str, optional): Temas de noticias separados por comas.
            
        Returns:
            list: Lista de noticias.
        """
        if not self.alpha_vantage_api_key:
            logger.error("No se ha configurado ALPHA_VANTAGE_API_KEY. No se pueden obtener noticias de Alpha Vantage.")
            return []
        
        url = 'https://www.alphavantage.co/query'
        
        params = {
            'function': 'NEWS_SENTIMENT',
            'apikey': self.alpha_vantage_api_key
        }
        
        if ticker:
            params['tickers'] = ticker
        
        if topics:
            params['topics'] = topics
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if 'feed' in data:
                return data['feed']
            else:
                logger.error(f"Error en la respuesta de Alpha Vantage: {data.get('Information', 'Desconocido')}")
                return []
                
        except Exception as e:
            logger.error(f"Error al obtener noticias de Alpha Vantage: {str(e)}")
            return []
    
    def extract_article_content(self, url):
        """
        Extrae el contenido de un artículo a partir de su URL.
        
        Args:
            url (str): URL del artículo.
            
        Returns:
            str: Contenido del artículo.
        """
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.error(f"Error al extraer contenido del artículo {url}: {str(e)}")
            return ""
    
    def analyze_sentiment_textblob(self, text):
        """
        Analiza el sentimiento del texto usando TextBlob.
        
        Args:
            text (str): Texto a analizar.
            
        Returns:
            dict: Resultados del análisis de sentimiento.
        """
        if not text:
            return {'polarity': 0, 'subjectivity': 0}
        
        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_sentiment_vader(self, text):
        """
        Analiza el sentimiento del texto usando VADER.
        
        Args:
            text (str): Texto a analizar.
            
        Returns:
            dict: Resultados del análisis de sentimiento.
        """
        if not text:
            return {'compound': 0, 'neg': 0, 'neu': 0, 'pos': 0}
        
        return self.vader.polarity_scores(text)
    
    def collect_news_for_ticker(self, ticker, from_date, to_date):
        """
        Recopila noticias para un ticker específico.
        
        Args:
            ticker (str): Símbolo del ticker.
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            
        Returns:
            pandas.DataFrame: DataFrame con las noticias recopiladas.
        """
        all_news = []
        
        # Obtener nombre de la empresa
        try:
            ticker_info = yf.Ticker(ticker)
            company_name = ticker_info.info.get('shortName', ticker)
        except:
            company_name = ticker
        
        # 1. Finnhub
        logger.info(f"Obteniendo noticias de Finnhub para {ticker}...")
        finnhub_news = self.get_finnhub_company_news(ticker, from_date, to_date)
        
        for news in finnhub_news:
            all_news.append({
                'source': 'Finnhub',
                'ticker': ticker,
                'company': company_name,
                'date': datetime.fromtimestamp(news.get('datetime', 0)).strftime('%Y-%m-%d'),
                'title': news.get('headline', ''),
                'description': news.get('summary', ''),
                'url': news.get('url', ''),
                'source_name': news.get('source', 'Finnhub')
            })
        
        # 2. NewsAPI
        logger.info(f"Obteniendo noticias de NewsAPI para {ticker} ({company_name})...")
        query = f"{ticker} OR \"{company_name}\""
        newsapi_articles = self.get_news_api_articles(query, from_date, to_date)
        
        for article in newsapi_articles:
            all_news.append({
                'source': 'NewsAPI',
                'ticker': ticker,
                'company': company_name,
                'date': article.get('publishedAt', '')[:10],
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source_name': article.get('source', {}).get('name', 'NewsAPI')
            })
        
        # 3. Alpha Vantage
        logger.info(f"Obteniendo noticias de Alpha Vantage para {ticker}...")
        av_news = self.get_alpha_vantage_news(ticker)
        
        for news in av_news:
            all_news.append({
                'source': 'AlphaVantage',
                'ticker': ticker,
                'company': company_name,
                'date': news.get('time_published', '')[:10],
                'title': news.get('title', ''),
                'description': news.get('summary', ''),
                'url': news.get('url', ''),
                'source_name': news.get('source', 'AlphaVantage'),
                'sentiment': news.get('overall_sentiment_score', 0)
            })
        
        # Crear DataFrame
        if all_news:
            df = pd.DataFrame(all_news)
            
            # Eliminar duplicados basados en el título
            df.drop_duplicates(subset=['title'], inplace=True)
            
            return df
        else:
            logger.warning(f"No se encontraron noticias para {ticker} en el período especificado.")
            return pd.DataFrame()
    
    def analyze_news_sentiment(self, news_df):
        """
        Analiza el sentimiento de las noticias en el DataFrame.
        
        Args:
            news_df (pandas.DataFrame): DataFrame con noticias.
            
        Returns:
            pandas.DataFrame: DataFrame con análisis de sentimiento añadido.
        """
        if news_df.empty:
            return news_df
        
        # Copiar el DataFrame para no modificar el original
        df = news_df.copy()
        
        # Añadir columnas de sentimiento
        df['textblob_sentiment'] = None
        df['vader_sentiment'] = None
        df['full_content'] = None
        
        # Procesar cada noticia
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Analizando sentimiento"):
            # Analizar sentimiento del título y descripción
            text = f"{row['title']} {row['description']}"
            
            # Análisis con TextBlob
            textblob_result = self.analyze_sentiment_textblob(text)
            df.at[idx, 'textblob_sentiment'] = textblob_result['polarity']
            df.at[idx, 'textblob_subjectivity'] = textblob_result['subjectivity']
            
            # Análisis con VADER
            vader_result = self.analyze_sentiment_vader(text)
            df.at[idx, 'vader_sentiment'] = vader_result['compound']
            df.at[idx, 'vader_neg'] = vader_result['neg']
            df.at[idx, 'vader_neu'] = vader_result['neu']
            df.at[idx, 'vader_pos'] = vader_result['pos']
            
            # Intentar extraer contenido completo si hay URL
            if row['url']:
                content = self.extract_article_content(row['url'])
                df.at[idx, 'full_content'] = content
                
                # Si se pudo extraer contenido, analizar su sentimiento
                if content:
                    # TextBlob
                    full_textblob = self.analyze_sentiment_textblob(content)
                    df.at[idx, 'full_textblob_sentiment'] = full_textblob['polarity']
                    df.at[idx, 'full_textblob_subjectivity'] = full_textblob['subjectivity']
                    
                    # VADER
                    full_vader = self.analyze_sentiment_vader(content)
                    df.at[idx, 'full_vader_sentiment'] = full_vader['compound']
            
            # Pequeña pausa para no sobrecargar los servidores
            time.sleep(0.1)
        
        return df
    
    def aggregate_daily_sentiment(self, news_df):
        """
        Agrega el sentimiento diario para cada ticker.
        
        Args:
            news_df (pandas.DataFrame): DataFrame con noticias y análisis de sentimiento.
            
        Returns:
            pandas.DataFrame: DataFrame con sentimiento diario agregado.
        """
        if news_df.empty:
            return pd.DataFrame()
        
        # Asegurarse de que la columna de fecha sea datetime
        news_df['date'] = pd.to_datetime(news_df['date'])
        
        # Agregar por fecha y ticker
        daily_sentiment = news_df.groupby(['date', 'ticker']).agg({
            'textblob_sentiment': 'mean',
            'vader_sentiment': 'mean',
            'full_vader_sentiment': 'mean',
            'full_textblob_sentiment': 'mean',
            'vader_pos': 'mean',
            'vader_neg': 'mean',
            'vader_neu': 'mean'
        }).reset_index()
        
        # Calcular sentimiento combinado (promedio de diferentes métodos)
        # Dar mayor peso al análisis de contenido completo si está disponible
        daily_sentiment['combined_sentiment'] = (
            daily_sentiment['vader_sentiment'] * 0.3 +
            daily_sentiment['textblob_sentiment'] * 0.2 +
            daily_sentiment.get('full_vader_sentiment', daily_sentiment['vader_sentiment']) * 0.3 +
            daily_sentiment.get('full_textblob_sentiment', daily_sentiment['textblob_sentiment']) * 0.2
        )
        
        return daily_sentiment
    
    def collect_market_news(self, from_date, to_date):
        """
        Recopila noticias generales del mercado.
        
        Args:
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            
        Returns:
            pandas.DataFrame: DataFrame con noticias del mercado.
        """
        all_news = []
        
        # Consultas para noticias del mercado
        queries = [
            "stock market", "financial markets", "wall street",
            "federal reserve", "interest rates", "inflation",
            "economic growth", "recession", "market crash",
            "bull market", "bear market", "market trend"
        ]
        
        for query in queries:
            logger.info(f"Obteniendo noticias de mercado para '{query}'...")
            newsapi_articles = self.get_news_api_articles(query, from_date, to_date)
            
            for article in newsapi_articles:
                all_news.append({
                    'source': 'NewsAPI',
                    'ticker': 'MARKET',
                    'company': 'Market',
                    'date': article.get('publishedAt', '')[:10],
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'source_name': article.get('source', {}).get('name', 'NewsAPI'),
                    'query': query
                })
        
        # Alpha Vantage para temas generales
        topics = "economy_macro,economy_fiscal,economy_monetary,finance,business_economy"
        logger.info("Obteniendo noticias de Alpha Vantage para temas económicos...")
        av_news = self.get_alpha_vantage_news(topics=topics)
        
        for news in av_news:
            all_news.append({
                'source': 'AlphaVantage',
                'ticker': 'MARKET',
                'company': 'Market',
                'date': news.get('time_published', '')[:10],
                'title': news.get('title', ''),
                'description': news.get('summary', ''),
                'url': news.get('url', ''),
                'source_name': news.get('source', 'AlphaVantage'),
                'sentiment': news.get('overall_sentiment_score', 0),
                'query': 'economy'
            })
        
        # Crear DataFrame
        if all_news:
            df = pd.DataFrame(all_news)
            
            # Eliminar duplicados basados en el título
            df.drop_duplicates(subset=['title'], inplace=True)
            
            return df
        else:
            logger.warning("No se encontraron noticias del mercado en el período especificado.")
            return pd.DataFrame()
    
    def process_ticker_news(self, ticker, from_date, to_date):
        """
        Procesa noticias para un ticker específico.
        
        Args:
            ticker (str): Símbolo del ticker.
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            
        Returns:
            pandas.DataFrame: DataFrame con sentimiento diario.
        """
        # Recopilar noticias
        news_df = self.collect_news_for_ticker(ticker, from_date, to_date)
        
        if news_df.empty:
            logger.warning(f"No se encontraron noticias para {ticker}.")
            return pd.DataFrame()
        
        # Analizar sentimiento
        news_with_sentiment = self.analyze_news_sentiment(news_df)
        
        # Agregar sentimiento diario
        daily_sentiment = self.aggregate_daily_sentiment(news_with_sentiment)
        
        # Guardar datos detallados
        os.makedirs('data/news', exist_ok=True)
        news_with_sentiment.to_csv(f"data/news/{ticker}_news_detailed.csv", index=False)
        
        return daily_sentiment
    
    def process_multiple_tickers(self, tickers, from_date, to_date):
        """
        Procesa noticias para múltiples tickers.
        
        Args:
            tickers (list): Lista de símbolos de tickers.
            from_date (str): Fecha de inicio en formato 'YYYY-MM-DD'.
            to_date (str): Fecha de fin en formato 'YYYY-MM-DD'.
            
        Returns:
            pandas.DataFrame: DataFrame con sentimiento diario para todos los tickers.
        """
        all_sentiment = []
        
        for ticker in tickers:
            logger.info(f"Procesando noticias para {ticker}...")
            daily_sentiment = self.process_ticker_news(ticker, from_date, to_date)
            
            if not daily_sentiment.empty:
                all_sentiment.append(daily_sentiment)
        
        # Procesar noticias de mercado
        logger.info("Procesando noticias generales del mercado...")
        market_news = self.collect_market_news(from_date, to_date)
        
        if not market_news.empty:
            market_with_sentiment = self.analyze_news_sentiment(market_news)
            market_daily = self.aggregate_daily_sentiment(market_with_sentiment)
            
            if not market_daily.empty:
                all_sentiment.append(market_daily)
            
            # Guardar datos detallados del mercado
            os.makedirs('data/news', exist_ok=True)
            market_with_sentiment.to_csv("data/news/MARKET_news_detailed.csv", index=False)
        
        # Combinar todos los resultados
        if all_sentiment:
            result = pd.concat(all_sentiment)
            
            # Convertir fecha a formato correcto
            result['date'] = pd.to_datetime(result['date'])
            
            # Ordenar por fecha y ticker
            result = result.sort_values(['date', 'ticker'])
            
            # Guardar resultado combinado
            result.to_csv("data/news_sentiment_daily.csv", index=False)
            
            return result
        else:
            logger.warning("No se encontraron noticias para ningún ticker.")
            return pd.DataFrame()

def main():
    """Función principal para la recopilación y análisis de noticias."""
    # Definir el rango de fechas (se limita a 1 mes por restricciones de las APIs gratuitas)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    # Lista de tickers a analizar
    tickers = [
        "^GSPC",  # S&P 500
        "AAPL",   # Apple
        "MSFT",   # Microsoft
        "GOOGL",  # Alphabet (Google)
        "AMZN",   # Amazon
        "META",   # Meta (Facebook)
        "TSLA"    # Tesla
    ]
    
    # Crear y ejecutar el analizador de noticias
    analyzer = FinancialNewsAnalyzer()
    sentiment_data = analyzer.process_multiple_tickers(tickers, start_date, end_date)
    
    if not sentiment_data.empty:
        logger.info(f"Análisis de sentimiento completado. Datos guardados en 'data/news_sentiment_daily.csv'.")
    else:
        logger.warning("No se pudo completar el análisis de sentimiento.")

if __name__ == "__main__":
    main()
