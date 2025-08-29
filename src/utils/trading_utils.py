#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de utilidades para el proyecto de predicción de inversiones.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configurar el estilo de las gráficas
sns.set(style='darkgrid')
plt.rcParams['figure.figsize'] = (15, 8)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("utils.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ensure_dir_exists(directory):
    """
    Asegura que un directorio exista, creándolo si es necesario.
    
    Args:
        directory (str): Ruta del directorio.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Directorio creado: {directory}")

def load_json(file_path):
    """
    Carga un archivo JSON.
    
    Args:
        file_path (str): Ruta del archivo JSON.
        
    Returns:
        dict: Contenido del archivo JSON.
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error al cargar {file_path}: {str(e)}")
        return None

def save_json(data, file_path):
    """
    Guarda datos en un archivo JSON.
    
    Args:
        data (dict): Datos a guardar.
        file_path (str): Ruta del archivo JSON.
    """
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Datos guardados en {file_path}")
    except Exception as e:
        logger.error(f"Error al guardar en {file_path}: {str(e)}")

def evaluate_predictions(y_true, y_pred):
    """
    Evalúa las predicciones con varias métricas.
    
    Args:
        y_true (array): Valores reales.
        y_pred (array): Valores predichos.
        
    Returns:
        dict: Diccionario con las métricas.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }
    
    return metrics

def plot_metrics_comparison(metrics_dict, title='Comparación de Modelos', save_path=None):
    """
    Genera un gráfico comparativo de métricas entre modelos.
    
    Args:
        metrics_dict (dict): Diccionario con las métricas por modelo.
            Formato: {model_name: {metric_name: value, ...}, ...}
        title (str): Título del gráfico.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    models = list(metrics_dict.keys())
    metrics = ['mse', 'rmse', 'mae', 'r2']
    
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    
    for i, metric in enumerate(metrics):
        values = [metrics_dict[model][metric] for model in models]
        
        axs[i].bar(models, values, color=sns.color_palette("muted", len(models)))
        axs[i].set_title(f'{metric.upper()}', fontsize=14)
        axs[i].set_xlabel('Modelo')
        axs[i].set_ylabel('Valor')
        
        # Mostrar valores
        for j, v in enumerate(values):
            axs[i].text(j, v, f"{v:.4f}", ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Gráfico guardado en {save_path}")
    
    plt.show()

def plot_prediction_vs_actual(dates, y_true, y_pred, ticker, model_name, save_path=None):
    """
    Genera un gráfico de predicciones vs valores reales.
    
    Args:
        dates (array): Fechas.
        y_true (array): Valores reales.
        y_pred (array): Valores predichos.
        ticker (str): Símbolo del ticker.
        model_name (str): Nombre del modelo.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    plt.figure(figsize=(16, 8))
    
    plt.plot(dates, y_true, label='Real', linewidth=2)
    plt.plot(dates, y_pred, label='Predicción', linewidth=2, linestyle='--')
    
    plt.title(f'Predicción vs Real para {ticker} usando {model_name}', fontsize=16)
    plt.xlabel('Fecha', fontsize=12)
    plt.ylabel('Precio', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Formatear eje x para fechas
    plt.gcf().autofmt_xdate()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Gráfico guardado en {save_path}")
    
    plt.show()

def calculate_trading_signals(df, strategy='sma_crossover'):
    """
    Calcula señales de trading basadas en una estrategia.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de precios e indicadores.
        strategy (str): Estrategia a utilizar ('sma_crossover', 'rsi', 'macd').
        
    Returns:
        pandas.DataFrame: DataFrame con señales de trading.
    """
    data = df.copy()
    
    if strategy == 'sma_crossover':
        # Estrategia de cruce de medias móviles (5 y 20 días)
        data['Signal'] = 0
        data.loc[data['sma_5'] > data['sma_20'], 'Signal'] = 1  # Compra
        data.loc[data['sma_5'] < data['sma_20'], 'Signal'] = -1  # Venta
        
    elif strategy == 'rsi':
        # Estrategia basada en RSI
        data['Signal'] = 0
        data.loc[data['rsi_14'] < 30, 'Signal'] = 1  # Compra (sobrevendido)
        data.loc[data['rsi_14'] > 70, 'Signal'] = -1  # Venta (sobrecomprado)
        
    elif strategy == 'macd':
        # Estrategia basada en MACD
        data['Signal'] = 0
        data.loc[data['macd'] > data['macd_signal'], 'Signal'] = 1  # Compra
        data.loc[data['macd'] < data['macd_signal'], 'Signal'] = -1  # Venta
    
    # Eliminar señales repetidas
    data['Signal_Change'] = data['Signal'].diff()
    data.loc[data['Signal_Change'] == 0, 'Signal'] = 0
    
    return data

def backtest_strategy(df, initial_capital=10000):
    """
    Realiza un backtest simple de una estrategia de trading.
    
    Args:
        df (pandas.DataFrame): DataFrame con datos de precios y señales.
        initial_capital (float): Capital inicial.
        
    Returns:
        pandas.DataFrame: DataFrame con resultados del backtest.
        dict: Métricas de rendimiento.
    """
    data = df.copy()
    
    # Asegurarse de que existe la columna 'Signal'
    if 'Signal' not in data.columns:
        logger.error("No se encontró la columna 'Signal' en los datos")
        return None, None
    
    # Inicializar columnas
    data['Position'] = data['Signal'].shift(1)
    data['Position'].fillna(0, inplace=True)
    
    # Calcular retornos
    data['Market_Return'] = data['Close'].pct_change()
    data['Strategy_Return'] = data['Position'] * data['Market_Return']
    
    # Calcular capital acumulado
    data['Market_Cumulative'] = (1 + data['Market_Return']).cumprod() * initial_capital
    data['Strategy_Cumulative'] = (1 + data['Strategy_Return']).cumprod() * initial_capital
    
    # Calcular drawdown
    data['Market_Peak'] = data['Market_Cumulative'].cummax()
    data['Strategy_Peak'] = data['Strategy_Cumulative'].cummax()
    data['Market_Drawdown'] = (data['Market_Cumulative'] - data['Market_Peak']) / data['Market_Peak']
    data['Strategy_Drawdown'] = (data['Strategy_Cumulative'] - data['Strategy_Peak']) / data['Strategy_Peak']
    
    # Calcular métricas de rendimiento
    total_days = len(data)
    trading_days_per_year = 252
    years = total_days / trading_days_per_year
    
    # Retorno total
    market_total_return = (data['Market_Cumulative'].iloc[-1] / initial_capital) - 1
    strategy_total_return = (data['Strategy_Cumulative'].iloc[-1] / initial_capital) - 1
    
    # Retorno anualizado
    market_annual_return = (1 + market_total_return) ** (1 / years) - 1
    strategy_annual_return = (1 + strategy_total_return) ** (1 / years) - 1
    
    # Volatilidad anualizada
    market_volatility = data['Market_Return'].std() * np.sqrt(trading_days_per_year)
    strategy_volatility = data['Strategy_Return'].std() * np.sqrt(trading_days_per_year)
    
    # Ratio de Sharpe (asumiendo tasa libre de riesgo del 0%)
    market_sharpe = market_annual_return / market_volatility if market_volatility != 0 else 0
    strategy_sharpe = strategy_annual_return / strategy_volatility if strategy_volatility != 0 else 0
    
    # Máximo drawdown
    market_max_drawdown = data['Market_Drawdown'].min()
    strategy_max_drawdown = data['Strategy_Drawdown'].min()
    
    # Crear diccionario de métricas
    metrics = {
        'market': {
            'total_return': market_total_return,
            'annual_return': market_annual_return,
            'volatility': market_volatility,
            'sharpe_ratio': market_sharpe,
            'max_drawdown': market_max_drawdown
        },
        'strategy': {
            'total_return': strategy_total_return,
            'annual_return': strategy_annual_return,
            'volatility': strategy_volatility,
            'sharpe_ratio': strategy_sharpe,
            'max_drawdown': strategy_max_drawdown
        }
    }
    
    return data, metrics

def plot_backtest_results(backtest_data, metrics, title=None, save_path=None):
    """
    Grafica los resultados de un backtest.
    
    Args:
        backtest_data (pandas.DataFrame): DataFrame con resultados del backtest.
        metrics (dict): Métricas de rendimiento.
        title (str, optional): Título del gráfico.
        save_path (str, optional): Ruta para guardar el gráfico.
    """
    data = backtest_data.copy()
    
    # Crear figura con subplots
    fig, axs = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Evolución del capital
    axs[0].plot(data.index, data['Market_Cumulative'], label='Mercado', linewidth=2)
    axs[0].plot(data.index, data['Strategy_Cumulative'], label='Estrategia', linewidth=2)
    
    if title:
        axs[0].set_title(title, fontsize=16)
    else:
        axs[0].set_title('Backtest de Estrategia vs Mercado', fontsize=16)
    
    axs[0].set_ylabel('Capital ($)', fontsize=12)
    axs[0].legend(fontsize=12)
    axs[0].grid(True, alpha=0.3)
    
    # 2. Drawdown
    axs[1].fill_between(data.index, data['Market_Drawdown'] * 100, 0, alpha=0.3, color='blue', label='Mercado')
    axs[1].fill_between(data.index, data['Strategy_Drawdown'] * 100, 0, alpha=0.3, color='orange', label='Estrategia')
    axs[1].set_title('Drawdown (%)', fontsize=14)
    axs[1].set_xlabel('Fecha', fontsize=12)
    axs[1].set_ylabel('Drawdown (%)', fontsize=12)
    axs[1].legend(fontsize=12)
    axs[1].grid(True, alpha=0.3)
    
    # Formatear eje x para fechas
    plt.gcf().autofmt_xdate()
    
    # Añadir tabla de métricas
    plt.figtext(0.01, 0.01, f"""
    Métricas de Rendimiento:
    
    Mercado:
        Retorno Total: {metrics['market']['total_return']:.2%}
        Retorno Anual: {metrics['market']['annual_return']:.2%}
        Volatilidad: {metrics['market']['volatility']:.2%}
        Ratio de Sharpe: {metrics['market']['sharpe_ratio']:.2f}
        Máximo Drawdown: {metrics['market']['max_drawdown']:.2%}
        
    Estrategia:
        Retorno Total: {metrics['strategy']['total_return']:.2%}
        Retorno Anual: {metrics['strategy']['annual_return']:.2%}
        Volatilidad: {metrics['strategy']['volatility']:.2%}
        Ratio de Sharpe: {metrics['strategy']['sharpe_ratio']:.2f}
        Máximo Drawdown: {metrics['strategy']['max_drawdown']:.2%}
    """, fontsize=10, va='bottom')
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path)
        logger.info(f"Gráfico guardado en {save_path}")
    
    plt.show()

if __name__ == "__main__":
    # Ejemplo de uso
    print("Este es un módulo de utilidades y no debe ejecutarse directamente.")
