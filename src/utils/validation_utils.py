#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidades para validaciones y verificaciones en el proyecto.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

def validate_ticker(ticker):
    """
    Valida si un ticker existe y tiene datos disponibles.
    
    Args:
        ticker (str): Símbolo del ticker
        
    Returns:
        tuple: (bool, str) - (es_válido, mensaje)
    """
    try:
        # Intentar descargar datos de los últimos 5 días
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            return False, f"No se encontraron datos para {ticker}"
        
        return True, f"Ticker {ticker} válido"
        
    except Exception as e:
        return False, f"Error al validar {ticker}: {str(e)}"

def validate_data_files(tickers, data_dir="data"):
    """
    Valida que existan los archivos de datos necesarios.
    
    Args:
        tickers (list): Lista de tickers a validar
        data_dir (str): Directorio de datos
        
    Returns:
        dict: Resultados de validación por ticker
    """
    results = {}
    
    for ticker in tickers:
        ticker_safe = ticker.replace('.', '-')
        files_to_check = [
            f"{ticker_safe}_data.csv",
            f"{ticker_safe}_processed.csv",
            f"{ticker_safe}_X_train.npy",
            f"{ticker_safe}_y_train.npy",
            f"{ticker_safe}_X_test.npy", 
            f"{ticker_safe}_y_test.npy"
        ]
        
        file_status = {}
        for file_name in files_to_check:
            file_path = os.path.join(data_dir, file_name)
            file_status[file_name] = os.path.exists(file_path)
        
        results[ticker] = file_status
    
    return results

def validate_model_files(tickers, model_types, models_dir="models"):
    """
    Valida que existan los archivos de modelos necesarios.
    
    Args:
        tickers (list): Lista de tickers
        model_types (list): Lista de tipos de modelos
        models_dir (str): Directorio de modelos
        
    Returns:
        dict: Resultados de validación por ticker y modelo
    """
    results = {}
    
    for ticker in tickers:
        results[ticker] = {}
        
        for model_type in model_types:
            files_to_check = [
                f"{ticker}_{model_type}_model.h5",
                f"scaler_{ticker}.pkl",
                f"features_{ticker}.txt"
            ]
            
            file_status = {}
            for file_name in files_to_check:
                file_path = os.path.join(models_dir, file_name)
                file_status[file_name] = os.path.exists(file_path)
            
            results[ticker][model_type] = file_status
    
    return results

def check_data_quality(data, ticker=None):
    """
    Verifica la calidad de los datos.
    
    Args:
        data (pd.DataFrame): DataFrame con datos
        ticker (str, optional): Nombre del ticker para logging
        
    Returns:
        dict: Reporte de calidad de datos
    """
    report = {
        'total_rows': len(data),
        'total_columns': len(data.columns),
        'missing_values': {},
        'infinite_values': {},
        'duplicated_rows': 0,
        'date_range': None,
        'issues': []
    }
    
    # Verificar valores faltantes
    missing = data.isnull().sum()
    report['missing_values'] = missing[missing > 0].to_dict()
    
    # Verificar valores infinitos
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        inf_count = np.isinf(data[col]).sum()
        if inf_count > 0:
            report['infinite_values'][col] = int(inf_count)
    
    # Verificar filas duplicadas
    report['duplicated_rows'] = data.duplicated().sum()
    
    # Verificar rango de fechas si existe columna de fecha
    if 'date' in data.columns:
        try:
            data['date'] = pd.to_datetime(data['date'])
            report['date_range'] = {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d'),
                'days': (data['date'].max() - data['date'].min()).days
            }
        except:
            report['issues'].append("Problema con formato de fechas")
    
    # Identificar problemas
    if report['missing_values']:
        report['issues'].append(f"Valores faltantes en {len(report['missing_values'])} columnas")
    
    if report['infinite_values']:
        report['issues'].append(f"Valores infinitos en {len(report['infinite_values'])} columnas")
    
    if report['duplicated_rows'] > 0:
        report['issues'].append(f"{report['duplicated_rows']} filas duplicadas")
    
    # Verificar datos suficientes para entrenamiento
    if len(data) < 100:
        report['issues'].append("Datos insuficientes para entrenamiento (< 100 filas)")
    
    return report

def validate_project_structure():
    """
    Valida la estructura completa del proyecto.
    
    Returns:
        dict: Reporte de validación de estructura
    """
    required_dirs = ['data', 'models', 'plots', 'results', 'src']
    required_files = ['main.py', 'requirements.txt', 'README.md']
    src_files = [
        'src/data_downloader.py',
        'src/data_processor.py', 
        'src/model_trainer.py',
        'src/utils/data_utils.py'
    ]
    
    report = {
        'directories': {},
        'required_files': {},
        'src_files': {},
        'issues': []
    }
    
    # Verificar directorios
    for dir_name in required_dirs:
        report['directories'][dir_name] = os.path.exists(dir_name)
        if not os.path.exists(dir_name):
            report['issues'].append(f"Directorio faltante: {dir_name}")
    
    # Verificar archivos principales
    for file_name in required_files:
        report['required_files'][file_name] = os.path.exists(file_name)
        if not os.path.exists(file_name):
            report['issues'].append(f"Archivo faltante: {file_name}")
    
    # Verificar archivos del código fuente
    for file_path in src_files:
        report['src_files'][file_path] = os.path.exists(file_path)
        if not os.path.exists(file_path):
            report['issues'].append(f"Archivo de código faltante: {file_path}")
    
    return report

def generate_validation_report(tickers, model_types=['lstm', 'gru', 'bilstm']):
    """
    Genera un reporte completo de validación del proyecto.
    
    Args:
        tickers (list): Lista de tickers
        model_types (list): Lista de tipos de modelos
        
    Returns:
        dict: Reporte completo de validación
    """
    print("Generando reporte de validación...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'project_structure': validate_project_structure(),
        'data_files': validate_data_files(tickers),
        'model_files': validate_model_files(tickers, model_types),
        'data_quality': {},
        'summary': {
            'total_issues': 0,
            'critical_issues': 0,
            'ready_for_training': False,
            'ready_for_prediction': False
        }
    }
    
    # Verificar calidad de datos si existen
    for ticker in tickers:
        ticker_safe = ticker.replace('.', '-')
        data_file = f"data/{ticker_safe}_processed.csv"
        
        if os.path.exists(data_file):
            try:
                data = pd.read_csv(data_file)
                report['data_quality'][ticker] = check_data_quality(data, ticker)
            except Exception as e:
                report['data_quality'][ticker] = {
                    'error': f"No se pudo leer {data_file}: {str(e)}"
                }
    
    # Calcular resumen
    total_issues = len(report['project_structure']['issues'])
    
    for ticker, files in report['data_files'].items():
        missing_files = [f for f, exists in files.items() if not exists]
        total_issues += len(missing_files)
    
    for ticker_data in report['data_quality'].values():
        if 'issues' in ticker_data:
            total_issues += len(ticker_data['issues'])
    
    report['summary']['total_issues'] = total_issues
    
    # Determinar si está listo para entrenamiento/predicción
    has_data = any(
        all(report['data_files'][ticker][f] for f in [
            f"{ticker.replace('.', '-')}_data.csv",
            f"{ticker.replace('.', '-')}_processed.csv"
        ])
        for ticker in tickers
    )
    
    has_models = any(
        all(report['model_files'][ticker][model_type][f] for f in [
            f"{ticker}_{model_type}_model.h5",
            f"scaler_{ticker}.pkl"
        ])
        for ticker in tickers
        for model_type in model_types
        if ticker in report['model_files'] and model_type in report['model_files'][ticker]
    )
    
    report['summary']['ready_for_training'] = has_data and total_issues < 5
    report['summary']['ready_for_prediction'] = has_models and total_issues < 3
    
    return report

def print_validation_report(report):
    """
    Imprime un reporte de validación de forma legible.
    
    Args:
        report (dict): Reporte de validación
    """
    print("\n" + "=" * 60)
    print("REPORTE DE VALIDACIÓN DEL PROYECTO")
    print("=" * 60)
    
    # Resumen
    summary = report['summary']
    print(f"\n📊 RESUMEN:")
    print(f"   • Issues totales: {summary['total_issues']}")
    print(f"   • Listo para entrenamiento: {'✅' if summary['ready_for_training'] else '❌'}")
    print(f"   • Listo para predicción: {'✅' if summary['ready_for_prediction'] else '❌'}")
    
    # Estructura del proyecto
    print(f"\n🏗️  ESTRUCTURA DEL PROYECTO:")
    structure = report['project_structure']
    for category, items in [
        ('Directorios', structure['directories']),
        ('Archivos principales', structure['required_files']),
        ('Archivos de código', structure['src_files'])
    ]:
        print(f"   {category}:")
        for name, exists in items.items():
            status = "✅" if exists else "❌"
            print(f"     {status} {name}")
    
    # Archivos de datos
    print(f"\n📁 ARCHIVOS DE DATOS:")
    for ticker, files in report['data_files'].items():
        print(f"   {ticker}:")
        for file_name, exists in files.items():
            status = "✅" if exists else "❌"
            print(f"     {status} {file_name}")
    
    # Calidad de datos
    if report['data_quality']:
        print(f"\n🔍 CALIDAD DE DATOS:")
        for ticker, quality in report['data_quality'].items():
            if 'error' in quality:
                print(f"   {ticker}: ❌ {quality['error']}")
            else:
                status = "✅" if not quality['issues'] else "⚠️"
                print(f"   {ticker}: {status} {quality['total_rows']} filas, {quality['total_columns']} columnas")
                if quality['issues']:
                    for issue in quality['issues']:
                        print(f"     • {issue}")
    
    # Issues críticos
    if structure['issues']:
        print(f"\n⚠️  PROBLEMAS DETECTADOS:")
        for issue in structure['issues']:
            print(f"   • {issue}")
    
    print("\n" + "=" * 60)