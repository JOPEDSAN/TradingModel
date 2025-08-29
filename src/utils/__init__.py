#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de inicialización para el paquete utils.
"""

# Importar las utilidades para hacerlas accesibles desde src.utils
from .trading_utils import (
    ensure_dir_exists,
    load_json,
    save_json,
    evaluate_predictions,
    plot_metrics_comparison,
    plot_prediction_vs_actual,
    calculate_trading_signals,
    backtest_strategy,
    plot_backtest_results
)

__all__ = [
    'ensure_dir_exists',
    'load_json',
    'save_json',
    'evaluate_predictions',
    'plot_metrics_comparison',
    'plot_prediction_vs_actual',
    'calculate_trading_signals',
    'backtest_strategy',
    'plot_backtest_results'
]
