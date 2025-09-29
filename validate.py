#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar validaciones del proyecto de predicción de inversiones.
"""

import sys
import argparse
from pathlib import Path

# Agregar el directorio src al path para imports
sys.path.append('src')

try:
    from src.utils.validation_utils import generate_validation_report, print_validation_report
    from config import DEFAULT_TICKERS, AVAILABLE_MODELS
except ImportError as e:
    print(f"Error al importar módulos: {e}")
    print("Asegúrate de ejecutar 'python setup.py' primero.")
    sys.exit(1)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description='Validación del proyecto de predicción de inversiones')
    
    parser.add_argument('--tickers', nargs='+', default=DEFAULT_TICKERS[:3], 
                        help='Lista de tickers a validar')
    parser.add_argument('--models', nargs='+', default=AVAILABLE_MODELS,
                        help='Lista de modelos a validar')
    parser.add_argument('--save-report', action='store_true',
                        help='Guardar reporte en archivo JSON')
    
    return parser.parse_args()

def main():
    """Función principal del script de validación."""
    print("🔍 Script de Validación del Proyecto")
    print("=" * 50)
    
    args = parse_args()
    
    # Generar reporte de validación
    try:
        report = generate_validation_report(args.tickers, args.models)
        
        # Mostrar reporte
        print_validation_report(report)
        
        # Guardar reporte si se solicita
        if args.save_report:
            import json
            with open('validation_report.json', 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print("\n💾 Reporte guardado en: validation_report.json")
        
        # Retornar código de salida basado en issues críticos
        if report['summary']['total_issues'] > 10:
            print("\n❌ Muchos problemas detectados. Ejecuta 'python setup.py' para corregir.")
            return 1
        elif not report['summary']['ready_for_training']:
            print("\n⚠️  El proyecto no está completamente listo. Considera descargar datos primero.")
            print("   Ejecuta: python main.py --download --tickers", ' '.join(args.tickers[:3]))
            return 1
        else:
            print("\n✅ El proyecto está en buen estado!")
            return 0
        
    except Exception as e:
        print(f"\n❌ Error durante la validación: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)