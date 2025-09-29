#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Trading Model - Modelo para trading incluyendo datos macroeconómicos y análisis de noticias.

Información del Proyecto:
------------------------
Nombre: Trading model
Versión: 0.1.0
Autor: José Francisco Pedrero Sánchez
Email: jopedsan@ibv.org
Descripción: Modelo para trading incluyendo datos macroeconómicos y análisis de noticias.

Script para verificar e instalar dependencias del proyecto de predicción de inversiones.
"""

import subprocess
import sys
import os
import importlib
from pathlib import Path

# Información del proyecto
PROJECT_INFO = {
    "name": "Trading model",
    "version": "0.1.0",
    "author": "José Francisco Pedrero Sánchez",
    "email": "jopedsan@ibv.org",
    "description": "Modelo para trading incluyendo datos macroeconómicos y análisis de noticias."
}

# Lista de dependencias críticas con sus comandos de instalación alternativos
DEPENDENCIES = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'scikit-learn': 'scikit-learn',
    'yfinance': 'yfinance',
    'tensorflow': 'tensorflow',
    'joblib': 'joblib',
    'tqdm': 'tqdm',
    'requests': 'requests',
    'python-dotenv': 'python-dotenv',
    'textblob': 'textblob',
    'vaderSentiment': 'vaderSentiment',
    'nltk': 'nltk',
    'newspaper3k': 'newspaper3k',
}

# Dependencias opcionales (APIs externas)
OPTIONAL_DEPENDENCIES = {
    'fredapi': 'fredapi',
    'finnhub-python': 'finnhub-python', 
    'alpha_vantage': 'alpha_vantage',
    'newsapi-python': 'newsapi-python',
    'plotly': 'plotly',
    'dash': 'dash',
    'TA-Lib': 'TA-Lib'
}

def check_dependency(package_name, import_name=None):
    """
    Verifica si una dependencia está instalada.
    
    Args:
        package_name (str): Nombre del paquete en pip
        import_name (str): Nombre para importar (si es diferente del paquete)
    
    Returns:
        bool: True si está instalada, False en caso contrario
    """
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name):
    """
    Instala un paquete usando pip.
    
    Args:
        package_name (str): Nombre del paquete a instalar
    
    Returns:
        bool: True si la instalación fue exitosa, False en caso contrario
    """
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def check_and_install_dependencies():
    """
    Verifica e instala todas las dependencias necesarias.
    """
    print("=== Verificación de Dependencias ===")
    
    missing_critical = []
    missing_optional = []
    
    # Verificar dependencias críticas
    print("\n--- Dependencias Críticas ---")
    for package, pip_name in DEPENDENCIES.items():
        import_name = package.replace('-', '_')
        if package == 'scikit-learn':
            import_name = 'sklearn'
        elif package == 'python-dotenv':
            import_name = 'dotenv'
        elif package == 'vaderSentiment':
            import_name = 'vaderSentiment'
        
        if check_dependency(package, import_name):
            print(f"✓ {package} - Instalado")
        else:
            print(f"✗ {package} - No encontrado")
            missing_critical.append((package, pip_name))
    
    # Verificar dependencias opcionales
    print("\n--- Dependencias Opcionales ---")
    for package, pip_name in OPTIONAL_DEPENDENCIES.items():
        import_name = package.replace('-', '_')
        if package == 'TA-Lib':
            import_name = 'talib'
        elif package == 'finnhub-python':
            import_name = 'finnhub'
        elif package == 'newsapi-python':
            import_name = 'newsapi'
        
        if check_dependency(package, import_name):
            print(f"✓ {package} - Instalado")
        else:
            print(f"- {package} - No encontrado (opcional)")
            missing_optional.append((package, pip_name))
    
    # Instalar dependencias críticas faltantes
    if missing_critical:
        print(f"\n--- Instalando {len(missing_critical)} dependencias críticas ---")
        for package, pip_name in missing_critical:
            print(f"Instalando {package}...")
            if install_package(pip_name):
                print(f"✓ {package} instalado exitosamente")
            else:
                print(f"✗ Error al instalar {package}")
                if package == 'TA-Lib':
                    print("  NOTA: TA-Lib puede requerir instalación especial.")
                    print("  En Windows: pip install --find-links https://github.com/cgohlke/pythonlibs/releases TA-Lib")
                    print("  En Linux/Mac: sudo apt-get install ta-lib / brew install ta-lib")
    
    # Mostrar información sobre dependencias opcionales
    if missing_optional:
        print(f"\n--- Dependencias opcionales no instaladas ({len(missing_optional)}) ---")
        print("Estas dependencias proporcionan funcionalidades adicionales:")
        for package, pip_name in missing_optional:
            print(f"  - {package}: Para instalar ejecuta 'pip install {pip_name}'")
    
    print("\n=== Verificación completada ===")
    
    if missing_critical:
        print(f"⚠️  Hay {len(missing_critical)} dependencias críticas faltantes.")
        print("El proyecto puede no funcionar correctamente hasta instalarlas.")
        return False
    else:
        print("✅ Todas las dependencias críticas están instaladas.")
        return True

def verify_environment():
    """
    Verifica la configuración del entorno.
    """
    print("\n=== Verificación del Entorno ===")
    
    # Verificar Python version
    python_version = sys.version_info
    print(f"Versión de Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("⚠️  Se recomienda Python 3.8 o superior")
    else:
        print("✅ Versión de Python compatible")
    
    # Verificar entorno virtual
    in_conda = 'CONDA_DEFAULT_ENV' in os.environ
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_conda:
        conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
        print(f"✅ Entorno Conda activo: {conda_env}")
    elif in_venv:
        print("✅ Entorno virtual activo (venv/virtualenv)")
    else:
        print("⚠️  No se detectó entorno virtual")
        print("   Se recomienda usar: conda create -n trading_model python=3.9")
    
    # Verificar directorios del proyecto
    project_dirs = ['data', 'models', 'plots', 'results', 'src']
    print("\nDirectorios del proyecto:")
    for dir_name in project_dirs:
        if Path(dir_name).exists():
            print(f"✓ {dir_name}/")
        else:
            print(f"- {dir_name}/ (se creará automáticamente)")
    
    # Verificar archivos clave
    key_files = ['main.py', 'requirements.txt', 'README.md']
    print("\nArchivos clave:")
    for file_name in key_files:
        if Path(file_name).exists():
            print(f"✓ {file_name}")
        else:
            print(f"✗ {file_name} - No encontrado")
    
    # Verificar archivo .env
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env - Archivo de configuración encontrado")
    else:
        print("- .env - Archivo de configuración no encontrado (opcional)")
        print("  Crear .env para configurar claves API:")
        print("  FRED_API_KEY=tu_clave")
        print("  NEWS_API_KEY=tu_clave")
        print("  FINNHUB_API_KEY=tu_clave")
        print("  ALPHAVANTAGE_API_KEY=tu_clave")

def show_project_info():
    """
    Muestra información detallada del proyecto.
    """
    print(f"\n📋 INFORMACIÓN DEL PROYECTO:")
    print(f"   • Nombre: {PROJECT_INFO['name']}")
    print(f"   • Versión: {PROJECT_INFO['version']}")
    print(f"   • Autor: {PROJECT_INFO['author']}")
    print(f"   • Email: {PROJECT_INFO['email']}")
    print(f"   • Descripción: {PROJECT_INFO['description']}")

def create_sample_env():
    """
    Crea un archivo .env de ejemplo.
    """
    env_content = """# Configuración de APIs para el proyecto de predicción de inversiones
# Descomenta y completa las claves que tengas disponibles

# Federal Reserve Economic Data (FRED) - Para datos macroeconómicos
# Obtén tu clave en: https://fred.stlouisfed.org/docs/api/api_key.html
# FRED_API_KEY=tu_clave_fred_aqui

# News API - Para noticias generales
# Obtén tu clave en: https://newsapi.org/register
# NEWS_API_KEY=tu_clave_newsapi_aqui

# Finnhub - Para noticias financieras
# Obtén tu clave en: https://finnhub.io/register
# FINNHUB_API_KEY=tu_clave_finnhub_aqui

# Alpha Vantage - Para datos adicionales
# Obtén tu clave en: https://www.alphavantage.co/support/#api-key
# ALPHAVANTAGE_API_KEY=tu_clave_alphavantage_aqui
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✓ Archivo .env.example creado con plantilla de configuración")

def main():
    """
    Función principal del script de setup.
    """
    print(f"🔧 {PROJECT_INFO['name']} - Setup v{PROJECT_INFO['version']}")
    print(f"👨‍💻 Autor: {PROJECT_INFO['author']} ({PROJECT_INFO['email']})")
    print(f"📊 {PROJECT_INFO['description']}")
    print("=" * 70)
    
    # Mostrar información del proyecto
    show_project_info()
    
    # Verificar e instalar dependencias
    deps_ok = check_and_install_dependencies()
    
    # Verificar entorno
    verify_environment()
    
    # Crear archivo .env de ejemplo
    if not Path('.env.example').exists():
        create_sample_env()
    
    print("\n" + "=" * 70)
    if deps_ok:
        print("🎉 Setup completado exitosamente!")
        print(f"\n📋 {PROJECT_INFO['name']} v{PROJECT_INFO['version']} está listo para usar")
        print("\n🚀 Próximos pasos:")
        print("1. (Opcional) Configura .env con claves API: copy .env.example .env")
        print("2. Valida la instalación: python validate.py --tickers AAPL MSFT")
        print("3. Ejecuta el pipeline: python main.py --all --tickers AAPL MSFT GOOGL")
        print("\n💡 Consejos:")
        print("   • Usa siempre un entorno virtual (conda recomendado)")
        print("   • Para ayuda: python main.py --help")
        print(f"   • Contacto: {PROJECT_INFO['email']}")
    else:
        print("⚠️  Setup completado con advertencias")
        print("Revisa las dependencias faltantes antes de continuar")
        print(f"Para soporte contacta: {PROJECT_INFO['email']}")

if __name__ == "__main__":
    main()