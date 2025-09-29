# Trading Model

**Modelo para trading incluyendo datos macroeconómicos y análisis de noticias**

- 👨‍💻 **Autor:** José Francisco Pedrero Sánchez
- 📧 **Email:** jopedsan@ib.org
- 🔢 **Versión:** 0.1.0

Este proyecto implementa un sistema avanzado de predicción para trading basado en datos históricos de Yahoo Finance, indicadores macroeconómicos y análisis de sentimiento de noticias, utilizando modelos de aprendizaje profundo con series temporales.

## 🚀 Mejoras Implementadas (v2.0)

- ✅ **Validaciones robustas** de datos y configuración
- ✅ **Manejo mejorado de errores** con logging detallado  
- ✅ **Script de setup automático** para dependencias
- ✅ **Configuración centralizada** en `config.py`
- ✅ **Utilidades de validación** para verificar integridad del proyecto
- ✅ **Compatibilidad mejorada** con versiones recientes de pandas
- ✅ **Manejo seguro de fechas** y secuencias temporales
- ✅ **Documentación expandida** con ejemplos prácticos

## Características

- Descarga automática de datos históricos de Yahoo Finance (20 años de historial)
- Integración de indicadores macroeconómicos (inflación, tipos de interés, PIB, etc.)
- Análisis de sentimiento de noticias financieras
- Procesamiento de datos y cálculo de indicadores técnicos
- Entrenamiento de múltiples modelos de series temporales (LSTM, GRU, BiLSTM)
- Evaluación y comparación de modelos
- Predicciones futuras para apoyo en decisiones de inversión
- Visualizaciones de resultados

## Estructura del Proyecto

```
prediccion_inversiones/
│
├── data/                  # Almacenamiento de datos crudos y procesados
│   ├── economic/          # Datos económicos (inflación, tipos de interés, etc.)
│   ├── news/              # Datos de noticias y análisis de sentimiento
│   ├── processed/         # Datos procesados para entrenamiento
│   └── raw/               # Datos crudos de Yahoo Finance
│
├── models/                # Modelos entrenados y escaladores
│
├── notebooks/             # Jupyter notebooks para análisis exploratorio
│
├── plots/                 # Visualizaciones generadas
│
├── results/               # Resultados y predicciones
│
├── src/                   # Código fuente
│   ├── data_downloader.py # Descarga de datos de Yahoo Finance
│   ├── data_processor.py  # Procesamiento y características
│   ├── economic_data.py   # Obtención y procesamiento de datos macroeconómicos
│   ├── news_analyzer.py   # Análisis de sentimiento de noticias financieras
│   ├── model_trainer.py   # Definición y entrenamiento de modelos
│   └── utils/             # Utilidades y funciones auxiliares
│       ├── data_utils.py  # Utilidades para el procesamiento de datos
│
├── .gitignore             # Archivos a ignorar por Git
├── main.py                # Script principal para ejecutar el pipeline
├── requirements.txt       # Dependencias del proyecto
└── README.md              # Documentación del proyecto
```

## 📋 Requisitos del Sistema

### Requisitos Mínimos
- **Python:** 3.8 o superior
- **RAM:** 4GB mínimo, 8GB recomendado
- **Espacio en disco:** 2GB para datos y modelos
- **Conexión a internet:** Para descarga de datos financieros

### Gestión de Entornos
Se **recomienda encarecidamente** usar un entorno virtual:
- **Conda** (recomendado): `conda create -n trading_model python=3.11.11`
- **venv**: `python -m venv trading_env`

### Dependencias Principales
Las siguientes librerías se instalan automáticamente con `setup.py`:

**Críticas (instalación automática):**
- pandas, numpy, matplotlib, seaborn, scikit-learn
- yfinance (datos financieros)
- tensorflow (modelos de deep learning)
- textblob, vaderSentiment, nltk (análisis de sentimiento)
- joblib, tqdm, requests, python-dotenv

**Opcionales (funcionalidades adicionales):**
- fredapi, finnhub-python, alpha_vantage, newsapi-python (APIs)
- TA-Lib (indicadores técnicos avanzados)
- plotly, dash (visualizaciones interactivas)

## Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
FRED_API_KEY=tu_clave_api_fred
NEWS_API_KEY=tu_clave_api_noticias
FINNHUB_API_KEY=tu_clave_api_finnhub
ALPHAVANTAGE_API_KEY=tu_clave_api_alphavantage
```

> Nota: Puedes obtener estas claves API gratuitas registrándote en sus respectivos sitios web.

## 🚀 Inicio Rápido

### Opción 1: Setup con Conda (Recomendado)

1. **Crea y activa un entorno conda:**
```bash
conda create -n trading_model python=3.11.11
conda activate trading_model
```

2. **Descarga o crea los archivos del proyecto:**
```bash
# Si tienes el código fuente, navega a la carpeta del proyecto
cd prediccion_inversiones

# O crea una nueva carpeta para el proyecto
mkdir trading_model
cd trading_model
```

3. **Ejecuta el setup automático:**
```bash
python setup.py
```
Este script verificará e instalará automáticamente todas las dependencias necesarias usando pip dentro del entorno conda.

4. **Configura las claves API (opcional):**
```bash
# Copia el archivo de ejemplo y edítalo con tus claves
copy .env.example .env
# Edita .env con tus claves API
```

5. **Ejecuta el pipeline completo:**
```bash
python main.py --all --tickers AAPL MSFT GOOGL
```

### Opción 2: Setup con pip (Alternativo)

1. **Crea un entorno virtual:**
```bash
python -m venv trading_env
# En Windows:
trading_env\Scripts\activate
# En Linux/Mac:
source trading_env/bin/activate
```

2. **Navega al directorio del proyecto y ejecuta setup:**
```bash
cd prediccion_inversiones
python setup.py
```

3. **Verifica la instalación:**
```bash
python validate.py --tickers AAPL MSFT
```

### ⚠️ Nota Importante
- **Siempre usa un entorno virtual** (conda o venv) para evitar conflictos de dependencias
- **El setup.py instalará automáticamente** todas las dependencias necesarias
- **Las claves API son opcionales** pero recomendadas para funcionalidad completa

## 💻 Uso del Sistema

### 🚀 Ejecución Rápida (Pipeline Completo)

**Para usuarios que quieren resultados inmediatos:**
```bash
# Activar el entorno conda
conda activate trading_model

# Ejecutar pipeline completo con tickers populares
python main.py --all --tickers AAPL MSFT GOOGL --years 3
```

### 📊 Ejecución Paso a Paso

### Descarga de Datos

Para descargar solo datos históricos, económicos y noticias:

```bash
python main.py --download --tickers AAPL MSFT GOOGL AMZN --years 20
```

### Procesamiento de Datos

Para procesar los datos descargados:

```bash
python main.py --process
```

### Entrenamiento de Modelos

Para entrenar los modelos con los datos procesados:

```bash
python main.py --train --models lstm gru bilstm
```

### Predicción

Para generar predicciones con los modelos entrenados:

```bash
python main.py --predict --days 30
```

## Módulos Principales

### Data Downloader

Descarga datos históricos de Yahoo Finance para los tickers especificados. Ahora incluye datos de 20 años de historia.

### Economic Data

Obtiene indicadores macroeconómicos de la API FRED y otras fuentes, incluyendo:
- Inflación (CPI)
- Tipos de interés
- Valor de las monedas
- PIB
- Desempleo
- Oferta monetaria
- Índices de volatilidad

### News Analyzer

Recopila y analiza noticias financieras para calcular el sentimiento del mercado:
- Análisis de sentimiento con TextBlob y VADER
- Categorización de noticias
- Agregación diaria de sentimientos
- Correlación con movimientos del mercado

Para ejecutar todo el pipeline (descarga, procesamiento, entrenamiento y predicción):

```bash
python main.py --all
```

### Operaciones Individuales

Descarga de datos:
```bash
python main.py --download --tickers AAPL MSFT GOOGL --years 5
```

Procesamiento de datos:
```bash
python main.py --process
```

Entrenamiento de modelos:
```bash
python main.py --train --models lstm gru
```

Predicciones:
```bash
python main.py --predict --future-days 30
```

### Parámetros

- `--tickers`: Lista de tickers a procesar (por defecto incluye índices principales y grandes tecnológicas)
- `--models`: Tipos de modelos a entrenar (`lstm`, `gru`, `bilstm`)
- `--years`: Años de datos históricos a descargar
- `--seq-length`: Longitud de la secuencia para los modelos
- `--horizon`: Horizonte de predicción (días en el futuro)
- `--future-days`: Días futuros a predecir

## Indicadores Técnicos Implementados

- Medias móviles (5, 10, 20, 50, 200 días)
- Cruces de medias móviles
- RSI (Índice de fuerza relativa)
- MACD (Convergencia/Divergencia de Medias Móviles)
- Bandas de Bollinger
- Volatilidad en diferentes períodos
- Momentum
- Indicadores de volumen

## Modelos Implementados

1. **LSTM (Long Short-Term Memory)**: Redes neuronales recurrentes especializadas en memoria de largo plazo.

2. **GRU (Gated Recurrent Unit)**: Variante de LSTM con estructura más simple y eficiente.

3. **BiLSTM (Bidirectional LSTM)**: LSTM bidireccional que procesa los datos en ambas direcciones.

## Próximas Mejoras

- Implementación de estrategias de trading basadas en las predicciones
- Incorporación de análisis de sentimiento de noticias financieras
- Optimización de hiperparámetros automatizada
- Interfaz web para visualización de resultados en tiempo real
- Backtesting de estrategias

## 🤝 Contribución

Las contribuciones son bienvenidas. Para cambios importantes, por favor contacta al autor antes de realizar modificaciones.

**Contacto del Autor:**
- 👨‍💻 José Francisco Pedrero Sánchez
- 📧 jopedsan@ib.org

## 📄 Licencia

[MIT](https://choosealicense.com/licenses/mit/)

## ⚠️ Descargo de Responsabilidad

Este proyecto es solo para fines educativos y de investigación. **No constituye asesoramiento financiero**. Invertir en mercados financieros conlleva riesgos, y las decisiones de inversión deben tomarse bajo tu propia responsabilidad y con el debido asesoramiento profesional.

---

### 📞 Soporte y Contacto

Para preguntas, problemas o sugerencias sobre Trading Model v0.1.0:
- 📧 **Email:** jopedsan@ib.org
- 🐛 **Reportar bugs:** Contacta directamente por email
- 💡 **Sugerencias:** Todas las ideas de mejora son bienvenidas
