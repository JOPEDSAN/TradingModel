# Proyecto de Predicción de Inversiones

Este proyecto implementa un sistema avanzado de predicción para trading basado en datos históricos de Yahoo Finance, indicadores macroeconómicos y análisis de sentimiento de noticias, utilizando modelos de aprendizaje profundo con series temporales.

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

## Requisitos

- Python 3.8+
- pandas, numpy, matplotlib, seaborn, scikit-learn
- yfinance, pandas-datareader (datos financieros)
- tensorflow, keras (modelos de deep learning)
- TA-Lib (indicadores técnicos)
- fredapi, requests (datos económicos)
- newspaper3k, textblob, vaderSentiment, nltk (análisis de noticias)
- python-dotenv (gestión de variables de entorno)
- finnhub-python, alpha_vantage, newsapi-python (APIs financieras adicionales)

## Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto con las siguientes variables:

```
FRED_API_KEY=tu_clave_api_fred
NEWS_API_KEY=tu_clave_api_noticias
FINNHUB_API_KEY=tu_clave_api_finnhub
ALPHAVANTAGE_API_KEY=tu_clave_api_alphavantage
```

> Nota: Puedes obtener estas claves API gratuitas registrándote en sus respectivos sitios web.

## Instalación

1. Clona este repositorio:
```bash
git clone https://github.com/tu-usuario/prediccion_inversiones.git
cd prediccion_inversiones
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Pipeline Completo

Para ejecutar el pipeline completo (descarga, procesamiento, entrenamiento y predicción):

```bash
python main.py --all --tickers AAPL MSFT GOOGL AMZN --years 20
```

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

## Contribución

Las contribuciones son bienvenidas. Para cambios importantes, por favor abre primero un issue para discutir lo que te gustaría cambiar.

## Licencia

[MIT](https://choosealicense.com/licenses/mit/)

## Descargo de Responsabilidad

Este proyecto es solo para fines educativos y de investigación. No constituye asesoramiento financiero. Invertir en mercados financieros conlleva riesgos, y las decisiones de inversión deben tomarse bajo tu propia responsabilidad.
