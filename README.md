# Proyecto de Predicción de Inversiones

Este proyecto implementa un sistema de predicción para trading basado en datos históricos de Yahoo Finance, utilizando modelos de aprendizaje profundo con series temporales.

## Características

- Descarga automática de datos históricos de Yahoo Finance
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
│   ├── model_trainer.py   # Definición y entrenamiento de modelos
│   └── utils/             # Utilidades y funciones auxiliares
│
├── .gitignore             # Archivos a ignorar por Git
├── main.py                # Script principal para ejecutar el pipeline
└── README.md              # Documentación del proyecto
```

## Requisitos

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- yfinance
- tensorflow
- statsmodels
- joblib
- tqdm
- pandas-datareader

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
