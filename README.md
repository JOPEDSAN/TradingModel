# Trading Model

**Modelo para trading incluyendo datos macroeconómicos y análisis de noticias**

- 👨‍💻 **Autor:** José Francisco Pedrero Sánchez
- 📧 **Email:** jopedsan@ibv.org
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

# Opción 1: Usar índices bursátiles principales (recomendado para principiantes)
python main.py --all --indices --years 3

# Opción 2: Usar acciones tecnológicas principales
python main.py --all --stocks --years 3

# Opción 3: Combinar índices y ETFs
python main.py --all --indices --etfs --years 3

# Opción 4: Especificar tickers específicos
python main.py --all --tickers ^GSPC ^DJI AAPL MSFT --years 3
```

### 🎯 Selección de Activos

**Trading Model** permite analizar diferentes tipos de activos financieros:

#### **📈 Índices Bursátiles (--indices)**
```bash
python main.py --download --indices
# Incluye: S&P 500, Dow Jones, NASDAQ, Russell 2000, VIX, FTSE, DAX, CAC 40, Nikkei 225, Hang Seng
```

#### **🏢 Acciones Individuales (--stocks)**
```bash
python main.py --download --stocks
# Incluye: FAANG, Microsoft, Tesla, NVIDIA, bancos principales, y otras empresas destacadas
```

#### **📊 ETFs (--etfs)**
```bash
python main.py --download --etfs
# Incluye: SPY, QQQ, VTI, fondos sectoriales, bonos, oro, mercados internacionales
```

#### **₿ Criptomonedas (--crypto)**
```bash
python main.py --download --crypto
# Incluye: Bitcoin, Ethereum, principales altcoins (a través de Yahoo Finance)
```

#### **🥇 Materias Primas (--commodities)**
```bash
python main.py --download --commodities
# Incluye: Oro, petróleo, gas natural, cobre, productos agrícolas
```

#### **🎛️ Combinaciones Personalizadas**
```bash
# Análisis diversificado (índices + ETFs + algunas acciones)
python main.py --all --indices --etfs --tickers AAPL TSLA BTC-USD

# Análisis de materias primas y criptomonedas
python main.py --all --commodities --crypto --years 2

# Solo tickers específicos
python main.py --all --tickers ^GSPC QQQ GLD BTC-USD CL=F
```

### 📊 Ejecución Paso a Paso

#### **1. Descarga de Datos**

**Descargar índices bursátiles principales:**
```bash
python main.py --download --indices --years 5
```

**Descargar diferentes tipos de activos:**
```bash
# Solo acciones tecnológicas
python main.py --download --stocks --years 3

# ETFs y materias primas
python main.py --download --etfs --commodities --years 5

# Criptomonedas (datos más limitados)
python main.py --download --crypto --years 2

# Combinación personalizada
python main.py --download --tickers ^GSPC SPY AAPL BTC-USD GC=F --years 5
```

#### **2. Procesamiento de Datos**

**Procesar todos los datos descargados:**
```bash
python main.py --process
```

#### **3. Entrenamiento de Modelos**

**Entrenar modelos para activos específicos:**
```bash
# Entrenar para índices principales
python main.py --train --indices --models lstm gru

# Entrenar para tickers específicos
python main.py --train --tickers ^GSPC SPY AAPL --models bilstm
```

#### **4. Predicciones**

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

### 🛠️ Parámetros Disponibles

#### **Selección de Activos:**
- `--indices`: Índices bursátiles principales (S&P 500, Dow Jones, NASDAQ, etc.)
- `--stocks`: Acciones individuales (FAANG, tecnológicas, financieras)
- `--etfs`: ETFs principales (SPY, QQQ, sectoriales, internacionales)
- `--crypto`: Criptomonedas (Bitcoin, Ethereum, altcoins principales)
- `--commodities`: Materias primas (oro, petróleo, productos agrícolas)
- `--tickers`: Lista específica de tickers (ej: `AAPL MSFT ^GSPC`)

#### **Configuración de Modelos:**
- `--models`: Tipos de modelos (`lstm`, `gru`, `bilstm`)
- `--years`: Años de datos históricos (por defecto: 5)
- `--seq-length`: Longitud de secuencia (por defecto: 60)
- `--horizon`: Horizonte de predicción en días (por defecto: 1)
- `--future-days`: Días futuros a predecir (por defecto: 30)

### 💡 Ejemplos Prácticos

#### **Análisis de Mercado General:**
```bash
# Análisis completo de índices principales (recomendado para principiantes)
python main.py --all --indices --years 3

# Monitoreo de volatilidad del mercado
python main.py --all --tickers ^GSPC ^VIX --years 2 --future-days 7
```

#### **Trading de Acciones Tecnológicas:**
```bash
# Análisis de las Big Tech
python main.py --all --tickers AAPL MSFT GOOGL AMZN META --years 5

# Comparación entre acciones tech y ETF tecnológico
python main.py --all --tickers AAPL MSFT NVDA QQQ XLK --years 3
```

#### **Diversificación de Portafolio:**
```bash
# Análisis diversificado: acciones, bonos, oro, petróleo
python main.py --all --tickers SPY TLT GLD CL=F --years 5

# ETFs sectoriales para diversificación
python main.py --all --tickers XLF XLK XLE XLV XLI --years 3
```

#### **Trading de Criptomonedas:**
```bash
# Principales criptomonedas
python main.py --all --crypto --years 2 --future-days 14

# Bitcoin vs mercado tradicional
python main.py --all --tickers BTC-USD ^GSPC GLD --years 3
```

#### **Análisis de Materias Primas:**
```bash
# Oro, petróleo y productos agrícolas
python main.py --all --commodities --years 5

# Correlación entre inflación y materias primas
python main.py --all --tickers GC=F CL=F DJP TIP --years 3
```

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
