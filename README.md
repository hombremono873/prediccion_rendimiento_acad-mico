# Predicción de la productividad estudiantil

Aplicación en Streamlit que recorre el flujo completo de un proyecto de regresión sobre el dataset de Kaggle [Student Performance Prediction](https://www.kaggle.com/competitions/student-performance-prediction-machine-learning-challenge/data): extracción, transformación y carga (ETL), análisis de variables, entrenamiento de una regresión lineal y predicción con datos ingresados por el usuario.

## Requisitos

- Python 3.11 o superior (probado con 3.13).
- Los datos de Kaggle, que no se incluyen en el repositorio (ver [Datos](#datos)).

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Datos

1. Descarga `train.csv` (y opcionalmente `test.csv`) desde la [página de datos de la competencia](https://www.kaggle.com/competitions/student-performance-prediction-machine-learning-challenge/data).
2. Copia los archivos en la carpeta `data/`.

Si prefieres no copiarlos, en la sección **Extracción** de la app puedes subir el CSV directamente.

## Ejecución

```bash
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`.

## Flujo de la aplicación

Cada sección de la barra lateral depende de la anterior; si falta un paso, la app indica a cuál ir.

| Sección | Qué hace |
|---|---|
| 1. Extracción | Carga `data/train.csv` o un CSV subido; muestra filas, columnas, tipos y una vista previa. |
| 2. Transformación | Traduce columnas y categorías al español, elimina `id` e `id_estudiante`, elimina nulos y duplicados, y crea dummies de `genero` (`drop_first=True`). |
| 3. Carga | Divide en entrenamiento y prueba con el porcentaje que elija el usuario (10 % a 50 %, 20 % por defecto, semilla 42), guarda `data/procesado.csv` y permite descargarlo. |
| 4. Análisis de variables | Histogramas (con curva KDE solo en variables continuas), tabla de Pearson, Spearman e información mutua, y gráfico de información mutua. Usa solo el conjunto de entrenamiento. |
| 5. Selección de variables | Modo "Variables seleccionadas" (6 por defecto, editables) o "Todas las variables", más una lista de variables a eliminar. Cambiar la selección descarta el modelo entrenado. |
| 6. Entrenamiento | Pipeline `StandardScaler` + `LinearRegression`, validación cruzada KFold (5), métricas, coeficientes y gráficos de real vs. predicho, coeficientes y residuos. Guarda `models/modelo.joblib`. |
| 7. Predicción | Formulario generado a partir de las variables del modelo (mínimo, máximo y mediana de entrenamiento) y predicción por lotes desde un CSV, que se descarga con las columnas `id` y `productivity_score`. |

## Estructura

```
app.py              Punto de entrada, barra lateral y secciones
src/config.py       Constantes: rutas, traducciones, variables por defecto, semilla
src/etl.py          Extracción, transformación y carga
src/analisis.py     Correlaciones e información mutua
src/modelo.py       Pipeline, validación cruzada, métricas y persistencia
src/prediccion.py   Predicción individual y por lotes
src/graficos.py     Visualizaciones con plotly
src/interfaz.py     Estilos, encabezados y créditos
.streamlit/         Tema visual (oscuro con acentos azules)
data/               Datos de entrada y dataset procesado
models/             Modelo entrenado
```

## Autor

**Omar Alberto Torres**, estudiante de la Maestría en Análisis de Datos.

## Nota sobre los resultados

El dataset es sintético: `puntaje_productividad` es una combinación lineal de seis variables redondeada a dos decimales. Por eso la regresión lineal obtiene R² ≈ 1 y un error cercano al del redondeo (MAE ≈ 0.0025), y las demás variables reciben coeficientes prácticamente nulos.
