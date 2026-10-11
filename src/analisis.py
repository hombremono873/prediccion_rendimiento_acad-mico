"""Análisis exploratorio: tendencia central, atípicos, correlaciones, relación con el objetivo
y reconstrucción de la fórmula que generó los datos sintéticos."""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score

from src import config


def es_discreta(serie: pd.Series) -> bool:
    """Indica si una variable es discreta (entera y con pocos valores distintos).

    Args:
        serie (pd.Series): Variable a evaluar.

    Returns:
        bool: True si la variable es discreta.
    """
    return pd.api.types.is_integer_dtype(serie) and serie.nunique() <= config.MAX_VALORES_DISCRETA


def calcular_informacion_mutua(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calcula la información mutua de cada predictor con la variable objetivo.

    Args:
        X (pd.DataFrame): Predictores numéricos.
        y (pd.Series): Variable objetivo.

    Returns:
        pd.Series: Información mutua por variable.
    """
    discretas = [es_discreta(X[c]) for c in X.columns]
    puntajes = mutual_info_regression(X, y, discrete_features=discretas, random_state=config.SEMILLA)
    return pd.Series(puntajes, index=X.columns)


@st.cache_data(show_spinner="Calculando correlaciones e información mutua...")
def tabla_relaciones(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Construye la tabla de Pearson, Spearman e información mutua respecto al objetivo.

    Args:
        X (pd.DataFrame): Predictores del conjunto de entrenamiento.
        y (pd.Series): Variable objetivo del conjunto de entrenamiento.

    Returns:
        pd.DataFrame: Tabla ordenada de mayor a menor información mutua.
    """
    tabla = pd.DataFrame({
        "Variable": X.columns,
        "Pearson": X.corrwith(y, method="pearson").values,
        "Spearman": X.corrwith(y, method="spearman").values,
        "Información mutua": calcular_informacion_mutua(X, y).values,
    })
    return tabla.sort_values("Información mutua", ascending=False).reset_index(drop=True)


def es_binaria(serie: pd.Series) -> bool:
    """Indica si una variable toma como mucho dos valores (por ejemplo, una dummy).

    Args:
        serie (pd.Series): Variable a evaluar.

    Returns:
        bool: True si la variable es binaria.
    """
    return serie.nunique() <= 2


def calcular_moda(serie: pd.Series) -> float:
    """Calcula la moda de una variable.

    En variables con pocos valores distintos es el valor más frecuente; en las continuas,
    donde casi ningún valor se repite, es el centro del intervalo más frecuente del histograma.

    Args:
        serie (pd.Series): Variable numérica.

    Returns:
        float: Moda de la variable.
    """
    if serie.nunique() <= config.MAX_VALORES_DISCRETA:
        return float(serie.mode().iloc[0])
    conteos, bordes = np.histogram(serie, bins=config.CONTENEDORES_HISTOGRAMA)
    i = int(conteos.argmax())
    return float((bordes[i] + bordes[i + 1]) / 2)


def contar_atipicos(serie: pd.Series) -> int:
    """Cuenta los valores atípicos con el criterio del rango intercuartílico (Tukey).

    Args:
        serie (pd.Series): Variable numérica.

    Returns:
        int: Número de valores fuera de [Q1 − k·IQR, Q3 + k·IQR].
    """
    q1, q3 = serie.quantile([0.25, 0.75])
    margen = config.FACTOR_IQR * (q3 - q1)
    return int(((serie < q1 - margen) | (serie > q3 + margen)).sum())


def tabla_tendencia_central(X: pd.DataFrame) -> pd.DataFrame:
    """Resume media, mediana, moda, dispersión, asimetría y atípicos de cada predictor.

    Las variables binarias (dummies) se omiten porque no tiene sentido buscarles atípicos.

    Args:
        X (pd.DataFrame): Predictores del conjunto de entrenamiento.

    Returns:
        pd.DataFrame: Una fila por variable numérica no binaria.
    """
    filas = []
    for nombre in [c for c in X.columns if not es_binaria(X[c])]:
        serie = X[nombre]
        media, mediana, desviacion = serie.mean(), serie.median(), serie.std()
        atipicos = contar_atipicos(serie)
        filas.append({
            "Variable": nombre,
            "Media": media,
            "Mediana": mediana,
            "Moda": calcular_moda(serie),
            "Desviación": desviacion,
            "(Media − mediana) / desviación": (media - mediana) / desviacion if desviacion else 0.0,
            "Asimetría": serie.skew(),
            "Curtosis": serie.kurt(),
            "Atípicos (IQR)": atipicos,
            "% atípicos": 100 * atipicos / len(serie),
        })
    return pd.DataFrame(filas)


def variables_no_aptas_para_estandarizar(tabla: pd.DataFrame) -> pd.DataFrame:
    """Filtra las variables con atípicos o con asimetría apreciable.

    Args:
        tabla (pd.DataFrame): Resultado de tabla_tendencia_central.

    Returns:
        pd.DataFrame: Filas de las variables que no cumplen los supuestos de StandardScaler.
    """
    return tabla[(tabla["Atípicos (IQR)"] > 0) | (tabla["Asimetría"].abs() > config.UMBRAL_ASIMETRIA)]


def matriz_correlacion(datos: pd.DataFrame, metodo: str) -> pd.DataFrame:
    """Calcula la matriz de correlación entre todas las variables.

    Args:
        datos (pd.DataFrame): Predictores y variable objetivo.
        metodo (str): "pearson" o "spearman".

    Returns:
        pd.DataFrame: Matriz cuadrada de correlaciones.
    """
    return datos.corr(method=metodo)


@st.cache_data(show_spinner="Comprobando si las demás variables dependen de las relevantes...")
def tabla_dependencia(datos: pd.DataFrame, relevantes: list[str]) -> pd.DataFrame:
    """Mide cuánto se puede explicar cada variable no relevante a partir de las relevantes.

    Si una variable fuera un agregado (suma, promedio, cociente...) de las relevantes,
    tendría una correlación, un R² o una información mutua altos.

    Args:
        datos (pd.DataFrame): Dataset con predictores y objetivo.
        relevantes (list[str]): Variables que explican el objetivo.

    Returns:
        pd.DataFrame: Una fila por variable no relevante.
    """
    X = datos[relevantes]
    otras = [c for c in datos.columns if c not in relevantes and c != config.OBJETIVO]
    filas = []
    for nombre in otras:
        y = datos[nombre]
        filas.append({
            "Variable": nombre,
            "|r| máximo con las relevantes": X.corrwith(y).abs().max(),
            "R² lineal (validación cruzada)": cross_val_score(
                LinearRegression(), X, y, cv=config.N_PARTICIONES, scoring="r2").mean(),
            "Información mutua total": mutual_info_regression(X, y, random_state=config.SEMILLA).sum(),
        })
    return pd.DataFrame(filas)


def pesos_simples(coeficientes: pd.Series) -> pd.Series | None:
    """Busca la escala que convierte los coeficientes en pesos "redondos".

    Una normalización min-max multiplica todos los pesos originales por el mismo factor, así
    que se prueba como peso del coeficiente más pequeño cada múltiplo de PASO_PESOS y se
    acepta el primero que deja a todos los demás sobre la rejilla de PASO_PESOS.

    Args:
        coeficientes (pd.Series): Coeficientes de la regresión en unidades originales.

    Returns:
        pd.Series | None: Pesos simples, o None si no se encuentran.
    """
    minimo = coeficientes.abs().min()
    if minimo == 0:
        return None
    paso = config.PASO_PESOS
    for multiplo in range(1, round(config.MAX_PESO_MINIMO / paso) + 1):
        pesos = coeficientes / (minimo / (multiplo * paso))
        rejilla = (pesos / paso).round() * paso
        if (pesos - rejilla).abs().max() <= config.TOLERANCIA_PESOS:
            return rejilla.round(6)
    return None


@st.cache_data(show_spinner="Reconstruyendo la ecuación que generó los datos...")
def ecuacion_generadora(datos: pd.DataFrame, relevantes: list[str]) -> dict:
    """Ajusta el objetivo con las variables relevantes e intenta reconstruir su fórmula exacta.

    Hipótesis: objetivo = min-max(Σ peso·variable) llevado al rango observado del objetivo y
    redondeado a DECIMALES_OBJETIVO. Se verifica fila por fila.

    Args:
        datos (pd.DataFrame): Dataset completo con predictores y objetivo.
        relevantes (list[str]): Variables que explican el objetivo.

    Returns:
        dict: Intercepto y coeficientes de la regresión, desviación del residuo y, si se
        encuentra, pesos, mínimo y máximo del puntaje bruto, rango del objetivo,
        proporción de filas reproducidas exactamente y error máximo.
    """
    X, y = datos[relevantes].astype(float), datos[config.OBJETIVO]
    modelo = LinearRegression().fit(X, y)
    coeficientes = pd.Series(modelo.coef_, index=relevantes)
    resultado = {"intercepto": float(modelo.intercept_), "coeficientes": coeficientes,
                 "desviacion_residuo": float((y - modelo.predict(X)).std()), "pesos": None}
    pesos = pesos_simples(coeficientes)
    if pesos is None:
        return resultado

    bruto = X @ pesos
    minimo, maximo = float(bruto.min()), float(bruto.max())
    y_min, y_max = float(y.min()), float(y.max())
    escalado = y_min + (y_max - y_min) * (bruto - minimo) / (maximo - minimo)
    reconstruido = escalado.round(config.DECIMALES_OBJETIVO)
    error = (reconstruido - y).abs()
    resultado.update(pesos=pesos, minimo=minimo, maximo=maximo, y_min=y_min, y_max=y_max,
                     coincidencia=float((error < 1e-9).mean()), error_maximo=float(error.max()))
    return resultado
