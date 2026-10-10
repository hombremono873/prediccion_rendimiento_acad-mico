"""Predicción individual y por lotes con el modelo entrenado."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.etl import codificar_categoricas, traducir_columnas


def calcular_rangos(X: pd.DataFrame) -> dict[str, dict]:
    """Obtiene mínimo, máximo, mediana y tipo de cada variable de entrenamiento.

    Args:
        X (pd.DataFrame): Predictores de entrenamiento.

    Returns:
        dict[str, dict]: Por variable, 'min', 'max', 'mediana' y 'entera'.
    """
    return {
        c: {
            "min": float(X[c].min()),
            "max": float(X[c].max()),
            "mediana": float(X[c].median()),
            "entera": bool(pd.api.types.is_integer_dtype(X[c])),
        }
        for c in X.columns
    }


def limitar_prediccion(valores: np.ndarray) -> np.ndarray:
    """Recorta las predicciones al rango válido y las redondea.

    Args:
        valores (np.ndarray): Predicciones del modelo.

    Returns:
        np.ndarray: Predicciones limitadas y redondeadas.
    """
    minimo, maximo = config.RANGO_PREDICCION
    return np.round(np.clip(valores, minimo, maximo), config.DECIMALES_PREDICCION)


def predecir_individual(pipeline: Pipeline, variables: list[str], valores: dict[str, float]) -> float:
    """Predice el puntaje de productividad de un estudiante.

    Args:
        pipeline (Pipeline): Pipeline entrenado.
        variables (list[str]): Variables en el orden de entrenamiento.
        valores (dict[str, float]): Valor ingresado para cada variable.

    Returns:
        float: Predicción limitada y redondeada.
    """
    fila = pd.DataFrame([valores], columns=variables)
    return float(limitar_prediccion(pipeline.predict(fila))[0])


def preparar_lote(df_crudo: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """Aplica al lote la misma transformación que al entrenamiento.

    Las dummies se generan sin eliminar la primera categoría y luego se alinean con
    las columnas del modelo, así un lote con una sola categoría no pierde información.

    Args:
        df_crudo (pd.DataFrame): Lote tal como se subió.
        variables (list[str]): Variables que usa el modelo.

    Returns:
        pd.DataFrame: Lote traducido, con la columna id y las variables del modelo.

    Raises:
        ValueError: Si faltan la columna id o alguna variable necesaria.
    """
    df = traducir_columnas(df_crudo)
    if config.COLUMNA_ID_LOTE not in df.columns:
        raise ValueError(f"El archivo debe tener la columna '{config.COLUMNA_ID_LOTE}'.")

    prefijos = {c: f"{c}_" for c in config.COLUMNAS_CATEGORICAS}
    necesarias = {next((c for c, p in prefijos.items() if v.startswith(p)), v) for v in variables}
    faltantes = sorted(necesarias - set(df.columns))
    if faltantes:
        raise ValueError(f"Al archivo le faltan estas columnas: {', '.join(faltantes)}.")

    codificado = codificar_categoricas(df[sorted(necesarias)], eliminar_primera=False)
    X = codificado.reindex(columns=variables, fill_value=0)
    if X.isna().any().any():
        raise ValueError("El archivo tiene valores vacíos en las variables del modelo.")
    return pd.concat([df[[config.COLUMNA_ID_LOTE]], X], axis=1)


def predecir_lote(pipeline: Pipeline, variables: list[str],
                  df_crudo: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predice un lote y devuelve el archivo de envío y la tabla de detalle.

    Args:
        pipeline (Pipeline): Pipeline entrenado.
        variables (list[str]): Variables en el orden de entrenamiento.
        df_crudo (pd.DataFrame): Lote tal como se subió.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Archivo de envío (id y productivity_score) y
            tabla con todas las columnas de entrada en español más la predicción junto al id.
    """
    lote = preparar_lote(df_crudo, variables)
    predicciones = limitar_prediccion(pipeline.predict(lote[variables]))

    envio = pd.DataFrame({
        config.COLUMNA_ID_LOTE: lote[config.COLUMNA_ID_LOTE],
        config.COLUMNA_PREDICCION_LOTE: predicciones,
    })
    detalle = traducir_columnas(df_crudo)
    posicion = detalle.columns.get_loc(config.COLUMNA_ID_LOTE) + 1
    detalle.insert(posicion, config.COLUMNA_PREDICCION_DETALLE, predicciones)
    return envio, detalle
