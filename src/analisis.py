"""Análisis de la relación entre los predictores y la variable objetivo."""

import pandas as pd
import streamlit as st
from sklearn.feature_selection import mutual_info_regression

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
