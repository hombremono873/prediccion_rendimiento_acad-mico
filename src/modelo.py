"""Pipeline de regresión, validación cruzada, métricas y persistencia del modelo."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.prediccion import calcular_rangos


def crear_pipeline() -> Pipeline:
    """Crea el pipeline de escalado y regresión lineal.

    El escalador vive dentro del pipeline, así se ajusta solo con los datos de
    entrenamiento de cada partición y no hay fuga de información.

    Returns:
        Pipeline: Pipeline sin entrenar.
    """
    return Pipeline([
        ("escalador", StandardScaler()),
        ("modelo", LinearRegression()),
    ])


def validacion_cruzada(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Evalúa el pipeline con validación cruzada KFold.

    Args:
        pipeline (Pipeline): Pipeline sin entrenar.
        X (pd.DataFrame): Predictores de entrenamiento.
        y (pd.Series): Objetivo de entrenamiento.

    Returns:
        pd.DataFrame: Media y desviación de MAE, RMSE y R².
    """
    kfold = KFold(n_splits=config.N_PARTICIONES, shuffle=True, random_state=config.SEMILLA)
    res = cross_validate(
        pipeline, X, y, cv=kfold,
        scoring={"MAE": "neg_mean_absolute_error",
                 "RMSE": "neg_root_mean_squared_error",
                 "R2": "r2"},
    )
    mae, rmse, r2 = -res["test_MAE"], -res["test_RMSE"], res["test_R2"]
    return pd.DataFrame({
        "Media": [mae.mean(), rmse.mean(), r2.mean()],
        "Desviación": [mae.std(), rmse.std(), r2.std()],
    }, index=pd.Index(["MAE", "RMSE", "R²"], name="Métrica"))


def calcular_metricas(y_real: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Calcula MAE, MSE, RMSE y R².

    Args:
        y_real (pd.Series): Valores reales.
        y_pred (np.ndarray): Valores predichos.

    Returns:
        dict[str, float]: Métricas por nombre.
    """
    mse = mean_squared_error(y_real, y_pred)
    return {
        "MAE": mean_absolute_error(y_real, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R²": r2_score(y_real, y_pred),
    }


def obtener_coeficientes(pipeline: Pipeline, columnas: list[str]) -> pd.DataFrame:
    """Extrae los coeficientes estandarizados ordenados por valor absoluto.

    Args:
        pipeline (Pipeline): Pipeline entrenado.
        columnas (list[str]): Nombres de las variables en el orden de entrenamiento.

    Returns:
        pd.DataFrame: Tabla de variables y coeficientes.
    """
    return pd.DataFrame({
        "Variable": columnas,
        "Coeficiente": pipeline.named_steps["modelo"].coef_,
    }).sort_values("Coeficiente", key=abs, ascending=False).reset_index(drop=True)


def entrenar(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series,
             y_test: pd.Series, variables: list[str], categorias: dict[str, list[str]]) -> dict:
    """Entrena y evalúa el pipeline con las variables indicadas.

    Args:
        X_train (pd.DataFrame): Predictores de entrenamiento.
        X_test (pd.DataFrame): Predictores de prueba.
        y_train (pd.Series): Objetivo de entrenamiento.
        y_test (pd.Series): Objetivo de prueba.
        variables (list[str]): Variables que usará el modelo.
        categorias (dict[str, list[str]]): Categorías originales de las variables categóricas.

    Returns:
        dict: Pipeline, variables, rangos, categorías, métricas, coeficientes y
            predicciones sobre el conjunto de prueba.
    """
    X_tr, X_te = X_train[variables], X_test[variables]
    pipeline = crear_pipeline()
    metricas_cv = validacion_cruzada(pipeline, X_tr, y_train)

    pipeline.fit(X_tr, y_train)
    y_pred_train = pipeline.predict(X_tr)
    y_pred_test = pipeline.predict(X_te)

    metricas = pd.DataFrame({
        "Entrenamiento": calcular_metricas(y_train, y_pred_train),
        "Prueba": calcular_metricas(y_test, y_pred_test),
    })
    metricas.index.name = "Métrica"

    return {
        "pipeline": pipeline,
        "variables": list(variables),
        "rangos": calcular_rangos(X_tr),
        "categorias": categorias,
        "metricas_cv": metricas_cv,
        "metricas": metricas,
        "coeficientes": obtener_coeficientes(pipeline, list(variables)),
        "y_test": y_test,
        "y_pred_test": y_pred_test,
    }


# Claves del resultado que se guardan en disco (las predicciones de prueba solo viven en la sesión)
CLAVES_PERSISTIDAS = ["pipeline", "variables", "rangos", "categorias", "metricas_cv", "metricas", "coeficientes"]


def guardar_modelo(resultado: dict, ruta: Path = config.RUTA_MODELO) -> Path:
    """Guarda el pipeline entrenado y la información necesaria para predecir.

    Args:
        resultado (dict): Resultado devuelto por `entrenar`.
        ruta (Path): Ruta del archivo joblib.

    Returns:
        Path: Ruta donde se guardó el modelo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({k: resultado[k] for k in CLAVES_PERSISTIDAS}, ruta)
    return ruta


@st.cache_resource(show_spinner="Cargando el modelo...")
def cargar_modelo(ruta: str, marca_modificacion: float) -> dict:
    """Carga el modelo guardado en disco.

    Args:
        ruta (str): Ruta del archivo joblib.
        marca_modificacion (float): Fecha de modificación del archivo; invalida la caché si cambia.

    Returns:
        dict: Pipeline y datos asociados guardados por `guardar_modelo`.
    """
    return joblib.load(ruta)
