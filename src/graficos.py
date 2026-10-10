"""Visualizaciones de la aplicación (plotly)."""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gaussian_kde

from src import config

COLOR_PRINCIPAL = config.COLORES["primario"]
COLOR_POSITIVO = config.COLORES["positivo"]
COLOR_NEGATIVO = config.COLORES["negativo"]
COLOR_REFERENCIA = config.COLORES["referencia"]
COLOR_EJE = config.COLORES["texto_suave"]


def _estilo(fig: go.Figure, titulo: str, eje_x: str, eje_y: str, alto: int = 320) -> go.Figure:
    """Aplica título, ejes y márgenes comunes a una figura.

    Args:
        fig (go.Figure): Figura a ajustar.
        titulo (str): Título del gráfico.
        eje_x (str): Etiqueta del eje X.
        eje_y (str): Etiqueta del eje Y.
        alto (int): Alto en píxeles.

    Returns:
        go.Figure: La misma figura con el estilo aplicado.
    """
    fig.update_layout(title=titulo, xaxis_title=eje_x, yaxis_title=eje_y, height=alto,
                      margin=dict(l=10, r=10, t=50, b=10), showlegend=False)
    return fig


def histograma(serie: pd.Series, discreta: bool) -> go.Figure:
    """Grafica la distribución de una variable.

    Las variables discretas se muestran como barras por valor y sin curva KDE;
    las continuas como histograma de densidad con su curva KDE.

    Args:
        serie (pd.Series): Variable a graficar.
        discreta (bool): Si la variable es discreta.

    Returns:
        go.Figure: Gráfico de distribución.
    """
    titulo = f"Distribución de {serie.name}"
    if discreta:
        conteo = serie.value_counts().sort_index()
        fig = go.Figure(go.Bar(x=conteo.index, y=conteo.values, marker_color=COLOR_PRINCIPAL))
        return _estilo(fig, titulo, serie.name, "Frecuencia")

    fig = go.Figure(go.Histogram(x=serie, histnorm="probability density", nbinsx=30,
                                 marker_color=COLOR_PRINCIPAL, opacity=0.75))
    if serie.nunique() > 1:
        rejilla = np.linspace(serie.min(), serie.max(), config.PUNTOS_KDE)
        fig.add_trace(go.Scatter(x=rejilla, y=gaussian_kde(serie)(rejilla), mode="lines",
                                 line=dict(color=COLOR_REFERENCIA, width=2)))
    return _estilo(fig, titulo, serie.name, "Densidad")


def barras_informacion_mutua(tabla: pd.DataFrame) -> go.Figure:
    """Grafica la información mutua de cada variable con el objetivo.

    Args:
        tabla (pd.DataFrame): Tabla con las columnas 'Variable' e 'Información mutua'.

    Returns:
        go.Figure: Gráfico de barras horizontales.
    """
    datos = tabla.sort_values("Información mutua")
    fig = px.bar(datos, x="Información mutua", y="Variable", orientation="h",
                 color_discrete_sequence=[COLOR_PRINCIPAL])
    return _estilo(fig, f"Información mutua con {config.OBJETIVO}", "Información mutua", "Variable",
                   alto=max(320, 28 * len(datos)))


def real_vs_predicho(y_real: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Compara valores reales y predichos con la diagonal de predicción perfecta.

    Args:
        y_real (pd.Series): Valores reales.
        y_pred (np.ndarray): Valores predichos.

    Returns:
        go.Figure: Gráfico de dispersión.
    """
    limites = [float(min(y_real.min(), y_pred.min())), float(max(y_real.max(), y_pred.max()))]
    fig = go.Figure([
        go.Scattergl(x=y_real, y=y_pred, mode="markers",
                     marker=dict(color=COLOR_PRINCIPAL, opacity=0.4, size=5)),
        go.Scatter(x=limites, y=limites, mode="lines",
                   line=dict(color=COLOR_REFERENCIA, dash="dash", width=2)),
    ])
    return _estilo(fig, "Valores reales vs. predicciones (prueba)",
                   "Puntaje de productividad real", "Puntaje de productividad predicho", alto=450)


def barras_coeficientes(coeficientes: pd.DataFrame) -> go.Figure:
    """Grafica los coeficientes estandarizados (verde positivos, rojo negativos).

    Args:
        coeficientes (pd.DataFrame): Tabla con 'Variable' y 'Coeficiente'.

    Returns:
        go.Figure: Gráfico de barras horizontales.
    """
    datos = coeficientes.iloc[::-1]
    colores = [COLOR_POSITIVO if c > 0 else COLOR_NEGATIVO for c in datos["Coeficiente"]]
    fig = go.Figure(go.Bar(x=datos["Coeficiente"], y=datos["Variable"], orientation="h",
                           marker_color=colores))
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_EJE)
    return _estilo(fig, "Impacto de las variables (coeficientes)", "Coeficiente estandarizado",
                   "Variable", alto=max(450, 28 * len(datos)))


def residuos(y_real: pd.Series, y_pred: np.ndarray) -> go.Figure:
    """Grafica los residuos frente a los valores predichos.

    Args:
        y_real (pd.Series): Valores reales.
        y_pred (np.ndarray): Valores predichos.

    Returns:
        go.Figure: Gráfico de dispersión de residuos.
    """
    fig = go.Figure(go.Scattergl(x=y_pred, y=np.asarray(y_real) - y_pred, mode="markers",
                                 marker=dict(color=COLOR_PRINCIPAL, opacity=0.4, size=5)))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_REFERENCIA)
    return _estilo(fig, "Residuos (prueba)", "Puntaje de productividad predicho",
                   "Residuo (real − predicho)", alto=400)
