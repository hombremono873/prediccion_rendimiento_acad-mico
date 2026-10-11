"""Estilos visuales, encabezados y créditos de la aplicación."""

import pandas as pd
import streamlit as st
from pandas.io.formats.style import Styler

from src import config

C = config.COLORES

ESTILOS = f"""
<style>
/* Barra lateral con degradado oscuro */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {C['fondo']} 0%, {C['superficie']} 100%);
    border-right: 1px solid {C['borde']};
}}
/* Texto blanco en la barra lateral, sin depender del tema elegido en el navegador */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label {{
    color: {C['texto_claro']} !important;
}}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
    color: {C['acento']} !important;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
[data-testid="stSidebar"] hr {{
    border-color: {C['borde']};
}}
.paso {{
    color: {C['texto_claro']};
    font-size: 0.95rem;
    margin: 0.25rem 0;
}}
.paso .marca-paso {{
    display: inline-block;
    width: 1.3rem;
    font-weight: 700;
}}
.paso .listo {{ color: {C['positivo']}; }}
.paso .pendiente {{ color: {C['texto_suave']}; }}

/* Encabezado de cada sección */
.encabezado {{
    background: linear-gradient(120deg, {C['primario_oscuro']} 0%, {C['superficie']} 70%);
    border: 1px solid {C['borde']};
    border-left: 4px solid {C['primario']};
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.5rem;
}}
.encabezado .etiqueta {{
    color: {C['acento']};
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.encabezado h1 {{
    color: {C['texto']};
    font-size: 1.9rem;
    margin: 0.2rem 0 0.3rem 0;
    padding: 0;
}}
.encabezado p {{
    color: {C['texto_suave']};
    margin: 0;
}}

/* Métricas como tarjetas */
[data-testid="stMetric"] {{
    background: {C['superficie_clara']};
    border: 1px solid {C['borde_claro']};
    border-left: 4px solid {C['primario']};
    border-radius: 10px;
    padding: 0.8rem 1rem;
}}
[data-testid="stMetricLabel"] p {{
    color: {C['texto_secundario']} !important;
}}
[data-testid="stMetricValue"] {{
    color: {C['primario']} !important;
}}

/* Marca y créditos de la barra lateral */
.marca h2 {{
    color: {C['texto']};
    font-size: 1.3rem;
    margin-bottom: 0;
}}
.marca p {{
    color: {C['texto_suave']};
    font-size: 0.85rem;
    margin-top: 0.2rem;
}}
.creditos {{
    border-top: 1px solid {C['borde']};
    color: {C['texto_suave']};
    font-size: 0.82rem;
    margin-top: 1.5rem;
    padding-top: 0.8rem;
}}
.creditos strong {{
    color: {C['acento']};
}}
.pie {{
    border-top: 1px solid {C['borde_claro']};
    color: {C['texto_secundario']};
    font-size: 0.85rem;
    margin-top: 3rem;
    padding-top: 1rem;
    text-align: center;
}}
.pie strong {{
    color: {C['texto_oscuro']};
}}

/* Barras de desplazamiento visibles en azul (tablas incluidas) */
::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}
::-webkit-scrollbar-track {{
    background: {C['superficie_clara']};
}}
::-webkit-scrollbar-thumb {{
    background: {C['primario']};
    border-radius: 6px;
    border: 2px solid {C['superficie_clara']};
}}
::-webkit-scrollbar-thumb:hover {{
    background: {C['acento']};
}}
* {{
    scrollbar-color: {C['primario']} {C['superficie_clara']};
}}
[data-testid="stDataFrame"] {{
    border: 1px solid {C['borde_claro']};
    border-radius: 8px;
}}

/* Control deslizante más fácil de agarrar (mismos colores del tema).
   Los márgenes negativos mantienen el pulgar centrado sobre la pista al agrandarlo. */
[data-testid="stSlider"] [role="slider"] {{
    width: {config.TAMANO_PULGAR_SLIDER}px !important;
    height: {config.TAMANO_PULGAR_SLIDER}px !important;
    margin-left: -{(config.TAMANO_PULGAR_SLIDER - 12) // 2}px;
    margin-top: {config.AJUSTE_VERTICAL_PULGAR}px;
    cursor: grab !important;
}}
[data-testid="stSlider"] [role="slider"]:active {{
    cursor: grabbing !important;
}}
[data-testid="stSlider"] div:has(> [role="slider"]) {{
    height: {config.ALTO_PISTA_SLIDER}px !important;
    border-radius: {config.ALTO_PISTA_SLIDER}px;
}}
[data-testid="stSlider"] div:has(> div > [role="slider"]) {{
    padding-top: 12px !important;
    padding-bottom: 12px !important;
    cursor: pointer;
}}
</style>
"""


def aplicar_estilos() -> None:
    """Inyecta la hoja de estilos de la aplicación."""
    st.markdown(ESTILOS, unsafe_allow_html=True)


def marca_barra_lateral() -> None:
    """Muestra el nombre de la aplicación en la parte superior de la barra lateral."""
    st.sidebar.markdown(
        f'<div class="marca"><h2>📊 {config.TITULO_APP}</h2><p>{config.SUBTITULO_APP}</p></div>',
        unsafe_allow_html=True,
    )


def encabezado(seccion: str) -> None:
    """Dibuja el encabezado de la sección actual con su descripción.

    Args:
        seccion (str): Nombre de la sección tal como aparece en config.SECCIONES.
    """
    numero, _, titulo = seccion.partition(". ")
    st.markdown(
        f'<div class="encabezado"><div class="etiqueta">Paso {numero} de {len(config.SECCIONES)}</div>'
        f'<h1>{titulo}</h1><p>{config.DESCRIPCIONES_SECCIONES[seccion]}</p></div>',
        unsafe_allow_html=True,
    )


def tabla(datos: pd.DataFrame | Styler, ocultar_indice: bool = False) -> None:
    """Muestra una tabla completa con scroll vertical y horizontal cuando no cabe.

    Las tablas pequeñas se muestran enteras; las grandes se limitan a
    config.ALTO_MAX_TABLA píxeles y se recorren con las barras de desplazamiento.

    Args:
        datos (pd.DataFrame | Styler): Tabla a mostrar, con o sin formato.
        ocultar_indice (bool): Si es True, no muestra la columna de índice.
    """
    filas = len(datos.data if isinstance(datos, Styler) else datos)
    alto = min(config.ALTO_FILA_TABLA * (filas + 1) + 3, config.ALTO_MAX_TABLA)
    st.caption(f"{filas:,} filas · desplázate dentro de la tabla para ver todas las filas y columnas"
               if alto == config.ALTO_MAX_TABLA else f"{filas:,} filas")
    st.dataframe(datos, hide_index=ocultar_indice, use_container_width=True, height=alto)


def etiqueta_seccion(seccion: str) -> str:
    """Escapa el número de la sección para que Markdown no lo convierta en una lista.

    Args:
        seccion (str): Nombre de la sección, por ejemplo "1. Carga de datos".

    Returns:
        str: Etiqueta lista para mostrarse en el menú.
    """
    return seccion.replace(". ", "\\. ", 1)


def estado_pasos(pasos: dict[str, bool]) -> None:
    """Muestra en la barra lateral qué pasos del flujo están completos.

    Args:
        pasos (dict[str, bool]): Nombre de cada paso y si está completo.
    """
    filas = "".join(
        f'<div class="paso"><span class="marca-paso {"listo" if listo else "pendiente"}">'
        f'{"✔" if listo else "○"}</span>{paso}</div>'
        for paso, listo in pasos.items()
    )
    st.sidebar.markdown(filas, unsafe_allow_html=True)


def creditos_barra_lateral() -> None:
    """Muestra los créditos del autor al final de la barra lateral."""
    st.sidebar.markdown(
        f'<div class="creditos">Desarrollado por<br><strong>{config.AUTOR}</strong><br>'
        f'{config.AUTOR_DESCRIPCION}</div>',
        unsafe_allow_html=True,
    )


def pie_de_pagina() -> None:
    """Muestra los créditos del autor al pie de la página."""
    st.markdown(
        f'<div class="pie">Desarrollado por <strong>{config.AUTOR}</strong> · '
        f'{config.AUTOR_DESCRIPCION}</div>',
        unsafe_allow_html=True,
    )
