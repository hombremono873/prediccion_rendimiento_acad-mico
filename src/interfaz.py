"""Estilos visuales, pantalla de presentación, encabezados y créditos de la aplicación."""

import base64

import numpy as np
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
.creditos a {{
    color: {C['acento']} !important;
}}
.pie a {{
    color: {C['primario']};
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


def enlace_fuente_datos(texto: str) -> str:
    """Construye el enlace a los datos de Kaggle que se abre en otra pestaña.

    Args:
        texto (str): Texto visible del enlace.

    Returns:
        str: Etiqueta HTML del enlace.
    """
    return f'<a href="{config.URL_FUENTE_DATOS}" target="_blank" rel="noopener noreferrer">{texto}</a>'


def creditos_barra_lateral() -> None:
    """Muestra los créditos del autor y de los datos al final de la barra lateral."""
    st.sidebar.markdown(
        f'<div class="creditos">Desarrollado por<br><strong>{config.AUTOR}</strong><br>'
        f'{config.AUTOR_PROFESION}<br>{config.AUTOR_DESCRIPCION}<br>{config.UNIVERSIDAD}'
        f'<br><br>Datos: {enlace_fuente_datos("Kaggle")}</div>',
        unsafe_allow_html=True,
    )


def pie_de_pagina() -> None:
    """Muestra los créditos del autor y de los datos al pie de la página."""
    st.markdown(
        f'<div class="pie">Desarrollado por <strong>{config.AUTOR}</strong> · {config.AUTOR_PROFESION} · '
        f'{config.AUTOR_DESCRIPCION}, {config.UNIVERSIDAD}<br>'
        f'Datos: {enlace_fuente_datos(config.FUENTE_DATOS)}</div>',
        unsafe_allow_html=True,
    )


# ================= PANTALLA DE PRESENTACIÓN =================
def _rgba(color_hex: str, opacidad: float) -> str:
    """Convierte un color hexadecimal en rgba con la opacidad indicada.

    Args:
        color_hex (str): Color en formato #RRGGBB.
        opacidad (float): Opacidad entre 0 y 1.

    Returns:
        str: Color en formato rgba(r, g, b, a).
    """
    r, g, b = (int(color_hex[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r}, {g}, {b}, {opacidad})"


def svg_fondo_presentacion() -> str:
    """Genera el fondo minimalista de la presentación con los colores de la aplicación.

    Combina una rejilla tenue, una nube de puntos con su recta de regresión (el modelo del
    proyecto) y una red neuronal de tres capas (aprendizaje automático). Las líneas se dibujan
    solas y los nodos laten; las animaciones se desactivan con "reducir movimiento".

    Returns:
        str: Documento SVG.
    """
    ancho, alto = 1600, 900
    rejilla = "".join(f'<line x1="{x}" y1="0" x2="{x}" y2="{alto}"/>' for x in range(0, ancho + 1, 80))
    rejilla += "".join(f'<line x1="0" y1="{y}" x2="{ancho}" y2="{y}"/>' for y in range(0, alto + 1, 80))

    # Nube de puntos alrededor de una recta descendente en la mitad izquierda
    rng = np.random.default_rng(config.SEMILLA)
    xs = rng.uniform(110, 480, 40)
    ys = 720 - 0.75 * (xs - 110) + rng.normal(0, 38, xs.size)
    puntos = "".join(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="4"/>' for x, y in zip(xs, ys))

    # Red neuronal en la mitad derecha
    capas = [(1180, 4), (1330, 5), (1480, 3)]
    nodos = [[(x, 450 + (i - (n - 1) / 2) * 95) for i in range(n)] for x, n in capas]
    conexiones, k = "", 0
    for origen, destino in zip(nodos, nodos[1:]):
        for x1, y1 in origen:
            for x2, y2 in destino:
                conexiones += (f'<line class="trazo" pathLength="1" style="animation-delay:{0.2 + 0.03 * k:.2f}s" '
                               f'x1="{x1}" y1="{y1:.0f}" x2="{x2}" y2="{y2:.0f}"/>')
                k += 1
    circulos = "".join(f'<circle class="nodo" style="animation-delay:{0.15 * i:.2f}s" cx="{x}" cy="{y:.0f}" r="11"/>'
                       for i, (x, y) in enumerate(p for capa in nodos for p in capa))

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" '
        f'preserveAspectRatio="xMidYMid slice">'
        f'<style>'
        f'.trazo{{stroke-dasharray:1;stroke-dashoffset:1;animation:dibujar 1.4s ease-out forwards}}'
        f'.nodo{{animation:latir 2.4s ease-in-out infinite}}'
        f'@keyframes dibujar{{to{{stroke-dashoffset:0}}}}'
        f'@keyframes latir{{0%,100%{{opacity:.45}}50%{{opacity:1}}}}'
        f'@media (prefers-reduced-motion:reduce){{.trazo{{animation:none;stroke-dashoffset:0}}'
        f'.nodo{{animation:none;opacity:.8}}}}'
        f'</style>'
        f'<rect width="{ancho}" height="{alto}" fill="{C["fondo"]}"/>'
        f'<g stroke="{C["borde"]}" stroke-width="1" opacity="0.6">{rejilla}</g>'
        f'<g fill="{C["texto_suave"]}" opacity="0.3">{puntos}</g>'
        f'<line class="trazo" pathLength="1" x1="90" y1="735" x2="500" y2="428" '
        f'stroke="{C["referencia"]}" stroke-width="3" stroke-linecap="round" opacity="0.55"/>'
        f'<g stroke="{C["primario"]}" stroke-width="1.5" opacity="0.35">{conexiones}</g>'
        f'<g fill="{C["fondo"]}" stroke="{C["acento"]}" stroke-width="2.5">{circulos}</g>'
        f'</svg>'
    )


def pantalla_presentacion() -> None:
    """Muestra una vez por sesión la presentación animada con los créditos.

    Es una capa fija a pantalla completa, solo HTML y CSS, que se desvanece sola y deja de
    capturar clics, dejando la aplicación lista debajo.
    """
    if st.session_state.get("presentacion_mostrada"):
        return
    st.session_state["presentacion_mostrada"] = True

    fondo = base64.b64encode(svg_fondo_presentacion().encode("utf-8")).decode("ascii")
    salida, duracion = config.INICIO_SALIDA_PRESENTACION, config.DURACION_SALIDA_PRESENTACION
    lineas = [
        '<div class="icono">📊</div>',
        f'<div class="nombre">{config.AUTOR}</div>',
        f'<div class="profesion">{config.AUTOR_PROFESION}</div>',
        f'<p>{config.AUTOR_DESCRIPCION}</p>',
        f'<p class="universidad">{config.UNIVERSIDAD}</p>',
        f'<div class="separador"></div><p class="pequeno">Datos: '
        f'{enlace_fuente_datos(config.FUENTE_DATOS)}<br>'
        f'<span class="url">{config.URL_FUENTE_DATOS.removeprefix("https://")}</span></p>',
        f'<p class="pequeno">Puesta en servicio: <strong>{config.FECHA_PUESTA_EN_SERVICIO}</strong></p>',
    ]
    contenido = "".join(
        f'<div class="linea" style="animation-delay:{0.1 + i * config.RETRASO_LINEAS_PRESENTACION:.2f}s">'
        f'{linea}</div>' for i, linea in enumerate(lineas))

    estilos = f"""<style>
#presentacion {{
    position: fixed;
    inset: 0;
    z-index: 1000000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px;
    background: {C['fondo']} url("data:image/svg+xml;base64,{fondo}") center / cover no-repeat;
    animation: salir-presentacion {duracion}s ease-in {salida}s forwards;
}}
#presentacion .panel {{
    background: {_rgba(C['superficie'], 0.82)};
    backdrop-filter: blur(4px);
    border: 1px solid {C['borde']};
    border-top: 3px solid {C['primario']};
    border-radius: 16px;
    box-shadow: 0 20px 60px {_rgba(C['fondo'], 0.6)};
    color: {C['texto']};
    box-sizing: border-box;
    max-width: 580px;
    width: 100%;
    padding: 2.2rem 2.5rem;
    text-align: center;
}}
@media (max-width: 600px) {{
    #presentacion .panel {{ padding: 1.6rem 1.2rem; }}
    #presentacion .nombre {{ font-size: 1.6rem; }}
}}
#presentacion .linea {{
    opacity: 0;
    transform: translateY(14px);
    animation: entrar-linea 0.6s ease-out forwards;
}}
#presentacion .icono {{ font-size: 2.4rem; }}
#presentacion .nombre {{
    color: {C['texto_claro']};
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1.2;
    margin: 0.3rem 0 0.4rem 0;
}}
#presentacion .profesion {{
    color: {C['acento']};
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
#presentacion p {{ margin: 0.6rem 0 0 0; }}
#presentacion .universidad {{
    color: {C['texto_claro']};
    font-weight: 700;
    margin-top: 0.2rem;
}}
#presentacion .separador {{
    border-top: 1px solid {C['borde']};
    margin: 1.2rem auto 0 auto;
    width: 60%;
}}
#presentacion .pequeno {{
    color: {C['texto_suave']};
    font-size: 0.85rem;
}}
#presentacion .pequeno strong {{ color: {C['texto_claro']}; }}
#presentacion a {{ color: {C['acento']}; }}
#presentacion .url {{ font-size: 0.75rem; word-break: break-all; }}
#presentacion .barra {{
    position: absolute;
    left: 0;
    bottom: 0;
    height: 4px;
    width: 0;
    background: linear-gradient(90deg, {C['primario']}, {C['acento']});
    animation: llenar-barra {salida}s linear forwards;
}}
@keyframes entrar-linea {{ to {{ opacity: 1; transform: none; }} }}
@keyframes llenar-barra {{ to {{ width: 100%; }} }}
@keyframes salir-presentacion {{ to {{ opacity: 0; visibility: hidden; pointer-events: none; }} }}
@media (prefers-reduced-motion: reduce) {{
    #presentacion {{ animation-delay: {config.INICIO_SALIDA_PRESENTACION_REDUCIDA}s; }}
    #presentacion .linea {{ animation: none; opacity: 1; transform: none; }}
    #presentacion .barra {{ animation-duration: {config.INICIO_SALIDA_PRESENTACION_REDUCIDA}s; }}
}}
</style>"""
    st.markdown(f'{estilos}<div id="presentacion"><div class="panel">{contenido}</div>'
                f'<div class="barra"></div></div>', unsafe_allow_html=True)
