"""Aplicación Streamlit: preparación, análisis, entrenamiento y predicción de la productividad estudiantil."""

import pandas as pd
import streamlit as st

from src import analisis, config, etl, graficos, interfaz, modelo, prediccion

ss = st.session_state

# Claves de estado en el orden del flujo; invalidar un paso borra también los siguientes
CLAVES_FLUJO = ["df_crudo", "transformacion", "division", "seleccion", "modelo"]


# ================= ESTADO =================
def invalidar_desde(clave: str) -> None:
    """Borra del estado la clave indicada y todas las que dependen de ella.

    Args:
        clave (str): Primera clave de CLAVES_FLUJO que se invalida.
    """
    for c in CLAVES_FLUJO[CLAVES_FLUJO.index(clave):]:
        ss.pop(c, None)


def requerir(clave: str, seccion: str) -> bool:
    """Comprueba que exista un paso previo y, si falta, muestra un aviso.

    Args:
        clave (str): Clave de estado que debe existir.
        seccion (str): Sección a la que debe ir el usuario.

    Returns:
        bool: True si el paso previo está completo.
    """
    if clave in ss:
        return True
    st.warning(f"Primero completa la sección **{seccion}**.")
    return False


def modelo_vigente() -> dict | None:
    """Devuelve el modelo entrenado que corresponde a la selección actual.

    Prioriza el modelo de la sesión; si no hay, usa el guardado en disco siempre que
    sus variables coincidan con la selección vigente (o no haya selección todavía).

    Returns:
        dict | None: Modelo con su pipeline y variables, o None si no hay uno válido.
    """
    if "modelo" in ss:
        return ss["modelo"]
    if not config.RUTA_MODELO.exists():
        return None
    try:
        guardado = modelo.cargar_modelo(str(config.RUTA_MODELO), config.RUTA_MODELO.stat().st_mtime)
    except Exception as error:  # archivo corrupto o de otra versión de scikit-learn
        st.error(f"No se pudo cargar el modelo guardado: {error}")
        return None
    seleccion = ss.get("seleccion")
    if seleccion is not None and seleccion != guardado["variables"]:
        return None
    return guardado


# ================= 1. CARGA DE DATOS =================
def render_carga_datos() -> None:
    """Carga el CSV del proyecto o uno subido y muestra su estructura."""
    fuente = st.radio("Origen de los datos",
                      [f"Archivo del proyecto ({config.RUTA_DATOS.relative_to(config.RAIZ_PROYECTO).as_posix()})",
                       "Subir un archivo CSV"], horizontal=True)

    df, origen = None, None
    if fuente.startswith("Archivo"):
        if not config.RUTA_DATOS.exists():
            st.error(f"No se encontró {config.RUTA_DATOS}. Copia ahí el train.csv de Kaggle o sube un archivo.")
        elif st.button("Cargar datos", type="primary"):
            marca = config.RUTA_DATOS.stat().st_mtime
            df, origen = etl.leer_csv_ruta(str(config.RUTA_DATOS), marca), f"disco:{marca}"
    else:
        archivo = st.file_uploader("Archivo CSV", type="csv")
        if archivo is not None and st.button("Cargar archivo subido", type="primary"):
            try:
                df, origen = etl.leer_csv_bytes(archivo.getvalue()), f"subido:{archivo.name}:{archivo.size}"
            except Exception as error:
                st.error(f"No se pudo leer el CSV: {error}")

    if df is not None and origen != ss.get("origen"):
        invalidar_desde("df_crudo")
        ss["df_crudo"], ss["origen"] = df, origen
        st.success("Datos cargados. Los pasos siguientes se reiniciaron.")

    if "df_crudo" not in ss:
        st.info("Elige un origen y pulsa el botón para cargar los datos.")
        return

    # Solo para visualizar: la traducción real se aplica en Verificación y limpieza
    vista = etl.traducir_columnas(ss["df_crudo"])
    c1, c2 = st.columns(2)
    c1.metric("Filas", f"{vista.shape[0]:,}")
    c2.metric("Columnas", vista.shape[1])
    st.caption("Los nombres se muestran traducidos; la traducción se aplica al dataset en "
               f"{config.SECCION_VERIFICACION}.")
    st.subheader("Tipos de datos")
    interfaz.tabla(etl.resumen_tipos(vista), ocultar_indice=True)
    st.subheader("Datos")
    interfaz.tabla(vista)


# ================= 2. VERIFICACIÓN Y LIMPIEZA =================
def render_verificacion() -> None:
    """Verifica nulos y duplicados, limpia y codifica los datos, y muestra sus estadísticas."""
    if not requerir("df_crudo", config.SECCION_CARGA):
        return

    if st.button("Verificar y limpiar", type="primary"):
        try:
            resultado = etl.transformar(ss["df_crudo"])
        except ValueError as error:
            st.error(str(error))
            return
        invalidar_desde("transformacion")
        ss["transformacion"] = resultado
        st.success("Datos verificados y limpios.")

    if "transformacion" not in ss:
        st.info("Pulsa el botón para verificar y limpiar los datos.")
        return

    t = ss["transformacion"]
    st.markdown(
        f"- Columnas traducidas al español.\n"
        f"- Columnas de identificación eliminadas: {', '.join(t['columnas_id']) or 'ninguna'}.\n"
        f"- Dummies creadas (drop_first=True): {', '.join(t['columnas_dummy']) or 'ninguna'}."
    )

    st.subheader("Calidad de los datos")
    c1, c2, c3 = st.columns(3)
    c1.metric("Columnas con nulos", len(t["tabla_nulos"]))
    c2.metric("Filas duplicadas", t["duplicados"])
    c3.metric("Filas eliminadas", t["filas_eliminadas"])
    if t["tabla_nulos"].empty:
        st.success("No hay valores nulos.")
    else:
        interfaz.tabla(t["tabla_nulos"], ocultar_indice=True)
        st.warning("Las filas con valores nulos se eliminaron.")
    if t["duplicados"]:
        st.warning("Las filas duplicadas se eliminaron.")

    st.subheader("Tabla resultante")
    interfaz.tabla(t["datos"])
    st.subheader("Estadísticas descriptivas")
    descriptivas = t["datos"].describe().T.rename(columns={
        "count": "conteo", "mean": "media", "std": "desviación", "min": "mínimo",
        "25%": "percentil 25", "50%": "mediana", "75%": "percentil 75", "max": "máximo"})
    interfaz.tabla(descriptivas)


# ================= 3. DIVISIÓN DE DATOS =================
def sincronizar_porcentaje(origen: str) -> None:
    """Iguala el control deslizante y el campo numérico del porcentaje de prueba.

    Redondea el valor al paso configurado (por si se escribe, por ejemplo, 23) y lo
    recuerda para restaurarlo al volver a la sección.

    Args:
        origen (str): Clave del widget que cambió el usuario.
    """
    paso = config.PORCENTAJE_PRUEBA_PASO
    valor = int(round(ss[origen] / paso) * paso)
    valor = min(max(valor, config.PORCENTAJE_PRUEBA_MIN), config.PORCENTAJE_PRUEBA_MAX)
    ss["w_porcentaje_slider"] = ss["w_porcentaje_numero"] = ss["porcentaje_prueba_elegido"] = valor


def render_division() -> None:
    """Divide en entrenamiento y prueba, y guarda el dataset procesado."""
    if not requerir("transformacion", config.SECCION_VERIFICACION):
        return
    datos = ss["transformacion"]["datos"]
    aplicado = ss.get("division", {}).get("porcentaje_prueba")

    elegido = ss.get("porcentaje_prueba_elegido", aplicado or config.PORCENTAJE_PRUEBA_DEFECTO)
    restaurar_widget("w_porcentaje_slider", elegido)
    restaurar_widget("w_porcentaje_numero", elegido)
    limites = (config.PORCENTAJE_PRUEBA_MIN, config.PORCENTAJE_PRUEBA_MAX)
    col_slider, col_numero = st.columns([5, 1], vertical_alignment="bottom")
    with col_slider:
        st.slider("Porcentaje de los datos para prueba", *limites, step=config.PORCENTAJE_PRUEBA_PASO,
                  format="%d%%", key="w_porcentaje_slider",
                  on_change=sincronizar_porcentaje, args=("w_porcentaje_slider",))
    with col_numero:
        st.number_input("Escribe o usa − / +", *limites, step=config.PORCENTAJE_PRUEBA_PASO,
                        key="w_porcentaje_numero",
                        on_change=sincronizar_porcentaje, args=("w_porcentaje_numero",))
    porcentaje = ss["w_porcentaje_slider"]
    filas_prueba = round(len(datos) * porcentaje / 100)
    st.caption(f"Entrenamiento {100 - porcentaje}% (~{len(datos) - filas_prueba:,} filas) · "
               f"Prueba {porcentaje}% (~{filas_prueba:,} filas) · semilla {config.SEMILLA}")

    if st.button("Dividir y guardar", type="primary"):
        X_train, X_test, y_train, y_test = etl.dividir_datos(datos, porcentaje)
        invalidar_desde("division")
        ss["division"] = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test,
                          "porcentaje_prueba": porcentaje}
        aplicado = porcentaje
        try:
            ruta = etl.guardar_procesado(datos)
            st.success(f"Dataset procesado guardado en {ruta.relative_to(config.RAIZ_PROYECTO).as_posix()}.")
        except OSError as error:
            st.error(f"No se pudo guardar el dataset procesado: {error}")

    if "division" not in ss:
        st.info("Elige el porcentaje de prueba y pulsa el botón para dividir los datos.")
        return
    if porcentaje != aplicado:
        st.warning(f"La división actual usa {aplicado}% para prueba. Pulsa **Dividir y guardar** "
                   f"para aplicar {porcentaje}% (el modelo entrenado se descartará).")

    d = ss["division"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("División aplicada", f"{100 - d['porcentaje_prueba']}/{d['porcentaje_prueba']}")
    c2.metric("Filas de entrenamiento", f"{len(d['X_train']):,}")
    c3.metric("Filas de prueba", f"{len(d['X_test']):,}")
    c4.metric("Predictores", d["X_train"].shape[1])
    st.download_button("Descargar dataset procesado", etl.a_csv_bytes(datos),
                       file_name=config.RUTA_PROCESADO.name, mime="text/csv")


# ================= 4. ANÁLISIS EXPLORATORIO =================
def pestana_distribuciones(datos: pd.DataFrame) -> None:
    """Dibuja un gráfico de distribución por variable.

    Args:
        datos (pd.DataFrame): Predictores y variable objetivo.
    """
    columnas = st.columns(config.COLUMNAS_HISTOGRAMAS)
    for i, nombre in enumerate(datos.columns):
        with columnas[i % config.COLUMNAS_HISTOGRAMAS]:
            serie = datos[nombre]
            st.plotly_chart(graficos.histograma(serie, analisis.es_discreta(serie)),
                            use_container_width=True)


def pestana_tendencia_central(X: pd.DataFrame) -> None:
    """Muestra media, mediana, moda y atípicos para justificar el escalado con StandardScaler.

    Args:
        X (pd.DataFrame): Predictores del conjunto de entrenamiento.
    """
    st.markdown(
        "`StandardScaler` resta la **media** y divide por la **desviación estándar**. Ambas son "
        "sensibles a los valores atípicos, por eso antes de escalar se comprueba que en cada "
        "variable la media y la mediana coincidan (distribución simétrica; en distribuciones con "
        "un solo pico también la moda) y que no haya atípicos según el criterio del rango "
        "intercuartílico "
        f"(fuera de Q1 − {config.FACTOR_IQR}·IQR y Q3 + {config.FACTOR_IQR}·IQR).")
    tabla = analisis.tabla_tendencia_central(X)
    if tabla.empty:
        st.info("No hay variables numéricas no binarias que analizar.")
        return
    formato = {c: "{:.4f}" for c in tabla.columns if c not in ("Variable", "Atípicos (IQR)")}
    formato["% atípicos"] = "{:.2f}%"
    interfaz.tabla(tabla.style.format(formato), ocultar_indice=True)
    st.caption("Se omiten las variables binarias (dummies). En variables continuas la moda es el "
               "centro del intervalo más frecuente del histograma. Una curtosis cercana a −1.2 "
               "indica una distribución casi uniforme: todos los valores son igual de frecuentes, "
               "la moda cae en cualquier punto del rango y no es representativa; ahí la simetría "
               "se juzga con la media, la mediana y la asimetría. La variable objetivo no se "
               "escala, por eso no aparece.")

    no_aptas = analisis.variables_no_aptas_para_estandarizar(tabla)
    if no_aptas.empty:
        st.success(
            "Ninguna variable tiene atípicos y todas son aproximadamente simétricas "
            f"(|asimetría| ≤ {config.UMBRAL_ASIMETRIA}, media ≈ mediana). La media y la desviación "
            "estándar representan bien a cada variable, así que **StandardScaler es adecuado**.")
    else:
        st.warning(
            f"Variables con atípicos o asimetría apreciable: {', '.join(no_aptas['Variable'])}. "
            "En ellas la media y la desviación estándar se desplazan hacia los extremos; conviene "
            "tratar los atípicos o usar un escalador robusto (RobustScaler) antes de aplicar "
            "StandardScaler.")
    st.plotly_chart(graficos.cajas_estandarizadas(X[tabla["Variable"]]), use_container_width=True)


def pestana_mapa_calor(datos: pd.DataFrame) -> None:
    """Dibuja la matriz de correlación con el método que elija el usuario.

    Args:
        datos (pd.DataFrame): Predictores y variable objetivo.
    """
    metodo = st.radio("Método de correlación", list(config.METODOS_CORRELACION), horizontal=True,
                      key="w_metodo_correlacion")
    matriz = analisis.matriz_correlacion(datos, config.METODOS_CORRELACION[metodo])
    st.plotly_chart(graficos.mapa_calor(matriz, metodo), use_container_width=True)
    st.caption("Valores cercanos a 1 o −1 entre dos predictores indican multicolinealidad; "
               f"la fila de {config.OBJETIVO} muestra la relación de cada variable con el objetivo.")


def pestana_relacion_objetivo(X: pd.DataFrame, y: pd.Series) -> None:
    """Muestra la tabla de Pearson, Spearman e información mutua y su gráfico.

    Args:
        X (pd.DataFrame): Predictores del conjunto de entrenamiento.
        y (pd.Series): Variable objetivo del conjunto de entrenamiento.
    """
    tabla = analisis.tabla_relaciones(X, y)
    interfaz.tabla(tabla.style.format(precision=4), ocultar_indice=True)
    st.plotly_chart(graficos.barras_informacion_mutua(tabla), use_container_width=True)


def render_analisis() -> None:
    """Explora distribuciones, tendencia central, atípicos y correlaciones en pestañas."""
    if not requerir("division", config.SECCION_DIVISION):
        return
    X, y = ss["division"]["X_train"], ss["division"]["y_train"]
    datos = pd.concat([X, y], axis=1)
    st.caption("El análisis usa solo el conjunto de entrenamiento.")

    distribuciones, tendencia, calor, relacion = st.tabs(config.PESTANAS_ANALISIS)
    with distribuciones:
        pestana_distribuciones(datos)
    with tendencia:
        pestana_tendencia_central(X)
    with calor:
        pestana_mapa_calor(datos)
    with relacion:
        pestana_relacion_objetivo(X, y)


# ================= 5. SELECCIÓN =================
def restaurar_widget(clave_widget: str, valor: object) -> None:
    """Recupera el valor de un widget que Streamlit borró al cambiar de sección.

    Args:
        clave_widget (str): Clave del widget.
        valor (object): Valor a restaurar si el widget no está en el estado.
    """
    if clave_widget not in ss:
        ss[clave_widget] = valor


def render_seleccion() -> None:
    """Permite elegir las variables del modelo."""
    if not requerir("division", config.SECCION_DIVISION):
        return
    disponibles = list(ss["division"]["X_train"].columns)
    por_defecto = [v for v in config.VARIABLES_POR_DEFECTO if v in disponibles]
    previa = ss.get("preferencias_seleccion", {})

    restaurar_widget("w_modo", previa.get("modo", config.MODO_SELECCIONADAS))
    modo = st.radio("Modo", [config.MODO_SELECCIONADAS, config.MODO_TODAS], key="w_modo", horizontal=True)

    if modo == config.MODO_SELECCIONADAS:
        restaurar_widget("w_incluidas", [v for v in previa.get("incluidas", por_defecto) if v in disponibles])
        base = st.multiselect("Variables a usar", disponibles, key="w_incluidas")
    else:
        base = disponibles

    # Se depura antes de dibujar el widget para que no contenga variables fuera de las opciones
    ss["w_eliminar"] = [v for v in ss.get("w_eliminar", previa.get("eliminar", [])) if v in base]
    eliminar = st.multiselect("Variables a eliminar del modelo", base, key="w_eliminar")

    finales = [v for v in base if v not in eliminar]
    incluidas = base if modo == config.MODO_SELECCIONADAS else previa.get("incluidas", por_defecto)
    ss["preferencias_seleccion"] = {"modo": modo, "incluidas": incluidas, "eliminar": eliminar}

    if not finales:
        ss.pop("seleccion", None)
        st.error("Debe quedar al menos una variable para entrenar el modelo.")
        return

    if ss.get("seleccion") != finales:
        if "modelo" in ss:
            ss.pop("modelo")
            st.warning("Cambiaste las variables: el modelo entrenado se descartó. Vuelve a entrenar.")
        ss["seleccion"] = finales

    st.subheader(f"Variables que usará el modelo ({len(finales)})")
    interfaz.tabla(pd.DataFrame({"Variable": finales}), ocultar_indice=True)


# ================= 6. ENTRENAMIENTO =================
def render_entrenamiento() -> None:
    """Entrena el pipeline y muestra métricas, coeficientes y gráficos."""
    if (not requerir("division", config.SECCION_DIVISION)
            or not requerir("seleccion", config.SECCION_SELECCION)):
        return
    variables = ss["seleccion"]
    st.write(f"**Variables ({len(variables)}):** {', '.join(variables)}")

    if st.button("Entrenar modelo", type="primary"):
        d = ss["division"]
        with st.spinner("Entrenando..."):
            resultado = modelo.entrenar(d["X_train"], d["X_test"], d["y_train"], d["y_test"],
                                        variables, ss["transformacion"]["categorias"])
        ss["modelo"] = resultado
        try:
            ruta = modelo.guardar_modelo(resultado)
            modelo.cargar_modelo.clear()
            st.success(f"Modelo entrenado y guardado en {ruta.relative_to(config.RAIZ_PROYECTO).as_posix()}.")
        except OSError as error:
            st.error(f"El modelo se entrenó, pero no se pudo guardar: {error}")

    if "modelo" not in ss:
        st.info("Pulsa el botón para entrenar con las variables seleccionadas.")
        return

    m = ss["modelo"]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Validación cruzada ({config.N_PARTICIONES} particiones)")
        interfaz.tabla(m["metricas_cv"].style.format(precision=4))
    with c2:
        st.subheader("Métricas")
        interfaz.tabla(m["metricas"].style.format(precision=4))

    st.subheader("Coeficientes estandarizados")
    interfaz.tabla(m["coeficientes"].style.format({"Coeficiente": "{:.4f}"}), ocultar_indice=True)

    c1, c2 = st.columns(2)
    c1.plotly_chart(graficos.real_vs_predicho(m["y_test"], m["y_pred_test"]), use_container_width=True)
    c2.plotly_chart(graficos.residuos(m["y_test"], m["y_pred_test"]), use_container_width=True)
    st.plotly_chart(graficos.barras_coeficientes(m["coeficientes"]), use_container_width=True)


# ================= 7. PREDICCIÓN =================
def formulario_prediccion(m: dict) -> None:
    """Genera un formulario con un campo por variable del modelo y muestra la predicción.

    Args:
        m (dict): Modelo vigente con pipeline, variables, rangos y categorías.
    """
    variables, rangos = m["variables"], m["rangos"]
    dummies = {c: [v for v in variables if v.startswith(f"{c}_")] for c in m["categorias"]}
    es_dummy = {v for grupo in dummies.values() for v in grupo}
    numericas = [v for v in variables if v not in es_dummy]

    with st.form("formulario_prediccion"):
        valores = {}
        columnas = st.columns(config.COLUMNAS_HISTOGRAMAS)
        for i, v in enumerate(numericas):
            r = rangos[v]
            with columnas[i % config.COLUMNAS_HISTOGRAMAS]:
                if r["entera"]:
                    valores[v] = st.number_input(v, min_value=int(r["min"]), max_value=int(r["max"]),
                                                 value=int(round(r["mediana"])), step=1)
                else:
                    valores[v] = st.number_input(v, min_value=r["min"], max_value=r["max"],
                                                 value=r["mediana"], step=0.01, format="%.2f")
        for categoria, grupo in dummies.items():
            if grupo:
                elegida = st.selectbox(categoria, m["categorias"][categoria])
                valores.update({v: int(v == f"{categoria}_{elegida}") for v in grupo})
        enviado = st.form_submit_button("Predecir", type="primary")

    if enviado:
        resultado = prediccion.predecir_individual(m["pipeline"], variables, valores)
        st.metric("Puntaje de productividad predicho", f"{resultado:.{config.DECIMALES_PREDICCION}f}")


def prediccion_por_lotes(m: dict) -> None:
    """Predice un CSV subido, muestra entrada y predicción, y ofrece descargar el resultado.

    Args:
        m (dict): Modelo vigente con pipeline y variables.
    """
    archivo = st.file_uploader("CSV con los estudiantes a predecir", type="csv", key="archivo_lote")
    if archivo is None:
        return
    try:
        envio, detalle = prediccion.predecir_lote(m["pipeline"], m["variables"],
                                                  etl.leer_csv_bytes(archivo.getvalue()))
    except ValueError as error:
        st.error(str(error))
        return
    except Exception as error:
        st.error(f"No se pudo procesar el archivo: {error}")
        return

    st.success(f"Se predijeron {len(envio):,} filas.")
    st.download_button("Descargar predicciones", etl.a_csv_bytes(envio),
                       file_name="predicciones.csv", mime="text/csv")
    st.caption(f"El archivo descargado contiene solo `{config.COLUMNA_ID_LOTE}` y "
               f"`{config.COLUMNA_PREDICCION_LOTE}` (formato de envío de Kaggle).")

    st.subheader("Datos de entrada y predicción")
    interfaz.tabla(detalle, ocultar_indice=True)
    st.plotly_chart(graficos.histograma_predicciones(detalle[config.COLUMNA_PREDICCION_DETALLE]),
                    use_container_width=True)


def render_prediccion() -> None:
    """Predice con datos ingresados por el usuario o con un lote."""
    m = modelo_vigente()
    if m is None:
        st.warning(f"No hay un modelo entrenado para la selección actual. Ve a **{config.SECCION_ENTRENAMIENTO}**.")
        return
    if "modelo" not in ss:
        st.info("Usando el último modelo guardado en disco.")
    st.caption(f"Variables del modelo: {', '.join(m['variables'])}")

    individual, lote = st.tabs(["Predicción individual", "Predicción por lotes"])
    with individual:
        formulario_prediccion(m)
    with lote:
        prediccion_por_lotes(m)


# ================= NAVEGACIÓN =================
SECCIONES = dict(zip(config.SECCIONES, [
    render_carga_datos, render_verificacion, render_division, render_analisis,
    render_seleccion, render_entrenamiento, render_prediccion,
]))


def barra_lateral() -> str:
    """Dibuja la marca y el menú de secciones de la barra lateral.

    Returns:
        str: Sección elegida.
    """
    interfaz.marca_barra_lateral()
    return st.sidebar.radio("Secciones", config.SECCIONES, format_func=interfaz.etiqueta_seccion)


def estado_barra_lateral() -> None:
    """Muestra el avance del flujo y los créditos; se dibuja después de la sección para estar al día."""
    st.sidebar.divider()
    interfaz.estado_pasos({
        "Datos cargados": "df_crudo" in ss,
        "Datos verificados y limpios": "transformacion" in ss,
        "Datos divididos": "division" in ss,
        "Variables seleccionadas": "seleccion" in ss,
        "Modelo entrenado": "modelo" in ss,
    })
    interfaz.creditos_barra_lateral()


def main() -> None:
    """Punto de entrada de la aplicación."""
    st.set_page_config(page_title=config.TITULO_APP, page_icon="📊", layout="wide")
    interfaz.aplicar_estilos()
    interfaz.pantalla_presentacion()
    seccion = barra_lateral()
    interfaz.encabezado(seccion)
    SECCIONES[seccion]()
    estado_barra_lateral()
    interfaz.pie_de_pagina()


main()
