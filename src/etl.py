"""Extracción, transformación y carga del dataset."""

from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from src import config


# ================= EXTRACCIÓN =================
@st.cache_data(show_spinner="Leyendo el archivo...")
def leer_csv_ruta(ruta: str, marca_modificacion: float) -> pd.DataFrame:
    """Lee un CSV desde disco.

    Args:
        ruta (str): Ruta del archivo CSV.
        marca_modificacion (float): Fecha de modificación del archivo; invalida la caché si cambia.

    Returns:
        pd.DataFrame: Datos leídos.
    """
    return pd.read_csv(ruta)


@st.cache_data(show_spinner="Leyendo el archivo...")
def leer_csv_bytes(contenido: bytes) -> pd.DataFrame:
    """Lee un CSV a partir de su contenido en bytes (archivo subido).

    Args:
        contenido (bytes): Contenido del archivo CSV.

    Returns:
        pd.DataFrame: Datos leídos.
    """
    return pd.read_csv(BytesIO(contenido))


def resumen_tipos(df: pd.DataFrame) -> pd.DataFrame:
    """Describe el tipo de dato y los valores no nulos de cada columna.

    Args:
        df (pd.DataFrame): Datos a describir.

    Returns:
        pd.DataFrame: Tabla con columna, tipo de dato, valores no nulos y valores únicos.
    """
    return pd.DataFrame({
        "Columna": df.columns,
        "Tipo de dato": df.dtypes.astype(str).values,
        "Valores no nulos": df.notna().sum().values,
        "Valores únicos": df.nunique().values,
    })


# ================= TRANSFORMACIÓN =================
def traducir_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Traduce al español los nombres de las columnas y los valores categóricos.

    Args:
        df (pd.DataFrame): Datos con nombres originales en inglés.

    Returns:
        pd.DataFrame: Copia de los datos con columnas y categorías en español.
    """
    df_traducido = df.rename(columns=config.TRADUCCION_COLUMNAS)
    for columna, mapa in config.TRADUCCION_VALORES.items():
        if columna in df_traducido.columns:
            df_traducido[columna] = df_traducido[columna].replace(mapa)
    return df_traducido


def reporte_calidad(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Cuenta los valores nulos por columna y los registros duplicados.

    Args:
        df (pd.DataFrame): Datos a revisar.

    Returns:
        tuple[pd.DataFrame, int]: Tabla de columnas con nulos y número de filas duplicadas.
    """
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    tabla_nulos = pd.DataFrame({"Columna": nulos.index, "Valores nulos": nulos.values})
    return tabla_nulos, int(df.duplicated().sum())


def codificar_categoricas(df: pd.DataFrame, eliminar_primera: bool = True) -> pd.DataFrame:
    """Convierte las variables categóricas en variables dummy.

    Args:
        df (pd.DataFrame): Datos con columnas categóricas.
        eliminar_primera (bool): Si es True, elimina la primera categoría (drop_first).

    Returns:
        pd.DataFrame: Datos con las categóricas reemplazadas por columnas 0/1.
    """
    categoricas = [c for c in config.COLUMNAS_CATEGORICAS if c in df.columns]
    return pd.get_dummies(df, columns=categoricas, drop_first=eliminar_primera, dtype=int)


def obtener_categorias(df: pd.DataFrame) -> dict[str, list[str]]:
    """Obtiene las categorías de cada variable categórica antes de codificarla.

    Args:
        df (pd.DataFrame): Datos traducidos sin codificar.

    Returns:
        dict[str, list[str]]: Categorías ordenadas por variable categórica.
    """
    return {
        c: sorted(df[c].dropna().astype(str).unique().tolist())
        for c in config.COLUMNAS_CATEGORICAS if c in df.columns
    }


def transformar(df_crudo: pd.DataFrame) -> dict:
    """Aplica la transformación completa: traducción, limpieza, eliminación de ids y dummies.

    Args:
        df_crudo (pd.DataFrame): Datos tal como se extrajeron.

    Returns:
        dict: Datos procesados ('datos'), categorías, tabla de nulos, duplicados,
            filas eliminadas, columnas id eliminadas y columnas dummy creadas.

    Raises:
        ValueError: Si falta la variable objetivo en los datos.
    """
    df = traducir_columnas(df_crudo)
    if config.OBJETIVO not in df.columns:
        raise ValueError(
            f"El archivo no contiene la variable objetivo '{config.OBJETIVO}'. "
            "Revisa que sea el dataset de entrenamiento."
        )

    tabla_nulos, duplicados = reporte_calidad(df)
    filas_iniciales = len(df)
    df = df.dropna().drop_duplicates()

    columnas_id = [c for c in config.COLUMNAS_ID if c in df.columns]
    df = df.drop(columns=columnas_id)

    categorias = obtener_categorias(df)
    columnas_antes = set(df.columns)
    df = codificar_categoricas(df)
    columnas_dummy = [c for c in df.columns if c not in columnas_antes]

    return {
        "datos": df.reset_index(drop=True),
        "categorias": categorias,
        "tabla_nulos": tabla_nulos,
        "duplicados": duplicados,
        "filas_eliminadas": filas_iniciales - len(df),
        "columnas_id": columnas_id,
        "columnas_dummy": columnas_dummy,
    }


# ================= CARGA =================
def dividir_datos(df: pd.DataFrame, porcentaje_prueba: int = config.PORCENTAJE_PRUEBA_DEFECTO
                  ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Separa predictores y objetivo, y divide en entrenamiento y prueba.

    Args:
        df (pd.DataFrame): Datos procesados con la variable objetivo.
        porcentaje_prueba (int): Porcentaje de filas que se reserva para prueba.

    Returns:
        tuple: X_train, X_test, y_train, y_test.
    """
    X = df.drop(columns=[config.OBJETIVO])
    y = df[config.OBJETIVO]
    return train_test_split(X, y, test_size=porcentaje_prueba / 100, random_state=config.SEMILLA)


def guardar_procesado(df: pd.DataFrame, ruta: Path = config.RUTA_PROCESADO) -> Path:
    """Guarda el dataset procesado en disco.

    Args:
        df (pd.DataFrame): Datos procesados.
        ruta (Path): Ruta del archivo de salida.

    Returns:
        Path: Ruta donde se guardó el archivo.
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return ruta


def a_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convierte un DataFrame en un CSV codificado en UTF-8 para descargarlo.

    Args:
        df (pd.DataFrame): Datos a exportar.

    Returns:
        bytes: Contenido del CSV.
    """
    return df.to_csv(index=False).encode("utf-8")
