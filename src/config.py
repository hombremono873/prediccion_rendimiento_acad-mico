"""Constantes compartidas por toda la aplicación."""

from pathlib import Path

# ================= RUTAS =================
RAIZ_PROYECTO = Path(__file__).resolve().parent.parent
RUTA_DATOS = RAIZ_PROYECTO / "data" / "train.csv"
RUTA_PROCESADO = RAIZ_PROYECTO / "data" / "procesado.csv"
RUTA_MODELO = RAIZ_PROYECTO / "models" / "modelo.joblib"

# ================= COLUMNAS =================
TRADUCCION_COLUMNAS = {
    "id": "id",
    "student_id": "id_estudiante",
    "age": "edad",
    "gender": "genero",
    "study_hours_per_day": "horas_estudio_por_dia",
    "sleep_hours": "horas_sueno",
    "phone_usage_hours": "horas_uso_telefono",
    "social_media_hours": "horas_redes_sociales",
    "youtube_hours": "horas_youtube",
    "gaming_hours": "horas_videojuegos",
    "breaks_per_day": "descansos_por_dia",
    "coffee_intake_mg": "consumo_cafe_mg",
    "exercise_minutes": "minutos_ejercicio",
    "assignments_completed": "tareas_completadas",
    "attendance_percentage": "porcentaje_asistencia",
    "stress_level": "nivel_estres",
    "focus_score": "puntaje_enfoque",
    "final_grade": "calificacion_final",
    "productivity_score": "puntaje_productividad",
}

# Traducción de los valores de las variables categóricas (así las dummies quedan en español)
TRADUCCION_VALORES = {
    "genero": {"Male": "masculino", "Female": "femenino", "Other": "otro"},
}

OBJETIVO = "puntaje_productividad"
COLUMNAS_ID = ["id", "id_estudiante"]
COLUMNAS_CATEGORICAS = ["genero"]

VARIABLES_POR_DEFECTO = [
    "horas_estudio_por_dia",
    "puntaje_enfoque",
    "horas_sueno",
    "horas_uso_telefono",
    "nivel_estres",
    "porcentaje_asistencia",
]

# Columnas del archivo de predicciones por lotes (formato de envío de Kaggle)
COLUMNA_ID_LOTE = "id"
COLUMNA_PREDICCION_LOTE = "productivity_score"
# Columna con la predicción en la tabla de detalle que se muestra en pantalla
COLUMNA_PREDICCION_DETALLE = "puntaje_productividad_predicho"

# ================= MODELADO =================
SEMILLA = 42
# Porcentaje del dataset que se reserva para prueba (el usuario lo elige en la sección Carga)
PORCENTAJE_PRUEBA_DEFECTO = 20
PORCENTAJE_PRUEBA_MIN = 10
PORCENTAJE_PRUEBA_MAX = 50
PORCENTAJE_PRUEBA_PASO = 5
N_PARTICIONES = 5
RANGO_PREDICCION = (0.0, 100.0)
DECIMALES_PREDICCION = 2

# ================= INTERFAZ =================
SECCIONES = [
    "1. Extracción",
    "2. Transformación",
    "3. Carga",
    "4. Análisis de variables",
    "5. Selección de variables",
    "6. Entrenamiento",
    "7. Predicción",
]
DESCRIPCIONES_SECCIONES = dict(zip(SECCIONES, [
    "Carga el dataset de Kaggle o sube tu propio archivo CSV.",
    "Traduce, limpia y codifica los datos para el modelo.",
    "Divide los datos en entrenamiento y prueba, y guarda el dataset procesado.",
    "Explora las distribuciones y la relación de cada variable con la productividad.",
    "Elige las variables con las que se entrenará el modelo.",
    "Entrena la regresión lineal y evalúa su desempeño.",
    "Estima la productividad de un estudiante o de un lote completo.",
]))
TITULO_APP = "Productividad estudiantil"
SUBTITULO_APP = "Regresión lineal sobre hábitos de estudio"
AUTOR = "Omar Alberto Torres"
AUTOR_DESCRIPCION = "Estudiante de la Maestría en Análisis de Datos"

# Paleta: área principal blanca (debe coincidir con .streamlit/config.toml) y
# barra lateral y encabezados en negro azulado con acentos azules
COLORES = {
    "fondo_principal": "#FFFFFF",
    "superficie_clara": "#F8FAFC",
    "borde_claro": "#E2E8F0",
    "texto_oscuro": "#0F172A",
    "texto_secundario": "#475569",
    "fondo": "#0B1120",
    "superficie": "#111827",
    "borde": "#1E293B",
    "texto": "#E5E7EB",
    "texto_claro": "#FFFFFF",
    "texto_suave": "#94A3B8",
    "primario": "#2563EB",
    "primario_oscuro": "#1E3A8A",
    "acento": "#38BDF8",
    "positivo": "#22C55E",
    "negativo": "#EF4444",
    "referencia": "#F59E0B",
}

MODO_SELECCIONADAS = "Variables seleccionadas"
MODO_TODAS = "Todas las variables"
# Las tablas muestran todas sus filas hasta ALTO_MAX_TABLA píxeles; a partir de ahí usan scroll
ALTO_FILA_TABLA = 35
ALTO_MAX_TABLA = 450
COLUMNAS_HISTOGRAMAS = 3
# Una variable entera con hasta este número de valores distintos se trata como discreta
MAX_VALORES_DISCRETA = 25
PUNTOS_KDE = 200
# Tamaño (px) del pulgar y alto de la pista del control deslizante; Streamlit usa 12 y 4
TAMANO_PULGAR_SLIDER = 24
ALTO_PISTA_SLIDER = 6
AJUSTE_VERTICAL_PULGAR = 0  # medido en el navegador para centrar el pulgar sobre la pista
