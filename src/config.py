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
# Porcentaje del dataset que se reserva para prueba (el usuario lo elige en la sección División de datos)
PORCENTAJE_PRUEBA_DEFECTO = 20
PORCENTAJE_PRUEBA_MIN = 10
PORCENTAJE_PRUEBA_MAX = 50
PORCENTAJE_PRUEBA_PASO = 5
N_PARTICIONES = 5
RANGO_PREDICCION = (0.0, 100.0)
DECIMALES_PREDICCION = 2

# ================= INTERFAZ =================
SECCION_CARGA = "1. Carga de datos"
SECCION_VERIFICACION = "2. Verificación y limpieza"
SECCION_DIVISION = "3. División de datos"
SECCION_ANALISIS = "4. Análisis exploratorio"
SECCION_SELECCION = "5. Selección de variables"
SECCION_ENTRENAMIENTO = "6. Entrenamiento"
SECCION_PREDICCION = "7. Predicción"
SECCIONES = [
    SECCION_CARGA,
    SECCION_VERIFICACION,
    SECCION_DIVISION,
    SECCION_ANALISIS,
    SECCION_SELECCION,
    SECCION_ENTRENAMIENTO,
    SECCION_PREDICCION,
]
DESCRIPCIONES_SECCIONES = dict(zip(SECCIONES, [
    "Carga el dataset de Kaggle o sube tu propio archivo CSV.",
    "Revisa nulos y duplicados, traduce, limpia y codifica los datos, y resume sus estadísticas.",
    "Divide los datos en entrenamiento y prueba, y guarda el dataset procesado.",
    "Explora distribuciones, tendencia central, atípicos, correlaciones y el origen de los datos.",
    "Elige las variables con las que se entrenará el modelo.",
    "Entrena la regresión lineal y evalúa su desempeño.",
    "Estima la productividad de un estudiante o de un lote completo.",
]))
PESTANAS_ANALISIS = ["Distribuciones", "Tendencia central y atípicos", "Mapa de calor",
                     "Relación con el objetivo", "Datos sintéticos"]
TITULO_APP = "Productividad estudiantil"
SUBTITULO_APP = "Regresión lineal sobre hábitos de estudio"
AUTOR = "Omar Alberto Torres"
AUTOR_PROFESION = "Ingeniero de Sistemas"
AUTOR_DESCRIPCION = "Estudiante de la Maestría en Análisis de Datos y Sistemas Inteligentes"
UNIVERSIDAD = "Universidad Santo Tomás"
FECHA_PUESTA_EN_SERVICIO = "11 de octubre de 2026"
FUENTE_DATOS = "Kaggle — Student Performance Prediction (Machine Learning Challenge)"
URL_FUENTE_DATOS = ("https://www.kaggle.com/competitions/"
                    "student-performance-prediction-machine-learning-challenge/data")

# Pantalla de presentación (una vez por sesión): segundos hasta que empieza a desvanecerse
# y duración del desvanecimiento
INICIO_SALIDA_PRESENTACION = 12.0
DURACION_SALIDA_PRESENTACION = 0.8
# Desvanecimiento al cerrarla antes con el botón o con un clic fuera del panel
DURACION_CIERRE_PRESENTACION = 0.3
RETRASO_LINEAS_PRESENTACION = 0.4

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
CONTENEDORES_HISTOGRAMA = 30
# Atípicos: valores fuera de [Q1 − FACTOR_IQR·IQR, Q3 + FACTOR_IQR·IQR] (criterio de Tukey)
FACTOR_IQR = 1.5
# Por debajo de este valor absoluto la asimetría se considera despreciable (distribución simétrica)
UMBRAL_ASIMETRIA = 0.5
# Curtosis (exceso) de una distribución uniforme
CURTOSIS_UNIFORME = -1.2
# Pestaña "Datos sintéticos": una variable se considera independiente de las relevantes si
# su |r| máximo y su R² validado quedan por debajo de estos umbrales
UMBRAL_CORRELACION_INDEPENDENCIA = 0.05
UMBRAL_R2_INDEPENDENCIA = 0.01
# Cabeceras de la tabla de dependencia: Z es la variable de la fila y Xⱼ cada variable relevante
COLUMNA_DEP_VARIABLE = "Variable (Z)"
COLUMNA_DEP_PEARSON = "Pearson: máx |r(Z, Xⱼ)|"
COLUMNA_DEP_SPEARMAN = "Spearman: máx |ρ(Z, Xⱼ)|"
COLUMNA_DEP_R2 = "R² validado = 1 − SC_res / SC_tot"
COLUMNA_DEP_IM = "Información mutua: Σⱼ I(Z; Xⱼ)"
# Búsqueda de la fórmula generadora: pesos múltiplos de PASO_PESOS, con el peso más pequeño
# entre PASO_PESOS y MAX_PESO_MINIMO, y TOLERANCIA_PESOS de margen frente a la rejilla
PASO_PESOS = 0.1
MAX_PESO_MINIMO = 10.0
TOLERANCIA_PESOS = 0.005
DECIMALES_OBJETIVO = 2
METODOS_CORRELACION = {"Pearson": "pearson", "Spearman": "spearman"}
# Escala divergente del mapa de calor: −1 rojo, 0 blanco, 1 azul
ESCALA_MAPA_CALOR = [[0.0, COLORES["negativo"]], [0.5, COLORES["fondo_principal"]],
                     [1.0, COLORES["primario"]]]
# Tamaño (px) del pulgar y alto de la pista del control deslizante; Streamlit usa 12 y 4
TAMANO_PULGAR_SLIDER = 24
ALTO_PISTA_SLIDER = 6
AJUSTE_VERTICAL_PULGAR = 0  # medido en el navegador para centrar el pulgar sobre la pista
