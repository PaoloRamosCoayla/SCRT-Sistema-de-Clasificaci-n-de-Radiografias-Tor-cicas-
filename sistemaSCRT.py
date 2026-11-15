# ============================================================
# INTERFAZ WEB MULTIMODELO JERÁRQUICO SCRT (CON VALIDADOR Y CRONÓMETRO)
# (VGG19 / INCEPTIONV3 / RESNET50 - DUAL MODEL M1+M2)
# ============================================================

import os
import json
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from PIL import Image
import io
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    TZ = ZoneInfo('America/Lima')
except Exception:
    TZ = None

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import vgg19, inception_v3, resnet50

# ------------------------------------------------------------
# 1. CONFIGURACIÓN DE ARCHIVOS Y CARGA DINÁMICA DE JERARQUÍAS
# ------------------------------------------------------------
BASE_DIR = "/content/dataset_por_paciente"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

CLASSES = ["bacteriana", "sano", "viral"]

MODELOS = {}
CONFIGS = {}
METRICAS = {}

# Mapeo de archivos jerárquicos (M1 y M2) por arquitectura
ARCH_FILES = {
    "VGG19": {
        "m1_model": "modelo_vgg19_m1_detector.keras",
        "m2_model": "modelo_vgg19_m2_patogeno.keras",
        "config": "config_jerarquico.json",
        "metrics": "metricas_vgg19_jerarquico.json",
        "preprocess_fn": vgg19.preprocess_input
    },
    "InceptionV3": {
        "m1_model": "modelo_inceptionv3_m1_detector.keras",
        "m2_model": "modelo_inceptionv3_m2_patogeno.keras",
        "config": "config_jerarquico_inceptionv3.json",
        "metrics": "metricas_inceptionv3_jerarquico.json",
        "preprocess_fn": inception_v3.preprocess_input
    },
    "ResNet50": {
        "m1_model": "modelo_resnet50_m1_detector.keras",
        "m2_model": "modelo_resnet50_m2_patogeno.keras",
        "config": "config_jerarquico_resnet50.json",
        "metrics": "metricas_resnet50_jerarquico.json",
        "preprocess_fn": resnet50.preprocess_input
    }
}

# Dónde buscar los archivos de cada arquitectura (en este orden):
# 1) carpeta actual, 2) /content (si los copiaste ahí), 3) carpeta de Drive de esa arquitectura.
RAIZ_DRIVE = "/content/drive/MyDrive/tesis"
CARPETA_ARCH = {"VGG19": "vgg19", "InceptionV3": "inceptionv3", "ResNet50": "resnet50"}

def resolver_archivos(arch, files):
    """Devuelve una copia de files con la ruta completa donde se encontró cada archivo."""
    carpetas = [os.getcwd(), "/content", os.path.join(RAIZ_DRIVE, CARPETA_ARCH[arch])]
    res = dict(files)
    for k in ("m1_model", "m2_model", "config", "metrics"):
        for c in carpetas:
            ruta = os.path.join(c, files[k])
            if os.path.exists(ruta):
                res[k] = ruta
                break
    return res

# Carga de recursos disponibles por arquitectura
for arch in list(ARCH_FILES.keys()):
    files = resolver_archivos(arch, ARCH_FILES[arch])
    ARCH_FILES[arch] = files   # deja las rutas resueltas (el preprocess_fn se conserva)
    if os.path.exists(files["m1_model"]) and os.path.exists(files["m2_model"]) and os.path.exists(files["config"]):
        try:
            MODELOS[arch] = {
                "m1": load_model(files["m1_model"], compile=False),
                "m2": load_model(files["m2_model"], compile=False)
            }
            with open(files["config"]) as f:
                CONFIGS[arch] = json.load(f)
            # Calentamiento: la primera llamada de Keras es lenta; así el cronómetro mide una inferencia normal
            _s1 = tuple(CONFIGS[arch].get("img_size_m1", [224, 224]))
            _s2 = tuple(CONFIGS[arch].get("img_size_m2", [299, 299]))
            MODELOS[arch]["m1"](np.zeros((1, _s1[1], _s1[0], 3), np.float32), training=False)
            MODELOS[arch]["m2"](np.zeros((1, _s2[1], _s2[0], 3), np.float32), training=False)
            print(f"✓ Sistema jerárquico {arch} (M1 + M2) cargado correctamente.")
        except Exception as e:
            MODELOS.pop(arch, None)
            print(f"⚠️ Error al cargar la jerarquía {arch}: {e}")
    else:
        faltan = [files[k] for k in ("m1_model", "m2_model", "config") if not os.path.exists(files[k])]
        print(f"· {arch}: no se encontró {faltan} (busqué en {os.getcwd()}, /content y {RAIZ_DRIVE}/{CARPETA_ARCH[arch]}); no estará disponible.")

    if os.path.exists(files["metrics"]):
        try:
            with open(files["metrics"]) as f:
                METRICAS[arch] = json.load(f)
        except Exception as e:
            print(f"⚠️ No se pudo leer {files['metrics']}: {e}")

historial_sesion = []

def ahora_str():
    return datetime.now(TZ).strftime("%d/%m/%Y %H:%M:%S") if TZ else datetime.now().strftime("%d/%m/%Y %H:%M:%S")

def tabla_historial_html(items, vacio):
    """Tabla HTML con columnas: Imagen, Fecha, Modelo usado, Clase, Confianza."""
    if not items:
        return f"<p style='color:#666'>{vacio}</p>"
    th = "style='text-align:left;padding:6px 10px;border-bottom:2px solid #999;background:#EEE'"
    td = "style='padding:6px 10px;border-bottom:1px solid #DDD'"
    filas = "".join(
        f"<tr><td {td}>{it['imagen']}</td><td {td}>{it['fecha']}</td><td {td}>{it['modelo']}</td>"
        f"<td {td}><b>{it['clase']}</b></td><td {td}>{it['confianza']:.2f}%</td></tr>"
        for it in items)
    return (f"<table style='border-collapse:collapse;font-family:sans-serif;font-size:13px;color:#222'>"
            f"<tr><th {th}>Imagen</th><th {th}>Fecha</th><th {th}>Modelo usado</th>"
            f"<th {th}>Clase</th><th {th}>Confianza</th></tr>{filas}</table>")

# ------------------------------------------------------------
# FUNCIONES AUXILIARES DE PROCESAMIENTO, VALIDADOR E INFERENCIA
# ------------------------------------------------------------
def obtener_datos_archivo(file_uploader):
    val = file_uploader.value
    if not val:
        return None, None
    if isinstance(val, (tuple, list)):
        item = val[0]
        nombre = item.get('name', 'radiografia.jpg') if isinstance(item, dict) else getattr(item, 'name', 'radiografia.jpg')
        contenido = item.get('content', b'') if isinstance(item, dict) else getattr(item, 'content', b'')
        return nombre, bytes(contenido)
    elif isinstance(val, dict):
        first_key = list(val.keys())[0]
        item = val[first_key]
        nombre = item.get('name', first_key) if isinstance(item, dict) else first_key
        contenido = item.get('content', b'') if isinstance(item, dict) else b''
        if hasattr(contenido, 'tobytes'):
            contenido = contenido.tobytes()
        return nombre, bytes(contenido)
    return "radiografia.jpg", b""

def es_imagen_valida(nombre_archivo, bytes_contenido):
    """Valida la extensión del archivo y que la estructura binaria sea una imagen procesable."""
    if not bytes_contenido or len(bytes_contenido) == 0:
        return False, "El archivo está vacío."

    extensiones_validas = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    if not nombre_archivo.lower().endswith(extensiones_validas):
        return False, f"Extensión no permitida. Formatos válidos: {', '.join(extensiones_validas)}"

    try:
        img = Image.open(io.BytesIO(bytes_contenido))
        img.verify()
        return True, "Imagen válida."
    except Exception:
        return False, "El archivo seleccionado está dañado o no es un formato de imagen válido."

def aplicar_preprocesamiento_cv(x, config):
    x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2GRAY)

    if config.get("use_clahe", True):
        clahe = cv2.createCLAHE(
            clipLimit=config.get("clahe_clip", 1.5),
            tileGridSize=tuple(config.get("clahe_grid", [8, 8]))
        )
        gray = clahe.apply(gray)

    return np.stack([gray, gray, gray], axis=-1)

def preprocesar_y_clasificar(bytes_imagen, arquitectura_seleccionada):
    if arquitectura_seleccionada not in MODELOS:
        return {"error": f"M-03: El sistema jerárquico {arquitectura_seleccionada} no está disponible."}

    # CRONÓMETRO: Inicio de medición de tiempo de inferencia
    tiempo_inicio = time.perf_counter()

    try:
        models_dict = MODELOS[arquitectura_seleccionada]
        model_m1 = models_dict["m1"]
        model_m2 = models_dict["m2"]

        config = CONFIGS[arquitectura_seleccionada]
        preprocess_fn = ARCH_FILES[arquitectura_seleccionada]["preprocess_fn"]

        img_pil = Image.open(io.BytesIO(bytes_imagen)).convert('RGB')

        # Tamaños de entrada requeridos por M1 y M2
        size_m1 = tuple(config.get("img_size_m1", [224, 224]))
        size_m2 = tuple(config.get("img_size_m2", [299, 299]))

        # Umbrales calibrados
        umbral_m1 = float(config.get("umbral_m1", 0.5))
        umbral_m2 = float(config.get("umbral_m2", 0.5))

        # --- PREPARACIÓN E INFERENCIA EN M1 (Detector Sano vs Neumonía) ---
        img_m1 = img_pil.resize(size_m1)
        x_m1 = np.array(img_m1, dtype=np.float32)
        proc_m1 = aplicar_preprocesamiento_cv(x_m1, config)
        input_m1 = np.expand_dims(preprocess_fn(proc_m1.astype(np.float32)), axis=0)

        pred_m1 = model_m1(input_m1, training=False).numpy()[0]
        prob_neumonia = float(pred_m1[1])  # P(neumonía)

        # --- PREPARACIÓN E INFERENCIA EN M2 (Patógeno Viral vs Bacteriana) ---
        img_m2 = img_pil.resize(size_m2)
        x_m2 = np.array(img_m2, dtype=np.float32)
        proc_m2 = aplicar_preprocesamiento_cv(x_m2, config)
        input_m2 = np.expand_dims(preprocess_fn(proc_m2.astype(np.float32)), axis=0)

        pred_m2 = model_m2(input_m2, training=False).numpy()[0]
        prob_bacteriana_cond = float(pred_m2[0]) if pred_m2.ndim == 1 else float(pred_m2)  # Sigmoid: P(bacteriana | neumonía)

        # --- LÓGICA DE DECISIÓN JERÁRQUICA ---
        if prob_neumonia < umbral_m1:
            clase_predicha = "sano"
            confianza = (1.0 - prob_neumonia) * 100.0
            prob_sano = 1.0 - prob_neumonia
            prob_bacteriana = prob_neumonia * prob_bacteriana_cond
            prob_viral = prob_neumonia * (1.0 - prob_bacteriana_cond)
        else:
            if prob_bacteriana_cond >= umbral_m2:
                clase_predicha = "bacteriana"
                confianza = prob_bacteriana_cond * 100.0
            else:
                clase_predicha = "viral"
                confianza = (1.0 - prob_bacteriana_cond) * 100.0

            prob_sano = 1.0 - prob_neumonia
            prob_bacteriana = prob_neumonia * prob_bacteriana_cond
            prob_viral = prob_neumonia * (1.0 - prob_bacteriana_cond)

        baja_confianza = 1 if (abs(prob_neumonia - umbral_m1) < 0.10) or (prob_neumonia >= umbral_m1 and abs(prob_bacteriana_cond - umbral_m2) < 0.10) else 0

        # CRONÓMETRO: Fin de medición de tiempo de inferencia
        tiempo_fin = time.perf_counter()
        tiempo_clasificacion = tiempo_fin - tiempo_inicio

        prob_dict = {
            "sano": prob_sano,
            "viral": prob_viral,
            "bacteriana": prob_bacteriana
        }

        # Confianza del nodo (solo la decisión final de M1 o M2) y probabilidad final de la clase (M1 x M2).
        # Las barras muestran la segunda; ahora la "confianza" mostrada y guardada es esa misma cifra.
        confianza_nodo = confianza
        confianza = prob_dict[clase_predicha] * 100.0

        return {
            "clase_predicha": clase_predicha,
            "confianza": confianza,
            "confianza_nodo": confianza_nodo,
            "probabilidades": prob_dict,
            "baja_confianza": baja_confianza,
            "img_pil": img_pil,
            "img_proc": proc_m1,
            "modelo_usado": arquitectura_seleccionada,
            "prob_m1_neumonia": prob_neumonia,
            "prob_m2_bacteriana": prob_bacteriana_cond,
            "umbral_m1": umbral_m1,
            "umbral_m2": umbral_m2,
            "tiempo_segundos": tiempo_clasificacion
        }
    except Exception as e:
        return {"error": f"M-03: Error al clasificar con jerarquía {arquitectura_seleccionada}:\n{str(e)}"}

# ------------------------------------------------------------
# 2. CONTENEDOR PRINCIPAL DE LA INTERFAZ
# ------------------------------------------------------------
header_html = widgets.HTML(
    value="""
    <div style="background-color: #2D2D2D; padding: 12px 20px; border-bottom: 2px solid #444; display: flex; justify-content: space-between; align-items: center;">
        <h2 style="margin:0; font-family: sans-serif; color: #FFF;">SCRT | Sistema Jerárquico de Clasificación de Radiografías Torácicas</h2>
        <span style="font-family: sans-serif; font-weight: bold; color: #BBB;">Usuario: Investigador</span>
    </div>
    """
)

footer_html = widgets.HTML(
    value="""
    <div style="background-color: #2D2D2D; padding: 10px; border-top: 1px solid #444; text-align: center; font-family: sans-serif; font-size: 13px; color: #CCC; margin-top: 15px;">
        <i>M-06: Herramienta de apoyo a la decisión clínica basada en arquitecturas jerárquicas (M1 + M2). No sustituye el criterio del especialista.</i>
    </div>
    """
)

content_area = widgets.VBox([], layout=widgets.Layout(flex='1', padding='15px', min_height='500px'))
temp_classification_data = {}

# ------------------------------------------------------------
# 3. FUNCIONES DE PANTALLA (P01 - P07)
# ------------------------------------------------------------

def render_P01():
    btn_c1 = widgets.Button(description="Abrir", button_style='primary')
    btn_c2 = widgets.Button(description="Abrir", button_style='primary')
    btn_c3 = widgets.Button(description="Abrir", button_style='primary')

    btn_c1.on_click(lambda b: render_P02())
    btn_c2.on_click(lambda b: render_P05())
    btn_c3.on_click(lambda b: render_P04())

    box1 = widgets.VBox([
        widgets.HTML("<b style='color:#333; font-size:14px;'>Clasificar radiografía</b><br><span style='color:#666; font-size:12px;'>Cargue una imagen y ejecute la jerarquía M1+M2</span>"),
        btn_c1
    ], layout=widgets.Layout(border='1px solid #CCC', padding='15px', width='30%', align_items='center'))

    box2 = widgets.VBox([
        widgets.HTML("<b style='color:#333; font-size:14px;'>Consultar métricas</b><br><span style='color:#666; font-size:12px;'>Revise el desempeño jerárquico calibrado</span>"),
        btn_c2
    ], layout=widgets.Layout(border='1px solid #CCC', padding='15px', width='30%', align_items='center'))

    box3 = widgets.VBox([
        widgets.HTML("<b style='color:#333; font-size:14px;'>Ver historial</b><br><span style='color:#666; font-size:12px;'>Consulte resultados procesados en la sesión</span>"),
        btn_c3
    ], layout=widgets.Layout(border='1px solid #CCC', padding='15px', width='30%', align_items='center'))

    ultimas = list(reversed(historial_sesion[-3:]))   # la más reciente primero
    hist_html = widgets.HTML(tabla_historial_html(ultimas, "(Aún no se han clasificado imágenes en esta sesión)"))

    content_area.children = [
        widgets.HTML("<h3>--- P01 · PANEL PRINCIPAL ---</h3>"),
        widgets.HBox([box1, box2, box3], layout=widgets.Layout(gap='15px', margin='0 0 25px 0')),
        widgets.HTML("<h4>Últimas 3 clasificaciones</h4>"),
        hist_html
    ]

def render_P02():
    file_uploader = widgets.FileUpload(accept='.png, .jpg, .jpeg, .bmp', multiple=False, description="Seleccionar archivo")

    opciones_modelos = list(MODELOS.keys()) if MODELOS else ['VGG19', 'InceptionV3', 'ResNet50']
    model_select = widgets.Dropdown(options=opciones_modelos, value=opciones_modelos[0], description='Arquitectura:')

    btn_cancel = widgets.Button(description="Cancelar")
    btn_classify = widgets.Button(description="Clasificar", button_style='success')

    preview_out = widgets.Output()
    info_out = widgets.Output()

    def on_upload(change):
        nombre, contenido = obtener_datos_archivo(file_uploader)

        # VALIDACIÓN DE FORMATO E IMAGEN
        es_valido, msj = es_imagen_valida(nombre, contenido)

        with preview_out:
            clear_output()
            if es_valido:
                img = Image.open(io.BytesIO(contenido))
                fig, ax = plt.subplots(figsize=(2.8, 2.8))
                ax.imshow(img)
                ax.axis('off')
                ax.set_title("Vista previa radiografía", fontsize=10)
                plt.show()

        with info_out:
            clear_output()
            if es_valido:
                print(f"✓ Archivo cargado: {nombre}")
                print(f"✓ Tamaño: {len(contenido)/1024:.1f} KB")
                print("✓ Estado: Listo para evaluación jerárquica.")
            else:
                print(f"❌ M-01 ERROR DE FORMATO: {msj}")

    file_uploader.observe(on_upload, names='value')

    def on_classify_click(b):
        nombre, contenido = obtener_datos_archivo(file_uploader)
        arch = model_select.value

        es_valido, msj = es_imagen_valida(nombre, contenido)
        if not es_valido:
            with info_out:
                clear_output()
                print(f"⚠️ M-01: No se puede clasificar. {msj}")
            return

        with info_out:
            clear_output()
            print(f"⏳ Evaluando imagen con sistema jerárquico {arch} (M1 + M2)...")

        res = preprocesar_y_clasificar(contenido, arch)
        if "error" in res:
            with info_out:
                clear_output()
                print(res["error"])
        else:
            global temp_classification_data
            temp_classification_data = {"nombre_imagen": nombre, "resultado": res}
            render_P03()

    btn_classify.on_click(on_classify_click)
    btn_cancel.on_click(lambda b: render_P01())

    content_area.children = [
        widgets.HTML("<h3>--- P02 · CARGAR RADIOGRAFÍA ---</h3>"),
        widgets.HBox([file_uploader, model_select]),
        widgets.HBox([preview_out, info_out], layout=widgets.Layout(margin='15px 0', min_height='200px')),
        widgets.HBox([btn_cancel, btn_classify])
    ]

def render_P03():
    if not temp_classification_data:
        content_area.children = [widgets.HTML("<p>No hay datos de clasificación activos.</p>")]
        return

    fname = temp_classification_data["nombre_imagen"]
    res = temp_classification_data["resultado"]
    clase = res["clase_predicha"].upper()
    conf = res["confianza"]
    probs = res["probabilidades"]
    modelo_usado = res["modelo_usado"]
    t_seg = res["tiempo_segundos"]

    plot_out = widgets.Output()
    with plot_out:
        if res["baja_confianza"]:
            print("⚠️ M-02 ADVERTENCIA: La probabilidad está muy cerca del umbral de decisión. Se sugiere revisión médica.\n")

        print(f"Archivo: {fname} | Arquitectura Jerárquica: {modelo_usado}")
        print(f"⏱️ Tiempo de Inferencia: {t_seg:.4f} segundos ({t_seg*1000:.1f} ms)")
        print(f"M1 Detector -> P(Neumonía): {res['prob_m1_neumonia']:.4f} (Umbral: {res['umbral_m1']:.2f})")
        print(f"M2 Patógeno -> P(Bacteriana|Neumonía): {res['prob_m2_bacteriana']:.4f} (Umbral: {res['umbral_m2']:.2f})")
        print(f"Diagnóstico Final: {clase} (Probabilidad final: {conf:.2f}%)")
        print(f"  (Probabilidad final = P(neumonía) x P(tipo|neumonía), igual que las barras; confianza solo del último nodo: {res['confianza_nodo']:.2f}%)\n")

        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
        axes[0].imshow(res["img_proc"])
        axes[0].set_title(f"Radiografía Procesada (CLAHE)", fontsize=9)
        axes[0].axis('off')

        labels = ["sano", "viral", "bacteriana"]
        values = [probs[k] * 100.0 for k in labels]
        colors = ['#1f77b4' if k.upper() == clase else '#cccccc' for k in labels]

        axes[1].barh(labels, values, color=colors)
        axes[1].set_xlim(0, 100)
        axes[1].set_xlabel("Probabilidad Ponderada (%)")
        axes[1].set_title(f"Distribución Jerárquica ({t_seg*1000:.0f} ms)", fontsize=9)
        plt.tight_layout()
        plt.show()

    btn_new = widgets.Button(description="Nueva clasificación")
    btn_save = widgets.Button(description="Guardar resultado", button_style='success')
    status_lbl = widgets.Label(value="")

    def on_save(b):
        historial_sesion.append({
            "imagen": fname,
            "fecha": ahora_str(),
            "modelo": modelo_usado,
            "clase": clase,
            "confianza": conf,
            "tiempo": t_seg
        })
        status_lbl.value = f"M-04: Resultado guardado correctamente en la sesión (Inferencia: {t_seg:.3f}s)."
        btn_save.disabled = True

    btn_new.on_click(lambda b: render_P02())
    btn_save.on_click(on_save)

    content_area.children = [
        widgets.HTML("<h3>--- P03 · RESULTADO DE LA CLASIFICACIÓN JERÁRQUICA ---</h3>"),
        plot_out,
        widgets.HBox([btn_new, btn_save]),
        status_lbl
    ]

def render_P04():
    tabla = widgets.HTML(tabla_historial_html(list(reversed(historial_sesion)),
                                              "M-05: No se encontraron resultados registrados en la sesión."))
    content_area.children = [
        widgets.HTML("<h3>--- P04 · HISTORIAL DE RESULTADOS ---</h3>"),
        tabla
    ]

def dibujar_matriz(arch):
    m = METRICAS.get(arch)
    if not m or "matriz_confusion" not in m:
        print(f"{arch}: no hay matriz de confusión disponible (falta {os.path.basename(ARCH_FILES[arch]['metrics'])}).")
        return
    cm = np.array(m["matriz_confusion"], dtype=float)
    clases = m.get("clases", CLASSES)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, data, titulo, fmt in [(axes[0], cm, "Conteos", "{:.0f}"),
                                  (axes[1], cm_norm, "Normalizada por fila (recall)", "{:.2f}")]:
        ax.imshow(data, cmap="Blues", vmin=0, vmax=data.max())
        ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases)
        ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
        ax.set_xlabel("Predicho"); ax.set_ylabel("Real"); ax.set_title(titulo, fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center",
                        color="white" if data[i, j] > data.max() * 0.55 else "black")
    fig.suptitle(f"Matriz de confusión · {arch} (conjunto test, n={int(cm.sum())})", fontsize=11)
    plt.tight_layout()
    plt.show()

def render_P05():
    met_out = widgets.Output()
    with met_out:
        print("Desempeño registrado del Ensamble Jerárquico (Conjunto Test):")
        print(f"{'Arquitectura':<13} | {'Accuracy':<9} | {'F1-Macro':<9} | {'F1-Pond.':<9} | {'Recall':<9} | {'Precision':<9}")
        print("-" * 72)
        for arch in ["VGG19", "InceptionV3", "ResNet50"]:
            if arch in METRICAS:
                m = METRICAS[arch]
                print(f"{arch:<13} | {m.get('accuracy', 0):<9.4f} | {m.get('f1_macro', 0):<9.4f} | "
                      f"{m.get('f1_score', 0):<9.4f} | {m.get('recall', 0):<9.4f} | {m.get('precision', 0):<9.4f}")
            else:
                print(f"{arch:<13} | {'No evaluado':<9} | {'-':<9} | {'-':<9} | {'-':<9} | {'-'}")
        print("\n(Recall, Precision y F1-Pond. son promedios ponderados por clase.)")

    cm_out = widgets.Output()
    selector = widgets.ToggleButtons(
        options=["VGG19", "InceptionV3", "ResNet50"],
        description="Matriz de:", style={"button_width": "110px"}, value=None)

    def on_select(change):
        if change["new"] is None:
            return
        with cm_out:
            clear_output(wait=True)
            dibujar_matriz(change["new"])

    selector.observe(on_select, names="value")
    with cm_out:
        print("Elija una arquitectura para ver su matriz de confusión.")

    content_area.children = [
        widgets.HTML("<h3>--- P05 · MÉTRICAS COMPARATIVAS JERÁRQUICAS ---</h3>"),
        met_out,
        widgets.HTML("<h4>Matriz de confusión por modelo</h4>"),
        selector,
        cm_out
    ]

def render_P06():
    ds_out = widgets.Output()
    with ds_out:
        print(f"Ruta configurada: {BASE_DIR}\n")
        def contar(dir_path):
            res = {}
            for c in CLASSES:
                p = os.path.join(dir_path, c)
                res[c] = len(os.listdir(p)) if os.path.exists(p) else 0
            return res

        c_train, c_val, c_test = contar(TRAIN_DIR), contar(VAL_DIR), contar(TEST_DIR)

        print(f"{'Partición':<15} | {'Bacteriana':<10} | {'Sano':<10} | {'Viral':<10} | {'Total':<10}")
        print("-" * 60)
        print(f"{'Entrenamiento':<15} | {c_train.get('bacteriana',0):<10} | {c_train.get('sano',0):<10} | {c_train.get('viral',0):<10} | {sum(c_train.values()):<10}")
        print(f"{'Validación':<15} | {c_val.get('bacteriana',0):<10} | {c_val.get('sano',0):<10} | {c_val.get('viral',0):<10} | {sum(c_val.values()):<10}")
        print(f"{'Prueba (Test)':<15} | {c_test.get('bacteriana',0):<10} | {c_test.get('sano',0):<10} | {c_test.get('viral',0):<10} | {sum(c_test.values()):<10}")

    content_area.children = [
        widgets.HTML("<h3>--- P06 · GESTIÓN DEL DATASET POR PACIENTE ---</h3>"),
        ds_out
    ]

def render_P07():
    train_out = widgets.Output()
    with train_out:
        print("Sistemas jerárquicos de 2 etapas configurados en el entorno:")
        print(" • VGG19       : M1 (224x224) Sano/Neumonía  | M2 (299x299) Viral/Bacteriana")
        print(" • InceptionV3 : M1 (299x299) Sano/Neumonía  | M2 (299x299) Viral/Bacteriana")
        print(" • ResNet50    : M1 (224x224) Sano/Neumonía  | M2 (299x299) Viral/Bacteriana")
        print("\nPreprocesamiento global: Ecualización adaptativa CLAHE (clip=1.5, grid=8x8)")
        print("Validación de Entrada: Filtro activo para formatos raster (.png, .jpg, .jpeg, .bmp)")

    content_area.children = [
        widgets.HTML("<h3>--- P07 · ENTRENAMIENTO Y JERARQUÍA ---</h3>"),
        train_out
    ]

# ------------------------------------------------------------
# 4. MENÚ LATERAL Y NAVEGACIÓN
# ------------------------------------------------------------
btn_m1 = widgets.Button(description="Inicio", layout=widgets.Layout(width='90%'))
btn_m2 = widgets.Button(description="Cargar radiografía", layout=widgets.Layout(width='90%'))
btn_m3 = widgets.Button(description="Historial", layout=widgets.Layout(width='90%'))
btn_m4 = widgets.Button(description="Métricas", layout=widgets.Layout(width='90%'))
btn_m5 = widgets.Button(description="Dataset", layout=widgets.Layout(width='90%'))
btn_m6 = widgets.Button(description="Entrenamiento", layout=widgets.Layout(width='90%'))

btn_m1.on_click(lambda b: render_P01())
btn_m2.on_click(lambda b: render_P02())
btn_m3.on_click(lambda b: render_P04())
btn_m4.on_click(lambda b: render_P05())
btn_m5.on_click(lambda b: render_P06())
btn_m6.on_click(lambda b: render_P07())

sidebar = widgets.VBox(
    [btn_m1, btn_m2, btn_m3, btn_m4, btn_m5, btn_m6],
    layout=widgets.Layout(width='190px', padding='10px')
)

main_body = widgets.HBox([sidebar, content_area], layout=widgets.Layout(border='1px solid #CCC'))
app_layout = widgets.VBox([header_html, main_body, footer_html])

# Inicialización
clear_output()
display(app_layout)
render_P01()