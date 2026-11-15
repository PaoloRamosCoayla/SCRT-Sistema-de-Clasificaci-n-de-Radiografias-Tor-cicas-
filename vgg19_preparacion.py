# ##############################################################
# VGG19 JERÁRQUICO — DIVIDIDO EN 4 CELDAS PARA COLAB
#
# Cada bloque entre las líneas "CELDA X" va en UNA celda de Colab.
#   CELDA A: preparación  -> correr SIEMPRE primero al abrir Colab
#   CELDA B: entrenar M1  -> solo si necesitas M1
#   CELDA C: entrenar M2  -> solo si necesitas M2
#   CELDA D: evaluación   -> carga M1 y M2 desde Drive y calcula todo
#
# Todo lo que se guarda (modelos, JSON, checkpoints, probabilidades)
# queda en la carpeta de Drive definida en CARPETA_DRIVE.
# ##############################################################


# ##############################################################
# CELDA A — PREPARACIÓN (ejecutar siempre primero en una sesión nueva) VGG19
# ##############################################################
import os
import json
import random
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight

# ---- Google Drive: todo lo que se guarde con nombre simple cae en esta carpeta ----
from google.colab import drive
drive.mount("/content/drive")
CARPETA_DRIVE = "/content/drive/MyDrive/tesis/vgg19"     # <-- ajusta a tu Drive
os.makedirs(CARPETA_DRIVE, exist_ok=True)
os.chdir(CARPETA_DRIVE)
print("Carpeta de trabajo (Drive):", os.getcwd())

print("=" * 70)
print("ENTRENAMIENTO JERÁRQUICO VGG19 - VERSIÓN CORREGIDA")
print("=" * 70)
print("TensorFlow:", tf.__version__)
print("OpenCV:", cv2.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))

# ============================================================
# 1. PARÁMETROS
# ============================================================
BASE_DIR = "/content/dataset_por_paciente"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "validation")
TEST_DIR = os.path.join(BASE_DIR, "test")

if not os.path.isdir(TRAIN_DIR):
    raise FileNotFoundError(
        f"No se encuentra {TRAIN_DIR}. Copia o descomprime tu dataset_por_paciente "
        f"en /content antes de continuar (el disco de Colab se borra en cada sesión).")

# Orden alfabético = mismo orden que usaba flow_from_directory (bacteriana, sano, viral)
CLASES_ORDEN = ["bacteriana", "sano", "viral"]
IDX = {c: i for i, c in enumerate(CLASES_ORDEN)}

IMG_SIZE_M1 = (224, 224)
IMG_SIZE_M2 = (299, 299)
BATCH_SIZE = 32
SEED = 42
PESOS = "imagenet"          # (None solo para pruebas rápidas sin descargar pesos)

EPOCAS_CABEZA = 6           # etapa 1: solo la cabeza (VGG19 congelada)
EPOCAS_FINE = 20            # etapa 2: fine-tuning (el EarlyStopping corta antes si no mejora)
PACIENCIA = 5
LR_CABEZA = 1e-3            # la cabeza es nueva: admite un LR mayor
LR_FINE = 1e-5

# Bloques de VGG19 que se descongelan en la etapa 2
FT_M1 = ("block5",)                 # Sano vs Neumonía es fácil: basta con el último bloque
FT_M2 = ("block4", "block5")        # Viral vs Bacteriana es sutil: se necesita adaptar más

USE_CLAHE = True
CLAHE_CLIP = 1.5
CLAHE_GRID = (8, 8)

# Rango inferior de la rejilla de umbrales (0.20 = igual que antes).
# Si quieres explorar umbrales más bajos, pon 0.05. Se decide solo con validación.
GRID_MIN = 0.20

RUTA_M1 = "modelo_vgg19_m1_detector.keras"
RUTA_M2 = "modelo_vgg19_m2_patogeno.keras"

tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ============================================================
# 2. LISTADO DEL DATASET (una sola fuente de verdad)
#    Ya NO se manipulan los atributos internos de los generadores.
# ============================================================
EXT = (".jpg", ".jpeg", ".png")


def listar_dataset(carpeta):
    filas = []
    for clase in CLASES_ORDEN:
        d = os.path.join(carpeta, clase)
        if not os.path.isdir(d):
            raise FileNotFoundError(f"No existe la carpeta: {d}")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(EXT):
                filas.append((os.path.join(d, f), clase))
    df = pd.DataFrame(filas, columns=["filepath", "clase"])
    df["nivel1"] = np.where(df["clase"] == "sano", "sano", "neumonia")
    return df


df_train = listar_dataset(TRAIN_DIR)
df_val = listar_dataset(VAL_DIR)
df_test = listar_dataset(TEST_DIR)

print("\nImágenes por partición y clase (los 3 grupos están en train, validation y test):")
resumen = pd.concat(
    {"train": df_train.clase.value_counts(),
     "validation": df_val.clase.value_counts(),
     "test": df_test.clase.value_counts()}, axis=1
).reindex(CLASES_ORDEN)
resumen.loc["TOTAL"] = resumen.sum()
print(resumen)

# Subconjuntos para M2 (solo neumonías)
df_train_m2 = df_train[df_train.clase != "sano"].reset_index(drop=True)
df_val_m2 = df_val[df_val.clase != "sano"].reset_index(drop=True)
print(f"\nM1 entrena con {len(df_train)} imágenes (valida con {len(df_val)}).")
print(f"M2 entrena con {len(df_train_m2)} imágenes (valida con {len(df_val_m2)}): solo viral + bacteriana.")

# ============================================================
# 3. PREPROCESAMIENTO (idéntico en entrenamiento e inferencia)
# ============================================================
def aplicar_preprocesamiento_cv(x):
    x_uint8 = np.clip(x, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(x_uint8, cv2.COLOR_RGB2GRAY)
    if USE_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
        gray = clahe.apply(gray)
    return np.stack([gray, gray, gray], axis=-1)


def preprocesar_radiografia(x):
    rgb = aplicar_preprocesamiento_cv(x).astype(np.float32)
    return preprocess_input(rgb)


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocesar_radiografia,
    rotation_range=8,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    brightness_range=(0.9, 1.1),
    horizontal_flip=False,
    fill_mode="nearest",
)
eval_datagen = ImageDataGenerator(preprocessing_function=preprocesar_radiografia)


def crear_gen(datagen, df, size, y_col=None, clases=None, class_mode=None, shuffle=False):
    return datagen.flow_from_dataframe(
        df, x_col="filepath", y_col=y_col, classes=clases,
        target_size=size, batch_size=BATCH_SIZE, class_mode=class_mode,
        shuffle=shuffle, seed=SEED, validate_filenames=False,
    )

# ============================================================
# 4. FUNCIONES DE MODELO Y ENTRENAMIENTO
# ============================================================
def respaldar(ruta):
    """Si el archivo ya existe, guarda una copia con fecha antes de sobrescribirlo."""
    if os.path.exists(ruta):
        copia = ruta.replace(".keras", f"_respaldo_{time.strftime('%Y%m%d_%H%M%S')}.keras")
        shutil.copy(ruta, copia)
        print("Respaldo del modelo anterior:", copia)


def construir_modelo(size, tipo, dropout, nombre):
    base = VGG19(weights=PESOS, include_top=False, input_shape=(size[0], size[1], 3))
    base.trainable = False
    salida = (layers.Dense(2, activation="softmax") if tipo == "softmax2"
              else layers.Dense(1, activation="sigmoid"))
    modelo = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        salida,
    ], name=nombre)
    return modelo, base


def compilar(modelo, lr, tipo):
    if tipo == "softmax2":
        loss, metricas = "categorical_crossentropy", ["accuracy"]
    else:
        loss, metricas = "binary_crossentropy", ["accuracy", tf.keras.metrics.AUC(name="auc")]
    modelo.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=loss, metrics=metricas)


def crear_callbacks(ruta, monitor, modo):
    return [
        ModelCheckpoint(ruta, monitor=monitor, mode=modo, save_best_only=True, verbose=1),
        EarlyStopping(monitor=monitor, mode=modo, patience=PACIENCIA,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1),
    ]


def mejor_valor(hist, monitor, modo):
    v = hist.history[monitor]
    return max(v) if modo == "max" else min(v)


def entrenar_modelo(nombre, tipo, size, df_tr, df_va, y_col, clases, class_mode,
                    bloques_ft, dropout, monitor, modo):
    gen_tr = crear_gen(train_datagen, df_tr, size, y_col, clases, class_mode, shuffle=True)
    gen_va = crear_gen(eval_datagen, df_va, size, y_col, clases, class_mode, shuffle=False)
    print("Mapeo de clases del generador:", gen_tr.class_indices)

    pesos = compute_class_weight("balanced", classes=np.array(clases), y=df_tr[y_col].values)
    class_weight = {i: float(p) for i, p in enumerate(pesos)}
    print("Pesos de clase:", class_weight)

    modelo, base = construir_modelo(size, tipo, dropout, nombre)

    # ---- Etapa 1: solo la cabeza ----
    compilar(modelo, LR_CABEZA, tipo)
    h1 = modelo.fit(
        gen_tr, validation_data=gen_va, epochs=EPOCAS_CABEZA, class_weight=class_weight,
        callbacks=crear_callbacks(f"best_{nombre}_etapa1.keras", monitor, modo))
    mejor1 = mejor_valor(h1, monitor, modo)
    pesos_etapa1 = modelo.get_weights()          # mejores pesos de la etapa 1 (EarlyStopping los restauró)

    # ---- Etapa 2: fine-tuning de los últimos bloques ----
    base.trainable = True
    for capa in base.layers:
        capa.trainable = capa.name.startswith(bloques_ft)
    n_ent = sum(c.trainable for c in base.layers)
    print(f"Fine-tuning: {n_ent} capas de VGG19 descongeladas {bloques_ft}")

    compilar(modelo, LR_FINE, tipo)
    ini = len(h1.epoch)
    h2 = modelo.fit(
        gen_tr, validation_data=gen_va, initial_epoch=ini, epochs=ini + EPOCAS_FINE,
        class_weight=class_weight,
        callbacks=crear_callbacks(f"best_{nombre}_etapa2.keras", monitor, modo))
    mejor2 = mejor_valor(h2, monitor, modo)

    # Nos quedamos con la mejor de las dos etapas (antes la etapa 2 siempre pisaba a la 1)
    peor_la_2 = (mejor2 < mejor1) if modo == "max" else (mejor2 > mejor1)
    if peor_la_2:
        modelo.set_weights(pesos_etapa1)
        print(f"-> Se conserva la ETAPA 1 ({monitor}={mejor1:.4f} vs {mejor2:.4f})")
    else:
        print(f"-> Se conserva la ETAPA 2 ({monitor}={mejor2:.4f} vs {mejor1:.4f})")

    historial = {k: h1.history[k] + h2.history[k] for k in h1.history if k in h2.history}
    return modelo, historial, len(h1.epoch)


def graficar_historial(historial, corte, titulo):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, clave, nombre in [(axes[0], "accuracy", "Accuracy"), (axes[1], "loss", "Loss")]:
        ax.plot(historial[clave], label="Entrenamiento")
        ax.plot(historial["val_" + clave], label="Validación")
        ax.axvline(corte - 0.5, color="gray", linestyle="--", label="Inicio fine-tuning")
        ax.set_title(f"{titulo} - {nombre}")
        ax.set_xlabel("Época")
        ax.legend()
    plt.tight_layout()
    plt.show()

print("\n✓ Celda A lista. Ahora corre la celda que necesites (B = M1, C = M2, D = evaluación).")