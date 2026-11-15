#CODIGO PARA PARTICIONAR EL DATASET Y QUE NO HAYA FUGA DE DATOS
# ============================================================
# REPARTO POR PACIENTE (sin tocar el dataset original)
#
# Lee /content/dataset (train / validation / test) y crea una COPIA en
# /content/dataset_por_paciente donde cada paciente queda en UNA sola
# partición. Intenta conservar los mismos totales por clase y partición
# que tenías (4098 / 879 / 879) y al final verifica que no haya pacientes
# repetidos entre particiones.
# ============================================================

import os
import re
import shutil
import random
import hashlib
import pandas as pd
from collections import defaultdict

# ------------------------------------------------------------
# 1. PARÁMETROS
# ------------------------------------------------------------
ORIGEN = "/content/dataset"
DESTINO = "/content/dataset_por_paciente"
SEED = 42
PARTICIONES = ["train", "validation", "test"]
CLASES = ["bacteriana", "sano", "viral"]
EXT = (".jpg", ".jpeg", ".png")
VERIFICAR_DUPLICADOS_EXACTOS = True   # compara el contenido (md5) entre particiones

# ------------------------------------------------------------
# 2. IDENTIFICAR AL PACIENTE POR EL NOMBRE DEL ARCHIVO
#    person441_virus_894.jpeg      -> person441
#    person946_bacteria_2871.jpeg  -> person946
#    IM-0115-0001.jpeg             -> im-0115
#    NORMAL2-IM-1226-0001.jpeg     -> im-1226
# ------------------------------------------------------------
def id_paciente(nombre):
    m = re.search(r"(person\d+)", nombre, re.I)
    if m:
        return m.group(1).lower()
    m = re.search(r"IM-(\d+)", nombre, re.I)
    if m:
        return "im-" + m.group(1)
    return None


# ------------------------------------------------------------
# 3. LEER EL DATASET ORIGINAL (todas las particiones juntas)
# ------------------------------------------------------------
def listar(raiz):
    filas = []
    for part in PARTICIONES:
        for clase in CLASES:
            d = os.path.join(raiz, part, clase)
            if not os.path.isdir(d):
                raise FileNotFoundError(f"No existe la carpeta: {d}")
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXT):
                    pid = id_paciente(f)
                    filas.append({
                        "ruta": os.path.join(d, f), "archivo": f, "clase": clase,
                        "particion": part,
                        # sin id reconocible: cada imagen cuenta como su propio "paciente"
                        "paciente": pid if pid else f"sinid::{clase}::{os.path.splitext(f)[0]}",
                        "reconocido": pid is not None,
                    })
    return pd.DataFrame(filas)


if os.path.exists(DESTINO) and os.listdir(DESTINO):
    raise FileExistsError(
        f"{DESTINO} ya existe y no está vacío. Bórralo o cambia DESTINO para no mezclar repartos.")

df = listar(ORIGEN)
print(f"Imágenes encontradas: {len(df)}")
print(f"Con paciente identificado: {int(df.reconocido.sum())} | sin identificar: {int((~df.reconocido).sum())}")
if (~df.reconocido).any():
    print("  (las no identificadas se tratan como pacientes individuales; revisa el patrón del nombre)")

orig = pd.crosstab(df.clase, df.particion).reindex(index=CLASES, columns=PARTICIONES)
print("\nDistribución ORIGINAL (imágenes):")
print(orig)

# ------------------------------------------------------------
# 4. GRUPOS = PACIENTES
# ------------------------------------------------------------
grupos = df.groupby("paciente").agg(
    n=("ruta", "size"),
    clase=("clase", lambda s: s.value_counts().idxmax()),     # clase dominante
    n_clases=("clase", "nunique"),
)
print(f"\nPacientes distintos: {len(grupos)}")
print("Pacientes con imágenes en más de una clase:", int((grupos.n_clases > 1).sum()))

# Objetivo: mantener los mismos totales por clase y partición que el dataset original
objetivo = {(c, p): int(orig.loc[c, p]) for c in CLASES for p in PARTICIONES}

# ------------------------------------------------------------
# 5. ASIGNAR PACIENTES A PARTICIONES (aleatorio con semilla, llenando por déficit)
# ------------------------------------------------------------
rng = random.Random(SEED)
asignacion = {}
for clase in CLASES:
    lista = [(pid, int(r.n)) for pid, r in grupos[grupos.clase == clase].iterrows()]
    rng.shuffle(lista)
    actual = {p: 0 for p in PARTICIONES}
    for pid, n in lista:
        # partición que está más lejos de su objetivo (en proporción)
        p = max(PARTICIONES,
                key=lambda p: (objetivo[(clase, p)] - actual[p]) / max(objetivo[(clase, p)], 1))
        asignacion[pid] = p
        actual[p] += n

df["particion_nueva"] = df.paciente.map(asignacion)

# ------------------------------------------------------------
# 6. COPIAR (el original no se modifica)
# ------------------------------------------------------------
for part in PARTICIONES:
    for clase in CLASES:
        os.makedirs(os.path.join(DESTINO, part, clase), exist_ok=True)

for fila in df.itertuples():
    destino = os.path.join(DESTINO, fila.particion_nueva, fila.clase, fila.archivo)
    if os.path.exists(destino):                       # mismo nombre en dos particiones originales
        destino = os.path.join(DESTINO, fila.particion_nueva, fila.clase,
                               f"{fila.particion}_{fila.archivo}")
    shutil.copy2(fila.ruta, destino)

# ------------------------------------------------------------
# 7. VERIFICACIÓN SOBRE EL DATASET NUEVO (volviendo a leer los archivos copiados)
# ------------------------------------------------------------
nuevo = listar(DESTINO)
print(f"\nImágenes copiadas: {len(nuevo)} (original: {len(df)})")
assert len(nuevo) == len(df), "El número de imágenes no coincide"

dist_nueva = pd.crosstab(nuevo.clase, nuevo.particion).reindex(index=CLASES, columns=PARTICIONES)
print("\nDistribución NUEVA (imágenes):")
print(dist_nueva)
print("\nCambio respecto al original (imágenes):")
print(dist_nueva - orig)
print("\nCambio relativo (%):")
print(((dist_nueva - orig) / orig * 100).round(1))

pac = {p: set(nuevo[(nuevo.particion == p) & nuevo.reconocido].paciente) for p in PARTICIONES}
print("\nPacientes distintos por partición:", {k: len(v) for k, v in pac.items()})
c1 = len(pac["train"] & pac["validation"])
c2 = len(pac["train"] & pac["test"])
c3 = len(pac["validation"] & pac["test"])
print("Pacientes compartidos train ∩ validation:", c1)
print("Pacientes compartidos train ∩ test      :", c2)
print("Pacientes compartidos validation ∩ test :", c3)
assert c1 == c2 == c3 == 0, "Quedaron pacientes repetidos entre particiones"
print("OK: 0 pacientes repetidos entre particiones.")

if VERIFICAR_DUPLICADOS_EXACTOS:
    print("\nBuscando imágenes idénticas (mismo contenido) en particiones distintas...")
    por_hash = defaultdict(set)
    for fila in nuevo.itertuples():
        with open(fila.ruta, "rb") as f:
            por_hash[hashlib.md5(f.read()).hexdigest()].add(fila.particion)
    repetidas = sum(1 for v in por_hash.values() if len(v) > 1)
    print(f"Imágenes con contenido idéntico en más de una partición: {repetidas}")
    if repetidas:
        print("  Aviso: hay copias exactas en particiones distintas; conviene revisarlas.")

print(f"\nListo. Dataset nuevo en: {DESTINO}")
print('Para entrenar con él, cambia en el script de entrenamiento: BASE_DIR = "/content/dataset_por_paciente"')