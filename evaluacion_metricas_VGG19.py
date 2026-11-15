# ##############################################################
# CELDA D — EVALUACIÓN (carga M1 y M2 desde Drive si no están en memoria) VGG19
# ##############################################################
if "model_m1" not in globals():
    model_m1 = tf.keras.models.load_model(RUTA_M1, compile=False)
if "model_m2" not in globals():
    model_m2 = tf.keras.models.load_model(RUTA_M2, compile=False)

# ============================================================
# 7. PREDICCIONES EN BATCH (ambos modelos sobre TODAS las imágenes)
#    Mismo orden de filas del DataFrame en M1 y M2 -> alineación garantizada.
# ============================================================
def predecir_df(modelo, df, size):
    gen = crear_gen(eval_datagen, df, size, class_mode=None, shuffle=False)
    p = modelo.predict(gen, verbose=1)
    assert len(p) == len(df), "El número de predicciones no coincide con el de imágenes"
    return p


def probs_jerarquia(df):
    p1 = predecir_df(model_m1, df, IMG_SIZE_M1)[:, 1]        # P(neumonia)
    p2 = predecir_df(model_m2, df, IMG_SIZE_M2).ravel()      # P(bacteriana | neumonia)
    return p1, p2


def decidir(p1, p2, t1=0.5, t2=0.5):
    """Ruteo jerárquico: M1 decide sano/neumonía; si es neumonía, M2 decide el tipo."""
    y = np.full(len(p1), IDX["sano"])
    neu = p1 >= t1
    y[neu & (p2 >= t2)] = IDX["bacteriana"]
    y[neu & (p2 < t2)] = IDX["viral"]
    return y


print("\nPrediciendo en validación...")
p1_val, p2_val = probs_jerarquia(df_val)
print("Prediciendo en test...")
p1_test, p2_test = probs_jerarquia(df_test)

y_val = df_val.clase.map(IDX).values
y_test = df_test.clase.map(IDX).values

# Guardar las probabilidades en Drive: permite ensamblar modelos, bootstrap, curvas ROC, etc.
# SIN volver a usar la GPU.
np.savez("probs_vgg19.npz", p1_val=p1_val, p2_val=p2_val, p1_test=p1_test, p2_test=p2_test,
         y_val=y_val, y_test=y_test)
print("✓ Probabilidades guardadas en probs_vgg19.npz")

# ============================================================
# 8. UMBRALES CALIBRADOS EN VALIDACIÓN (nunca en test)
#    Maximizan el F1-macro de la jerarquía completa.
# ============================================================
GRID = np.round(np.arange(GRID_MIN, 0.801, 0.02), 2)
mejor = (-1.0, 0.5, 0.5)
for t1 in GRID:
    for t2 in GRID:
        f = f1_score(y_val, decidir(p1_val, p2_val, t1, t2), average="macro")
        if f > mejor[0]:
            mejor = (f, float(t1), float(t2))
f1_val_cal, UMBRAL_M1, UMBRAL_M2 = mejor
f1_val_def = f1_score(y_val, decidir(p1_val, p2_val), average="macro")
print(f"\nUmbrales calibrados en validación -> M1: {UMBRAL_M1:.2f} | M2: {UMBRAL_M2:.2f}")
print(f"F1-macro en validación: {f1_val_def:.4f} (umbral 0.5) -> {f1_val_cal:.4f} (calibrado)")

# ============================================================
# 9. DIAGNÓSTICO POR ETAPA (para saber dónde está el cuello de botella)
# ============================================================
print("\n" + "=" * 70)
print(">>> DIAGNÓSTICO EN TEST")
print("=" * 70)

# 9.1 M1 aislado
y1 = (df_test.clase != "sano").astype(int).values
print("\n[M1 solo] Sano vs Neumonía (umbral calibrado):")
print(classification_report(y1, (p1_test >= UMBRAL_M1).astype(int),
                            target_names=["sano", "neumonia"], digits=4, zero_division=0))
auc_m1 = roc_auc_score(y1, p1_test)
print(f"AUC M1: {auc_m1:.4f}")

# 9.2 M2 aislado ("oráculo": solo imágenes que realmente son neumonía)
mask = (df_test.clase != "sano").values
y2 = (df_test.clase[mask] == "bacteriana").astype(int).values
print("\n[M2 solo] Viral vs Bacteriana sobre neumonías reales (umbral calibrado):")
print(classification_report(y2, (p2_test[mask] >= UMBRAL_M2).astype(int),
                            target_names=["viral", "bacteriana"], digits=4, zero_division=0))
auc_m2 = roc_auc_score(y2, p2_test[mask])
print(f"AUC M2: {auc_m2:.4f}  (0.5 = azar, 1.0 = perfecto)")

# ============================================================
# 10. EVALUACIÓN FINAL DE LA JERARQUÍA EN TEST
# ============================================================
y_pred_def = decidir(p1_test, p2_test)                          # umbrales 0.5 / 0.5
y_pred_full = decidir(p1_test, p2_test, UMBRAL_M1, UMBRAL_M2)   # umbrales calibrados
class_names_full = CLASES_ORDEN

print("\n" + "=" * 70)
print(">>> REPORTE FINAL DE LA JERARQUÍA (umbrales 0.5 / 0.5, para comparar)")
print("=" * 70)
print(classification_report(y_test, y_pred_def, target_names=class_names_full,
                            digits=4, zero_division=0))

print("=" * 70)
print(f">>> REPORTE FINAL DE LA JERARQUÍA (umbrales calibrados {UMBRAL_M1:.2f} / {UMBRAL_M2:.2f})")
print("=" * 70)
print(classification_report(y_test, y_pred_full, target_names=class_names_full,
                            digits=4, zero_division=0))

acc_final = accuracy_score(y_test, y_pred_full)
prec_final = precision_score(y_test, y_pred_full, average="weighted", zero_division=0)
rec_final = recall_score(y_test, y_pred_full, average="weighted", zero_division=0)
f1_final = f1_score(y_test, y_pred_full, average="weighted", zero_division=0)
f1_macro_final = f1_score(y_test, y_pred_full, average="macro", zero_division=0)

print(f"Accuracy Global : {acc_final:.4f}")
print(f"F1 ponderado    : {f1_final:.4f}")
print(f"F1 macro        : {f1_macro_final:.4f}")

cm = confusion_matrix(y_test, y_pred_full, labels=list(range(3)))
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True).clip(min=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names_full,
            yticklabels=class_names_full, ax=axes[0])
axes[0].set_title("Matriz de Confusión (Conteos)")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names_full,
            yticklabels=class_names_full, ax=axes[1])
axes[1].set_title("Matriz de Confusión (Normalizada por clase)")
for ax in axes:
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
plt.tight_layout()
plt.show()

# ============================================================
# 11. GUARDAR CONFIGURACIONES Y MÉTRICAS PARA LA INTERFAZ
# ============================================================
def a_json(o):
    if isinstance(o, dict):
        return {str(k): a_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [a_json(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    return o


# Se agrega "arquitectura" (antes faltaba en VGG19; Inception y ResNet ya lo traían)
prep = {"use_clahe": USE_CLAHE, "clahe_clip": CLAHE_CLIP, "clahe_grid": list(CLAHE_GRID),
        "arquitectura": "VGG19"}

with open("config_vgg19_m1.json", "w") as f:
    json.dump({"class_indices": {"sano": 0, "neumonia": 1}, "img_size": list(IMG_SIZE_M1),
               "salida": "softmax2", "umbral": UMBRAL_M1, **prep}, f, indent=2)

with open("config_vgg19_m2.json", "w") as f:
    json.dump({"class_indices": {"viral": 0, "bacteriana": 1}, "img_size": list(IMG_SIZE_M2),
               "salida": "sigmoid1", "umbral": UMBRAL_M2, **prep}, f, indent=2)

with open("config_jerarquico.json", "w") as f:
    json.dump({"clases_orden": CLASES_ORDEN, "umbral_m1": UMBRAL_M1, "umbral_m2": UMBRAL_M2,
               "modelo_m1": "modelo_vgg19_m1_detector.keras",
               "modelo_m2": "modelo_vgg19_m2_patogeno.keras",
               "img_size_m1": list(IMG_SIZE_M1), "img_size_m2": list(IMG_SIZE_M2), **prep}, f, indent=2)

metricas_reales = a_json({
    "accuracy": acc_final,
    "precision": prec_final,
    "recall": rec_final,
    "f1_score": f1_final,
    "f1_macro": f1_macro_final,
    "reporte_clases": classification_report(y_test, y_pred_full, target_names=class_names_full,
                                            output_dict=True, zero_division=0),
    "clases": class_names_full,
    "matriz_confusion": cm.tolist(),
    "umbral_m1": UMBRAL_M1,
    "umbral_m2": UMBRAL_M2,
    "sin_calibrar": {"accuracy": accuracy_score(y_test, y_pred_def),
                     "f1_macro": f1_score(y_test, y_pred_def, average="macro", zero_division=0)},
    "diagnostico": {"auc_m1_sano_vs_neumonia": auc_m1,
                    "auc_m2_viral_vs_bacteriana": auc_m2},
})

with open("metricas_vgg19_jerarquico.json", "w") as f:
    json.dump(metricas_reales, f, indent=2)
with open("metricas_entrenamiento.json", "w") as f:      # compatibilidad con el nombre anterior
    json.dump(metricas_reales, f, indent=2)

print("\nArchivos generados en Drive: modelo_vgg19_m1_detector.keras, modelo_vgg19_m2_patogeno.keras,")
print("config_vgg19_m1.json, config_vgg19_m2.json, config_jerarquico.json,")
print("metricas_vgg19_jerarquico.json, metricas_entrenamiento.json, probs_vgg19.npz")