# ##############################################################
# CELDA B — ENTRENAR M1 (Sano vs Neumonía) VGG19
# ##############################################################
REENTRENAR_M1 = False    # True = entrenar de nuevo aunque ya exista el archivo en Drive

# ============================================================
# 5. MODELO 1 (M1): SANO vs NEUMONÍA   (softmax 2: [sano, neumonia])
# ============================================================
if os.path.exists(RUTA_M1) and not REENTRENAR_M1:
    print(f"{RUTA_M1} ya existe en Drive: se carga sin entrenar. (REENTRENAR_M1 = True para forzar)")
    model_m1 = tf.keras.models.load_model(RUTA_M1, compile=False)
else:
    print("\n" + "=" * 70)
    print(">>> FASE 1: ENTRENANDO M1 (Sano vs Neumonía)")
    print("=" * 70)

    model_m1, hist_m1, corte_m1 = entrenar_modelo(
        nombre="VGG19_M1_Detector", tipo="softmax2", size=IMG_SIZE_M1,
        df_tr=df_train, df_va=df_val, y_col="nivel1", clases=["sano", "neumonia"],
        class_mode="categorical", bloques_ft=FT_M1, dropout=0.3,
        monitor="val_loss", modo="min")

    graficar_historial(hist_m1, corte_m1, "M1")
    respaldar(RUTA_M1)
    model_m1.save(RUTA_M1)
    print("✓ Modelo M1 guardado en Drive.")