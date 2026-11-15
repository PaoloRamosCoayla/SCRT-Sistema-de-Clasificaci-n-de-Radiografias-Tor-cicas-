# ##############################################################
# CELDA C — ENTRENAR M2 (Viral vs Bacteriana)
# ##############################################################
REENTRENAR_M2 = False    # True = entrenar de nuevo (se hace copia de respaldo del M2 anterior)

# ============================================================
# 6. MODELO 2 (M2): VIRAL vs BACTERIANA   (sigmoid: P(bacteriana))
#    Mapeo: 0 = viral, 1 = bacteriana. Mejor checkpoint por AUC.
# ============================================================
if os.path.exists(RUTA_M2) and not REENTRENAR_M2:
    print(f"{RUTA_M2} ya existe en Drive: se carga sin entrenar. (REENTRENAR_M2 = True para forzar)")
    model_m2 = tf.keras.models.load_model(RUTA_M2, compile=False)
else:
    print("\n" + "=" * 70)
    print(">>> FASE 2: ENTRENANDO M2 (Viral vs Bacteriana)")
    print("=" * 70)

    model_m2, hist_m2, corte_m2 = entrenar_modelo(
        nombre="ResNet50_M2_Patogeno", tipo="sigmoid1", size=IMG_SIZE_M2,
        df_tr=df_train_m2, df_va=df_val_m2, y_col="clase", clases=["viral", "bacteriana"],
        class_mode="binary", ft_prefijos=FT_M2, dropout=0.4,
        monitor="val_auc", modo="max")

    graficar_historial(hist_m2, corte_m2, "M2")
    respaldar(RUTA_M2)
    model_m2.save(RUTA_M2)
    print("✓ Modelo M2 guardado en Drive.")