# Análisis: ¿Por qué las features NLP tienen poco impacto?

## Conclusión rápida

Las predicciones varían **mínimamente** al activar `--nlp`. Los cambios son de ~1pp en partidos donde uno de los dos equipos no tenía NLP previamente. En general, el aporte es marginal.

## Por qué

### 1. No hay datos NLP históricos

El meta-modelo (Logistic Regression) se entrena con split temporal: 80% train → 20% validation (~1634 partidos). Las features NLP (news embeddings + YouTube sentiment) existen para los **32 equipos actuales del Mundial**, pero no para los partidos históricos.

Durante el entrenamiento de validación, las features NLP se setean a **0.0** para todos los partidos:

```python
meta_val_nlp = np.zeros((len(y_val), len(NLP_FEATURE_COLS)))
```

El meta-modelo aprende pesos para las 24 dimensiones NLP a partir de **puros ceros**, sin correlación con el resultado. Los coeficientes son esencialmente ruido.

### 2. Features NLP ≈ ruido en predicción

Para los 16 partidos actuales, las features NLP tienen valores reales (PCs + YouTube sentiment). Pero como el meta-modelo fue entrenado con NLP=0, estos valores se multiplican por coeficientes sin señal → el aporte es despreciable.

Los modelos base (RF, XGBoost, SVM) solo usan las 25 features deportivas. El meta-modelo recibe 9 probabilidades + 24 NLP. Las 9 probabilidades dominan porque provienen de modelos entrenados con patrones reales.

### 3. Lo que haría falta

Para que NLP tenga impacto real:
1. **Datos NLP históricos**: news y YouTube comments para cada partido del dataset (~8000 partidos desde 2018)
2. **Embeddings por partido**, no por equipo: la cobertura de Brasil vs Argentina es distinta a Brasil vs Japón
3. **Pipeline re-entrenado**: el meta-modelo necesita ver correlaciones reales entre NLP y resultado

Sin eso, las features NLP son mayormente decorativas.

### 4. Validación del modelo (RF + XGBoost + MLP → LogisticRegression)

| Métrica | Valor |
|---------|-------|
| Accuracy temporal | 61.44% |
| Log-loss | 0.8235 |
| F1 macro | 0.5847 |
| MCC | 0.4059 |
| Top-2 accuracy | 86.94% |
| ECE | 0.0254 |

Sobre ~1639 partidos futuros (sin leakage). Modelo usa `class_weight='balanced'` y score prediction con Dixon-Coles (corrección τ).

## Conclusión

Las features NLP existen, se computan, se cargan, y el meta-modelo les asigna pesos — pero al no haber datos históricos con NLP, esos pesos se aprendieron de ceros y no afectan la salida. Es un requisito académico (variable NLP) implementado correctamente, pero que requiere recolección de datos históricos para ser efectivo.
