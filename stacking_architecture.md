# Stacking Architecture — predWC

## Vista general

```
                    ┌─────────────────────┐
                    │  25 features base   │
                    │  (escaladas con      │
                    │   StandardScaler)    │
                    └──────┬──────┬───────┘
                           │      │
               ┌───────────┘      └───────────┐
               ▼                               ▼
       ┌───────────────┐             ┌───────────────┐
       │ RandomForest  │             │    XGBoost    │
       │ 300 árboles   │             │ 300 árboles   │
       │ max_depth=12  │             │ max_depth=6   │
       │ class_weight= │             │ η=0.03        │
       │ 'balanced'    │             │ subsample=0.8 │
       └───────┬───────┘             └───────┬───────┘
               ▼                             ▼
       ┌───────────────┐             ┌───────────────┐
       │  p(L) p(E) p(V)│             │  p(L) p(E) p(V)│
       └───────┬───────┘             └───────┬───────┘
               │                             │
               │         ┌───────────────┐   │
               │         │     MLP       │   │
               │         │ (25 → 12)     │   │
               │         │ ReLU          │   │
               │         │ α=0.003       │   │
               │         └───────┬───────┘   │
               │                 │           │
               └─────────┬───────┴───────────┘
                         ▼
               ┌─────────────────────┐
               │  9 probabilidades   │
               │ (3 modelos × 3 cls) │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │  LogisticRegression │  ← meta-model (Level 1)
               │  C=0.1              │
               │  class_weight=      │
               │  'balanced'         │
               │  solver='lbfgs'     │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │   Predicción final  │
               │  Local / Empate /   │
               │  Visitante          │
               └─────────────────────┘
```

## Nivel 0 — Modelos base

Los 3 modelos base reciben **exactamente los mismos 25 features escalados** (con `StandardScaler` ajustado en el split de entrenamiento).

### RandomForestClassifier

```
Parámetros:
  n_estimators      = 300
  max_depth         = 12
  class_weight      = "balanced"
  random_state      = 42
  n_jobs            = -1

Entrada:  25 features escalados (X_tr_scaled)
Salida:   p(L), p(E), p(V)  — 3 probabilidades
```

- Árboles de decisión con bagging.
- `class_weight='balanced'` asigna pesos inversamente proporcionales a la frecuencia de cada clase (sube la importancia de los empates).
- Captura interacciones no lineales entre features.

### XGBoost

```
Parámetros:
  n_estimators      = 300
  max_depth         = 6
  learning_rate     = 0.03
  subsample         = 0.8
  colsample_bytree  = 0.8
  reg_lambda        = 1.0
  reg_alpha         = 0.1
  eval_metric       = "mlogloss"
  random_state      = 42

Entrada:  25 features escalados (X_tr_scaled)
Sample weight:  compute_class_weight("balanced", ...)  →  vector de pesos por clase

Salida:   p(L), p(E), p(V)
```

- Gradient boosting con regularización L1+L2.
- `subsample=0.8` y `colsample_bytree=0.8` reducen overfitting.
- `sample_weight` se pasa aparte (XGBoost no tiene `class_weight` nativo).
- Pesos calculados con `sklearn.utils.class_weight.compute_class_weight`.

### MLP (Multi-Layer Perceptron)

```
Parámetros:
  hidden_layer_sizes     = (25, 12)
  activation             = "relu"
  solver                 = "adam"
  max_iter               = 5000
  early_stopping         = True
  validation_fraction    = 0.15
  alpha                  = 0.003
  random_state           = 42

Arquitectura:
  Input(25) → Dense(25, ReLU) → Dense(12, ReLU) → Output(3, Softmax)

Entrada:  25 features escalados (X_tr_scaled)
Salida:   p(L), p(E), p(V)
```

- Red neuronal de 2 capas ocultas.
- `early_stopping` reserva 15% del train para validación interna.
- `alpha=0.003` es regularización L2 (weight decay).

## Nivel 1 — Meta-modelo (LogisticRegression)

### Construcción de la matriz de meta-features

Para cada partido `i` del conjunto de validación:

```
meta_val_base[i] = [  rf_pL,  rf_pE,  rf_pV,
                     xgb_pL, xgb_pE, xgb_pV,
                     mlp_pL, mlp_pE, mlp_pV  ]
```

**Sin NLP:** 9 columnas (3 modelos × 3 clases).

```
Dimensión:  (n_validación, 9)
```

**Con NLP:** se concatenan 24 features adicionales:

```
meta_val = [ meta_val_base | meta_val_nlp ]
            ──── 9 cols ──   ── 24 cols ──

meta_val_nlp[i] = [  home_pc1, ..., home_pc11, home_yt_sent,
                     away_pc1, ..., away_pc11, away_yt_sent  ]
```

```
Dimensión:  (n_validación, 33)
```

### LogisticRegression meta

```
Parámetros:
  solver          = "lbfgs"
  max_iter        = 1000
  C               = 0.1
  class_weight    = "balanced"
  random_state    = 42

Entrada:  9 (o 33 con NLP) meta-features
Salida:   p(L), p(E), p(V)  →  argmax → predicción final
```

- `C=0.1` es regularización L2 fuerte (inverso de la fuerza de regularización).
- `class_weight='balanced'` para compensar desbalance en validación.

### Flujo de entrenamiento

```python
# 1. Split temporal 80/20
X_tr, X_val = X[train_idx], X[val_idx]
y_tr, y_val = y[train_idx], y[val_idx]

# 2. Escalar features
scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

# 3. Entrenar modelos base
rf.fit(X_tr_scaled, y_tr)
xgb.fit(X_tr_scaled, y_tr, sample_weight=sw_tr)
mlp.fit(X_tr_scaled, y_tr)

# 4. Generar meta-features desde validación
for name, model in [rf, xgb, mlp]:
    meta_val[:, offset:offset+3] = model.predict_proba(X_val_scaled)
    offset += 3

# 5. Entrenar meta-modelo
meta.fit(meta_val, y_val)

# 6. Retrain final: modelos base en todos los datos
rf.fit(X_full_scaled, y)
xgb.fit(X_full_scaled, y, sample_weight=sw_full)
mlp.fit(X_full_scaled, y)

# 7. Predecir: generar meta-features → meta.predict_proba
```

## Integración de NLP

### Estado actual

Las 24 features NLP existen para los **32 equipos del Mundial 2026** pero **no están disponibles para los partidos históricos** (2018-2024). Esto significa:

1. En entrenamiento, todas las features NLP son **0.0**.
2. El meta-modelo aprende pesos únicamente de las 9 probabilidades base.
3. En predicción, los 32 equipos sí tienen valores NLP reales (PCA + sentiment).

### Impacto medido

```
Sin NLP:   accuracy 61.89%,  F1 macro 0.5911
Con NLP:   accuracy ~61%,    F1 macro ~0.58   (diferencia dentro del ruido)
```

El NLP añade ~1pp de cambio pero no mejora consistente porque:

- El meta-modelo nunca ve features NLP durante training (son ceros).
- En inferencia, los valores NLP son distintos de cero pero el meta-modelo no aprendió a usarlos.

### Arquitectura ideal (pendiente de implementar)

```
                    ┌──────────────────────┐
                    │     25 features      │
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │    StandardScaler    │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
       ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
       │     RF       │ │    XGB       │ │    MLP       │
       └──────┬───────┘ └──────┬───────┘ └──────┬───────┘
              ▼                ▼                ▼
       ┌─────────────────────────────────────────────────┐
       │          9 probabilidades base                  │
       └──────────────────────┬──────────────────────────┘
                              │
       ┌──────────────────────▼──────────────────────────┐
       │              24 features NLP                   │
       │  (home_pc1..11 + home_yt + away_pc1..11 + yt)  │
       └──────────────────────┬──────────────────────────┘
                              │
       ┌──────────────────────▼──────────────────────────┐
       │         meta_input = [probas | nlp]             │
       │                (33 columnas)                    │
       │                                               │
       │    LogisticRegression(C=0.1, balanced)         │
       └──────────────────────┬──────────────────────────┘
                              ▼
                     Predicción final
```

Para que el NLP funcione correctamente se necesita:

```python
# 1. Para CADA partido histórico, generar features NLP en esa fecha
#    (no solo para los 32 equipos actuales)

# 2. Incluir esas features como entrada directa a los modelos base,
#    no solo al meta-modelo:
BASE_FEATURE_COLS + NLP_FEATURE_COLS  →  RF / XGB / MLP

# 3. Opcional: standard scaler sobre las NLP features también
```

### Resumen de datos que recibe cada componente

| Componente | Datos de entrada | Dimensión |
|---|---|---|
| RF | 25 features escaladas | `(n, 25)` |
| XGB | 25 features escaladas + sample_weight | `(n, 25)` + `(n,)` |
| MLP | 25 features escaladas | `(n, 25)` |
| Meta-modelo (sin NLP) | 9 probabilidades (3 modelos × 3 clases) | `(n, 9)` |
| Meta-modelo (con NLP) | 9 probabilidades + 24 NLP features | `(n, 33)` |
