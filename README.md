# predWC — World Cup 2026 Knockout Predictor

Stacking ensemble (Random Forest + XGBoost + MLP → Logistic Regression) que predice los 16avos de final del Mundial 2026. Versión con features NLP opcionales (news embeddings + YouTube sentiment).

## Requisitos

- **Python ≥ 3.12** (verificar con `python3 --version`)
- **curl** (para instalar uv)

Probado en **Manjaro** y **Ubuntu**.

## Instalación

### Opción recomendada: auto-instalador

```bash
python install_all.py
```

Esto hace todo automáticamente:
1. Instala `uv` si no lo tienes
2. Crea entorno virtual e instala todas las dependencias con `uv sync`
3. Instala Playwright + Chromium (necesario para actualizar datos)

### Opción manual

```bash
# 1. Instalar uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Cierra y abre la terminal, o: source ~/.bashrc

# 2. Verificar Python 3.12
python3 --version

# 3. Instalar dependencias
uv sync

# 4. Instalar Chromium para Playwright
uv run playwright install --with-deps chromium
```

## Uso

### Modelo base (sin NLP)

```bash
uv run python stacking_model.py
```

Construye 25 features por partido (ELO, forma reciente, h2h, peso del torneo), entrena stacking con split temporal 80/20, evalúa en ~1634 partidos futuros (sin leakage) y predice los 16 cruces con scores Poisson + Monte Carlo.

### Modelo con NLP

```bash
uv run python stacking_model.py --nlp
```

Agrega 24 features NLP (11 componentes PCA de news embeddings + YouTube sentiment por equipo). Requiere que exista `data/team_nlp_features.json` (generado con los scripts de NLP).

### Ver resultados gráficos

```bash
uv run python show_results.py            # resultados base
uv run python show_results.py --nlp      # resultados NLP
uv run python show_results.py --save     # guarda PNG en vez de mostrar ventanas
```

Abre 4 ventanas: avance, probabilidades, confianza y scores Poisson.

### Tracker en vivo de resultados

```bash
uv run python bracket_tracker.py                  # tabla textual
uv run python bracket_tracker.py --nlp            # con predicciones NLP
uv run python bracket_tracker.py --graphs         # gráficos interactivos
uv run python bracket_tracker.py --graphs --save  # guarda PNG
```

Muestra cada partido con 3 dimensiones: **T.Regular** (ganador en 90'), **Marcador** (score exacto) y **Clasificación** (quién avanzó, incluyendo penales). Datos reales obtenidos de las fuentes oficiales vía bracketmundial2026.com.

```
Germany vs Paraguay       T. Regular       Germany (44%)       Empate              [x]
                          Marcador         1-1 (11%)           1-1                 [+]
                          Clasificación    Germany (61%)       Paraguay (3-4 pen)  [x]
```

Los gráficos incluyen:
- **Pizarra de resultados** — 16 cards con TR/SC/AV por partido
- **Perfiles de confianza** — barras L/E/V con resultado real destacado
- **Marcador: Pred vs Real** — comparación de goles (grid 2×2)
- **Panel de estadísticas** — aciertos de T.Regular y Clasificación

### Evaluación extendida del modelo

```bash
uv run python evaluate_model.py
```

Calcula F1 macro/weighted, MCC, Brier score, matriz de confusión, calibración por bins y comparación por modelo base (RF / XGB / MLP / Stacking).

## Actualizar datos

### ELO rankings + historial

```bash
uv run python scripts/update_data.py
```

### Pipeline NLP completo (noticias + YouTube + embeddings)

```bash
# 1. Scrapea noticias (Wikipedia + BBC/ESPN → trafilatura)
uv run python scripts/fetch_team_news.py

# 2. Comentarios YouTube (necesita API key en apis.txt)
uv run python scripts/fetch_team_comments.py

# 3. Recalcula embeddings → PCA → sentiment
uv run python scripts/compute_team_embeddings.py

# 4. Re-entrenar con NLP
uv run python stacking_model.py --nlp
```

La API key de YouTube va en `apis.txt` con este formato:
```
YOUTUBE:
 - AIzaSy...
```

## Archivos

| Archivo | Descripción |
|---------|-------------|
| `stacking_model.py` | Modelo stacking (25 features base; `--nlp` agrega 24 NLP) |
| `bracket_tracker.py` | Tracker en vivo — tabla + gráficos con métricas duales (TR + Clasificación) |
| `evaluate_model.py` | Evaluación extendida (F1, MCC, Brier, calibración) |
| `show_results.py` | Visualización interactiva |
| `install_all.py` | Auto-instalador |
| `analisis_nlp.md` | Explicación del impacto (limitado) de features NLP |
| `pyproject.toml` | Dependencias del proyecto |
| `apis.txt` | API keys (tracked solo por ser repo privado) |
| `data/` | ELO, partidos, predicciones, raw news/comments |
| `data/actual_knockout_results.json` | Resultados reales de 16avos con info de penales |
| `scripts/` | Scrapers y utilidades |

## Salida del modelo

Ambos modos muestran:
- Accuracy y log-loss en validación temporal
- Probabilidad local / empate / visitante para cada partido
- λ Poisson y top 5 scores exactos más probables
- Probabilidad de avance (empate repartido 50/50)
- Features NLP por equipo (solo con `--nlp`)

Guardan:
- `data/knockout_predictions.csv` — predicciones base
- `data/knockout_predictions_nlp.csv` — predicciones NLP

## Tracker en vivo — métricas duales

`bracket_tracker.py` compara las predicciones contra los resultados reales en **2 dimensiones**:

| Métrica | Mide | Ejemplo |
|---------|------|---------|
| **T.Regular** | Ganador en 90' (Local/Empate/Visitante) | Alemania (44%) → Empate ✗ |
| **Clasificación** | Quién avanzó (incluye penales) | Germany → Paraguay ✗ |
| **Marcador** | Score exacto | 1-1 → 1-1 ✓ |

Los resultados reales se obtienen de bracketmundial2026.com (ver `data/actual_knockout_results.json`). En fase eliminatoria, un empate en 90' puede definirse por penales — la métrica de Clasificación captura ese desenlace mientras que T.Regular solo refleja el resultado en tiempo reglamentario.

## Notas técnicas

- `MAX_DATE` se calcula como `hoy - 1 día` — los datos se actualizan solos al ejecutar
- Accuracy temporal: **61.44%**, log-loss: **0.8235** (Temporal split, ~1639 partidos futuros)
- F1 macro: **0.5847**, MCC: **0.4059**, Top-2 accuracy: **86.94%**, ECE: **0.0254**
- Modelo: RF + XGBoost + MLP (red neuronal) → LogisticRegression con `class_weight='balanced'`
- Predicción de goles con **Dixon-Coles** (corrección τ para undervaluación de empates bajos, ρ=-0.13) en vez de Poisson simple
- Distribución predicha: empates 27.2% (real 23.2%)
- Live tracking 16avos: T.Regular **6/10 (60%)**, Clasificación **6/10 (60%)**
- La gráfica de confianza en `show_results.py` incluye una línea horizontal en 50% como umbral
- `bracket_tracker.py` usa `actual_knockout_results.json` como fallback cuando results.csv de GitHub no tiene el partido (ej: partidos de julio que no están subidos al dataset público)
- Creado con `uv init --python 3.12`
- En Manjaro, si Playwright falla: `sudo pacman -S atk at-spi2-atk cups libdrm libxkbcommon libxcomposite libxdamage libxrandr mesa nss pango cairo gtk3`
- Las features NLP existen para los 32 equipos pero tienen impacto limitado porque no hay datos NLP históricos para entrenar el meta-modelo (ver `analisis_nlp.md`)
- `apis.txt` se incluye en el repo porque es privado — si se hace público, descomentar en `.gitignore`
