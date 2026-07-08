# predWC — World Cup 2026 Knockout Predictor

Stacking ensemble (Random Forest + XGBoost + MLP → Logistic Regression) que predice 16avos, 8avos y 4tos del Mundial 2026. Soporta `--8avos` y `--4tos` para rondas posteriores. Versión con features NLP opcionales (news embeddings + YouTube sentiment).

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
uv run python stacking_model.py                         # 16avos (default)
uv run python stacking_model.py --8avos                 # 8avos
uv run python stacking_model.py --4tos                  # 4tos (cuartos de final)
uv run python stacking_model.py --max-date 2026-07-03   # entrenar hasta fecha específica
```

Construye 25 features por partido (ELO, forma reciente, h2h, peso del torneo), entrena stacking con split temporal 80/20 y evalúa en ~1640 partidos futuros (sin leakage). Predice scores con Dixon-Coles + Monte Carlo.

### Modelo con NLP

```bash
uv run python stacking_model.py --nlp
```

Agrega 24 features NLP (11 componentes PCA de news embeddings + YouTube sentiment por equipo). Requiere que exista `data/team_nlp_features.json` (generado con los scripts de NLP).

### Ver resultados gráficos

```bash
uv run python show_results.py            # resultados 16avos
uv run python show_results.py --8avos    # resultados 8avos
uv run python show_results.py --4tos     # resultados 4tos
uv run python show_results.py --nlp      # resultados NLP
uv run python show_results.py --save     # guarda PNG en vez de mostrar ventanas
```

Abre 4 ventanas: avance, probabilidades, confianza y scores Dixon-Coles. Con `--8avos` o `--4tos`, los títulos se adaptan (ej: "Avance a Semifinales").

### Tracker en vivo de resultados

```bash
uv run python bracket_tracker.py                  # tabla textual 16avos
uv run python bracket_tracker.py --8avos          # tracker 8avos
uv run python bracket_tracker.py --4tos           # tracker 4tos
uv run python bracket_tracker.py --nlp            # con predicciones NLP
uv run python bracket_tracker.py --graphs         # gráficos interactivos
uv run python bracket_tracker.py --graphs --save  # guarda PNG
```

Muestra cada partido con 3 dimensiones: **T.Regular** (ganador en 90'), **Marcador** (score exacto) y **Clasificación** (quién avanzó, incluyendo penales). Datos reales de bracketmundial2026.com.

Cuando results.csv de GitHub no tiene el partido aún, usa `actual_knockout_results.json` como fallback. Si el ganador en 90' es claro, la clasificación se auto-detecta sin necesidad de info de penales.

```
Germany vs Paraguay       T. Regular       Germany (44%)       Empate              [x]
                          Marcador         1-1 (11%)           1-1                 [+]
                          Clasificación    Germany (61%)       Paraguay (3-4 pen)  [x]
```

Los gráficos incluyen:
- **Pizarra de resultados** — cards con TR/SC/AV por partido
- **Perfiles de confianza** — barras L/E/V con resultado real destacado + línea 50%
- **Marcador: Pred vs Real** — comparación de goles
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

Esto ejecuta `fetch_elo.py` (244 equipos de eloratings.net) y `fetch_elo_history.py` (103k+ registros históricos hasta la fecha actual).

Para actualizar ELO y generar predicciones en un solo paso:
```bash
uv run scripts/update_data.py && uv run stacking_model.py --max-date 2026-07-03 --8avos
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
| `stacking_model.py` | Modelo stacking (25 features base + 24 NLP opcionales, Dixon-Coles, WC2026 boost ×3) |
| `bracket_tracker.py` | Tracker en vivo — tabla + gráficos con métricas duales (TR + Clasificación), soporta `--8avos` y `--4tos` |
| `evaluate_model.py` | Evaluación extendida (F1, MCC, Brier, calibración, per-model comparison) |
| `show_results.py` | Visualización interactiva con títulos dinámicos según ronda |
| `install_all.py` | Auto-instalador (uv + dependencias + Playwright) |
| `analisis_nlp.md` | Explicación del impacto de features NLP |
| `pyproject.toml` | Dependencias: scikit-learn, xgboost, polars, scipy, matplotlib, seaborn |
| `apis.txt` | API keys (gitignored, requiere setup manual) |
| `data/knockout_matches.json` | Bracket 16avos |
| `data/8avos_matches.json` | Bracket 8avos |
| `data/4tos_matches.json` | Bracket 4tos |
| `data/actual_knockout_results.json` | Resultados reales con info de penales |
| `scripts/` | Scrapers (ELO, noticias, YouTube) y utilidades |

## Predicciones 16avos — Octavos de Final

### Pronóstico vs Realidad

| Partido | L | E | V | TR | Marcador Pred | Marcador Real | SC | Avance Pred | Avance Real | AV |
|---|---|---|---|---|---|---|---|---|---|---|
| Germany vs Paraguay | **64%** | 25% | 10% | ✗ | 1-0 | **1-1** | ✗ | **Germany** (77%) | Paraguay (3-4 pen) | ✗ |
| France vs Sweden | **76%** | 18% | 5% | ✓ | 2-0 | **3-0** | ✗ | **France** (85%) | France | ✓ |
| South Africa vs Canada | 19% | 30% | **51%** | ✓ | 1-1 | **0-1** | ✗ | **Canada** (66%) | Canada | ✓ |
| Netherlands vs Morocco | **42%** | 35% | 23% | ✗ | 1-0 | **1-1** | ✗ | **Netherlands** (59%) | Morocco (2-3 pen) | ✗ |
| Portugal vs Croatia | **46%** | 34% | 20% | ✓ | 1-1 | **2-1** | ✗ | **Portugal** (63%) | Portugal | ✓ |
| Spain vs Austria | **76%** | 18% | 5% | ✓ | 2-0 | **3-0** | ✗ | **Spain** (86%) | Spain | ✓ |
| US vs Bosnia-H. | **45%** | 36% | 19% | ✓ | 1-0 | **2-0** | ✗ | **US** (63%) | United States | ✓ |
| Belgium vs Senegal | 28% | 35% | **37%** | ✗ | 1-1 | **3-2** | ✗ | **Senegal** (55%) | Belgium | ✗ |
| Brazil vs Japan | **71%** | 21% | 8% | ✓ | 1-0 | **2-1** | ✗ | **Brazil** (81%) | Brazil | ✓ |
| Ivory Coast vs Norway | 8% | 18% | **74%** | ✓ | 0-2 | **1-2** | ✗ | **Norway** (83%) | Norway | ✓ |
| Mexico vs Ecuador | 30% | 36% | **34%** | ✗ | 1-1 | **2-0** | ✗ | **Ecuador** (52%) | Mexico | ✗ |
| England vs Congo DR | **74%** | 20% | 6% | ✓ | 2-0 | **2-1** | ✗ | **England** (84%) | England | ✓ |
| Argentina vs Cape Verde | **76%** | 18% | 6% | ✓ | 2-0 | **3-2** | ✗ | **Argentina** (86%) | Argentina | ✓ |
| Australia vs Egypt | **42%** | 37% | 21% | ✗ | 1-1 | **1-1** | ✓ | **Australia** (61%) | — | — |
| Switzerland vs Algeria | **39%** | 35% | 26% | ✓ | 1-0 | **2-0** | ✗ | **Switzerland** (57%) | Switzerland | ✓ |
| Colombia vs Ghana | **82%** | 14% | 4% | ✓ | 2-0 | **1-0** | ✗ | **Colombia** (89%) | Colombia | ✓ |

**Resumen:** T.Regular **11/16 (69%)** | Clasificación **11/15 (73%)** | Marcador **1/16 (6%)**

## Predicciones 8avos — Cuartos de Final

### Pronóstico vs Realidad

| Partido | L | E | V | TR | Marcador Pred | Marcador Real | SC | Avance Pred | Avance Real | AV |
|---|---|---|---|---|---|---|---|---|---|---|
| Paraguay vs France | 7% | 25% | **68%** | ✓ | 0-2 | **0-1** | ✗ | **France** (81%) | France | ✓ |
| Canada vs Morocco | 10% | 37% | **53%** | ✓ | 1-1 | **0-3** | ✗ | **Morocco** (72%) | Morocco | ✓ |
| Portugal vs Spain | 11% | **43%** | 45% | ✓ | 1-1 | **0-1** | ✗ | **Spain** (67%) | Spain | ✓ |
| US vs Belgium | 13% | **43%** | 44% | ✓ | 1-1 | **1-4** | ✗ | **Belgium** (66%) | Belgium | ✓ |
| Brazil vs Norway | 30% | **49%** | 22% | ✗ | 1-1 | **1-2** | ✗ | **Brazil** (54%) | Norway | ✗ |
| Mexico vs England | 14% | **45%** | 41% | ✗ | 1-1 | **2-3** | ✗ | **England** (63%) | England | ✓ |
| Argentina vs Egypt | **64%** | 30% | 6% | ✓ | 2-0 | **3-2** | ✗ | **Argentina** (79%) | Argentina | ✓ |
| Switzerland vs Colombia | 10% | 42% | **47%** | ✗ | 1-1 | **0-0** | ✗ | **Colombia** (69%) | Switzerland (4-3 pen) | ✗ |

**Resumen:** T.Regular **5/8 (62%)** | Clasificación **6/8 (75%)** | Marcador **0/8 (0%)**

## Predicciones 4tos — Semifinales

### Pronóstico

| Partido | Local | Empate | Visitante | Avance Local | Avance Visit. | Score más probable |
|---------|-------|--------|-----------|-------------|--------------|-------------------|
| France vs Morocco | **50.0%** | 38.9% | 11.0% | **69.5%** | 30.5% | 2-0 (13.9%) |
| Spain vs Belgium | **61.0%** | 31.2% | 7.8% | **76.6%** | 23.4% | 2-0 (16.8%) |
| Norway vs England | 13.8% | **43.4%** | 42.8% | 35.5% | **64.5%** | 1-1 (13.3%) |
| Argentina vs Switzerland | **50.7%** | 39.0% | 10.3% | **70.2%** | 29.8% | 2-0 (14.3%) |

Partidos a jugarse el 9-11 de julio de 2026.

## Gráficas

Las gráficas se generan con:

```bash
# 16avos
uv run python show_results.py --save
uv run python bracket_tracker.py --graphs --save

# 8avos
uv run python show_results.py --8avos --save
uv run python bracket_tracker.py --8avos --graphs --save

# 4tos
uv run python show_results.py --4tos --save
uv run python bracket_tracker.py --4tos --graphs --save
```

Los PNGs se guardan en `data/` con prefijo según ronda (`8avos_*.png`, `4tos_*.png`). Gitignored — disponibles localmente al ejecutar los comandos.

## Tracker en vivo — métricas duales

`bracket_tracker.py` compara las predicciones contra los resultados reales en **2 dimensiones**:

| Métrica | Mide | Ejemplo |
|---------|------|---------|
| **T.Regular** | Ganador en 90' (Local/Empate/Visitante) | Alemania (44%) → Empate ✗ |
| **Clasificación** | Quién avanzó (incluye penales) | Germany → Paraguay ✗ |
| **Marcador** | Score exacto | 1-1 → 1-1 ✓ |

Los resultados reales se obtienen de bracketmundial2026.com (ver `data/actual_knockout_results.json`). En fase eliminatoria, un empate en 90' puede definirse por penales — la métrica de Clasificación captura ese desenlace mientras que T.Regular solo refleja el resultado en tiempo reglamentario.

## Notas técnicas

- `MAX_DATE` se calcula como `hoy - 1 día`; overridable con `--max-date YYYY-MM-DD`
- Accuracy temporal: **61.89%**, log-loss: **0.8185** (split temporal 80/20, ~1640 partidos)
- F1 macro: **0.5911**, MCC: **0.4147**, Top-2 accuracy: **~87%**, ECE: **~0.025**
- Modelo: RF (`class_weight='balanced'`) + XGBoost (`subsample=0.8`, `max_depth=6`, `reg_lambda=1.0`) + MLP (`(25,12)`, `alpha=0.003`) → LogisticRegression (`class_weight='balanced'`)
- **25 features base**: ELO, rolling stats (10 partidos), h2h, peso de torneo, local/neutral
- **WC2026 boost (×3)**: partidos del Mundial actual pesan el triple en el rolling window
- **Dixon-Coles**: corrección τ (ρ=-0.13) para scores 0-0, 1-0, 0-1, 1-1 reemplazando Poisson simple
- Distribución predicha de empates: ~27% (real: 23.2%), gracias a `class_weight='balanced'`
- `--8avos` y `--4tos`: disponibles en `stacking_model.py`, `show_results.py` y `bracket_tracker.py`
- Las features NLP existen para los 32 equipos pero tienen impacto limitado (ver `analisis_nlp.md`)
- Creado con `uv init --python 3.12`
- En Manjaro, si Playwright falla: `sudo pacman -S atk at-spi2-atk cups libdrm libxkbcommon libxcomposite libxdamage libxrandr mesa nss pango cairo gtk3`
