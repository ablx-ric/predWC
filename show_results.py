import matplotlib
import sys

if "--save" in sys.argv:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
import os

# Verify interactive display is available (skip in --save mode)
if "--save" not in sys.argv:
    if os.environ.get("DISPLAY") is None and sys.platform != "darwin":
        print("  ERROR: No display available (DISPLAY not set).")
        print("  Tienes una terminal gráfica? Asegúrate de tener un servidor X corriendo.")
        print("  Sugerencias:")
        print("    - En Linux local: verifica que estás en una sesión gráfica")
        print("    - Por SSH: usa ssh -X o ssh -Y")
        print("    - En WSL: instala y ejecuta un X server (VcXsrv, Xming)")
        print()
        print("  Para guardar PNG sin display: python show_results.py --save")
        sys.exit(1)

# ── Professional theme ──────────────────────────────────────────────
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        10,
    "axes.titlesize":   14,
    "axes.labelsize":   11,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "grid.linestyle":   "--",
    "figure.dpi":       150,
    "savefig.dpi":      150,
    "savefig.bbox":     "tight",
    "legend.frameon":   True,
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.fontsize":  9,
})

# ── Color palette ───────────────────────────────────────────────────
LOCAL_COLOR  = "#1a5092"
AWAY_COLOR   = "#b22222"
DRAW_COLOR   = "#7f8c8d"
WIN_COLORS   = [LOCAL_COLOR, DRAW_COLOR, AWAY_COLOR]

GREEN  = "#27ae60"
AMBER  = "#f39c12"
RED    = "#c0392b"
GRAY   = "#95a5a6"

PREDICTIONS           = "data/knockout_predictions.csv"
PREDICTIONS_NLP       = "data/knockout_predictions_nlp.csv"
PREDICTIONS_8AVOS     = "data/8avos_predictions.csv"
PREDICTIONS_8AVOS_NLP = "data/8avos_predictions_nlp.csv"
PREDICTIONS_4TOS      = "data/4tos_predictions.csv"
PREDICTIONS_4TOS_NLP  = "data/4tos_predictions_nlp.csv"

RONDA = "16avos"
RONDA_LABELS = {"16avos": "Octavos", "8avos": "Cuartos", "4tos": "Semifinales"}


def _set_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass


def _to_pct(x, _):
    return f"{x:.0f}%"


def _shade(v):
    if v >= 65:
        return GREEN
    if v >= 50:
        return AMBER
    return RED


# ── 1. Barra de avance (local vs visitante) ────────────────────────
def plot_advancement(df):
    if df.is_empty():
        print("  No data to plot")
        return None

    fig, ax = plt.subplots(figsize=(10, 6.5))
    _set_title(fig, "Avance" if RONDA == "16avos" else "Avance 8avos")

    matches = [m.replace(" vs ", "\nvs\n") for m in df["match"]]
    y = np.arange(len(matches))
    w = 0.32

    ax.barh(y + w / 2, df["local_advance_pct"], w,
            label="Local", color=LOCAL_COLOR, zorder=3)
    ax.barh(y - w / 2, df["away_advance_pct"], w,
            label="Visitante", color=AWAY_COLOR, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(matches, fontsize=8.5)
    ax.set_xlabel("Probabilidad de avance")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_to_pct))
    titulo = f"Avance a {RONDA_LABELS.get(RONDA, '')} de Final — Mundial 2026"
    ax.set_title(titulo, fontsize=14, fontweight="bold", pad=14)
    ax.legend(loc="lower right", frameon=True)
    ax.set_xlim(0, 105)

    for i, (l, a) in enumerate(zip(
            df["local_advance_pct"], df["away_advance_pct"])):
        ax.text(l + 1.5, i + w / 2, f"{l:.0f}%",
                va="center", fontsize=7.5, color=LOCAL_COLOR, fontweight="bold")
        ax.text(a + 1.5, i - w / 2, f"{a:.0f}%",
                va="center", fontsize=7.5, color=AWAY_COLOR, fontweight="bold")

    fig.tight_layout()
    return fig


# ── 2. Probabilidades L/E/V por partido ─────────────────────────────
def plot_match_probabilities(df):
    if df.is_empty():
        print("  No data to plot")
        return None

    n = len(df)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.8 * rows))
    _set_title(fig, "Probabilidades por partido")

    for ax, row in zip(axes.flatten(), df.iter_rows(named=True)):
        vals = [row["local_win_pct"], row["draw_pct"], row["away_win_pct"]]
        bars = ax.bar(["Local", "Empate", "Visitante"], vals,
                      color=WIN_COLORS, width=0.55,
                      edgecolor="white", linewidth=0.6, zorder=3)
        ax.set_ylim(0, 105)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(_to_pct))
        ax.set_title(row["match"], fontsize=9, fontweight="bold", pad=6)
        for b, v in zip(bars, vals):
            if v >= 5:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 2,
                        f"{v:.0f}%", ha="center", fontsize=7.5, fontweight="bold")

    for j in range(n, len(axes.flatten())):
        fig.delaxes(axes.flatten()[j])

    fig.suptitle(f"Probabilidades por partido — {RONDA_LABELS.get(RONDA, RONDA)} Mundial 2026",
                 fontsize=15, fontweight="bold", y=1.015)
    fig.tight_layout()
    return fig


# ── 3. Medidor de confianza ─────────────────────────────────────────
def plot_confidence_gauge(df):
    df = df.with_columns(
        pl.max_horizontal("local_win_pct", "away_win_pct").alias("max_prob")
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    _set_title(fig, "Confianza del modelo")

    matches_short = [m.replace(" vs ", "\n") for m in df["match"]]
    max_p = df["max_prob"].to_numpy()
    colors = [_shade(v) for v in max_p]

    ax.bar(range(len(df)), max_p, color=colors, width=0.55,
           edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(matches_short, fontsize=7.5)
    ax.set_ylabel("Confianza (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_to_pct))
    ax.set_title("Confianza del modelo por partido",
                 fontsize=14, fontweight="bold", pad=10)
    ax.axhline(y=50, color=RED, linestyle="--", linewidth=0.8, alpha=0.35, zorder=2)
    ax.set_ylim(0, 105)

    for bar, v in zip(ax.containers[0], max_p):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                f"{v:.0f}%", ha="center", fontsize=8, fontweight="bold")

    fig.tight_layout()
    return fig


# ── 4. Panel Dixon-Coles (goles esperados + score más probable) ─────
def plot_poisson_panel(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    _set_title(fig, "Dixon-Coles")

    # ── Izquierda: goles esperados ──
    x = np.arange(len(df))
    w = 0.32
    ax1.bar(x - w / 2, df["expected_goals_local"], w,
            label="Local", color=LOCAL_COLOR, zorder=3)
    ax1.bar(x + w / 2, df["expected_goals_away"], w,
            label="Visitante", color=AWAY_COLOR, zorder=3)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(" vs ", "\n") for m in df["match"]],
                        fontsize=6.5)
    ax1.set_ylabel("Goles esperados (λ)")
    ax1.set_title("Goles esperados por equipo (Dixon-Coles)",
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(frameon=True)
    ax1.tick_params(axis="x", labelsize=6.5)

    for i, (l, a) in enumerate(zip(
            df["expected_goals_local"], df["expected_goals_away"])):
        ax1.text(i - w / 2, l + 0.04, f"{l:.2f}",
                 ha="center", fontsize=7, color=LOCAL_COLOR, fontweight="bold")
        ax1.text(i + w / 2, a + 0.04, f"{a:.2f}",
                 ha="center", fontsize=7, color=AWAY_COLOR, fontweight="bold")

    # ── Derecha: score más probable ──
    matches_short = [
        m[:14] + "\n" + m.split(" vs ")[1][:14] if " vs " in m else m
        for m in df["match"]
    ]
    score_labels = [
        f"{s} ({p:.0f}%)"
        for s, p in zip(df["most_likely_score"], df["most_likely_score_pct"])
    ]
    colors_bar = [
        LOCAL_COLOR if int(s.split("-")[0]) > int(s.split("-")[1])
        else AWAY_COLOR if int(s.split("-")[0]) < int(s.split("-")[1])
        else DRAW_COLOR
        for s in df["most_likely_score"]
    ]

    ax2.barh(range(len(df)), df["most_likely_score_pct"],
             color=colors_bar, zorder=3)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(matches_short, fontsize=7.5)
    ax2.set_xlabel("Probabilidad")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(_to_pct))
    ax2.set_title("Score más probable (Dixon-Coles + Monte Carlo)",
                  fontsize=13, fontweight="bold", pad=10)

    for i, (label, pct) in enumerate(zip(
            score_labels, df["most_likely_score_pct"])):
        ax2.text(pct + 0.6, i, label, va="center", fontsize=7.5)
    ax2.set_xlim(0, df["most_likely_score_pct"].max() + 12)

    fig.tight_layout()
    return fig


# ── Main ────────────────────────────────────────────────────────────
def main():
    save_mode = "--save" in sys.argv
    use_nlp = "--nlp" in sys.argv
    es_8avos = "--8avos" in sys.argv
    es_4tos = "--4tos" in sys.argv

    global RONDA
    ronda = "4tos" if es_4tos else ("8avos" if es_8avos else "16avos")
    RONDA = ronda

    csv_path = (
        PREDICTIONS_4TOS_NLP if es_4tos else
        PREDICTIONS_8AVOS_NLP if es_8avos else PREDICTIONS_NLP
    ) if use_nlp else (
        PREDICTIONS_4TOS if es_4tos else
        PREDICTIONS_8AVOS if es_8avos else PREDICTIONS
    )

    print("=" * 50)
    print(f"  SHOW RESULTS — World Cup 2026 {ronda}")
    if use_nlp:
        print("  (NLP-enhanced predictions)")
    print("=" * 50)
    print(f"\nReading {csv_path}...")

    try:
        df = pl.read_csv(csv_path)
    except FileNotFoundError:
        flag = " --nlp" if use_nlp else ""
        print(f"  ERROR: {csv_path} not found. Run stacking_model.py{flag} first.")
        return
    print(f"  {df.height} matches loaded\n")

    plots = [
        ("advancement.png", plot_advancement(df)),
        ("probabilities.png", plot_match_probabilities(df)),
        ("confidence.png", plot_confidence_gauge(df)),
        ("poisson_scores.png", plot_poisson_panel(df)),
    ]

    if save_mode:
        prefix = f"{ronda}_" if (es_8avos or es_4tos) else ""
        print("Saving charts to data/ ...")
        for name, fig in plots:
            if fig is None:
                continue
            fig.savefig(f"data/{prefix}{name}", dpi=150,
                        facecolor=fig.get_facecolor())
            print(f"  Saved data/{prefix}{name}")
            plt.close(fig)
        print("\nDone.")
    else:
        print("Opening interactive windows...")
        print("\nClose the plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
