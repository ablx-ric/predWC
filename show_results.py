import matplotlib
import sys

# Force non-interactive backend if --save flag is used
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

sns.set_style("whitegrid")
plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

PREDICTIONS = "data/knockout_predictions.csv"
PREDICTIONS_NLP = "data/knockout_predictions_nlp.csv"


LOCAL_COLOR = "#3498db"
AWAY_COLOR = "#e67e22"


def plot_advancement(df):
    if df.is_empty():
        print("  No data to plot")
        return None
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title("Avance a 16avos")
    matches = [m.replace(" vs ", "\nvs\n") for m in df["match"]]
    y = np.arange(len(matches))
    width = 0.35
    ax.barh(y + width / 2, df["local_advance_pct"], width, label="Local", color=LOCAL_COLOR)
    ax.barh(y - width / 2, df["away_advance_pct"], width, label="Visitante", color=AWAY_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels(matches, fontsize=8)
    ax.set_xlabel("Probabilidad de avance (%)")
    ax.set_title("Avance a Octavos de Final — Mundial 2026")
    ax.legend()
    ax.set_xlim(0, 105)
    for i, (l, a) in enumerate(zip(df["local_advance_pct"], df["away_advance_pct"])):
        ax.text(l + 1, i + width / 2, f"{l:.0f}%", va="center", fontsize=7, color=LOCAL_COLOR)
        ax.text(a + 1, i - width / 2, f"{a:.0f}%", va="center", fontsize=7, color=AWAY_COLOR)
    fig.tight_layout()
    return fig


def plot_match_probabilities(df):
    if df.is_empty():
        print("  No data to plot")
        return None
    n = len(df)
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(14, 3 * rows))
    fig.canvas.manager.set_window_title("Probabilidades por partido")
    axes = axes.flatten()
    colors = [LOCAL_COLOR, "#f39c12", AWAY_COLOR]
    for i, row in enumerate(df.iter_rows(named=True)):
        ax = axes[i]
        labels = ["Local", "Empate", "Visitante"]
        vals = [row["local_win_pct"], row["draw_pct"], row["away_win_pct"]]
        bars = ax.bar(labels, vals, color=colors, width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v:.0f}%", ha="center", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_ylabel("%")
        ax.set_title(row["match"], fontsize=8)
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle("Probabilidades por partido — 16avos Mundial 2026", fontsize=14, y=1.02)
    fig.tight_layout()
    return fig


def plot_confidence_gauge(df):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.canvas.manager.set_window_title("Confianza del modelo")
    df = df.with_columns(
        (pl.max_horizontal("local_win_pct", "away_win_pct")).alias("max_prob")
    )
    matches_short = [m.replace(" vs ", "\n") for m in df["match"]]
    colors = ["#2ecc71" if l > a else "#e74c3c" for l, a in zip(df["local_advance_pct"], df["away_advance_pct"])]
    bars = ax.bar(range(len(df)), df["max_prob"], color=colors, width=0.6)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(matches_short, fontsize=7)
    ax.set_ylabel("Confianza (%)")
    ax.set_title("Confianza del modelo por partido")
    ax.axhline(y=50, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylim(0, 105)
    for bar, v in zip(bars, df["max_prob"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    return fig


def plot_poisson_panel(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    fig.canvas.manager.set_window_title("Poisson — Scores Esperados")

    # Left: Expected goals comparison
    x = np.arange(len(df))
    width = 0.35
    ax1.bar(x - width / 2, df["expected_goals_local"], width, label="Local", color=LOCAL_COLOR)
    ax1.bar(x + width / 2, df["expected_goals_away"], width, label="Visitante", color=AWAY_COLOR)
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace(" vs ", "\n") for m in df["match"]], fontsize=7)
    ax1.set_ylabel("Goles esperados (λ)")
    ax1.set_title("Goles esperados por equipo (Poisson)")
    ax1.legend()
    ax1.tick_params(axis="x", labelsize=6)

    # Right: Most likely score per match
    matches_short = [m[:15] + "\n" + m.split(" vs ")[1][:15] if " vs " in m else m for m in df["match"]]
    score_labels = [f"{s}\n({p:.0f}%)" for s, p in zip(df["most_likely_score"], df["most_likely_score_pct"])]
    colors = [LOCAL_COLOR if int(s.split("-")[0]) > int(s.split("-")[1]) else
              AWAY_COLOR if int(s.split("-")[0]) < int(s.split("-")[1]) else
              "#f39c12"
              for s in df["most_likely_score"]]
    ax2.barh(range(len(df)), df["most_likely_score_pct"], color=colors)
    ax2.set_yticks(range(len(df)))
    ax2.set_yticklabels(matches_short, fontsize=7)
    ax2.set_xlabel("Probabilidad (%)")
    ax2.set_title("Score más probable (Poisson + Monte Carlo)")
    for i, (label, pct) in enumerate(zip(score_labels, df["most_likely_score_pct"])):
        ax2.text(pct + 0.5, i, label, va="center", fontsize=7)
    ax2.set_xlim(0, df["most_likely_score_pct"].max() + 10)
    fig.tight_layout()
    return fig


def main():
    save_mode = "--save" in sys.argv
    use_nlp = "--nlp" in sys.argv

    csv_path = PREDICTIONS_NLP if use_nlp else PREDICTIONS
    print("=" * 50)
    print("  SHOW RESULTS — World Cup 2026 Predictions")
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
    print(f"  {df.height} matches loaded")
    print()

    if save_mode:
        print("Saving charts to data/ (use without --save for interactive windows)...")
        figs = [
            ("advancement.png", plot_advancement(df)),
            ("probabilities.png", plot_match_probabilities(df)),
            ("confidence.png", plot_confidence_gauge(df)),
            ("poisson_scores.png", plot_poisson_panel(df)),
        ]
        for name, fig in figs:
            if fig is None:
                continue
            print(f"  Saved data/{name}")
            plt.close(fig)
        print("\nDone.")
    else:
        print("Opening interactive windows...")
        plot_advancement(df)
        plot_match_probabilities(df)
        plot_confidence_gauge(df)
        plot_poisson_panel(df)
        print("\nClose the plot windows to exit.")
        plt.show()


if __name__ == "__main__":
    main()
