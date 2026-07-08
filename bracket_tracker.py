import json
import sys
from datetime import datetime
from io import StringIO

import numpy as np
import polars as pl
import requests

# Must come before other matplotlib imports
if "--graphs" in sys.argv:
    if "--save" in sys.argv:
        import matplotlib
        matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch
    import seaborn as sns

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
KNOCKOUT_MATCHES = "data/knockout_matches.json"
MATCHES_8AVOS = "data/8avos_matches.json"
PREDICTIONS = "data/knockout_predictions.csv"
PREDICTIONS_NLP = "data/knockout_predictions_nlp.csv"
PREDICTIONS_8AVOS = "data/8avos_predictions.csv"
PREDICTIONS_8AVOS_NLP = "data/8avos_predictions_nlp.csv"
ACTUAL_KNOCKOUT = "data/actual_knockout_results.json"
WC_YEAR = 2026

LOCAL_COLOR = "#3498db"
AWAY_COLOR = "#e67e22"
DRAW_COLOR = "#f39c12"
OK_COLOR = "#2ecc71"
ERROR_COLOR = "#e74c3c"
PENDING_COLOR = "#95a5a6"
BG_COLOR = "#f5f0eb"
CARD_BG = "#ffffff"


def normalize_team_name(name):
    name = name.strip()
    name = name.replace("Korea Republic", "South Korea")
    name = name.replace("Czechia", "Czech Republic")
    name = name.replace("Turkey", "T\u00fcrkiye")
    name = name.replace("DR Congo", "Congo DR")
    name = name.replace("United States of America", "United States")
    name = name.replace("Bosnia and Herzegovina", "Bosnia-Herzegovina")
    name = name.replace("C\u00f4te d'Ivoire", "Ivory Coast")
    name = name.replace("Curacao", "Cura\u00e7ao")
    if name == "Congo":
        name = "Congo DR"
    return name


def fetch_actual_results():
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    df = pl.read_csv(
        StringIO(r.text),
        schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32},
        null_values=["NA", ""],
        try_parse_dates=True,
    )
    df = df.drop_nulls(subset=["home_score", "away_score"])
    df = df.with_columns(
        pl.col("home_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
    )
    df = df.filter(pl.col("date") >= datetime(WC_YEAR, 1, 1))
    return df


def find_actual_match(df, local, away):
    matches = df.filter(
        (
            ((pl.col("home_team") == local) & (pl.col("away_team") == away))
            | ((pl.col("home_team") == away) & (pl.col("away_team") == local))
        )
        & (pl.col("tournament") == "FIFA World Cup")
    ).sort("date", descending=True)
    if matches.height == 0:
        return None
    row = matches.row(0, named=True)
    if row["home_team"] == local:
        return {"date": row["date"], "local_score": row["home_score"], "away_score": row["away_score"]}
    else:
        return {"date": row["date"], "local_score": row["away_score"], "away_score": row["home_score"]}


def result_label(local, away, ls, aws):
    if ls > aws:
        return local, "Local"
    if ls < aws:
        return away, "Visitante"
    return "Empate", "Empate"


def winner_from_probs(local_pct, draw_pct, away_pct):
    if local_pct > away_pct and local_pct > draw_pct:
        return "Local"
    if away_pct > local_pct and away_pct > draw_pct:
        return "Visitante"
    return "Empate"


def format_score(goal_str):
    parts = goal_str.split("-")
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def load_actual_knockout():
    with open(ACTUAL_KNOCKOUT) as f:
        return json.load(f)


def find_advance_info(actual_ko, local, away):
    for r in actual_ko:
        if r["local"] == local and r["away"] == away:
            return r
    return None


# ────────────────────────────────────────────────────────────────
#  PLOTTING
# ────────────────────────────────────────────────────────────────


def _set_title(fig, title):
    try:
        fig.canvas.manager.set_window_title(title)
    except AttributeError:
        pass


def short_name(name, ml=10): return name if len(name) <= ml else name[:ml-1] + "."


def _sc_flag(r):
    if r["rt_status"] == "pending":
        return None
    pg, pa = format_score(r["pred_score"])
    return (pg == r["actual_local_goals"] and pa == r["actual_away_goals"])


def _mark(clr):
    return "[+]" if clr is True else ("[x]" if clr is False else "[ ]")


def draw_match_card(ax, x, y, r, card_w, card_h):
    av_s = r["av_status"]
    border = OK_COLOR if av_s == "correct" else (ERROR_COLOR if av_s == "incorrect" else PENDING_COLOR)
    rect = FancyBboxPatch(
        (x, y), card_w, card_h, boxstyle="round,pad=0.05", facecolor=CARD_BG,
        edgecolor=border, linewidth=2.5 if av_s != "pending" else 1.5)
    ax.add_patch(rect)

    cx = x + card_w / 2
    top = y + card_h - 0.015
    gap = 0.045

    ax.text(cx, top, f"{short_name(r['local'])} vs {short_name(r['away'])}",
            fontsize=5.5, fontweight="bold", ha="center", va="top", color="#2c3e50")

    div = top - 0.028
    ax.plot([x + 0.02, x + card_w - 0.02], [div, div], color="#e0e0e0", linewidth=0.5)

    ly = div - 0.010

    # TR
    if r["pred_winner"] == "Local":
        pw_n, pw_p = r["local"], r["local_win_pct"]
    elif r["pred_winner"] == "Visitante":
        pw_n, pw_p = r["away"], r["away_win_pct"]
    else:
        pw_n, pw_p = "Emp", r["draw_pct"]
    tr_ok = r["rt_correct"]
    tr_c = OK_COLOR if tr_ok else (ERROR_COLOR if r["rt_status"] == "incorrect" else "#aaa")
    if r["rt_status"] != "pending":
        _, awn = result_label(r["local"], r["away"], r["actual_local_goals"], r["actual_away_goals"])
        tr_l = f"{_mark(tr_ok)} TR {pw_n}({pw_p:.0f}%) \u2192 {awn}"
    else:
        tr_l = f"{_mark(tr_ok)} TR {pw_n}({pw_p:.0f}%)"
    ax.text(x + 0.02, ly, tr_l, fontsize=4.8, ha="left", va="center", color=tr_c, fontfamily="monospace")
    ly -= gap

    # SC
    sc_ok = _sc_flag(r)
    sc_c = OK_COLOR if sc_ok else (ERROR_COLOR if sc_ok is False else "#aaa")
    pg, pa = format_score(r["pred_score"]) if r["rt_status"] != "pending" else (None, None)
    if r["rt_status"] != "pending":
        sc_l = f"{_mark(sc_ok)} SC {pg}-{pa}({r['pred_score_pct']:.0f}%) \u2192 {r['actual_local_goals']}-{r['actual_away_goals']}"
    else:
        sc_l = f"{_mark(sc_ok)} SC {r['pred_score']}({r['pred_score_pct']:.0f}%)"
    ax.text(x + 0.02, ly, sc_l, fontsize=4.8, ha="left", va="center", color=sc_c, fontfamily="monospace")
    ly -= gap

    # AV
    av_p = r["local_advance_pct"] if r["pred_advance_label"] == "Local" else r["away_advance_pct"]
    av_ok = r["av_correct"]
    av_c = OK_COLOR if av_ok else (ERROR_COLOR if r["av_status"] == "incorrect" else "#aaa")
    if r["av_status"] != "pending":
        av_l = f"{_mark(av_ok)} AV {r['pred_advance_name']}({av_p:.0f}%) \u2192 {r['actual_advance_name']}"
    else:
        av_l = f"{_mark(av_ok)} AV {r['pred_advance_name']}({av_p:.0f}%)"
    ax.text(x + 0.02, ly, av_l, fontsize=4.8, ha="left", va="center", color=av_c, fontfamily="monospace")


def plot_bracket_overview(rows, ronda="16avos"):
    fig, ax = plt.subplots(figsize=(16, 10))
    _set_title(fig, "Bracket Tracker \u2014 Pizarra")
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    n = len(rows)
    cols = 4 if n > 4 else 2
    card_w = 0.235 if n > 4 else 0.47
    card_h = 0.19
    gap_x = 0.012
    gap_y = 0.025

    for idx, r in enumerate(rows):
        col = idx % cols
        row = idx // cols
        x = col * (card_w + gap_x) + 0.025
        y = 1.0 - (row + 1) * (card_h + gap_y)
        draw_match_card(ax, x, y, r, card_w, card_h)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 0.995,
            f"Bracket {ronda} \u2014 Mundial 2026    |    [+]=acierto  [x]=fallo  [ ]=pendiente",
            fontsize=8, fontweight="bold", ha="center", va="top", color="#2c3e50")

    # Footer stats
    av_p = [r for r in rows if r["av_status"] != "pending"]
    av_c = sum(1 for r in rows if r["av_status"] == "correct")
    rt_p = [r for r in rows if r["rt_status"] != "pending"]
    rt_c = sum(1 for r in rows if r["rt_status"] == "correct")
    na = len(av_p); nr = len(rt_p)
    if nr > 0:
        footer = (f"T.Regular: {rt_c}/{nr} ({rt_c/nr*100:.0f}%)  |  "
                  f"Clasif: {av_c}/{na} ({av_c/na*100:.0f}%)  |  "
                  f"Pend: {n-nr}")
    else:
        footer = "Sin partidos jugados"
    ax.text(0.5, 0.008, footer, fontsize=8, ha="center", va="bottom",
            color="#555", fontweight="bold")

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot_confidence_vs_outcome(rows):
    played = [r for r in rows if r["rt_status"] != "pending"]
    if not played:
        return None

    n = len(played)
    fig_w = min(3.2 * n, 14)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 4.2))
    _set_title(fig, "Confianza vs Resultado")
    fig.patch.set_facecolor(BG_COLOR)
    if n == 1:
        axes = [axes]

    bar_colors = [LOCAL_COLOR, DRAW_COLOR, AWAY_COLOR]

    for idx, r in enumerate(played):
        ax = axes[idx]
        ax.set_facecolor(BG_COLOR)

        vals = [r["local_win_pct"], r["draw_pct"], r["away_win_pct"]]
        bars = ax.bar(["L", "E", "V"], vals, color=bar_colors, width=0.6,
                       edgecolor="white", linewidth=1.2)

        actual_idx = 0 if r["actual_winner"] == "Local" else (2 if r["actual_winner"] == "Visitante" else 1)
        bars[actual_idx].set_edgecolor("#2c3e50")
        bars[actual_idx].set_linewidth(3.5)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.8,
                    f"{v:.0f}%", ha="center", fontsize=9, fontweight="bold", color="#2c3e50")

        ax.axhline(y=50, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylim(0, 110)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=7)

        # Winner + score as title
        awn, _ = result_label(r["local"], r["away"], r["actual_local_goals"], r["actual_away_goals"])
        av_icon = "+" if r["av_correct"] else "x"
        av_c = OK_COLOR if r["av_correct"] else ERROR_COLOR
        ax.set_title(f"{short_name(r['local'],7)}-{short_name(r['away'],7)}\n"
                     f"{awn}  {r['actual_local_goals']}-{r['actual_away_goals']} [{av_icon}]",
                     fontsize=8, fontweight="bold", color=av_c, linespacing=1.2)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")

    fig.tight_layout()
    return fig


def plot_accuracy_summary(rows):
    av_correct = sum(1 for r in rows if r["av_status"] == "correct")
    av_incorrect = sum(1 for r in rows if r["av_status"] == "incorrect")
    av_pending = sum(1 for r in rows if r["av_status"] == "pending")
    rt_correct = sum(1 for r in rows if r["rt_status"] == "correct")
    rt_incorrect = sum(1 for r in rows if r["rt_status"] == "incorrect")
    rt_pending = sum(1 for r in rows if r["rt_status"] == "pending")
    n = len(rows)

    fig, (ax_left, ax_mid, ax_right) = plt.subplots(1, 3, figsize=(13, 4.5))
    _set_title(fig, "Accuracy \u2014 Resumen")
    fig.patch.set_facecolor(BG_COLOR)

    # ── Left: Clasificación ──
    ax_left.set_facecolor(BG_COLOR)
    ax_left.axis("off")
    n_av = av_correct + av_incorrect
    if n_av > 0:
        av_pct = av_correct / n_av * 100
        ax_left.text(0.5, 0.78, f"{av_pct:.0f}%", fontsize=40, fontweight="bold",
                     ha="center", va="center",
                     color=OK_COLOR if av_pct >= 50 else ERROR_COLOR)
        ax_left.text(0.5, 0.55, f"{av_correct}/{n_av} aciertos", fontsize=11,
                     ha="center", va="center", color="#555")
    else:
        ax_left.text(0.5, 0.67, "\u2014", fontsize=40, fontweight="bold",
                     ha="center", va="center", color=PENDING_COLOR)
    ax_left.text(0.5, 0.30, "Clasificaci\u00f3n", fontsize=9,
                 ha="center", va="center", color="#999", fontweight="bold", style="italic")
    ax_left.set_xlim(0, 1); ax_left.set_ylim(0, 1)

    # ── Middle: Tiempo Regular ──
    ax_mid.set_facecolor(BG_COLOR)
    ax_mid.axis("off")
    n_rt = rt_correct + rt_incorrect
    if n_rt > 0:
        rt_pct = rt_correct / n_rt * 100
        ax_mid.text(0.5, 0.78, f"{rt_pct:.0f}%", fontsize=40, fontweight="bold",
                    ha="center", va="center",
                    color=OK_COLOR if rt_pct >= 50 else ERROR_COLOR)
        ax_mid.text(0.5, 0.55, f"{rt_correct}/{n_rt} aciertos", fontsize=11,
                    ha="center", va="center", color="#555")
    else:
        ax_mid.text(0.5, 0.67, "\u2014", fontsize=40, fontweight="bold",
                    ha="center", va="center", color=PENDING_COLOR)
    ax_mid.text(0.5, 0.30, "Tiempo Regular", fontsize=9,
                ha="center", va="center", color="#999", fontweight="bold", style="italic")
    ax_mid.set_xlim(0, 1); ax_mid.set_ylim(0, 1)

    # ── Right: stacked horizontal bar (avance) ──
    ax_right.set_facecolor(BG_COLOR)
    ax_right.axis("off")
    y_pos = 0.5
    bar_height = 0.35
    segments = []
    if av_correct > 0:
        segments.append((av_correct / n, OK_COLOR, f"Aciertos {av_correct}"))
    if av_incorrect > 0:
        segments.append((av_incorrect / n, ERROR_COLOR, f"Fallos {av_incorrect}"))
    if av_pending > 0:
        segments.append((av_pending / n, PENDING_COLOR, f"Pendientes {av_pending}"))
    x_start = 0
    for width_seg, color_seg, label_seg in segments:
        ax_right.barh(y_pos, width_seg, bar_height, left=x_start,
                      color=color_seg, edgecolor="white", linewidth=1.5)
        if width_seg > 0.08:
            pc = width_seg * 100
            ax_right.text(x_start + width_seg / 2, y_pos, f"{pc:.0f}%",
                          ha="center", va="center", fontsize=11,
                          fontweight="bold", color="white")
        x_start += width_seg

    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)
    legend_y = 0.12
    x_lgnd = 0.05
    for width_seg, color_seg, label_seg in segments:
        w = 0.03
        ax_right.barh(legend_y, w, 0.04, left=x_lgnd,
                      color=color_seg, edgecolor="white", linewidth=1)
        ax_right.text(x_lgnd + w + 0.01, legend_y, label_seg,
                      fontsize=8, va="center", color="#555")
        x_lgnd += w + 0.12
    ax_right.text(0.5, 0.85, "Distribuci\u00f3n global (Clasif.)", fontsize=9,
                  ha="center", va="center", fontweight="bold", color="#555")
    ax_right.set_xlim(0, 1); ax_right.set_ylim(0, 1)

    fig.tight_layout()
    return fig


def plot_score_comparison(rows):
    played = [r for r in rows if r["rt_status"] != "pending"]
    if not played:
        return None

    n = len(played)
    ncols = min(n, 2)
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5, 3.5 * nrows))
    _set_title(fig, "Score: Predicci\u00f3n vs Real")
    fig.patch.set_facecolor(BG_COLOR)
    axes_flat = [axes] if n == 1 else axes.flatten()
    for ax in axes_flat:
        ax.set_facecolor(BG_COLOR)

    for idx, r in enumerate(played):
        ax = axes_flat[idx]
        pg, pa = format_score(r["pred_score"])
        pg = pg or 0; pa = pa or 0

        cats = ["Pred", "Real"]
        l_gls = [pg, r["actual_local_goals"]]
        a_gls = [pa, r["actual_away_goals"]]
        x = np.arange(len(cats))
        w = 0.28
        sn_loc = short_name(r["local"],6)
        sn_awy = short_name(r["away"],6)

        bh = ax.bar(x - w/2, l_gls, w, label=sn_loc, color=LOCAL_COLOR, edgecolor="white", linewidth=1)
        ba = ax.bar(x + w/2, a_gls, w, label=sn_awy, color=AWAY_COLOR, edgecolor="white", linewidth=1)

        for bars, vals, clr in [(bh, l_gls, LOCAL_COLOR), (ba, a_gls, AWAY_COLOR)]:
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                        str(v), ha="center", fontsize=9, fontweight="bold", color=clr)

        max_g = max(max(l_gls), max(a_gls)) + 1
        ax.set_ylim(0, max(max_g, 3))
        ax.set_xticks(x)
        ax.set_xticklabels(cats, fontsize=7)
        ax.set_yticks(range(0, max(max_g, 3)))
        ax.tick_params(axis="y", labelsize=7)

        av_icon = "+" if r["av_correct"] else "x"
        av_c = OK_COLOR if r["av_correct"] else ERROR_COLOR
        ax.set_title(f"{sn_loc}-{sn_awy} [{av_icon}]", fontsize=8, fontweight="bold", color=av_c)
        ax.legend(fontsize=6, loc="upper right", framealpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#ccc")
        ax.spines["bottom"].set_color("#ccc")

    # hide unused subplots
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.tight_layout()
    return fig


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────


def main():
    use_graphs = "--graphs" in sys.argv
    use_nlp = "--nlp" in sys.argv
    save_mode = "--save" in sys.argv
    es_8avos = "--8avos" in sys.argv

    ronda = "8avos" if es_8avos else "16avos"
    matches_file = MATCHES_8AVOS if es_8avos else KNOCKOUT_MATCHES
    csv_path = (PREDICTIONS_8AVOS_NLP if es_8avos else PREDICTIONS_NLP) if use_nlp else (PREDICTIONS_8AVOS if es_8avos else PREDICTIONS)
    label = "NLP" if use_nlp else "BASE"

    print("=" * 68)
    print(f"  BRACKET TRACKER \u2014 {ronda} World Cup 2026  [{label}]")
    print("=" * 68)

    if use_graphs:
        import os
        if "--save" not in sys.argv:
            if os.environ.get("DISPLAY") is None and sys.platform != "darwin":
                print("\n  ERROR: No display available. Use --save to generate PNGs.")
                print("    uv run python bracket_tracker.py --graphs --save")
                sys.exit(1)
        sns.set_style("whitegrid")
        try:
            plt.rcParams["font.family"] = "DejaVu Sans"
        except Exception:
            pass
        plt.rcParams.update({"font.size": 11, "figure.dpi": 130})

    with open(matches_file) as f:
        bracket = json.load(f)

    try:
        pred_df = pl.read_csv(csv_path)
    except FileNotFoundError:
        fl = " --nlp" if use_nlp else ""
        print(f"\n  ERROR: {csv_path} not found.")
        print(f"  Run: uv run python stacking_model.py{fl}")
        sys.exit(1)

    print("\n  Fetching actual results...")
    actual_df = fetch_actual_results()
    print(f"  {actual_df.height} matches in {WC_YEAR}")

    actual_ko = load_actual_knockout()

    rows = []
    rt_correct = 0
    rt_played = 0
    av_correct = 0
    av_played = 0

    for match in bracket:
        local = match["local"]
        away = match["visitante"]

        pred_row = pred_df.filter(
            (pl.col("match") == f"{local} vs {away}") |
            (pl.col("match") == f"{away} vs {local}")
        )
        if pred_row.height == 0:
            continue
        pred = pred_row.row(0, named=True)

        actual = find_actual_match(actual_df, local, away)
        advance = find_advance_info(actual_ko, local, away)

        # Fallback: use JSON scores when results.csv doesn't have the match yet
        if actual is None and advance is not None:
            actual = {
                "local_score": advance["local_goals_regular"],
                "away_score": advance["away_goals_regular"],
            }

        is_played = actual is not None
        is_played_av = advance is not None

        pred_winner = winner_from_probs(pred["local_win_pct"], pred["draw_pct"], pred["away_win_pct"])

        pred_advance = "Local" if pred["local_advance_pct"] >= pred["away_advance_pct"] else "Visitante"
        predicted_advance_name = local if pred_advance == "Local" else away

        # ── Regular time ──
        if is_played:
            _, actual_winner_label = result_label(local, away, actual["local_score"], actual["away_score"])
            rt_correct_flag = pred_winner == actual_winner_label
            if rt_correct_flag:
                rt_correct += 1
            rt_played += 1
            rt_status = "correct" if rt_correct_flag else "incorrect"
        else:
            actual_winner_label = ""
            rt_status = "pending"
            rt_correct_flag = None

        # ── Advance ──
        if is_played_av:
            actual_advancing = advance["advancing_team"]
            av_correct_flag = predicted_advance_name == actual_advancing
            if av_correct_flag:
                av_correct += 1
            av_played += 1
            av_status = "correct" if av_correct_flag else "incorrect"
            pen_str = ""
            if advance["local_penalties"] is not None:
                pen_str = f" ({advance['local_penalties']}-{advance['away_penalties']} pen)"
            actual_advance_str = f"{actual_advancing}{pen_str}"
        elif is_played and actual["local_score"] != actual["away_score"]:
            # No JSON entry but clear 90-min winner → auto-detect advance
            if actual["local_score"] > actual["away_score"]:
                actual_advancing = local
            else:
                actual_advancing = away
            av_correct_flag = predicted_advance_name == actual_advancing
            if av_correct_flag:
                av_correct += 1
            av_played += 1
            av_status = "correct" if av_correct_flag else "incorrect"
            actual_advance_str = actual_advancing
        else:
            av_correct_flag = None
            av_status = "pending"
            actual_advance_str = ""

        rows.append({
            "local": local,
            "away": away,
            "rt_status": rt_status,
            "av_status": av_status,
            "played": is_played,
            "actual_local_goals": actual["local_score"] if is_played else None,
            "actual_away_goals": actual["away_score"] if is_played else None,
            "actual_winner": actual_winner_label if is_played else "",
            "pred_winner": pred_winner,
            "rt_correct": rt_correct_flag,
            "pred_advance_label": pred_advance,
            "pred_advance_name": predicted_advance_name,
            "actual_advance_name": actual_advance_str if is_played_av else "",
            "av_correct": av_correct_flag,
            "local_win_pct": pred["local_win_pct"],
            "draw_pct": pred["draw_pct"],
            "away_win_pct": pred["away_win_pct"],
            "local_advance_pct": pred.get("local_advance_pct", 50),
            "away_advance_pct": pred.get("away_advance_pct", 50),
            "pred_score": pred.get("most_likely_score", ""),
            "pred_score_pct": pred.get("most_likely_score_pct", 0),
            "exp_goals_local": pred.get("expected_goals_local", 0),
            "exp_goals_away": pred.get("expected_goals_away", 0),
        })

    # ── Text output ──
    print()
    print(f"  {'=' * 68}")
    print(f"  {'Partido':24s}  {'Dimensi\u00f3n':15s}  {'Predicci\u00f3n':18s}  {'Real':18s}")
    print(f"  {'=' * 68}")

    for r in rows:
        match_shown = f"{r['local']} vs {r['away']}"

        # Build the predicted winner name (team or "Empate")
        if r["pred_winner"] == "Local":
            pred_winner_name = r["local"]
            pred_winner_pct = r["local_win_pct"]
        elif r["pred_winner"] == "Visitante":
            pred_winner_name = r["away"]
            pred_winner_pct = r["away_win_pct"]
        else:
            pred_winner_name = "Empate"
            pred_winner_pct = r["draw_pct"]

        # Build the actual winner name for regular time
        if r["played"]:
            actual_winner_name, _ = result_label(
                r["local"], r["away"], r["actual_local_goals"], r["actual_away_goals"]
            )
        else:
            actual_winner_name = "\u2014"

        # ── Tiempo regular ──
        pred_tr = f"{pred_winner_name} ({pred_winner_pct:.0f}%)"
        real_tr = actual_winner_name
        tr_flag = r["rt_correct"]
        if r["played"]:
            if tr_flag:
                tr_mark = "  [+]"
            else:
                tr_mark = "  [x]"
        else:
            tr_mark = ""

        print(f"  {match_shown:24s}  {'T. Regular':15s}  {pred_tr:18s}  {real_tr:18s}{tr_mark}")

        # ── Marcador ──
        pred_sc = f"{r['pred_score']} ({r['pred_score_pct']:.0f}%)"
        if r["played"]:
            real_sc = f"{r['actual_local_goals']}-{r['actual_away_goals']}"
            pg, pa = format_score(r["pred_score"])
            sc_ok = pg == r["actual_local_goals"] and pa == r["actual_away_goals"]
            sc_mark = "  [+]" if sc_ok else "  [x]"
        else:
            real_sc = "\u2014"
            sc_mark = ""

        print(f"  {'':24s}  {'Marcador':15s}  {pred_sc:18s}  {real_sc:18s}{sc_mark}")

        # ── Clasificación ──
        pred_av_name = r["pred_advance_name"]
        pred_av_pct = (r["local_advance_pct"] if r["pred_advance_label"] == "Local"
                       else r["away_advance_pct"])
        pred_av = f"{pred_av_name} ({pred_av_pct:.0f}%)"
        if r["av_status"] != "pending":
            real_av = r["actual_advance_name"]
            av_flag = r["av_correct"]
            av_mark = "  [+]" if av_flag else "  [x]"
        else:
            real_av = "\u2014"
            av_mark = ""

        print(f"  {'':24s}  {'Clasificaci\u00f3n':15s}  {pred_av:18s}  {real_av:18s}{av_mark}")
        print(f"  {'-' * 68}")

    rt_pct = f"{rt_correct / rt_played * 100:.0f}%" if rt_played > 0 else "—"
    av_pct = f"{av_correct / av_played * 100:.0f}%" if av_played > 0 else "—"
    print(f"  {'=' * 68}")
    print(f"  T.Regular: {rt_correct}/{rt_played} ({rt_pct})  |  "
          f"Clasificaci\u00f3n: {av_correct}/{av_played} ({av_pct})  |  "
          f"Pendientes: {len(rows)-rt_played}/{len(rows)}")
    print(f"  {'=' * 68}")

    # ── Graphs ──
    if use_graphs:
        prefix = f"{ronda}_" if es_8avos else ""
        figs = [
            (f"{prefix}bracket_overview.png", plot_bracket_overview(rows, ronda)),
            (f"{prefix}accuracy_summary.png", plot_accuracy_summary(rows)),
            (f"{prefix}confidence_vs_outcome.png", plot_confidence_vs_outcome(rows)),
            (f"{prefix}score_comparison.png", plot_score_comparison(rows)),
        ]

        if save_mode:
            print("\nSaving charts...")
            for name, fig in figs:
                if fig is None:
                    continue
                fig.savefig(f"data/{name}", dpi=130, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
                print(f"  Saved data/{name}")
                plt.close(fig)
            print("\nDone.")
        else:
            print("\nOpening interactive windows...")
            for _, fig in figs:
                if fig is None:
                    continue
            plt.show()


if __name__ == "__main__":
    main()
