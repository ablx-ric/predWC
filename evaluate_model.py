"""
Extended model evaluation: additional metrics beyond accuracy and log-loss.
Run standalone or import the functions for use in other scripts.

Usage:
    uv run python evaluate_model.py
"""

import bisect
import json
import warnings
from datetime import datetime, timedelta
from io import StringIO

import numpy as np
import polars as pl
import requests
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, log_loss, f1_score, matthews_corrcoef,
    brier_score_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
ELO_RANKINGS = "data/elo_rankings.json"
ELO_HISTORY = "data/elo_history.parquet"
MAX_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 5.0,
    "UEFA Euro": 4.0,
    "Copa América": 4.0,
    "African Cup of Nations": 3.5,
    "AFC Asian Cup": 3.5,
    "Gold Cup": 3.0,
    "FIFA World Cup qualification": 3.0,
    "UEFA Euro qualification": 2.5,
    "CONCACAF Nations League": 2.0,
    "UEFA Nations League": 2.0,
    "African Cup of Nations qualification": 2.0,
    "Friendly": 1.0,
}

FEATURE_COLS = [
    "home_matches_played", "home_goals_for_avg", "home_goals_against_avg",
    "home_goal_diff_avg", "home_win_pct", "home_draw_pct", "home_loss_pct",
    "home_recent_form",
    "away_matches_played", "away_goals_for_avg", "away_goals_against_avg",
    "away_goal_diff_avg", "away_win_pct", "away_draw_pct", "away_loss_pct",
    "away_recent_form",
    "home_elo", "away_elo", "elo_diff", "goal_diff_strength",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "tournament_weight", "is_neutral",
]

CLASS_NAMES = ["Local", "Empate", "Visitante"]


def normalize_team_name(name):
    name = name.strip()
    name = name.replace("Korea Republic", "South Korea")
    name = name.replace("Czechia", "Czech Republic")
    name = name.replace("Turkey", "Türkiye")
    name = name.replace("DR Congo", "Congo DR")
    name = name.replace("United States of America", "United States")
    name = name.replace("Bosnia and Herzegovina", "Bosnia-Herzegovina")
    name = name.replace("C\u00f4te d'Ivoire", "Ivory Coast")
    name = name.replace("Curacao", "Cura\u00e7ao")
    if name == "Congo":
        name = "Congo DR"
    return name


def normalize_team_for_elo(name):
    m = {
        "Czech Republic": "Czechia",
        "Bosnia-Herzegovina": "Bosnia and Herzegovina",
        "Congo DR": "DR Congo",
        "United States of America": "United States",
        "Korea Republic": "South Korea",
        "C\u00f4te d'Ivoire": "Ivory Coast",
        "Curacao": "Cura\u00e7ao",
        "T\u00fcrkiye": "Turkey",
        "Republic of Ireland": "Ireland",
        "German DR": "Germany DR",
        "Vietnam Republic": "Vietnam",
        "United States Virgin Islands": "US Virgin Islands",
        "Timor-Leste": "East Timor",
    }
    return m.get(name, name)


def load_static_elo():
    elo = {}
    with open(ELO_RANKINGS) as f:
        for team in json.load(f):
            elo[team["team"]] = float(team["rating"])
    return elo


def load_elo_history():
    df = pl.read_parquet(ELO_HISTORY)
    lookup = {}
    for row in df.iter_rows(named=True):
        team = row["team"]
        try:
            dt = datetime.strptime(row["date"], "%Y-%m-%d")
        except ValueError:
            continue
        date_int = dt.toordinal()
        row_list = lookup.setdefault(team, [])
        elo_after = row["elo_before"] + row["elo_change"]
        row_list.append((date_int, float(row["elo_before"]), float(elo_after)))
    return lookup


def get_historical_elo(team, match_date, elo_lookup, static_elo=None):
    elo_rows = elo_lookup.get(team)
    if elo_rows:
        date_int = match_date.toordinal()
        idx = bisect.bisect_right(elo_rows, (date_int, float("inf")))
        if idx > 0:
            matched_date, elo_before, elo_after = elo_rows[idx - 1]
            if matched_date == date_int:
                return elo_before
            return elo_after
    if static_elo:
        return static_elo.get(team, 1500)
    return 1500


def compute_rolling_stats(matches, team_col, date, elo_lookup, static_elo=None, window_matches=10, decay_halflife=180):
    if isinstance(date, datetime):
        date = date.date()
    recent = matches.filter(
        ((pl.col("home_team") == team_col) | (pl.col("away_team") == team_col))
        & (pl.col("date") < pl.lit(date))
    ).sort("date", descending=True).head(window_matches)
    total = recent.height
    if total == 0:
        return {"matches_played": 0, "goals_for_avg": 0, "goals_against_avg": 0,
                "goal_diff_avg": 0, "win_pct": 0, "draw_pct": 0, "loss_pct": 0,
                "recent_form": 0}

    w_gf = w_ga = w_w = w_d = w_l = 0.0
    total_weight = 0.0
    form_points = 0

    for j, row in enumerate(recent.iter_rows(named=True)):
        days_ago = max((date - row["date"]).days, 1)
        time_weight = 2 ** (-days_ago / decay_halflife)

        if row["home_team"] == team_col:
            opp = row["away_team"]
            gf = row["home_score"]
            ga = row["away_score"]
        else:
            opp = row["home_team"]
            gf = row["away_score"]
            ga = row["home_score"]
        won = gf > ga
        drawn = gf == ga

        opp_elo = get_historical_elo(normalize_team_for_elo(opp), row["date"], elo_lookup, static_elo)
        weight = time_weight * (opp_elo / 2000.0)
        total_weight += weight
        w_gf += weight * gf
        w_ga += weight * ga
        if won:
            w_w += weight
        elif drawn:
            w_d += weight
        else:
            w_l += weight

        if j < 5:
            form_points += 3 if won else (1 if drawn else 0)

    tw = total_weight if total_weight > 0 else 1
    return {
        "matches_played": total, "goals_for_avg": w_gf / tw, "goals_against_avg": w_ga / tw,
        "goal_diff_avg": (w_gf - w_ga) / tw, "win_pct": w_w / tw, "draw_pct": w_d / tw, "loss_pct": w_l / tw,
        "recent_form": form_points / 5,
    }


def compute_additional_metrics(y_true, y_proba, class_names=None):
    if class_names is None:
        class_names = ["Local", "Empate", "Visitante"]
    n_classes = y_proba.shape[1]
    y_pred = np.argmax(y_proba, axis=1)

    metrics = {}

    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["log_loss"] = log_loss(y_true, y_proba)
    metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
    metrics["f1_weighted"] = f1_score(y_true, y_pred, average="weighted")
    metrics["mcc"] = matthews_corrcoef(y_true, y_pred)

    # Brier score per class + multiclass average
    y_true_bin = np.eye(n_classes)[y_true]
    brier_per_class = [
        brier_score_loss(y_true_bin[:, c], y_proba[:, c])
        for c in range(n_classes)
    ]
    metrics["brier_per_class"] = dict(zip(class_names, [round(b, 4) for b in brier_per_class]))
    metrics["brier_multi"] = np.mean(brier_per_class)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm

    # Per-class metrics from classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    metrics["per_class"] = {
        cls: {
            "precision": round(report[cls]["precision"], 4),
            "recall": round(report[cls]["recall"], 4),
            "f1": round(report[cls]["f1-score"], 4),
            "support": int(report[cls]["support"]),
        }
        for cls in class_names
    }

    # Calibration: how often the predicted class matches actual, binned by confidence
    max_probs = np.max(y_proba, axis=1)
    correct = (y_pred == y_true).astype(float)
    bins = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    calibration = {}
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        mask = (max_probs > lo) & (max_probs <= hi)
        if mask.sum() > 0:
            avg_conf = max_probs[mask].mean()
            avg_acc = correct[mask].mean()
            calibration[f"{lo:.0%}-{hi:.0%}"] = {
                "count": int(mask.sum()),
                "avg_confidence": round(avg_conf, 4),
                "accuracy": round(avg_acc, 4),
                "gap": round(avg_conf - avg_acc, 4),
            }
    metrics["calibration"] = calibration

    # Top-2 accuracy: true class is among the 2 most probable
    top2 = np.argsort(-y_proba, axis=1)[:, :2]
    metrics["top2_accuracy"] = np.mean([y_true[i] in top2[i] for i in range(len(y_true))])

    # Class distribution: predicted vs actual
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    true_counts = np.bincount(y_true, minlength=n_classes)
    metrics["predicted_distribution"] = {
        cls: int(pred_counts[c]) for c, cls in enumerate(class_names)
    }
    metrics["actual_distribution"] = {
        cls: int(true_counts[c]) for c, cls in enumerate(class_names)
    }

    return metrics


def print_metrics(metrics, indent=0):
    pad = "  " * indent
    print(f"\n{pad}{'=' * 55}")
    print(f"{pad}  MODEL EVALUATION METRICS")
    print(f"{pad}{'=' * 55}")

    # Main metrics
    print(f"\n{pad}  Main metrics:")
    print(f"{pad}    Accuracy:         {metrics['accuracy']:.4f}")
    print(f"{pad}    Log-loss:         {metrics['log_loss']:.4f}")
    print(f"{pad}    F1 (macro):       {metrics['f1_macro']:.4f}")
    print(f"{pad}    F1 (weighted):    {metrics['f1_weighted']:.4f}")
    print(f"{pad}    MCC:              {metrics['mcc']:.4f}")
    print(f"{pad}    Brier (multi):    {metrics['brier_multi']:.4f}")
    print(f"{pad}    Top-2 accuracy:   {metrics['top2_accuracy']:.4f}")

    # Per-class
    print(f"\n{pad}  Per-class breakdown:")
    print(f"{pad}    {'Class':<12} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<8}")
    print(f"{pad}    {'─' * 44}")
    for cls, vals in metrics["per_class"].items():
        print(f"{pad}    {cls:<12} {vals['precision']:<8.4f} {vals['recall']:<8.4f} {vals['f1']:<8.4f} {vals['support']:<8d}")

    # Brier per class
    print(f"\n{pad}  Brier score per class:")
    for cls, b in metrics["brier_per_class"].items():
        print(f"{pad}    {cls:<12} {b:.4f}")

    # Distribution
    print(f"\n{pad}  Actual vs Predicted distribution:")
    total_actual = sum(metrics["actual_distribution"].values())
    for cls in metrics["actual_distribution"]:
        act = metrics["actual_distribution"][cls]
        pred = metrics["predicted_distribution"][cls]
        pct_act = act / total_actual * 100
        pct_pred = pred / total_actual * 100
        print(f"{pad}    {cls:<12} actual: {act:5d} ({pct_act:5.1f}%)  pred: {pred:5d} ({pct_pred:5.1f}%)")

    # Calibration
    print(f"\n{pad}  Calibration (confidence bins):")
    print(f"{pad}    {'Bin':<12} {'Count':<7} {'Avg Conf':<10} {'Accuracy':<10} {'Gap':<8}")
    print(f"{pad}    {'─' * 47}")
    total_gap = 0
    total_weight = 0
    for bin_label, cal in metrics["calibration"].items():
        print(f"{pad}    {bin_label:<12} {cal['count']:<7d} {cal['avg_confidence']:<10.4f} {cal['accuracy']:<10.4f} {cal['gap']:<+8.4f}")
        total_gap += cal["gap"] * cal["count"]
        total_weight += cal["count"]
    if total_weight > 0:
        print(f"{pad}    {'─' * 47}")
        print(f"{pad}    ECE (Expected Calibration Error): {total_gap / total_weight:.4f}")

    # Confusion matrix
    print(f"\n{pad}  Confusion matrix:")
    cm = metrics["confusion_matrix"]
    classes = list(metrics["per_class"].keys())
    header = f"{pad}    {'':<12}" + "".join(f"{c:<8}" for c in classes)
    print(header)
    print(f"{pad}    {'─' * (12 + 8 * len(classes))}")
    for i, cls in enumerate(classes):
        row = f"{pad}    {cls:<12}" + "".join(f"{cm[i, j]:<8d}" for j in range(len(classes)))
        print(row)

    print(f"\n{pad}{'=' * 55}\n")


def main():
    print()
    print("=" * 60)
    print("  MODEL EVALUATION — Additional Metrics")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1] Loading training data...")
    r = requests.get(RESULTS_URL, timeout=30)
    r.raise_for_status()
    df = pl.read_csv(StringIO(r.text), schema_overrides={"home_score": pl.Int32, "away_score": pl.Int32},
                      null_values=["NA", ""], try_parse_dates=True)
    df = df.drop_nulls(subset=["home_score", "away_score"])
    max_dt = datetime.strptime(MAX_DATE, "%Y-%m-%d")
    df = df.filter((pl.col("date") >= datetime(2018, 1, 1)) & (pl.col("date") <= max_dt))
    df = df.with_columns(
        pl.col("home_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
        pl.col("away_team").map_elements(normalize_team_name, return_dtype=pl.Utf8),
    )
    df = df.with_columns(
        tournament_weight=pl.col("tournament").replace_strict(TOURNAMENT_WEIGHTS, default=1.0).cast(pl.Float32),
    )
    df = df.sort("date")
    print(f"   Training matches: {df.height}")

    # ------------------------------------------------------------------
    # 2. Build feature vectors
    # ------------------------------------------------------------------
    print("\n[2] Building feature vectors...")
    static_elo = load_static_elo()
    elo_lookup = load_elo_history()
    features = []
    targets = []
    match_dates = []
    matches_list = df.to_dicts()

    for i, m in enumerate(matches_list):
        if i % 2000 == 0 and i > 0:
            print(f"   {i}/{len(matches_list)}...")

        match_date = m["date"]
        home, away = m["home_team"], m["away_team"]

        hs = compute_rolling_stats(df, home, match_date, elo_lookup, static_elo)
        as_ = compute_rolling_stats(df, away, match_date, elo_lookup, static_elo)

        h2h = df.filter(
            ((pl.col("home_team") == home) & (pl.col("away_team") == away)) |
            ((pl.col("home_team") == away) & (pl.col("away_team") == home))
        ).filter(pl.col("date") < match_date).sort("date", descending=True).head(5)

        hw = aw = hd = 0
        for r in h2h.iter_rows(named=True):
            if r["home_team"] == home and r["home_score"] > r["away_score"]: hw += 1
            elif r["away_team"] == home and r["away_score"] > r["home_score"]: hw += 1
            elif r["home_team"] == away and r["home_score"] > r["away_score"]: aw += 1
            elif r["away_team"] == away and r["away_score"] > r["home_score"]: aw += 1
            else: hd += 1

        he = get_historical_elo(normalize_team_for_elo(home), match_date, elo_lookup, static_elo)
        ae = get_historical_elo(normalize_team_for_elo(away), match_date, elo_lookup, static_elo)

        features.append({
            "home_matches_played": hs["matches_played"],
            "home_goals_for_avg": hs["goals_for_avg"],
            "home_goals_against_avg": hs["goals_against_avg"],
            "home_goal_diff_avg": hs["goal_diff_avg"],
            "home_win_pct": hs["win_pct"],
            "home_draw_pct": hs["draw_pct"],
            "home_loss_pct": hs["loss_pct"],
            "home_recent_form": hs["recent_form"],
            "away_matches_played": as_["matches_played"],
            "away_goals_for_avg": as_["goals_for_avg"],
            "away_goals_against_avg": as_["goals_against_avg"],
            "away_goal_diff_avg": as_["goal_diff_avg"],
            "away_win_pct": as_["win_pct"],
            "away_draw_pct": as_["draw_pct"],
            "away_loss_pct": as_["loss_pct"],
            "away_recent_form": as_["recent_form"],
            "home_elo": he, "away_elo": ae,
            "elo_diff": he - ae,
            "goal_diff_strength": hs["goal_diff_avg"] - as_["goal_diff_avg"],
            "h2h_home_wins": hw, "h2h_away_wins": aw, "h2h_draws": hd,
            "tournament_weight": m["tournament_weight"],
            "is_neutral": m["neutral"],
        })
        match_dates.append(match_date)

        if m["home_score"] > m["away_score"]:
            targets.append(0)
        elif m["home_score"] == m["away_score"]:
            targets.append(1)
        else:
            targets.append(2)

    X = pl.DataFrame(features).to_numpy()
    y = np.array(targets)
    dates_arr = np.array(match_dates)
    print(f"   Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # 3. Train with temporal split
    # ------------------------------------------------------------------
    n = len(dates_arr)
    split_idx = int(n * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, n)

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    classes = np.array([0, 1, 2])
    tr_weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    sw_tr = tr_weights[y_tr]

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_val_scaled = scaler.transform(X_val)

    models = {
        "rf": RandomForestClassifier(n_estimators=300, max_depth=12, random_state=42,
                                     class_weight="balanced", n_jobs=-1),
        "xgb": xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
                                 random_state=42, eval_metric="mlogloss"),
        "mlp": MLPClassifier(hidden_layer_sizes=(15,), activation="relu",
                              solver="adam", max_iter=5000, random_state=42,
                              early_stopping=True, validation_fraction=0.1,
                              alpha=0.001),
    }

    print("\n[3] Training base models...")
    models["rf"].fit(X_tr_scaled, y_tr)
    models["xgb"].fit(X_tr_scaled, y_tr, sample_weight=sw_tr)
    models["mlp"].fit(X_tr_scaled, y_tr)

    meta_val = np.zeros((len(y_val), 9))
    offset = 0
    for name, model in models.items():
        meta_val[:, offset:offset + 3] = model.predict_proba(X_val_scaled)
        offset += 3

    # Meta-model trained on VAL predictions (out-of-sample)
    meta = LogisticRegression(solver="lbfgs", max_iter=1000, C=0.1,
                               class_weight="balanced", random_state=42)
    meta.fit(meta_val, y_val)

    val_probs = meta.predict_proba(meta_val)

    # ------------------------------------------------------------------
    # 4. Compute all metrics
    # ------------------------------------------------------------------
    print("\n[4] Computing extended metrics...")
    metrics = compute_additional_metrics(y_val, val_probs, CLASS_NAMES)
    print_metrics(metrics)

    # ------------------------------------------------------------------
    # 5. Per-model metrics for comparison
    # ------------------------------------------------------------------
    print("[5] Per-model validation metrics:")
    offset = 0
    for name, model in models.items():
        proba = model.predict_proba(X_val_scaled)
        acc = accuracy_score(y_val, np.argmax(proba, axis=1))
        ll = log_loss(y_val, proba)
        f1 = f1_score(y_val, np.argmax(proba, axis=1), average="macro")
        print(f"   {name.upper():5s} | accuracy: {acc:.4f} | logloss: {ll:.4f} | F1: {f1:.4f}")
    print(f"   STACKING | accuracy: {metrics['accuracy']:.4f} | logloss: {metrics['log_loss']:.4f} | F1: {metrics['f1_macro']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
