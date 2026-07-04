import argparse
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
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/refs/heads/master/results.csv"
ELO_RANKINGS = "data/elo_rankings.json"
ELO_HISTORY = "data/elo_history.parquet"
KNOCKOUT_MATCHES = "data/knockout_matches.json"
NLP_FEATURES = "data/team_nlp_features.json"
MAX_DATE = None  # set in main() based on --max-date or today-1

TOURNAMENT_WEIGHTS = {
    "FIFA World Cup": 5.0,
    "UEFA Euro": 4.0,
    "Copa Am\u00e9rica": 4.0,
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

BASE_FEATURE_COLS = [
    "home_matches_played", "home_goals_for_avg", "home_goals_against_avg",
    "home_goal_diff_avg", "home_win_pct", "home_draw_pct", "home_loss_pct",
    "home_recent_form",
    "away_matches_played", "away_goals_for_avg", "away_goals_against_avg",
    "away_goal_diff_avg", "away_win_pct", "away_draw_pct", "away_loss_pct",
    "away_recent_form",
    "home_elo", "away_elo", "elo_diff", "goal_diff_strength",
    "h2h_home_wins", "h2h_away_wins", "h2h_draws", "tournament_weight", "is_neutral",
]

NLP_FEATURE_COLS = None  # built dynamically after loading NLP data


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


def load_static_elo():
    elo = {}
    with open(ELO_RANKINGS) as f:
        for team in json.load(f):
            elo[team["team"]] = float(team["rating"])
    return elo


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
        # Store (date_int, elo_before, elo_after) where elo_after = rating after this match
        elo_after = row["elo_before"] + row["elo_change"]
        row_list.append((date_int, float(row["elo_before"]), float(elo_after)))
    return lookup


def get_historical_elo(team, match_date, elo_lookup, static_elo=None):
    elo_rows = elo_lookup.get(team)
    if elo_rows:
        date_int = match_date.toordinal()
        # Use float('inf') so that (date_int, inf) > (date_int, elo_before, elo_after)
        # This finds the LAST entry at or before the given date.
        idx = bisect.bisect_right(elo_rows, (date_int, float("inf")))
        if idx > 0:
            matched_date, elo_before, elo_after = elo_rows[idx - 1]
            if matched_date == date_int:
                # Exact match on this date → team played. Return ELO entering the match.
                return elo_before
            # No exact match → return ELO after their last match (current rating)
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


def load_nlp_data():
    global NLP_FEATURE_COLS
    with open(NLP_FEATURES) as f:
        data = json.load(f)
    if NLP_FEATURE_COLS is None and data:
        sample = next(iter(data.values()))
        n_pc = sum(1 for k in sample if k.startswith("news_pc"))
        NLP_FEATURE_COLS = (
            [f"home_news_pc{j+1}" for j in range(n_pc)]
            + ["home_yt_sent"]
            + [f"away_news_pc{j+1}" for j in range(n_pc)]
            + ["away_yt_sent"]
        )
    return data


HOST_NATIONS = {"Mexico", "United States", "Canada"}


def build_features_for_match(match, matches_df, match_date, elo_lookup, static_elo=None, nlp_data=None):
    local = normalize_team_name(match["local"])
    visitante = normalize_team_name(match["visitante"])

    # Host nations playing at home get home advantage
    # unless the match is explicitly marked as neutral (e.g., Canada's 16avos en USA)
    if match.get("neutral", False):
        home = local
        away = visitante
        is_neutral = True
        swapped = False
    else:
        host_local = local in HOST_NATIONS
        host_visitante = visitante in HOST_NATIONS
        if host_visitante and not host_local:
            home = visitante
            away = local
            swapped = True
        else:
            home = local
            away = visitante
            swapped = False
        is_neutral = not (host_local or host_visitante)

    date = datetime.strptime(MAX_DATE, "%Y-%m-%d")
    date_d = date.date()
    hs = compute_rolling_stats(matches_df, home, date, elo_lookup, static_elo)
    as_ = compute_rolling_stats(matches_df, away, date, elo_lookup, static_elo)

    h2h = matches_df.filter(
        ((pl.col("home_team") == home) & (pl.col("away_team") == away)) |
        ((pl.col("home_team") == away) & (pl.col("away_team") == home))
    ).filter(pl.col("date") < pl.lit(date_d)).sort("date", descending=True).head(5)

    hw = aw = hd = 0
    for r in h2h.iter_rows(named=True):
        if r["home_team"] == home and r["home_score"] > r["away_score"]: hw += 1
        elif r["away_team"] == home and r["away_score"] > r["home_score"]: hw += 1
        elif r["home_team"] == away and r["home_score"] > r["away_score"]: aw += 1
        elif r["away_team"] == away and r["away_score"] > r["home_score"]: aw += 1
        else: hd += 1

    he = get_historical_elo(normalize_team_for_elo(home), date, elo_lookup, static_elo)
    ae = get_historical_elo(normalize_team_for_elo(away), date, elo_lookup, static_elo)

    fv = {
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
        "home_elo": he,
        "away_elo": ae,
        "elo_diff": he - ae,
        "goal_diff_strength": hs["goal_diff_avg"] - as_["goal_diff_avg"],
        "h2h_home_wins": hw, "h2h_away_wins": aw, "h2h_draws": hd,
        "tournament_weight": 5.0,
        "is_neutral": is_neutral,
    }

    if nlp_data and NLP_FEATURE_COLS:
        def safe_get(team, key):
            d = nlp_data.get(team) or nlp_data.get(normalize_team_name(team), {})
            return d.get(key, 0.0)

        n_pc = sum(1 for k in (next(iter(nlp_data.values())) if nlp_data else {}) if k.startswith("news_pc"))
        for j in range(n_pc):
            fv[f"home_news_pc{j+1}"] = safe_get(home, f"news_pc{j+1}")
            fv[f"away_news_pc{j+1}"] = safe_get(away, f"news_pc{j+1}")
        fv["home_yt_sent"] = safe_get(home, "youtube_sentiment")
        fv["away_yt_sent"] = safe_get(away, "youtube_sentiment")
    elif NLP_FEATURE_COLS:
        for k in NLP_FEATURE_COLS:
            fv[k] = 0.0

    return fv, home, away, swapped


def predict_poisson_scores(home_elo, away_elo, overall_avg, n_sim=50000, seed=42):
    rng = np.random.default_rng(seed)

    elo_diff = home_elo - away_elo
    goal_ratio = 10 ** (elo_diff / 400)
    total_goals = 2 * overall_avg

    lambda_home = max(total_goals * goal_ratio / (1 + goal_ratio), 0.05)
    lambda_away = max(total_goals / (1 + goal_ratio), 0.05)

    hg = rng.poisson(lambda_home, n_sim)
    ag = rng.poisson(lambda_away, n_sim)

    unique, counts = np.unique(np.column_stack([hg, ag]), axis=0, return_counts=True)
    probs = counts / n_sim
    idx = np.argsort(-probs)

    scores = [(f"{int(unique[i][0])}-{int(unique[i][1])}", float(probs[i])) for i in idx[:10]]
    return scores, float(lambda_home), float(lambda_away)


def main():
    parser = argparse.ArgumentParser(description="Stacking Model for WC 2026 Predictions")
    parser.add_argument("--nlp", action="store_true", help="Include NLP features (news + YouTube sentiment)")
    parser.add_argument("--max-date", type=str, default=None,
                        help="Limitar datos de entrenamiento hasta esta fecha (YYYY-MM-DD). "
                             "Default: ayer. Ej: --max-date 2026-06-27 para excluir 16avos")
    parser.add_argument("--8avos", action="store_true", help="Predecir octavos de final (usa data/8avos_matches.json)")
    args = parser.parse_args()
    use_nlp = args.nlp
    es_8avos = getattr(args, "8avos", False)

    global MAX_DATE
    MAX_DATE = args.max_date if args.max_date else (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    ronda = "8avos" if es_8avos else "16avos"
    title = "STACKING MODEL + NLP" if use_nlp else "STACKING MODEL"
    print("=" * 60)
    print(f"{title} - {ronda} World Cup 2026")
    print("=" * 60)

    nlp_data = None
    if use_nlp:
        import os
        if not os.path.exists(NLP_FEATURES):
            print("\n  ERROR: NLP features file not found.")
            print("  ************************************************************")
            print("  Para usar --nlp necesitas generar los datos primero.")
            print()
            print("  Instala las dependencias adicionales:")
            print("    uv add sentence-transformers textblob")
            print("    uv add feedparser trafilatura google-api-python-client")
            print()
            print("  Luego ejecuta los scripts en orden:")
            print("    uv run python scripts/fetch_team_news.py")
            print("    uv run python scripts/fetch_team_comments.py")
            print("    uv run python scripts/compute_team_embeddings.py")
            print("  ************************************************************")
            import sys
            sys.exit(22)

        print("\n[0] Loading NLP data...")
        nlp_data = load_nlp_data()
        print(f"   {len(nlp_data)} teams with NLP features loaded")

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

    overall_avg_goals = df.select(
        (pl.col("home_score").mean() + pl.col("away_score").mean()) / 2
    ).item()
    print(f"   Overall avg goals per match: {overall_avg_goals:.3f}")

    print("\n[2] Building feature vectors...")
    static_elo = load_static_elo()
    print(f"   Static ELO: {len(static_elo)}")
    elo_lookup = load_elo_history()
    print(f"   {len(elo_lookup)} teams with ELO history")
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

    n = len(dates_arr)
    split_idx = int(n * 0.8)
    train_idx = np.arange(split_idx)
    val_idx = np.arange(split_idx, n)

    train_date_cutoff = dates_arr[split_idx - 1].strftime("%Y-%m-%d")
    val_date_range = f"{dates_arr[split_idx].strftime('%Y-%m-%d')} \u2192 {dates_arr[-1].strftime('%Y-%m-%d')}"
    print(f"\n[3] Training stacking model (temporal split)...")
    print(f"   Train: {n - len(val_idx)} matches (\u2192 {train_date_cutoff})")
    print(f"   Valid: {len(val_idx)} matches ({val_date_range})")

    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y[train_idx], y[val_idx]

    # Compute balanced class weights for XGBoost (no native class_weight)
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
        "svm": SVC(kernel="rbf", probability=True, random_state=42,
                   class_weight="balanced"),
    }

    print("   Training base models...")
    models["rf"].fit(X_tr_scaled, y_tr)
    models["xgb"].fit(X_tr_scaled, y_tr, sample_weight=sw_tr)
    models["svm"].fit(X_tr_scaled, y_tr)

    meta_val_base = np.zeros((len(y_val), 3 * 3))
    offset = 0
    for name, model in models.items():
        meta_val_base[:, offset:offset + 3] = model.predict_proba(X_val_scaled)
        offset += 3

    if use_nlp and NLP_FEATURE_COLS:
        meta_val_nlp = np.zeros((len(y_val), len(NLP_FEATURE_COLS)))
        meta_val = np.concatenate([meta_val_base, meta_val_nlp], axis=1)
    else:
        meta_val = meta_val_base

    meta = LogisticRegression(solver="lbfgs", max_iter=1000, C=0.1,
                               class_weight="balanced", random_state=42)
    meta.fit(meta_val, y_val)

    val_probs = meta.predict_proba(meta_val)
    val_acc = accuracy_score(y_val, np.argmax(val_probs, axis=1))
    val_ll = log_loss(y_val, val_probs)
    print(f"\n   Temporal validation \u2014 accuracy: {val_acc:.4f}, log_loss: {val_ll:.4f}")

    print("\n   Retraining base models on full data...")
    X_full_scaled = scaler.fit_transform(X)
    full_weights = compute_class_weight("balanced", classes=classes, y=y)
    sw_full = full_weights[y]
    models["rf"].fit(X_full_scaled, y)
    models["xgb"].fit(X_full_scaled, y, sample_weight=sw_full)
    models["svm"].fit(X_full_scaled, y)

    matches_file = "data/8avos_matches.json" if es_8avos else "data/knockout_matches.json"
    output_csv = f"data/{ronda}_predictions_nlp.csv" if use_nlp else f"data/{ronda}_predictions.csv"

    print(f"\n[4] Predicting {ronda} matchups...")
    with open(matches_file) as f:
        matches = json.load(f)

    predictions_rows = []

    for i, match in enumerate(matches, 1):
        fv, home_team, away_team, swapped = build_features_for_match(match, df, MAX_DATE, elo_lookup, static_elo, nlp_data)
        fv_df = pl.DataFrame([fv])
        fv_base = scaler.transform(fv_df.select(BASE_FEATURE_COLS).to_numpy())

        offset = 0
        meta_input_base = np.zeros((1, 9))

        nlp_str = ""
        if use_nlp and NLP_FEATURE_COLS:
            meta_input_nlp = fv_df.select(NLP_FEATURE_COLS).to_numpy()
            n_pc = sum(1 for k in fv if k.startswith("home_news_pc"))
            pc_str_h = "  ".join(f"PC{j+1}={fv.get(f'home_news_pc{j+1}',0):+.2f}" for j in range(min(n_pc, 3)))
            pc_str_a = "  ".join(f"PC{j+1}={fv.get(f'away_news_pc{j+1}',0):+.2f}" for j in range(min(n_pc, 3)))
            nlp_str = (
                f"\n   NLP: {home_team}  {pc_str_h}  ...  YT={fv.get('home_yt_sent',0):+.3f}"
                f"\n        {away_team}  {pc_str_a}  ...  YT={fv.get('away_yt_sent',0):+.3f}"
            )

        print(f"\n   {'\u2500' * 60}")
        print(f"   Match {i:02d}: {home_team} vs {away_team}")
        print(f"   {'\u2500' * 60}")
        if nlp_str:
            print(nlp_str)

        for name, model in models.items():
            proba = model.predict_proba(fv_base)[0]
            meta_input_base[0, offset:offset + 3] = proba
            offset += 3
            local_win = proba[0] * 100
            draw = proba[1] * 100
            away_win = proba[2] * 100
            print(f"   {name.upper():5s} | Local: {local_win:5.1f}%  Empate: {draw:5.1f}%  Visitante: {away_win:5.1f}%")

        if use_nlp and NLP_FEATURE_COLS:
            meta_input = np.concatenate([meta_input_base, meta_input_nlp], axis=1)
        else:
            meta_input = meta_input_base

        final_proba = meta.predict_proba(meta_input)[0]
        local_w = final_proba[0] * 100
        draw_p = final_proba[1] * 100
        away_w = final_proba[2] * 100
        print(f"   {'\u2500' * 60}")
        print(f"   STACKING  | Local: {local_w:5.1f}%  Empate: {draw_p:5.1f}%  Visitante: {away_w:5.1f}%")

        poisson_scores, lambda_home, lambda_away = predict_poisson_scores(
            fv["home_elo"], fv["away_elo"], overall_avg_goals,
        )
        print(f"\n   POISSON   | \u03bb local={lambda_home:.2f}  \u03bb visitante={lambda_away:.2f}")
        print(f"   {'\u2500' * 32}")
        for score, prob in poisson_scores[:5]:
            print(f"   {score:>5s} \u2192 {prob*100:5.2f}%")

        local_adv = final_proba[0] + final_proba[1] * 0.5
        away_adv = final_proba[2] + final_proba[1] * 0.5
        total_adv = local_adv + away_adv
        print(f"   {'\u2500' * 60}")
        print(f"   AVANCE    | {home_team}: {local_adv / total_adv * 100:.1f}%  |  {away_team}: {away_adv / total_adv * 100:.1f}%")

        # Store CSV in bracket order (match['local'] first) regardless of feature swap
        # When swapped: feature home = match['visitante'] (host), feature away = match['local']
        if swapped:
            csv_local = match["local"]
            csv_away = match["visitante"]
            csv_local_w = away_w      # feature away = match['local']'s roll stats
            csv_away_w = local_w      # feature home = match['visitante']'s roll stats
            csv_local_adv = away_adv / total_adv * 100
            csv_away_adv = local_adv / total_adv * 100
        else:
            csv_local = match["local"]
            csv_away = match["visitante"]
            csv_local_w = local_w
            csv_away_w = away_w
            csv_local_adv = local_adv / total_adv * 100
            csv_away_adv = away_adv / total_adv * 100

        predictions_rows.append({
            "match": f"{csv_local} vs {csv_away}",
            "local_win_pct": round(csv_local_w, 1),
            "draw_pct": round(draw_p, 1),
            "away_win_pct": round(csv_away_w, 1),
            "local_advance_pct": round(csv_local_adv, 1),
            "away_advance_pct": round(csv_away_adv, 1),
            "expected_goals_local": round(lambda_home, 2),
            "expected_goals_away": round(lambda_away, 2),
            "most_likely_score": poisson_scores[0][0] if poisson_scores else "",
            "most_likely_score_pct": round(poisson_scores[0][1] * 100, 1) if poisson_scores else 0,
        })

    print(f"\n   {'=' * 60}")
    pl.DataFrame(predictions_rows).write_csv(output_csv)
    print(f"   Predictions saved to {output_csv}")


if __name__ == "__main__":
    main()
