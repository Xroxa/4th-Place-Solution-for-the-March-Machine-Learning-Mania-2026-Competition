#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings("ignore")

SEED = 42
DEFAULT_DATA_DIR = "./data/march-machine-learning-mania-2026"


def build_team_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["PointDiff"] = df["WScore"] - df["LScore"]
    df["TotalPoints"] = df["WScore"] + df["LScore"]

    df["W_Poss"] = df["WFGA"] - df["WOR"] + df["WTO"] + 0.44 * df["WFTA"]
    df["L_Poss"] = df["LFGA"] - df["LOR"] + df["LTO"] + 0.44 * df["LFTA"]

    df["W_OE"] = 100 * df["WScore"] / df["W_Poss"].clip(1)
    df["L_OE"] = 100 * df["LScore"] / df["L_Poss"].clip(1)

    df["W_TS"] = df["WScore"] / (2 * (df["WFGA"] + 0.44 * df["WFTA"])).clip(1)
    df["L_TS"] = df["LScore"] / (2 * (df["LFGA"] + 0.44 * df["LFTA"])).clip(1)

    total_reb = (df["WDR"] + df["WOR"] + df["LDR"] + df["LOR"]).clip(1)
    df["W_RebRate"] = (df["WDR"] + df["WOR"]) / total_reb
    df["L_RebRate"] = 1 - df["W_RebRate"]

    wins = (
        df.groupby(["Season", "WTeamID"])
        .agg(
            AvgScore=("WScore", "mean"),
            AvgDiff=("PointDiff", "mean"),
            FGM=("WFGM", "mean"),
            FGM3=("WFGM3", "mean"),
            FTA=("WFTA", "mean"),
            TO=("WTO", "mean"),
            Wins=("WScore", "count"),
            AvgOE=("W_OE", "mean"),
            AvgTS=("W_TS", "mean"),
            AvgRebRate=("W_RebRate", "mean"),
            AvgDE=("L_OE", "mean"),
        )
        .reset_index()
        .rename(columns={"WTeamID": "TeamID"})
    )

    losses = (
        df.groupby(["Season", "LTeamID"])
        .agg(
            AvgScoreAllowed=("LScore", "mean"),
            FGM_allowed=("LFGM", "mean"),
            FGM3_allowed=("LFGM3", "mean"),
            FTA_allowed=("LFTA", "mean"),
            TO_forced=("LTO", "mean"),
            Losses=("LScore", "count"),
            AvgTS_loss=("L_TS", "mean"),
            AvgRebRate_loss=("L_RebRate", "mean"),
        )
        .reset_index()
        .rename(columns={"LTeamID": "TeamID"})
    )

    stats = wins.merge(losses, on=["Season", "TeamID"], how="outer").fillna(0)
    stats["Games"] = stats["Wins"] + stats["Losses"]
    stats["WinRate"] = stats["Wins"] / stats["Games"].replace(0, 1)
    stats["NetRating"] = stats["AvgOE"] - stats["AvgDE"]
    stats["AvgTS_combined"] = (
        stats["AvgTS"] * stats["Wins"] + stats["AvgTS_loss"] * stats["Losses"]
    ) / stats["Games"].replace(0, 1)
    stats["AvgRebRate_combined"] = (
        stats["AvgRebRate"] * stats["Wins"] + stats["AvgRebRate_loss"] * stats["Losses"]
    ) / stats["Games"].replace(0, 1)
    return stats


def compute_elo(regular_df: pd.DataFrame, k: float = 20, home_adv: float = 100, init_elo: float = 1500) -> pd.DataFrame:
    elo_dict: dict[tuple[int, int], float] = {}
    current_elo: dict[int, float] = {}
    df = regular_df.sort_values(["Season", "DayNum"]).copy()

    for season in sorted(df["Season"].unique()):
        current_elo = {team_id: init_elo + 0.75 * (elo - init_elo) for team_id, elo in current_elo.items()}
        for _, row in df[df["Season"] == season].iterrows():
            winner = int(row["WTeamID"])
            loser = int(row["LTeamID"])
            loc = row.get("WLoc", "N")
            elo_w = current_elo.get(winner, init_elo)
            elo_l = current_elo.get(loser, init_elo)
            adj_elo_w = elo_w + (home_adv if loc == "H" else (-home_adv if loc == "A" else 0))
            expected_w = 1 / (1 + 10 ** ((elo_l - adj_elo_w) / 400))
            k_adj = k * np.log1p(abs(row["WScore"] - row["LScore"])) / 2
            current_elo[winner] = elo_w + k_adj * (1 - expected_w)
            current_elo[loser] = elo_l + k_adj * (0 - (1 - expected_w))
        for team_id, elo in current_elo.items():
            elo_dict[(season, team_id)] = float(elo)

    return pd.DataFrame(
        [{"Season": season, "TeamID": team_id, "Elo": elo} for (season, team_id), elo in elo_dict.items()]
    )


def build_context_features(regular_df: pd.DataFrame, conf_df: pd.DataFrame, elo_df: pd.DataFrame) -> pd.DataFrame:
    elo_conf = elo_df.merge(conf_df[["Season", "TeamID", "ConfAbbrev"]], on=["Season", "TeamID"], how="left")
    conf_strength = (
        elo_conf.groupby(["Season", "ConfAbbrev"])["Elo"]
        .mean()
        .reset_index()
        .rename(columns={"Elo": "Conf_Avg_Elo"})
    )
    team_conf_strength = (
        conf_df[["Season", "TeamID", "ConfAbbrev"]]
        .merge(conf_strength, on=["Season", "ConfAbbrev"], how="left")
        .drop(columns=["ConfAbbrev"])
    )

    late_season = regular_df[regular_df["DayNum"] > 100].copy()
    late_wins = (
        late_season.groupby(["Season", "WTeamID"])
        .size()
        .reset_index(name="LateWins")
        .rename(columns={"WTeamID": "TeamID"})
    )
    late_losses = (
        late_season.groupby(["Season", "LTeamID"])
        .size()
        .reset_index(name="LateLosses")
        .rename(columns={"LTeamID": "TeamID"})
    )
    late_stats = late_wins.merge(late_losses, on=["Season", "TeamID"], how="outer").fillna(0)
    late_stats["LateWinRate"] = late_stats["LateWins"] / (late_stats["LateWins"] + late_stats["LateLosses"]).clip(1)

    return team_conf_strength.merge(
        late_stats[["Season", "TeamID", "LateWinRate"]], on=["Season", "TeamID"], how="left"
    ).fillna(0)


def process_gender_pipeline(data_dir: Path, gender_prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    regular = pd.read_csv(data_dir / f"{gender_prefix}RegularSeasonDetailedResults.csv")
    tourney = pd.read_csv(data_dir / f"{gender_prefix}NCAATourneyDetailedResults.csv")
    seeds = pd.read_csv(data_dir / f"{gender_prefix}NCAATourneySeeds.csv")
    confs = pd.read_csv(data_dir / f"{gender_prefix}TeamConferences.csv")

    features = build_team_stats(regular)

    seeds["Seed"] = seeds["Seed"].astype(str).str.extract(r"(\d+)").astype(float)
    features = features.merge(seeds[["Season", "TeamID", "Seed"]], on=["Season", "TeamID"], how="left")
    features["Seed"] = features["Seed"].fillna(20)

    elo_df = compute_elo(regular)
    features = features.merge(elo_df, on=["Season", "TeamID"], how="left").fillna({"Elo": 1400})

    context_feats = build_context_features(regular, confs, elo_df)
    features = features.merge(context_feats, on=["Season", "TeamID"], how="left").fillna(0)

    if gender_prefix == "M":
        massey = pd.read_csv(data_dir / "MMasseyOrdinals.csv")
        season_max = massey.groupby("Season")["RankingDayNum"].max().reset_index().rename(columns={"RankingDayNum": "MaxDay"})
        massey = massey.merge(season_max, on="Season")
        massey_late = massey[massey["RankingDayNum"] >= massey["MaxDay"] - 30]
        massey_agg = (
            massey_late.groupby(["Season", "TeamID"])
            .agg(
                OrdinalRank_mean=("OrdinalRank", "mean"),
                OrdinalRank_min=("OrdinalRank", "min"),
                OrdinalRank_std=("OrdinalRank", "std"),
            )
            .reset_index()
            .fillna(0)
        )
        features = features.merge(massey_agg, on=["Season", "TeamID"], how="left")
        features["OrdinalRank_mean"] = features["OrdinalRank_mean"].fillna(175)
        features["OrdinalRank_min"] = features["OrdinalRank_min"].fillna(175)
        features["OrdinalRank_std"] = features["OrdinalRank_std"].fillna(0)
    else:
        features["OrdinalRank_mean"] = 175
        features["OrdinalRank_min"] = 175
        features["OrdinalRank_std"] = 0

    return features, tourney


def build_pairs(df: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = df[["Season", "WTeamID", "LTeamID"]].copy()
    df = df.merge(features, left_on=["Season", "WTeamID"], right_on=["Season", "TeamID"], how="left")
    df = df.merge(
        features,
        left_on=["Season", "LTeamID"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("_1", "_2"),
    )

    base_feature_cols = [
        "AvgScore",
        "AvgDiff",
        "FGM",
        "FGM3",
        "FTA",
        "TO",
        "AvgScoreAllowed",
        "WinRate",
        "Seed",
        "AvgOE",
        "AvgDE",
        "NetRating",
        "AvgTS_combined",
        "AvgRebRate_combined",
        "Elo",
        "Conf_Avg_Elo",
        "LateWinRate",
        "OrdinalRank_mean",
        "OrdinalRank_min",
        "OrdinalRank_std",
    ]
    feature_cols = [col for col in base_feature_cols if f"{col}_1" in df.columns and f"{col}_2" in df.columns]

    for col in feature_cols:
        df[col] = df[f"{col}_1"] - df[f"{col}_2"]

    df["elo_ratio"] = df["Elo_1"] / df["Elo_2"].clip(1)
    df["seed_product"] = df["Seed_1"] * df["Seed_2"]

    all_cols = feature_cols + ["elo_ratio", "seed_product"]
    x = df[all_cols].fillna(0)
    y = np.ones(len(df), dtype=float)
    return x, y, all_cols


def build_test_matrix(sample: pd.DataFrame, features_all: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    ids = sample["ID"].str.split("_", expand=True)
    ids.columns = ["Season", "Team1", "Team2"]
    ids = ids.astype(int)

    test = ids.merge(features_all, left_on=["Season", "Team1"], right_on=["Season", "TeamID"], how="left")
    test = test.merge(
        features_all,
        left_on=["Season", "Team2"],
        right_on=["Season", "TeamID"],
        how="left",
        suffixes=("_1", "_2"),
    )

    for col in [c for c in feature_cols if c not in ["elo_ratio", "seed_product"]]:
        test[col] = test[f"{col}_1"] - test[f"{col}_2"]

    test["elo_ratio"] = test["Elo_1"] / test["Elo_2"].clip(1)
    test["seed_product"] = test["Seed_1"] * test["Seed_2"]
    return test[feature_cols].fillna(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the final March Machine Learning Mania 2026 submission.")
    parser.add_argument("--data-dir", type=Path, default=Path(DEFAULT_DATA_DIR), help="Path to the Kaggle data folder.")
    parser.add_argument("--output", type=Path, default=Path("submission.csv"), help="Output CSV path.")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    features_m, tourney_m = process_gender_pipeline(data_dir, "M")
    features_w, tourney_w = process_gender_pipeline(data_dir, "W")
    features_all = pd.concat([features_m, features_w], ignore_index=True)
    tourney_all = pd.concat([tourney_m, tourney_w], ignore_index=True)

    x_pos, y_pos, feature_cols = build_pairs(tourney_all, features_all)
    x_neg = x_pos.copy()
    for col in [c for c in feature_cols if c not in ["elo_ratio", "seed_product"]]:
        x_neg[col] = -x_neg[col]
    x_neg["elo_ratio"] = 1 / x_neg["elo_ratio"].clip(0.01)
    y_neg = np.zeros(len(x_neg), dtype=float)

    x_train = pd.concat([x_pos, x_neg], ignore_index=True)
    y_train = np.concatenate([y_pos, y_neg])

    xgb_final = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.5,
        reg_lambda=1.5,
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        verbosity=0,
        n_jobs=8,
    )
    xgb_final.fit(x_train, y_train)

    lgbm_final = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=1.5,
        random_state=SEED,
        verbosity=-1,
        n_jobs=8,
    )
    lgbm_final.fit(x_train, y_train)

    sample = pd.read_csv(data_dir / "SampleSubmissionStage2.csv")
    x_test = build_test_matrix(sample, features_all, feature_cols)

    preds_xgb = xgb_final.predict_proba(x_test)[:, 1]
    preds_lgb = lgbm_final.predict_proba(x_test)[:, 1]
    final_preds = 0.65 * preds_xgb + 0.35 * preds_lgb

    sample["Pred"] = np.clip(final_preds, 0.025, 0.975)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Saved submission to {args.output}")
    print(f"Rows: {len(sample)} | Mean: {sample['Pred'].mean():.6f} | Std: {sample['Pred'].std():.6f}")


if __name__ == "__main__":
    main()
