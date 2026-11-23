# =============================================================================
# models/collective.py — Étape 10 : la contribution du joueur à son ÉQUIPE.
#
# Question : combien de POINTS au classement un joueur rapporte-t-il ?
# Méthode :
#   1. Pour chaque club et chaque trimestre, on calcule les points par match
#      en championnat, et le "niveau moyen de l'effectif" = moyenne des indices
#      de performance ajustés de ses joueurs, pondérée par leurs minutes.
#   2. Régression : points par match ~ niveau de l'effectif + force Elo du club.
#      Le coefficient b dit combien de points par match rapporte +1 de niveau
#      moyen de l'effectif.
#   3. Un joueur pèse environ 1/11e de l'effectif quand il joue tout le temps.
#      Sa contribution par match = b x son indice x sa part de temps de jeu / 11.
#      Sur une saison de 38 matchs, on obtient des "points apportés" par rapport
#      à un joueur moyen du top 5 (indice 0).
# =============================================================================

import pickle

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

import config
from collectors import transfermarkt as tm
from features import quarterly as qt
from models import performance as pf

MODEL_PATH = config.PROCESSED_DIR / "collective.pkl"
SEASON_GAMES = 38


def club_points():
    # Points par match de championnat, par club et par trimestre (Transfermarkt).
    codes = "', '".join(tm.BIG5_CODES)
    season, quarter = qt._quarter_sql("g.date")
    return tm.query(
        f"""
        SELECT c.club_id, {season} AS season, {quarter} AS quarter,
               AVG(CASE WHEN c.own_goals > c.opponent_goals THEN 3
                        WHEN c.own_goals = c.opponent_goals THEN 1 ELSE 0 END) AS ppg,
               COUNT(*) AS games
        FROM club_games c JOIN games g ON g.game_id = c.game_id
        WHERE g.competition_id IN ('{codes}')
        GROUP BY ALL
        """
    )


def fit(cutoff_t=None):
    q = pd.read_parquet(qt.OUT_PATH)[["player_id", "t", "season", "quarter", "club_id", "minutes_league", "own_elo"]]
    p = pd.read_parquet(pf.OUT_PATH)[["player_id", "t", "perf_adj"]]
    df = q.merge(p, on=["player_id", "t"])
    if cutoff_t is not None:
        df = df[df.t <= cutoff_t]
    df = df[df.perf_adj.notna() & (df.minutes_league > 0)]
    df["w_perf"] = df.perf_adj * df.minutes_league
    squad = df.groupby(["club_id", "season", "quarter"]).agg(
        w_perf=("w_perf", "sum"), minutes=("minutes_league", "sum"), elo=("own_elo", "mean")).reset_index()
    squad["squad_perf"] = squad.w_perf / squad.minutes
    data = squad.merge(club_points(), on=["club_id", "season", "quarter"])
    data = data[(data.games >= 4) & data.elo.notna()]
    data["elo_c"] = (data.elo - 1600) / 100
    model = smf.wls("ppg ~ squad_perf + elo_c", data, weights=data.games).fit()
    return {"beta": float(model.params["squad_perf"]), "model": model, "n": len(data)}


def points_added(perf, share, beta):
    # Points apportés sur une saison, par rapport à un joueur moyen (indice 0).
    return beta * perf * share / 11 * SEASON_GAMES


def build():
    result = fit()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({k: v for k, v in result.items() if k != "model"}, f)
    return result


def load():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    r = build()
    print(r["model"].summary())
