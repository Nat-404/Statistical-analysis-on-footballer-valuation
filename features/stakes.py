# =============================================================================
# features/stakes.py — Indice d'enjeu de chaque match (0 = sans enjeu, 1 = enjeu maximal).
#
# Règles (simples et transparentes, ajustables) :
#   Coupes d'Europe : phase finale = 1.0 ; phase de groupes / ligue = 0.6 ; qualifications = 0.3
#   Coupes nationales : finale = 1.0 ; demi-finale = 0.8 ; quart = 0.5 ; autres tours = 0.2
#   Championnat : dans les 8 dernières journées, un club en course pour le titre
#     (1er-2e), l'Europe (3e-7e) ou le maintien (4 derniers) joue un match à
#     enjeu = 0.8 ; sinon fin de saison = 0.3 ; reste de la saison = 0.1.
#     Le classement est celui indiqué par Transfermarkt au moment du match.
# Un match "à enjeu" est un match dont l'indice est >= 0.7.
# =============================================================================

import pandas as pd

import config
from collectors import transfermarkt as tm

OUT_PATH = config.PROCESSED_DIR / "stakes.parquet"
HIGH_STAKES = 0.7


def build():
    games = tm.query(
        """
        SELECT CAST(game_id AS INTEGER) AS game_id, competition_id, competition_type, season, round,
               home_club_id, away_club_id, home_club_position, away_club_position
        FROM games
        """
    )
    r = games["round"].fillna("").str.lower()

    # --- Championnats : numéro de journée et nombre total de journées de la saison
    games["matchday"] = pd.to_numeric(r.str.extract(r"^(\d+)\.")[0], errors="coerce")
    games["n_matchdays"] = games.groupby(["competition_id", "season"]).matchday.transform("max")
    games["n_clubs"] = games.groupby(["competition_id", "season"]).home_club_id.transform("nunique")

    def league_stakes(position, g):
        # Enjeu d'un match de championnat pour UN des deux clubs
        late = g.matchday > g.n_matchdays - 8
        racing = (position <= 7) | (position > g.n_clubs - 4)
        return pd.Series(0.1, index=g.index).mask(late, 0.3).mask(late & racing, 0.8)

    league = games.competition_type == "domestic_league"
    home = league_stakes(games.home_club_position, games)
    away = league_stakes(games.away_club_position, games)

    # --- Coupes d'Europe
    europe = games.competition_type == "international_cup"
    qualif = r.str.contains("qualif|play-off|playoff|preliminary")
    group = r.str.contains("group|league phase|league stage|matchday")
    europe_value = pd.Series(1.0, index=games.index).mask(group, 0.6).mask(qualif, 0.3)

    # --- Coupes nationales
    cup = games.competition_type == "domestic_cup"
    cup_value = (pd.Series(0.2, index=games.index)
                 .mask(r.str.contains("quarter"), 0.5)
                 .mask(r.str.contains("semi"), 0.8)
                 .mask(r.str.fullmatch("final"), 1.0))

    games["stakes_home"] = home.where(league, europe_value.where(europe, cup_value.where(cup, 0.1)))
    games["stakes_away"] = away.where(league, europe_value.where(europe, cup_value.where(cup, 0.1)))
    out = games[["game_id", "home_club_id", "away_club_id", "stakes_home", "stakes_away"]]
    out.to_parquet(OUT_PATH)
    return out


if __name__ == "__main__":
    build()
