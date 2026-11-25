# =============================================================================
# features/onoff.py — Impact "on/off" : l'équipe est-elle meilleure avec lui ?
#
# Pour chaque joueur, on compare la différence d'xG de son équipe (xG pour
# moins xG contre) :
#   - quand il est SUR le terrain ("on") ;
#   - quand il n'y est PAS, mais qu'il figure sur la feuille de match
#     (remplaçant pas encore entré, ou déjà sorti) ("off").
# Un joueur dont l'équipe domine nettement plus quand il joue a un impact
# collectif que ses statistiques individuelles ne montrent pas toujours
# (pressing, placement, organisation).
#
# Sources : FotMob (minute et xG de chaque tir ; titulaires, minutes d'entrée et
# de sortie, minutes jouées). Les buts contre son camp sont exclus (pas de xG).
#
# Sortie (une ligne par joueur Transfermarkt et par trimestre) :
#   oo_on_min, oo_on_xgf, oo_on_xga    minutes et xG pour / contre quand il joue
#   oo_team_min, oo_team_xgf, oo_team_xga   la même chose pour les matchs entiers
#                                      où il figure sur la feuille de match
# Le "off" s'obtient par différence (équipe - on).
# =============================================================================

import numpy as np
import pandas as pd

import config
from collectors import fotmob as fm
from collectors import matching

COLUMNS = ["oo_on_min", "oo_on_xgf", "oo_on_xga", "oo_team_min", "oo_team_xgf", "oo_team_xga"]
MATCH_MINUTES = 90


def season_onoff(matches, players, shots, lineups):
    # Calcule, pour une saison d'un championnat, les totaux on / équipe par joueur et par match.
    if matches.empty or lineups.empty:
        return pd.DataFrame()
    # 1. Intervalle passé sur le terrain [début, fin[ pour chaque joueur ayant joué.
    #    Seuls les joueurs de la FEUILLE DE MATCH comptent (titulaires et remplaçants) :
    #    les listes d'absents de FotMob ne sont pas fiables pour 2020-2023 (voir quarterly.py).
    squad = lineups[lineups.role.isin(["starters", "subs"])]
    squad = squad[["match_id", "team_id", "player_id", "role", "sub_in_minute"]].dropna(subset=["player_id"])
    minutes = players.set_index(["match_id", "player_id"])["minutes_played"]
    squad = squad.join(minutes, on=["match_id", "player_id"])
    start = np.where(squad.role == "starters", 0.0, squad.sub_in_minute.astype(float))
    played = squad.minutes_played.fillna(0) > 0
    start = np.where(played, start, np.nan)
    end = start + squad.minutes_played.fillna(0).to_numpy()
    # Un joueur encore présent à la 90e couvre aussi le temps additionnel (les tirs à 90+3 ont min = 90)
    squad["start"] = start
    squad["end"] = np.where(end >= MATCH_MINUTES, np.inf, end)
    squad["on_min"] = np.where(played, np.minimum(squad.minutes_played.fillna(0), MATCH_MINUTES), 0.0)

    # 2. xG de chaque tir, attribué à l'équipe qui tire ; buts contre son camp exclus
    s = shots[~shots.isOwnGoal.fillna(False).astype(bool)][["match_id", "teamId", "min", "expectedGoals"]]
    s = s.rename(columns={"teamId": "shot_team", "expectedGoals": "xg"}).dropna(subset=["xg"])

    # 3. Totaux de l'équipe par match (pour et contre)
    teams = pd.concat([matches[["match_id", "home_id"]].rename(columns={"home_id": "team_id"}),
                       matches[["match_id", "away_id"]].rename(columns={"away_id": "team_id"})])
    totals = s.groupby(["match_id", "shot_team"]).xg.sum()
    match_xg = s.groupby("match_id").xg.sum()
    teams["team_xgf"] = [totals.get((m, t), 0.0) for m, t in zip(teams.match_id, teams.team_id)]
    teams["team_xga"] = teams.match_id.map(match_xg).fillna(0) - teams.team_xgf
    squad = squad.merge(teams, on=["match_id", "team_id"], how="inner")

    # 4. xG pendant la présence du joueur : tirs dont la minute tombe dans [début, fin[
    on = squad[squad.on_min > 0]
    pairs = on[["match_id", "team_id", "player_id", "start", "end"]].merge(s, on="match_id")
    inside = (pairs["min"] >= pairs.start) & (pairs["min"] < pairs.end)
    pairs = pairs[inside]
    pairs["for"] = np.where(pairs.shot_team == pairs.team_id, pairs.xg, 0.0)
    pairs["against"] = np.where(pairs.shot_team != pairs.team_id, pairs.xg, 0.0)
    on_xg = pairs.groupby(["match_id", "player_id"])[["for", "against"]].sum()

    squad = squad.join(on_xg, on=["match_id", "player_id"])
    squad["oo_on_xgf"] = squad["for"].fillna(0)
    squad["oo_on_xga"] = squad["against"].fillna(0)
    squad = squad.drop_duplicates(["match_id", "player_id"])
    dates = pd.to_datetime(matches.set_index("match_id")["date"]).dt.tz_localize(None)
    squad["date"] = squad.match_id.map(dates)
    return squad.assign(oo_on_min=squad.on_min, oo_team_min=float(MATCH_MINUTES),
                        oo_team_xgf=squad.team_xgf, oo_team_xga=squad.team_xga)[
        ["player_id", "date"] + COLUMNS]


def build():
    # Agrège toutes les saisons FotMob par joueur Transfermarkt et trimestre.
    id_map = matching.fm_to_tm()
    frames = []
    for key in {**config.LEAGUES, **config.SECOND_DIVISIONS}:
        for season in range(fm.FIRST_SEASON_D1, config.CURRENT_SEASON + 1):
            try:
                matches, players, shots, lineups = fm.load_season(key, season)
            except Exception:
                continue
            if not shots.empty:
                frames.append(season_onoff(matches, players, shots, lineups))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    df["player_id"] = df.player_id.map(id_map)
    df = df.dropna(subset=["player_id", "date"])
    df["season"] = np.where(df.date.dt.month >= 8, df.date.dt.year, df.date.dt.year - 1)
    df["quarter"] = df.date.dt.month.map({m: q for q, ms in config.QUARTERS.items() for m in ms})
    return df.groupby(["player_id", "season", "quarter"])[COLUMNS].sum().reset_index()


def onoff_xgd_90(df, prior_minutes=600):
    # Différence d'xG par 90 minutes, "avec lui" moins "sans lui", rétrécie vers 0
    # quand les minutes sont peu nombreuses (un trimestre est bruité : une
    # différence mesurée sur 30 minutes "off" ne veut presque rien dire).
    off_min = df.oo_team_min - df.oo_on_min
    on_rate = (df.oo_on_xgf - df.oo_on_xga) / df.oo_on_min.where(df.oo_on_min > 0) * 90
    off_rate = ((df.oo_team_xgf - df.oo_on_xgf) - (df.oo_team_xga - df.oo_on_xga)) / off_min.where(off_min > 0) * 90
    # Précision de la différence ~ 1 / (1/on + 1/off) : rétrécissement de type bayésien
    effective = 1 / (1 / df.oo_on_min.where(df.oo_on_min > 0) + 1 / off_min.where(off_min > 0))
    return (on_rate - off_rate) * effective / (effective + prior_minutes)


if __name__ == "__main__":
    out = build()
    print(out.describe().T)
