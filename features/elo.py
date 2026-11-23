# =============================================================================
# features/elo.py — Étape 3a : force des équipes (classement Elo maison).
#
# L'API de ClubElo ne répond plus : on calcule notre propre Elo à partir des
# ~89 000 matchs de Transfermarkt (championnats, coupes, coupes d'Europe).
#
# LE PRINCIPE DE L'ELO
#   Chaque club a une note (ex. 1700). Avant un match, la différence de notes
#   donne une probabilité de victoire attendue. Après le match, chaque club
#   gagne ou perd des points selon la SURPRISE :
#       nouvelle note = ancienne note + K x (résultat réel - résultat attendu)
#   Battre un club bien plus fort rapporte beaucoup ; battre un club faible,
#   presque rien. Les coupes d'Europe relient les championnats entre eux :
#   c'est ce qui permet de comparer un club français et un club allemand.
# =============================================================================

import numpy as np
import pandas as pd

import config
from collectors import transfermarkt as tm

OUT_PATH = config.PROCESSED_DIR / "elo.parquet"
INITIAL_RATING = 1500           # note de départ d'un club jamais vu (avant calibrage par championnat)


BURN_IN_END = pd.Timestamp("2014-07-01")   # avant cette date : période de "rodage" des notes


def load_matches():
    # Tous les matchs de clubs avec un score, du plus ancien au plus récent.
    # On exclut les matchs de sélections nationales (ce ne sont pas des clubs).
    return tm.query(
        """
        SELECT CAST(game_id AS INTEGER) AS game_id, date, season, competition_id, competition_type,
               home_club_id AS home_id, away_club_id AS away_id,
               home_club_goals AS home_goals, away_club_goals AS away_goals
        FROM games
        WHERE competition_type IN ('domestic_league', 'domestic_cup', 'international_cup', 'other')
          AND home_club_goals IS NOT NULL
        ORDER BY date, game_id
        """
    )


def league_seasons(matches):
    # Composition de chaque championnat, saison par saison :
    # {(championnat, saison): ensemble des clubs}. Sert à repérer promus et relégués.
    league = matches[matches.competition_type == "domestic_league"]
    members = {}
    for (comp, season), grp in league.groupby(["competition_id", "season"]):
        members[(comp, season)] = set(grp.home_id) | set(grp.away_id)
    return members


def goal_multiplier(goal_diff):
    # Un 4-0 en dit plus qu'un 1-0 : on amplifie la mise à jour selon l'écart.
    # Formule standard du "World Football Elo" : x1 pour 0-1 but d'écart,
    # x1.5 pour 2 buts, puis (11 + écart) / 8 au-delà.
    d = abs(goal_diff)
    if d <= 1:
        return 1.0
    if d == 2:
        return 1.5
    return (11 + d) / 8


def run_elo(matches, k=20.0, home_advantage=65.0, scale=400.0, league_start=None):
    # Calcule l'Elo match après match.
    # Renvoie, pour chaque match, les notes AVANT le match (c'est ce qui compte
    # pour juger la difficulté d'un adversaire) et la victoire attendue.
    # league_start : note de départ des clubs de chaque championnat ({"GB1": 1650, ...}).
    league_start = league_start or {}
    members = league_seasons(matches)
    previous = _previous_seasons(members)       # {(champ, saison): saison précédente du même champ}
    ratings = {}                                # note actuelle de chaque club : {club_id: note}
    placed = set()                              # (club, champ, saison) déjà initialisés

    home_pre = np.full(len(matches), np.nan)
    away_pre = np.full(len(matches), np.nan)
    expected = np.full(len(matches), np.nan)

    # itertuples parcourt le tableau ligne par ligne, rapidement
    for i, m in enumerate(matches.itertuples(index=False)):
        if m.competition_type == "domestic_league":
            key = (m.competition_id, m.season)
            for club in (m.home_id, m.away_id):
                if (club, *key) not in placed:
                    placed.add((club, *key))
                    _place_club(club, key, previous, members, ratings, league_start)

        # Règle "pas de fuite de points" : si l'un des deux clubs n'a jamais joué
        # dans un championnat de notre base (petit club de coupe, club d'un pays
        # non couvert), on ne met pas les notes à jour.
        if m.home_id not in ratings or m.away_id not in ratings:
            continue

        r_home, r_away = ratings[m.home_id], ratings[m.away_id]

        # Probabilité "attendue" que l'équipe à domicile gagne (formule logistique de l'Elo).
        # L'avantage du terrain s'ajoute à la note de l'équipe qui reçoit.
        # "scale" règle la sensibilité : avec 400, 400 points d'écart = 91 % attendus ;
        # une échelle plus grande rend les prédictions moins catégoriques.
        e_home = 1 / (1 + 10 ** (-(r_home + home_advantage - r_away) / scale))

        # Résultat réel : 1 = victoire, 0.5 = nul, 0 = défaite (vu de l'équipe à domicile)
        diff = m.home_goals - m.away_goals
        s_home = 1.0 if diff > 0 else 0.5 if diff == 0 else 0.0

        # Mise à jour : ce que gagne l'un, l'autre le perd (somme nulle)
        delta = k * goal_multiplier(diff) * (s_home - e_home)
        ratings[m.home_id] = r_home + delta
        ratings[m.away_id] = r_away - delta

        home_pre[i], away_pre[i], expected[i] = r_home, r_away, e_home

    result = matches.copy()
    result["elo_home_pre"] = home_pre
    result["elo_away_pre"] = away_pre
    result["expected_home"] = expected
    return result


def _previous_seasons(members):
    # Pour chaque (championnat, saison), retrouve la saison précédente disponible.
    previous = {}
    by_league = {}
    for comp, season in sorted(members):
        by_league.setdefault(comp, []).append(season)
    for comp, seasons in by_league.items():
        for before, after in zip(seasons, seasons[1:]):
            previous[(comp, after)] = (comp, before)
    return previous


def _place_club(club, key, previous, members, ratings, league_start):
    # Donne sa note à un club qui apparaît dans un championnat pour une saison.
    prev = previous.get(key)
    if prev is None:
        # Première saison du championnat dans nos données : note de départ du championnat
        ratings.setdefault(club, league_start.get(key[0], INITIAL_RATING))
        return
    if club in members[prev]:
        return  # il était déjà dans ce championnat la saison d'avant : il garde sa note

    # C'est un PROMU (ou un club qui revient). Règle de conservation des points :
    # il hérite de la note moyenne des clubs qui ont quitté ce championnat
    # (les relégués). Ainsi le total de points du championnat ne change pas,
    # et seuls les matchs européens peuvent faire monter ou baisser un championnat.
    departed = [ratings[c] for c in members[prev] - members[key] if c in ratings]
    if departed:
        ratings[club] = float(np.mean(departed))
    else:
        current = [ratings[c] for c in members[key] if c in ratings and c != club]
        ratings[club] = float(np.percentile(current, 20)) if current else INITIAL_RATING


def estimate_league_start(matches, k, home_advantage, scale, iterations=4):
    # Problème : au tout début (2012), tous les championnats partent à 1500.
    # Il faudrait des années de matchs européens pour que la Premier League
    # se détache de l'Écosse. Solution : on rejoue plusieurs fois la période
    # de rodage (2012-2014) ; à chaque passage, chaque championnat repart avec
    # le niveau moyen qu'il avait atteint à la fin du passage précédent.
    # On n'utilise QUE des matchs d'avant juillet 2014 : aucune information du futur.
    burn_in = matches[matches.date < BURN_IN_END].reset_index(drop=True)
    start = {}
    for _ in range(iterations):
        res = run_elo(burn_in, k, home_advantage, scale, start)
        final = _final_ratings(res)
        last_league = (
            burn_in[burn_in.competition_type == "domestic_league"]
            .melt(id_vars=["date", "competition_id"], value_vars=["home_id", "away_id"], value_name="club_id")
            .sort_values("date").groupby("club_id").competition_id.last()
        )
        means = final.groupby(last_league.reindex(final.index)).mean()
        start = (means - means.mean() + INITIAL_RATING).to_dict()   # recentré autour de 1500
    return start


def _final_ratings(result):
    # Note de chaque club avant son dernier match (à un match près : suffisant ici).
    r = result.dropna(subset=["expected_home"])
    long = pd.concat([
        pd.DataFrame({"date": r.date, "game_id": r.game_id, "club_id": r[f"{side}_id"], "elo": r[f"elo_{side}_pre"]})
        for side in ("home", "away")
    ]).sort_values(["date", "game_id"])
    return long.groupby("club_id").elo.last()


def score(result, start="2014-07-01"):
    # Qualité des prédictions : erreur quadratique moyenne entre le résultat
    # attendu et le résultat réel (plus c'est bas, mieux c'est).
    # On ignore les 2 premières saisons (le temps que les notes se stabilisent).
    # Les matchs non notés (club hors base) sont exclus.
    r = result[result["date"] >= pd.Timestamp(start)].dropna(subset=["expected_home"])
    diff = r.home_goals - r.away_goals
    s = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    return float(np.mean((s - r.expected_home) ** 2))


def calibrate(matches):
    # On teste plusieurs combinaisons de K (vitesse de réaction), d'avantage
    # du terrain et d'échelle, et on garde celle qui prédit le mieux les résultats.
    rows = []
    for k in [20, 30, 40]:
        for h in [65, 80]:
            for scale in [500, 600, 700, 800, 900]:
                start = estimate_league_start(matches, k, h, scale)
                rows.append({"k": k, "home_advantage": h, "scale": scale,
                             "erreur": score(run_elo(matches, k, h, scale, start))})
    return pd.DataFrame(rows).sort_values("erreur")


def build():
    # Calibre, calcule l'Elo final et l'enregistre.
    matches = load_matches()
    grid = calibrate(matches)
    best = grid.iloc[0]
    print(grid.head(5).to_string(index=False))
    start = estimate_league_start(matches, best.k, best.home_advantage, best.scale)
    result = run_elo(matches, best.k, best.home_advantage, best.scale, start)
    result.to_parquet(OUT_PATH)
    return result, best


def history():
    # Format "long" : une ligne par club et par match, avec sa note avant le match.
    # Pratique pour retrouver la note d'un club à n'importe quelle date.
    elo = pd.read_parquet(OUT_PATH)
    home = elo[["date", "home_id", "elo_home_pre"]].set_axis(["date", "club_id", "elo"], axis=1)
    away = elo[["date", "away_id", "elo_away_pre"]].set_axis(["date", "club_id", "elo"], axis=1)
    return pd.concat([home, away]).sort_values(["club_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    build()
