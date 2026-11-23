# =============================================================================
# collectors/matching.py — Étape 4 : relier les sources entre elles.
#
# Problème : Transfermarkt et FotMob ont chacun leurs propres identifiants.
# "Kylian Mbappé" est le n° 342229 chez l'un et le n° 701154 chez l'autre.
# Comparer les noms seuls est risqué (homonymes, accents, surnoms : "Rodri"
# s'appelle "Rodrigo Hernández" ailleurs).
#
# Méthode, de la plus sûre à la moins sûre :
#   1. On relie d'abord les MATCHS : même date (à 1 jour près, à cause des
#      fuseaux horaires), même championnat, même score.
#   2. Dans chaque match relié, on relie les CLUBS (domicile <-> domicile).
#   3. Dans chaque club d'un match relié, on relie les JOUEURS en combinant
#      la ressemblance des noms ET les minutes jouées (un joueur entré à la
#      78e a 12 minutes dans les deux sources). C'est une "affectation
#      optimale" : chaque joueur d'un côté reçoit au plus un joueur de l'autre.
#   4. Un joueur apparaît dans des dizaines de matchs : chaque match est un
#      "vote". On garde l'association qui recueille le plus de votes.
#   5. Pour les joueurs sans aucun match commun (2es divisions, saison 2026/27),
#      on se rabat sur nom + âge, avec une confiance plus faible.
# =============================================================================

import unicodedata

import numpy as np
import pandas as pd
from rapidfuzz import fuzz                          # comparaison de textes, rapide
from scipy.optimize import linear_sum_assignment    # "affectation optimale" (méthode hongroise)

import config
from collectors import fotmob as fm
from collectors import transfermarkt as tm

PLAYER_MAP_PATH = config.PROCESSED_DIR / "player_map.parquet"
CLUB_MAP_PATH = config.PROCESSED_DIR / "club_map.parquet"


def normalize(name):
    # "Kylian Mbappé-Lottin" -> "kylian mbappe lottin" : sans accents, minuscules, sans ponctuation.
    text = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalnum() else " " for c in text).split())


def name_similarity(a, b):
    # Ressemblance entre deux noms, de 0 (rien à voir) à 1 (identiques).
    # token_set_ratio ignore l'ordre et les mots en trop ("Vinicius Junior" ~ "Vinicius Jose Junior") ;
    # partial_ratio repère un nom contenu dans l'autre.
    a, b = normalize(a), normalize(b)
    return max(fuzz.token_set_ratio(a, b), fuzz.partial_ratio(a, b)) / 100


# -----------------------------------------------------------------------------
# 1. Relier les matchs
# -----------------------------------------------------------------------------
def tm_games(league_key, season):
    # Matchs Transfermarkt d'un championnat / saison.
    code = config.LEAGUES[league_key]["transfermarkt"]
    return tm.query(
        """
        SELECT CAST(game_id AS INTEGER) AS game_id, date, home_club_id, away_club_id,
               home_club_name, away_club_name, home_club_goals, away_club_goals
        FROM games WHERE competition_id = ? AND season = ?
        """,
        [code, str(season)],
    )


def match_games(fm_matches, tm_matches):
    # Pour chaque match FotMob, trouve le match Transfermarkt correspondant.
    fmx = fm_matches.copy()
    fmx["day"] = pd.to_datetime(fmx["date"]).dt.tz_localize(None).dt.normalize()
    goals = fmx["score"].str.split(" - ", expand=True)
    fmx["hg"], fmx["ag"] = pd.to_numeric(goals[0]), pd.to_numeric(goals[1])
    tmx = tm_matches.assign(day=pd.to_datetime(tm_matches["date"]))

    pairs = []
    for m in fmx.itertuples():
        # Candidats : même score, date à +/- 1 jour
        cand = tmx[(tmx.home_club_goals == m.hg) & (tmx.away_club_goals == m.ag)
                   & ((tmx.day - m.day).abs() <= pd.Timedelta(days=1))]
        if cand.empty:
            continue
        # S'il y en a plusieurs (deux 1-1 le même jour), on départage par les noms d'équipes
        sims = [name_similarity(m.home, c.home_club_name) + name_similarity(m.away, c.away_club_name)
                for c in cand.itertuples()]
        best = cand.iloc[int(np.argmax(sims))]
        if max(sims) >= 1.0:   # au moins une ressemblance moyenne de 50 % par équipe
            pairs.append({"fm_match_id": m.match_id, "tm_game_id": best.game_id,
                          "fm_home": m.home_id, "fm_away": m.away_id,
                          "tm_home": best.home_club_id, "tm_away": best.away_club_id})
    return pd.DataFrame(pairs)


# -----------------------------------------------------------------------------
# 2-3. Relier les joueurs à l'intérieur de chaque match relié
# -----------------------------------------------------------------------------
def tm_appearances(game_ids):
    # Joueurs Transfermarkt ayant joué ces matchs (avec leurs minutes).
    ids = ", ".join(str(int(g)) for g in game_ids)
    return tm.query(
        f"""
        SELECT game_id, player_id, player_club_id, player_name, minutes_played, goals
        FROM appearances WHERE game_id IN ({ids})
        """
    )


def match_players_in_game(fm_side, tm_side):
    # fm_side / tm_side : les joueurs d'UNE équipe dans UN match, dans chaque source.
    # On construit une matrice de "coût" : faible coût = forte probabilité que ce
    # soit le même joueur. Puis l'algorithme hongrois trouve l'affectation
    # globale de coût minimal (chaque joueur utilisé au plus une fois).
    if fm_side.empty or tm_side.empty:
        return []
    sim = np.array([[name_similarity(f, t) for t in tm_side.player_name] for f in fm_side.player_name])
    minute_gap = np.abs(fm_side.minutes_played.fillna(0).to_numpy()[:, None]
                        - tm_side.minutes_played.fillna(0).to_numpy()[None, :])
    # Score combiné : 70 % le nom, 30 % la cohérence des minutes (écart de 0 à 90 min)
    score = 0.7 * sim + 0.3 * (1 - np.minimum(minute_gap, 90) / 90)
    rows, cols = linear_sum_assignment(-score)       # on maximise le score
    out = []
    for r, c in zip(rows, cols):
        # Garde-fous : nom assez proche, OU nom moyennement proche avec minutes identiques
        if sim[r, c] >= 0.75 or (sim[r, c] >= 0.4 and minute_gap[r, c] <= 2):
            out.append((int(fm_side.player_id.iloc[r]), int(tm_side.player_id.iloc[c]), float(score[r, c])))
    return out


def match_season(league_key, season):
    # Relie matchs, clubs et joueurs pour un championnat / saison.
    # Renvoie les "votes" joueurs (une ligne par association trouvée dans un match)
    # et les votes clubs.
    fm_matches, fm_players, _, _ = fm.load_season(league_key, season)
    if fm_matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    games = match_games(fm_matches, tm_games(league_key, season))
    if games.empty:
        return pd.DataFrame(), pd.DataFrame()
    apps = tm_appearances(games.tm_game_id)

    player_votes, club_votes = [], []
    for g in games.itertuples():
        for fm_team, tm_club in ((g.fm_home, g.tm_home), (g.fm_away, g.tm_away)):
            club_votes.append({"fm_team_id": fm_team, "tm_club_id": tm_club})
            f = fm_players[(fm_players.match_id == g.fm_match_id) & (fm_players.team_id == fm_team)
                           & (fm_players.minutes_played.fillna(0) > 0)]
            t = apps[(apps.game_id == g.tm_game_id) & (apps.player_club_id == tm_club)]
            for fm_id, tm_id, s in match_players_in_game(f, t):
                player_votes.append({"fm_player_id": fm_id, "tm_player_id": tm_id, "score": s})
    return pd.DataFrame(player_votes), pd.DataFrame(club_votes)


# -----------------------------------------------------------------------------
# 4. Dépouillement des votes
# -----------------------------------------------------------------------------
def resolve(votes, left, right):
    # Pour chaque identifiant de gauche, garde l'identifiant de droite le plus voté.
    # "confidence" = part des votes obtenus par ce choix (1.0 = unanimité).
    counts = votes.groupby([left, right]).size().rename("votes").reset_index()
    totals = counts.groupby(left).votes.transform("sum")
    counts["confidence"] = counts.votes / totals
    best = counts.sort_values("votes", ascending=False).drop_duplicates(left)
    # Un identifiant de droite revendiqué par deux de gauche : on garde le plus voté
    return best.drop_duplicates(right).reset_index(drop=True)


# -----------------------------------------------------------------------------
# 5. Repli : nom + âge (joueurs sans match commun)
# -----------------------------------------------------------------------------
def fallback_by_name_age(fm_lineups, already_matched):
    # Pour les joueurs FotMob non reliés : on cherche dans la fiche Transfermarkt
    # un joueur au nom très proche (>= 90 %) ET au même âge (+/- 1 an).
    # On n'accepte que s'il y a UN SEUL candidat : en cas de doute, on ne relie pas.
    todo = (fm_lineups[~fm_lineups.player_id.isin(already_matched) & fm_lineups.age.notna()]
            .drop_duplicates("player_id")[["player_id", "player_name", "age"]])
    if todo.empty:
        return pd.DataFrame()
    players = tm.query("SELECT player_id, name, date_of_birth FROM players WHERE date_of_birth IS NOT NULL")
    players["norm"] = players.name.map(normalize)
    players["birth_year"] = pd.to_datetime(players.date_of_birth).dt.year
    # On pré-filtre par la première lettre du nom de famille pour aller plus vite
    players["key"] = players.norm.str.split().str[-1].str[:1]

    rows = []
    reference_year = config.CURRENT_SEASON + 1   # âge FotMob = âge actuel environ
    for p in todo.itertuples():
        norm = normalize(p.player_name)
        pool = players[players.key == norm.split()[-1][:1]] if norm else players
        pool = pool[(reference_year - pool.birth_year - p.age).abs() <= 1]
        sims = pool.norm.map(lambda n: max(fuzz.token_set_ratio(norm, n), fuzz.ratio(norm, n)) / 100)
        good = pool[sims >= 0.9]
        if len(good) == 1:
            rows.append({"fm_player_id": p.player_id, "tm_player_id": int(good.player_id.iloc[0]),
                         "votes": 0, "confidence": 0.5, "method": "nom+age"})
    return pd.DataFrame(rows)


def build(seasons=None):
    # Construit les tables de correspondance joueurs et clubs.
    seasons = seasons or range(fm.FIRST_SEASON_D1, config.CURRENT_SEASON)
    all_player_votes, all_club_votes = [], []
    for season in seasons:
        for key in config.LEAGUES:
            pv, cv = match_season(key, season)
            all_player_votes.append(pv)
            all_club_votes.append(cv)
            print(f"{key} {season}: {len(pv)} votes joueurs")
    player_votes = pd.concat(all_player_votes, ignore_index=True)
    club_votes = pd.concat(all_club_votes, ignore_index=True)

    players = resolve(player_votes, "fm_player_id", "tm_player_id").assign(method="matchs")
    clubs = resolve(club_votes, "fm_team_id", "tm_club_id")

    # Repli pour les joueurs vus uniquement en 2es divisions ou en 2026/27
    lineups = []
    for key in {**config.LEAGUES, **config.SECOND_DIVISIONS}:
        for season in range(fm.FIRST_SEASON_D2, config.CURRENT_SEASON + 1):
            lineups.append(fm.load_season(key, season)[3])
    lineups = pd.concat([l for l in lineups if not l.empty], ignore_index=True)
    extra = fallback_by_name_age(lineups, set(players.fm_player_id))
    extra = extra[~extra.tm_player_id.isin(players.tm_player_id)] if not extra.empty else extra
    players = pd.concat([players, extra], ignore_index=True)

    players.to_parquet(PLAYER_MAP_PATH)
    clubs.to_parquet(CLUB_MAP_PATH)
    return players, clubs


def fm_to_tm():
    # Dictionnaire {id FotMob: id Transfermarkt}, pour les autres modules.
    m = pd.read_parquet(PLAYER_MAP_PATH)
    return dict(zip(m.fm_player_id, m.tm_player_id))


if __name__ == "__main__":
    build()
