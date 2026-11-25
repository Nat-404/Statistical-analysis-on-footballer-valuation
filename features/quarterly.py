# =============================================================================
# features/quarterly.py — Étape 5 : la table "joueur x trimestre".
#
# C'est la table centrale du projet : UNE ligne par joueur et par trimestre,
# avec tout ce qu'on sait de lui à ce moment-là. Tous les modèles suivants
# (facteur club, chaîne de Markov cachée, valeur...) apprennent sur cette table.
#
# Important : les trimestres SANS minute jouée sont conservés (blessure, mise
# à l'écart...). Ce sont justement eux qui alimentent les états "Blessé" et
# "Marginal" de la chaîne de Markov.
#
# Sources combinées :
#   Transfermarkt -> minutes, buts, passes D, titularisations, club, valeur, transferts
#   Elo maison    -> force du club et des adversaires
#   Stakes        -> matchs à enjeu
#   FotMob        -> xG, xA, passes, défense, gardiens, absences (quand disponible)
#   Wikipédia     -> popularité
# =============================================================================

import duckdb
import numpy as np
import pandas as pd

import config
from collectors import fotmob as fm
from collectors import matching
from collectors import transfermarkt as tm
from collectors import wikipedia as wk
from features import elo, stakes

OUT_PATH = config.PROCESSED_DIR / "quarterly.parquet"
ABSENCE_RELIABILITY_PATH = config.PROCESSED_DIR / "absence_reliability.parquet"
BIG_GAME_ELO_GAP = 100     # adversaire "beaucoup plus fort" : au moins 100 points Elo de plus


def _quarter_sql(col):
    # Traduit une date en (saison, trimestre) selon le découpage de config.QUARTERS.
    # Ex. : 15 mars 2024 -> saison 2023 (2023/24), trimestre 3.
    cases = " ".join(f"WHEN month({col}) IN ({', '.join(map(str, months))}) THEN {q}"
                     for q, months in config.QUARTERS.items())
    season = f"CASE WHEN month({col}) >= 8 THEN year({col}) ELSE year({col}) - 1 END"
    return season, f"CASE {cases} END"


def quarter_bounds(season, quarter):
    # Dates de début et de fin d'un trimestre.
    first_month = config.QUARTERS[quarter][0]
    year = season if first_month >= 8 else season + 1
    start = pd.Timestamp(year=year, month=first_month, day=1)
    return start, start + pd.DateOffset(months=3) - pd.Timedelta(days=1)


def _connect():
    # Une base DuckDB en mémoire qui "voit" à la fois la base Transfermarkt
    # et nos fichiers parquet (Elo, enjeux) : on peut tout croiser en SQL.
    con = duckdb.connect()
    con.execute(f"ATTACH '{tm.DUCKDB_PATH}' AS tmdb (READ_ONLY)")
    con.execute(f"CREATE VIEW elo AS SELECT * FROM '{elo.OUT_PATH}'")
    con.execute(f"CREATE VIEW stakes AS SELECT * FROM '{stakes.OUT_PATH}'")
    return con


# -----------------------------------------------------------------------------
# 1. Matchs joués (Transfermarkt) -> agrégats trimestriels
# -----------------------------------------------------------------------------
def tm_quarterly(con, player_ids):
    season, quarter = _quarter_sql("a.date")
    con.register("targets", pd.DataFrame({"player_id": player_ids}))
    return con.execute(
        f"""
        WITH apps AS (
            SELECT a.player_id, a.player_club_id AS club_id, a.date,
                   -- minutes en CLUB (les sélections nationales viennent de collectors/tournaments.py)
                   CASE WHEN g.competition_type = 'national_team_competition' THEN 0 ELSE a.minutes_played END AS minutes,
                   (g.competition_type = 'national_team_competition') AS is_national,
                   -- buts, passes D et cartons en CLUB uniquement (comme les minutes)
                   CASE WHEN g.competition_type = 'national_team_competition' THEN 0 ELSE a.goals END AS goals,
                   CASE WHEN g.competition_type = 'national_team_competition' THEN 0 ELSE a.assists END AS assists,
                   CASE WHEN g.competition_type = 'national_team_competition' THEN 0 ELSE a.yellow_cards END AS yellow_cards,
                   CASE WHEN g.competition_type = 'national_team_competition' THEN 0 ELSE a.red_cards END AS red_cards,
                   g.competition_id, g.competition_type,
                   {season} AS season, {quarter} AS quarter,
                   -- Elo du club du joueur et de l'adversaire (avant le match)
                   CASE WHEN a.player_club_id = g.home_club_id THEN e.elo_home_pre ELSE e.elo_away_pre END AS own_elo,
                   CASE WHEN a.player_club_id = g.home_club_id THEN e.elo_away_pre ELSE e.elo_home_pre END AS opp_elo,
                   CASE WHEN a.player_club_id = g.home_club_id THEN s.stakes_home ELSE s.stakes_away END AS stakes,
                   (l.type = 'starting_lineup') AS started
            FROM tmdb.appearances a
            JOIN targets t USING (player_id)
            JOIN tmdb.games g ON a.game_id = CAST(g.game_id AS INTEGER)
            LEFT JOIN elo e ON e.game_id = a.game_id
            LEFT JOIN stakes s ON s.game_id = a.game_id
            LEFT JOIN tmdb.game_lineups l ON l.game_id = a.game_id AND l.player_id = a.player_id
        )
        SELECT player_id, season, quarter,
               -- club principal du trimestre = celui où il a joué le plus de minutes
               arg_max(club_id, minutes)                                                  AS club_id,
               arg_max(competition_id, minutes) FILTER (WHERE competition_type = 'domestic_league') AS league_id,
               COUNT(*) FILTER (WHERE NOT is_national)                                    AS matches,
               SUM(CASE WHEN started AND NOT is_national THEN 1 ELSE 0 END)               AS starts,
               SUM(minutes)                                                               AS minutes,
               SUM(minutes) FILTER (WHERE competition_type = 'domestic_league')           AS minutes_league,
               SUM(minutes) FILTER (WHERE competition_type = 'international_cup')         AS minutes_europe,
               SUM(goals) AS goals, SUM(assists) AS assists,
               SUM(yellow_cards) AS yellow, SUM(red_cards) AS red,
               -- contre des adversaires beaucoup plus forts
               COUNT(*)     FILTER (WHERE opp_elo - own_elo >= {BIG_GAME_ELO_GAP})       AS n_big,
               SUM(minutes) FILTER (WHERE opp_elo - own_elo >= {BIG_GAME_ELO_GAP})       AS minutes_big,
               SUM(goals + assists) FILTER (WHERE opp_elo - own_elo >= {BIG_GAME_ELO_GAP}) AS ga_big,
               -- matchs à enjeu
               COUNT(*)     FILTER (WHERE stakes >= {stakes.HIGH_STAKES})                AS n_stakes,
               SUM(minutes) FILTER (WHERE stakes >= {stakes.HIGH_STAKES})                AS minutes_stakes,
               SUM(goals + assists) FILTER (WHERE stakes >= {stakes.HIGH_STAKES})        AS ga_stakes,
               -- forces moyennes, pondérées par les minutes
               SUM(own_elo * minutes) / NULLIF(SUM(minutes) FILTER (WHERE own_elo IS NOT NULL), 0) AS own_elo,
               SUM(opp_elo * minutes) / NULLIF(SUM(minutes) FILTER (WHERE opp_elo IS NOT NULL), 0) AS opp_elo
        FROM apps
        GROUP BY player_id, season, quarter
        """
    ).df()


def team_league_games(con):
    # Nombre de matchs de championnat joués par chaque club chaque trimestre :
    # sert à calculer la PART du temps de jeu possible (minutes / (matchs x 90)).
    season, quarter = _quarter_sql("g.date")
    return con.execute(
        f"""
        SELECT club_id, {season} AS season, {quarter} AS quarter, COUNT(*) AS team_league_games
        FROM (SELECT date, home_club_id AS club_id FROM tmdb.games WHERE competition_type = 'domestic_league'
              UNION ALL
              SELECT date, away_club_id FROM tmdb.games WHERE competition_type = 'domestic_league') g
        GROUP BY ALL
        """
    ).df()


# -----------------------------------------------------------------------------
# 2. La grille complète joueur x trimestre (y compris les trimestres à 0 minute)
# -----------------------------------------------------------------------------
def build_grid(con, player_ids):
    # Chaque joueur est suivi de son premier match senior (ou de sa première
    # valeur marchande) jusqu'à sa dernière valeur ou son dernier match.
    con.register("targets", pd.DataFrame({"player_id": player_ids}))
    span = con.execute(
        """
        SELECT player_id, MIN(d) AS first_date, MAX(d) AS last_date FROM (
            SELECT player_id, date AS d FROM tmdb.appearances JOIN targets USING (player_id)
            UNION ALL
            SELECT player_id, date FROM tmdb.player_valuations JOIN targets USING (player_id)
        ) GROUP BY player_id
        """
    ).df()
    first_t = max(config.FIRST_SEASON, 2012) * 4
    rows = []
    for p in span.itertuples():
        start = _to_index(p.first_date)
        end = _to_index(min(p.last_date, pd.Timestamp(config.TM_SNAPSHOT_DATE)))
        for t in range(max(start, first_t), end + 1):
            rows.append((p.player_id, t // 4, t % 4 + 1))
    return pd.DataFrame(rows, columns=["player_id", "season", "quarter"])


def _to_index(date):
    # Date -> index de trimestre continu (saison x 4 + trimestre - 1), pratique pour les boucles.
    date = pd.Timestamp(date)
    season = date.year if date.month >= 8 else date.year - 1
    quarter = next(q for q, months in config.QUARTERS.items() if date.month in months)
    return season * 4 + quarter - 1


# -----------------------------------------------------------------------------
# 3. Valeur marchande, transferts, fiche joueur
# -----------------------------------------------------------------------------
def attach_values(grid, con):
    # Valeur marchande "connue à la fin du trimestre" : la dernière valeur publiée
    # avant cette date (jointure "as-of"). On garde aussi sa date, pour savoir
    # si elle est récente ou ancienne.
    vals = con.execute(
        "SELECT player_id, date AS value_date, market_value_in_eur AS market_value, "
        "current_club_id AS value_club_id FROM tmdb.player_valuations ORDER BY date"
    ).df()
    grid = grid.copy()
    grid["q_end"] = [quarter_bounds(s, q)[1] for s, q in zip(grid.season, grid.quarter)]
    grid = grid.sort_values("q_end")
    # merge_asof exige des types strictement identiques des deux côtés
    grid["player_id"] = grid.player_id.astype("int64")
    grid["q_end"] = pd.to_datetime(grid.q_end).astype("datetime64[ns]")
    vals["player_id"] = vals.player_id.astype("int64")
    vals["value_date"] = pd.to_datetime(vals["value_date"]).astype("datetime64[ns]")
    out = pd.merge_asof(grid, vals.sort_values("value_date"), left_on="q_end", right_on="value_date",
                        by="player_id", direction="backward")
    return out


def attach_transfers(grid, con):
    # Transferts du trimestre : montant payé, et indicateur "a changé de club".
    season, quarter = _quarter_sql("transfer_date")
    tr = con.execute(
        f"""
        SELECT player_id, {season} AS season, {quarter} AS quarter,
               SUM(COALESCE(transfer_fee, 0)) AS transfer_fee, COUNT(*) AS n_transfers,
               arg_max(to_club_id, transfer_date) AS to_club_id
        FROM tmdb.transfers WHERE NOT regexp_matches(to_club_name, 'Yth|Youth|U1[0-9]|U2[0-3]|Jgd|B$')
        GROUP BY ALL
        """
    ).df()
    return grid.merge(tr, on=["player_id", "season", "quarter"], how="left")


def player_profiles(con):
    # Données fixes du joueur + parcours jeune (premier club de jeunes, nombre de clubs de jeunes).
    profiles = con.execute(
        """
        SELECT player_id, name, date_of_birth, country_of_citizenship AS nationality, position,
               sub_position, foot, height_in_cm AS height, contract_expiration_date, international_caps
        FROM tmdb.players
        """
    ).df()
    youth = con.execute(
        """
        SELECT player_id,
               arg_min(from_club_name, transfer_date) AS first_youth_club,
               COUNT(DISTINCT to_club_name) AS n_youth_clubs
        FROM tmdb.transfers WHERE regexp_matches(to_club_name, 'Yth|Youth|U1[0-9]|U2[0-3]')
        GROUP BY player_id
        """
    ).df()
    return profiles.merge(youth, on="player_id", how="left")


# -----------------------------------------------------------------------------
# 4. FotMob -> agrégats trimestriels (relié aux identifiants Transfermarkt)
# -----------------------------------------------------------------------------
FOTMOB_SUMS = {
    # nom final          : colonne FotMob
    "fm_minutes": "minutes_played", "xg": "expected_goals_xg", "npxg": "xg_non_penalty",
    "xa": "expected_assists_xa", "xgot": "expected_goals_on_target_xgot", "shots": "total_shots",
    "chances_created": "chances_created", "big_chances_created": "big_chances_created",
    "touches": "touches", "touches_box": "touches_in_opposition_box",
    "passes_acc": "accurate_passes", "passes_tot": "accurate_passes_total",
    "final_third_passes": "passes_into_final_third",
    "long_acc": "accurate_long_balls", "long_tot": "accurate_long_balls_total",
    "dribbles_acc": "successful_dribbles", "dribbles_tot": "successful_dribbles_total",
    "tackles": "tackles", "interceptions": "interceptions", "recoveries": "recoveries",
    "clearances": "clearances", "blocks": "blocks", "defensive_actions": "defensive_actions",
    "aerial_won": "aerial_duels_won", "aerial_tot": "aerial_duels_won_total",
    "duels_won": "duels_won", "duels_lost": "duels_lost", "dribbled_past": "dribbled_past",
    "fouls": "fouls_committed", "dispossessed": "dispossessed",
    "saves": "saves", "xgot_faced": "xgot_faced", "goals_conceded_gk": "goals_conceded",
    "errors_to_goal": "error_led_to_goal",
}


# Statistiques issues de la carte des tirs (xG) : FotMob ne les a qu'à partir de
# 2020/21 en 1re division (2021/22 en D2 anglaise et allemande, 2025/26 pour les
# autres D2). Dans un match sans carte des tirs, elles sont INCONNUES (et non nulles).
XG_COLUMNS = ["xg", "npxg", "xa", "xgot", "xgot_faced"]

# Listes d'absents FotMob : pour certaines saisons (2020/21 à 2023/24), FotMob les a
# réécrites après coup (joueurs listés chez un club qu'ils avaient quitté, retours
# prévus des années plus tard). On ne garde une saison que si au moins 70 % de
# ses absences concernent un joueur réellement au club ce trimestre-là.
ABSENCE_RELIABILITY = 0.70


def fotmob_quarterly():
    # Agrège les stats FotMob par joueur et trimestre, pour toutes les saisons
    # disponibles (complètes ou en cours de téléchargement), 1res et 2es divisions.
    # Renvoie (agrégats trimestriels, absences match par match à filtrer ensuite).
    if not matching.PLAYER_MAP_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()     # rapprochement pas encore fait : on continue sans FotMob
    id_map = matching.fm_to_tm()
    frames = []
    for key in {**config.LEAGUES, **config.SECOND_DIVISIONS}:
        for season in range(fm.FIRST_SEASON_D1, config.CURRENT_SEASON + 1):
            try:
                matches, players, _, lineups = fm.load_season(key, season)
            except Exception:
                continue          # saison pas encore listée en cache
            if players.empty:
                continue
            dates = matches.set_index("match_id")["date"]
            p = players.assign(date=pd.to_datetime(players.match_id.map(dates)).dt.tz_localize(None),
                               division=2 if key in config.SECOND_DIVISIONS else 1, league_key=key)
            # Absences (blessure / suspension) : une ligne par match manqué, avec l'équipe
            # concernée (pour vérifier ensuite que le joueur y était vraiment)
            absences = lineups[lineups.role == "unavailable"][["match_id", "team_id", "player_id", "absence_type"]]
            absences = absences.assign(date=pd.to_datetime(absences.match_id.map(dates)).dt.tz_localize(None),
                                       league_key=key, fm_season=season)
            # Joueurs ayant JOUÉ le match : une absence le même jour est une erreur
            on_sheet = set(zip(players.match_id, players.player_id))
            absences = absences[[(m, i) not in on_sheet for m, i in zip(absences.match_id, absences.player_id)]]
            frames.append((p, absences))
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    players = pd.concat([f[0] for f in frames], ignore_index=True)
    absences = pd.concat([f[1] for f in frames], ignore_index=True)
    for frame in (players, absences):
        frame["player_id"] = frame.player_id.map(id_map)          # FotMob -> Transfermarkt
        frame.dropna(subset=["player_id"], inplace=True)
        frame["player_id"] = frame.player_id.astype(int)
        frame["season"] = np.where(frame.date.dt.month >= 8, frame.date.dt.year, frame.date.dt.year - 1)
        frame["quarter"] = frame.date.dt.month.map({m: q for q, ms in config.QUARTERS.items() for m in ms})

    cols = {k: v for k, v in FOTMOB_SUMS.items() if v in players.columns}
    players = players.rename(columns={v: k for k, v in cols.items()})
    players[list(cols)] = players[list(cols)].apply(pd.to_numeric, errors="coerce")
    # Le match a-t-il des données xG ? (au moins un joueur avec un xG ou un xA renseigné)
    xg_present = players[[c for c in ("xg", "xa") if c in players]].notna().any(axis=1)
    players["match_has_xg"] = xg_present.groupby(players.match_id).transform("any")
    # Une case vide FotMob = la stat ne s'est pas produite (ex. aucun tir -> xG vide) -> 0,
    # SAUF les stats xG d'un match sans carte des tirs : elles restent inconnues.
    other = [c for c in cols if c not in XG_COLUMNS]
    players[other] = players[other].fillna(0)
    xg_cols = [c for c in XG_COLUMNS if c in cols]
    players[xg_cols] = players[xg_cols].fillna(0).where(players.match_has_xg)
    players["xg_minutes"] = players.fm_minutes.where(players.match_has_xg, 0)
    # Buts encaissés par le gardien dans les SEULS matchs avec xG (à comparer aux xGOT concédés)
    if "goals_conceded_gk" in players:
        players["goals_conceded_gk_xg"] = players.goals_conceded_gk.where(players.match_has_xg)
        xg_cols = xg_cols + ["goals_conceded_gk_xg"]
    players["rating_x_min"] = pd.to_numeric(players.get("fotmob_rating"), errors="coerce") * players.fm_minutes

    keys = ["player_id", "season", "quarter"]
    sums = players.groupby(keys)[other + ["xg_minutes", "rating_x_min"]].sum()
    xg_sums = players.groupby(keys)[xg_cols].sum(min_count=1)      # inconnu si aucun match avec xG
    extra = players.groupby(keys).agg(
        fm_matches=("match_id", "nunique"), n_positions=("position_id", "nunique"),
        minutes_d2=("fm_minutes", lambda s: s[players.loc[s.index, "division"] == 2].sum()))
    agg = pd.concat([sums, xg_sums, extra], axis=1).reset_index()
    agg["rating"] = agg.pop("rating_x_min") / agg.fm_minutes.replace(0, np.nan)
    return agg, absences


def absences_quarterly(absences, clubs):
    # Matchs manqués (blessure / suspension) par joueur et trimestre, en ne gardant
    # que les absences FIABLES : l'équipe FotMob est bien le club Transfermarkt du
    # joueur ce trimestre-là, dans une saison dont les listes d'absents sont cohérentes.
    # Une saison non fiable -> valeur inconnue (NaN), et non "aucune blessure".
    keys = ["player_id", "season", "quarter"]
    club_map = pd.read_parquet(matching.CLUB_MAP_PATH)
    a = absences.assign(tm_club=absences.team_id.map(dict(zip(club_map.fm_team_id, club_map.tm_club_id))))
    a = a.merge(clubs, on=keys, how="left")
    known = a.tm_club.notna() & a.club_id.notna()
    a["consistent"] = known & (a.tm_club == a.club_id)
    rate = a[known].groupby(["league_key", "fm_season"]).consistent.mean()
    reliable = rate[rate >= ABSENCE_RELIABILITY].index
    a = a[a.set_index(["league_key", "fm_season"]).index.isin(reliable) & a.consistent]
    miss = (a.assign(missed_injury=a.absence_type.eq("injury"), missed_suspension=a.absence_type.eq("suspension"))
            .groupby(keys)[["missed_injury", "missed_suspension"]].sum().reset_index())
    return miss, rate


def tournament_quarterly():
    # Minutes jouées en grand tournoi de sélections (FotMob, 2014 -> aujourd'hui),
    # par joueur Transfermarkt et trimestre.
    from collectors import tournaments as tn
    if not tn.OUT_PATH.exists():
        print("⚠ tournois non collectés (python -m collectors.tournaments) : minutes en sélection = 0")
        return pd.DataFrame(columns=["player_id", "season", "quarter", "minutes_national"])
    t = tn.load().dropna(subset=["tm_player_id"])
    t["player_id"] = t.tm_player_id.astype(int)
    t["season"] = np.where(t.date.dt.month >= 8, t.date.dt.year, t.date.dt.year - 1)
    t["quarter"] = t.date.dt.month.map({m: q for q, ms in config.QUARTERS.items() for m in ms})
    return (t.groupby(["player_id", "season", "quarter"]).minutes.sum()
            .rename("minutes_national").reset_index())


def _reliable_quarters(reliable):
    # (joueur, saison, trimestre) appartenant à une saison FotMob aux absences fiables :
    # on les repère via les matchs de ces saisons (joueurs présents sur les feuilles).
    rows = []
    for key, season in reliable:
        matches, players, _, _ = fm.load_season(key, season)
        d = pd.to_datetime(matches.set_index("match_id")["date"]).dt.tz_localize(None)
        p = players[["match_id", "player_id"]].assign(date=players.match_id.map(d))
        rows.append(p)
    if not rows:
        return set()
    p = pd.concat(rows)
    p["player_id"] = p.player_id.map(matching.fm_to_tm())
    p = p.dropna(subset=["player_id", "date"])
    season = np.where(p.date.dt.month >= 8, p.date.dt.year, p.date.dt.year - 1)
    quarter = p.date.dt.month.map({m: q for q, ms in config.QUARTERS.items() for m in ms})
    return set(zip(p.player_id.astype(int), season, quarter))


def wiki_quarterly():
    # Pages vues Wikipédia par trimestre (somme des 5 langues).
    if not wk.VIEWS_PATH.exists():
        return pd.DataFrame(columns=["player_id", "season", "quarter", "wiki_views"])
    v = pd.read_parquet(wk.VIEWS_PATH)
    month = v.month.str[4:6].astype(int)
    year = v.month.str[:4].astype(int)
    v["season"] = np.where(month >= 8, year, year - 1)
    v["quarter"] = month.map({m: q for q, ms in config.QUARTERS.items() for m in ms})
    return v.groupby(["player_id", "season", "quarter"]).views.sum().rename("wiki_views").reset_index()


# -----------------------------------------------------------------------------
# 5. Assemblage
# -----------------------------------------------------------------------------
def build():
    con = _connect()
    from collectors.wikipedia import target_players
    player_ids = target_players()

    grid = build_grid(con, player_ids)
    keys = ["player_id", "season", "quarter"]
    df = grid.merge(tm_quarterly(con, player_ids), on=keys, how="left")

    # Club du trimestre : celui des matchs joués, sinon celui de la valeur marchande
    df = attach_values(df, con)
    df["club_id"] = df.club_id.fillna(df.value_club_id)
    df = df.merge(team_league_games(con), on=["club_id", "season", "quarter"], how="left")
    df = attach_transfers(df, con)

    fmq, absences = fotmob_quarterly()
    if not fmq.empty:
        df = df.merge(fmq, on=keys, how="left")
        miss, rate = absences_quarterly(absences, df[keys + ["club_id"]].dropna())
        rate.rename("consistent_share").reset_index().to_parquet(ABSENCE_RELIABILITY_PATH)
        df = df.merge(miss, on=keys, how="left")
        # Absences connues seulement pour les trimestres couverts par une saison FIABLE :
        # ailleurs, la colonne reste vide (inconnue) même si FotMob a des stats de match.
        # (trimestres où le joueur a joué un match d'une saison fiable, ou y a une absence fiable)
        covered = _reliable_quarters(set(rate[rate >= ABSENCE_RELIABILITY].index))
        known = df.set_index(keys).index.isin(covered) | df.missed_injury.notna()
        for col in ("missed_injury", "missed_suspension"):
            df[col] = df[col].fillna(0).where(known)
        from features import onoff
        df = df.merge(onoff.build(), on=keys, how="left")
    df = df.merge(tournament_quarterly(), on=keys, how="left")
    df = df.merge(wiki_quarterly(), on=keys, how="left")
    df = df.merge(player_profiles(con), on="player_id", how="left")

    # Variables dérivées
    count_cols = ["matches", "starts", "minutes", "minutes_league", "minutes_europe", "minutes_national",
                  "goals", "assists", "yellow", "red", "n_big", "minutes_big", "ga_big",
                  "n_stakes", "minutes_stakes", "ga_stakes", "transfer_fee", "n_transfers"]
    df[count_cols] = df[count_cols].fillna(0)
    df["q_start"] = [quarter_bounds(s, q)[0] for s, q in zip(df.season, df.quarter)]
    df["age"] = (df.q_start - pd.to_datetime(df.date_of_birth)).dt.days / 365.25
    df["minutes_share"] = (df.minutes_league / (df.team_league_games * 90)).clip(0, 1)
    df["t"] = df.season * 4 + df.quarter - 1          # index de temps continu
    df["moved"] = df.n_transfers > 0

    df = df.sort_values(["player_id", "t"]).reset_index(drop=True)
    df.to_parquet(OUT_PATH)
    return df


if __name__ == "__main__":
    out = build()
    print(out.shape)
